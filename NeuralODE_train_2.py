"""
[1] 使用 dopri5 自适应步长的 Runge-Kutta 方法
[2] 使用 adjoint (省显存)
[3] 动态学习率：连续 15 轮 loss 不下降, lr 减半
[4] 保存 best 模型 + 每 50 轮额外保存一次模型
[5] 从现有模型中加载权重参数继续训练
"""


import os
import time
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchdiffeq import odeint_adjoint as odeint  # 强制使用 adjoint


# =========================
# (0) 所有超参数写在这里
# =========================
@dataclass
class Config:
    # ---- Reproducibility ----
    SEED: int = 42
    DETERMINISTIC: bool = False  # True 会更可复现，但可能更慢

    # ---- Paths ----
    DATA_ROOT: str = "./data"
    SAVE_DIR: str = "./runs_neural_ode"
    SAVE_BEST: bool = True
    BEST_NAME: str = "best_neural_ode_mnist.pth"

    # ---- Resume Training（新增：按你的要求）----
    RESUME: bool = True
    RESUME_PATH: str = r"G:\7.Python_DeepLearning\NeuralODE\runs_neural_ode\epoch0050_neural_ode_mnist.pth"
    LOAD_OPTIMIZER: bool = False # 继续训练建议 True；只想加载权重可设 False
    #####
    #  LOAD_OPTIMIZER=True ⇒ 继续训（完全续上）：权重 + 动量 + step + lr + ……
    # LOAD_OPTIMIZER=False ⇒ 只加载权重：optimizer 用新建的（lr=CFG.LR）

    # ---- Training ----
    EPOCHS: int = 5000
    BATCH_SIZE: int = 256
    TEST_BATCH_SIZE: int = 512
    LR: float = 1e-3
    WEIGHT_DECAY: float = 0.0
    NUM_WORKERS: int = 0
    GRAD_CLIP: Optional[float] = 1.0  # None 表示不裁剪
    USE_AMP: bool = False  # 自适应 ODE 求解 + fp16 有时会不稳，默认关

    # ---- Dynamic LR ----
    LR_PATIENCE: int = 15     # 连续多少轮 loss 不下降
    LR_FACTOR: float = 0.5    # 降低倍数（减半）
    LR_EPS: float = 1e-6      # 判断“下降”的最小阈值（避免浮点抖动）

    # ---- Checkpoint Periodic Save ----
    SAVE_EVERY_EPOCHS: int = 50  # 每 50 轮额外保存一次

    # ---- ODE Solver ----
    METHOD: str = "dopri5"
    RTOL: float = 1e-3
    ATOL: float = 1e-4
    ADJOINT_RTOL: float = 1e-3
    ADJOINT_ATOL: float = 1e-4
    MAX_NUM_STEPS: int = 1000

    # ---- Time interval ----
    T0: float = 0.0
    T1: float = 1.0

    # ---- Model ----
    STEM_C: int = 64
    ODE_HIDDEN_C: int = 64
    GN_GROUPS: int = 8


CFG = Config()


# =========================
# (1) Utils
# =========================
def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int):
    seed = CFG.SEED + worker_id
    np.random.seed(seed)
    random.seed(seed)


@dataclass
class AverageMeter:
    sum: float = 0.0
    count: int = 0

    def update(self, val, n=1):
        self.sum += float(val) * int(n)
        self.count += int(n)

    @property
    def avg(self):
        return self.sum / max(1, self.count)


def accuracy(logits, targets):
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()


def save_checkpoint(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def set_optimizer_lr(optimizer: torch.optim.Optimizer, new_lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = float(new_lr)


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    """
    很关键：optimizer.load_state_dict 后，state 里的动量/二阶矩往往还在 CPU，
    在 CUDA 上继续训练会 device mismatch。
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def load_checkpoint_resume(
    ckpt_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    load_optimizer: bool = True,
):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[Resume] checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "model" not in ckpt:
        raise KeyError(f"[Resume] checkpoint missing key 'model': {ckpt_path}")

    # 1) load model weights
    model.load_state_dict(ckpt["model"], strict=True)

    # 2) load optimizer (optional)
    if load_optimizer and (optimizer is not None) and ("optimizer" in ckpt):
        optimizer.load_state_dict(ckpt["optimizer"])
        optimizer_to_device(optimizer, device)

        # 有些保存/加载后 lr 可能被 state_dict 覆盖或不一致，优先用 checkpoint 里记录的 current_lr（若存在）
        if "current_lr" in ckpt:
            set_optimizer_lr(optimizer, float(ckpt["current_lr"]))

    # 3) restore training states
    ckpt_epoch = int(ckpt.get("epoch", 0))
    best_acc = float(ckpt.get("best_acc", 0.0))
    best_monitored_loss = float(ckpt.get("monitored_best_loss", float("inf")))
    current_lr = float(ckpt.get("current_lr", get_current_lr(optimizer) if optimizer is not None else CFG.LR))

    # bad_epochs 没保存的话，从 0 重新计数最稳妥
    bad_epochs = 0

    info = {
        "ckpt_epoch": ckpt_epoch,
        "start_epoch": ckpt_epoch + 1,
        "best_acc": best_acc,
        "best_monitored_loss": best_monitored_loss,
        "current_lr": current_lr,
        "bad_epochs": bad_epochs,
    }
    return info


# =========================
# (2) ODE Function
# =========================
class ODEFuncConv(nn.Module):
    """
    输入/输出形状一致：(B, C, H, W)
    用 GroupNorm + Conv，batch 小也稳
    """
    def __init__(self, channels: int, hidden_channels: int, gn_groups: int):
        super().__init__()
        self.nfe = 0

        self.norm1 = nn.GroupNorm(gn_groups, channels)
        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(gn_groups, hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1)

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = F.relu(out, inplace=True)
        out = self.conv1(out)

        out = self.norm2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        return out


# =========================
# (3) ODEBlock
# =========================
class ODEBlock(nn.Module):
    def __init__(self, odefunc: nn.Module, cfg: Config):
        super().__init__()
        self.odefunc = odefunc
        self.cfg = cfg

    def forward(self, x):
        self.odefunc.nfe = 0

        t = torch.tensor([self.cfg.T0, self.cfg.T1], device=x.device, dtype=x.dtype)
        options = {"max_num_steps": int(self.cfg.MAX_NUM_STEPS)}

        out = odeint(
            self.odefunc,
            x,
            t,
            method=self.cfg.METHOD,
            rtol=self.cfg.RTOL,
            atol=self.cfg.ATOL,
            options=options,
            adjoint_rtol=self.cfg.ADJOINT_RTOL,
            adjoint_atol=self.cfg.ADJOINT_ATOL,
            adjoint_method=self.cfg.METHOD,
            adjoint_options=options,
        )
        return out[-1]


# =========================
# (4) Model
# =========================
class NeuralODEMNIST(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        C = cfg.STEM_C

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),   # 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(32, C, 4, stride=2, padding=1),   # 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 4, stride=2, padding=1),    # 7x7
            nn.ReLU(inplace=True),
        )

        self.odefunc = ODEFuncConv(channels=C, hidden_channels=cfg.ODE_HIDDEN_C, gn_groups=cfg.GN_GROUPS)
        self.odeblock = ODEBlock(self.odefunc, cfg)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C, 10),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.odeblock(x)
        x = self.head(x)
        return x


# =========================
# (5) Train / Eval
# =========================
def train_one_epoch(model, loader, optimizer, device, use_amp: bool, grad_clip: Optional[float]):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    nfe_meter = AverageMeter()

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            logits = model(x)
            loss = F.cross_entropy(logits, y)

        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))

        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(accuracy(logits.detach(), y), bs)
        nfe_meter.update(model.odefunc.nfe, bs)

    return loss_meter.avg, acc_meter.avg, nfe_meter.avg


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    nfe_meter = AverageMeter()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        bs = x.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(accuracy(logits, y), bs)
        nfe_meter.update(model.odefunc.nfe, bs)

    return loss_meter.avg, acc_meter.avg, nfe_meter.avg


# =========================
# (6) Main
# =========================
def main():
    set_seed(CFG.SEED, CFG.DETERMINISTIC)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CFG.SAVE_DIR, exist_ok=True)
    best_path = os.path.join(CFG.SAVE_DIR, CFG.BEST_NAME)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST(root=CFG.DATA_ROOT, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=CFG.DATA_ROOT, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_set,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=worker_init_fn if CFG.NUM_WORKERS > 0 else None,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=CFG.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=worker_init_fn if CFG.NUM_WORKERS > 0 else None,
    )

    model = NeuralODEMNIST(CFG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)

    print("=" * 80)
    print(f"Device: {device}")
    print(f"Solver: method={CFG.METHOD} (fixed), adjoint=True (fixed)")
    print(f"rtol={CFG.RTOL}, atol={CFG.ATOL}, adjoint_rtol={CFG.ADJOINT_RTOL}, adjoint_atol={CFG.ADJOINT_ATOL}")
    print(f"max_num_steps={CFG.MAX_NUM_STEPS}, time=[{CFG.T0}, {CFG.T1}]")
    print(f"Train: epochs={CFG.EPOCHS}, batch={CFG.BATCH_SIZE}, lr={CFG.LR}, wd={CFG.WEIGHT_DECAY}")
    print(f"AMP: {CFG.USE_AMP}, grad_clip={CFG.GRAD_CLIP}")
    print(f"Dynamic LR: patience={CFG.LR_PATIENCE}, factor={CFG.LR_FACTOR}, eps={CFG.LR_EPS}")
    print(f"Periodic Save: every {CFG.SAVE_EVERY_EPOCHS} epochs")
    print(f"Resume: {CFG.RESUME}, path={CFG.RESUME_PATH}, load_optimizer={CFG.LOAD_OPTIMIZER}")
    print("=" * 80)

    # -------------------------
    # Resume states（新增）
    # -------------------------
    start_epoch = 1
    best_acc = 0.0
    best_monitored_loss = float("inf")
    bad_epochs = 0

    if CFG.RESUME:
        info = load_checkpoint_resume(
            ckpt_path=CFG.RESUME_PATH,
            model=model,
            optimizer=optimizer,
            device=device,
            load_optimizer=CFG.LOAD_OPTIMIZER,
        )
        start_epoch = int(info["start_epoch"])
        best_acc = float(info["best_acc"])
        best_monitored_loss = float(info["best_monitored_loss"])
        bad_epochs = int(info["bad_epochs"])

        print("=" * 80)
        print("[Resume] Loaded checkpoint successfully:")
        print(f"  ckpt_epoch          = {info['ckpt_epoch']}")
        print(f"  start_epoch         = {start_epoch}")
        print(f"  best_acc            = {best_acc:.6f}")
        print(f"  best_monitored_loss = {best_monitored_loss:.6f}")
        print(f"  current_lr          = {get_current_lr(optimizer):.6e}")
        print("=" * 80)

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(start_epoch, CFG.EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_acc, tr_nfe = train_one_epoch(
            model, train_loader, optimizer, device,
            use_amp=CFG.USE_AMP,
            grad_clip=CFG.GRAD_CLIP
        )
        te_loss, te_acc, te_nfe = evaluate(model, test_loader, device)

        dt = time.time() - t0

        # (A) 动态学习率：监控 te_loss
        lr_reduced = False
        if te_loss < best_monitored_loss - float(CFG.LR_EPS):
            best_monitored_loss = te_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(CFG.LR_PATIENCE):
                old_lr = get_current_lr(optimizer)
                new_lr = old_lr * float(CFG.LR_FACTOR)
                set_optimizer_lr(optimizer, new_lr)
                bad_epochs = 0
                lr_reduced = True

        # (B) 保存 best checkpoint（按 te_acc）
        saved_best = False
        if te_acc > best_acc:
            best_acc = te_acc
            if CFG.SAVE_BEST:
                ckpt = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": CFG.__dict__,
                    "monitored_best_loss": best_monitored_loss,
                    "current_lr": get_current_lr(optimizer),
                }
                save_checkpoint(best_path, ckpt)
                saved_best = True

        # (C) 每 50 轮额外保存一次
        saved_periodic = False
        if int(CFG.SAVE_EVERY_EPOCHS) > 0 and (epoch % int(CFG.SAVE_EVERY_EPOCHS) == 0):
            periodic_name = f"epoch{epoch:04d}_neural_ode_mnist.pth"
            periodic_path = os.path.join(CFG.SAVE_DIR, periodic_name)
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
                "config": CFG.__dict__,
                "monitored_best_loss": best_monitored_loss,
                "current_lr": get_current_lr(optimizer),
            }
            save_checkpoint(periodic_path, ckpt)
            saved_periodic = True

        curr_lr = get_current_lr(optimizer)
        extra = []
        if saved_best:
            extra.append("best_saved")
        if saved_periodic:
            extra.append("periodic_saved")
        if lr_reduced:
            extra.append(f"lr_halved-> {curr_lr:.2e}")

        extra_str = ("  [" + ", ".join(extra) + "]") if len(extra) > 0 else ""

        print(
            f"Epoch {epoch:04d} | "
            f"train loss {tr_loss:.4f}, acc {tr_acc:.4f}, NFE {tr_nfe:.1f} | "
            f"test loss {te_loss:.4f}, acc {te_acc:.4f}, NFE {te_nfe:.1f} | "
            f"lr {curr_lr:.2e} | "
            f"time {dt:.1f}s | best_acc {best_acc:.4f}"
            f"{extra_str}"
        )

    print("Done.")
    if CFG.SAVE_BEST:
        print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
