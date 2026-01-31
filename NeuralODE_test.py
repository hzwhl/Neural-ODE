"""
[1] 当 checkpoint 中 ckpt["config"] 与当前脚本 Config 不一致时：默认策略 CKPT_CONFIG_POLICY="prefer_local" => 永远保留当前脚本 CFG，不会被 ckpt 覆盖, 同时会打印差异列表（local CFG vs ckpt config）
    例如：将 T0: float = 0.0, T1: float = 1.0 改为 T0: float = 0.0, T1: float = 100

"""

import os
import sys
import time
import platform
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from torchdiffeq import odeint_adjoint as odeint  # 与训练一致：强制 adjoint


# =========================
# (0) Config
# =========================
@dataclass
class Config:
    # ---- Reproducibility ----
    SEED: int = 42
    DETERMINISTIC: bool = False

    # ---- Paths ----
    DATA_ROOT: str = "./data"
    CKPT_PATH: str = r"G:\7.Python_DeepLearning\NeuralODE\runs_neural_ode\epoch0050_neural_ode_mnist.pth"
    SAVE_DIR: str = r"G:\7.Python_DeepLearning\NeuralODE\run"
    VIZ_DIRNAME: str = "test_viz"

    # ---- DataLoader ----
    TEST_BATCH_SIZE: int = 512
    NUM_WORKERS: int = 0

    # ---- ODE Solver (must match training for strict reproducibility) ----
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

    # ---- Visualization ----
    GRID_N: int = 25          # 5x5
    TOPK_WRONG: int = 20      # Top-K most confident wrong preds
    ENABLE_TOPK_WRONG: bool = False  # 是否启用 top-k wrong（遍历逐样本会慢）

    # ---- Verbose controls ----
    PRINT_MODEL_PARAM_DETAILS: bool = True   # 是否逐参数打印 name/shape/numel
    PRINT_MODEL_MODULE_REPR: bool = True     # 是否打印 model.__repr__()

    # ---- Checkpoint config policy ----
    # 取值：
    #   "prefer_local"  : 当前脚本 CFG 永远优先（只对比并打印差异，不覆盖）  <-- 你要的默认行为
    #   "prefer_ckpt"   : 用 ckpt["config"] 覆盖 CFG（用于严格复现训练；会尊重 blocklist）
    #   "ignore_ckpt"   : 完全忽略 ckpt["config"]（连对比都不做）
    CKPT_CONFIG_POLICY: str = "prefer_local"

    # 不允许 ckpt 覆盖的字段（仅 prefer_ckpt 时生效）
    CKPT_OVERRIDE_BLOCKLIST: Tuple[str, ...] = (
        "CKPT_PATH", "SAVE_DIR", "DATA_ROOT", "VIZ_DIRNAME",
        "NUM_WORKERS", "TEST_BATCH_SIZE",
        # 如需本地固定随机性，也可加：
        # "SEED", "DETERMINISTIC",
    )


CFG = Config()


# =========================
# (A) Pretty print helpers
# =========================
def _section(title: str):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


def _kv(k: str, v: Any, k_width: int = 28):
    print(f"{k:<{k_width}} : {v}")


def _print_dict_sorted(d: Dict[str, Any], k_width: int = 28):
    for k in sorted(d.keys()):
        _kv(k, d[k], k_width=k_width)


def _safe_resolve(p: str) -> str:
    try:
        return str(Path(p).expanduser().resolve())
    except Exception:
        return str(p)


def print_env_info():
    _section("Environment")
    _kv("time", time.strftime("%Y-%m-%d %H:%M:%S"))
    _kv("platform", platform.platform())
    _kv("python", sys.version.replace("\n", " "))
    _kv("torch", torch.__version__)
    _kv("torch.cuda.is_available", torch.cuda.is_available())
    _kv("torch.version.cuda", torch.version.cuda)
    _kv("torch.backends.cudnn.version", torch.backends.cudnn.version())
    _kv("cudnn.enabled", torch.backends.cudnn.enabled)
    _kv("cudnn.benchmark", torch.backends.cudnn.benchmark)
    _kv("cudnn.deterministic", torch.backends.cudnn.deterministic)

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        _kv("cuda.device_count", n)
        for i in range(n):
            prop = torch.cuda.get_device_properties(i)
            _kv(f"gpu[{i}].name", prop.name)
            _kv(f"gpu[{i}].capability", f"{prop.major}.{prop.minor}")
            _kv(f"gpu[{i}].total_mem_GB", f"{prop.total_memory / (1024**3):.2f}")


def print_cfg(cfg: Config, title: str = "Config (CFG)"):
    _section(title)
    d = asdict(cfg)

    # 额外打印：路径字段的绝对路径解析
    _kv("DATA_ROOT(abs)", _safe_resolve(d["DATA_ROOT"]))
    _kv("CKPT_PATH(abs)", _safe_resolve(d["CKPT_PATH"]))
    _kv("SAVE_DIR(abs)", _safe_resolve(d["SAVE_DIR"]))

    print("\n[All CFG fields]")
    _print_dict_sorted(d)


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


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def print_model_params(model: nn.Module):
    _section("Model Summary")
    stats = count_parameters(model)
    _kv("total_params", stats["total"])
    _kv("trainable_params", stats["trainable"])

    if CFG.PRINT_MODEL_MODULE_REPR:
        print("\n[model repr]")
        print(model)

    if CFG.PRINT_MODEL_PARAM_DETAILS:
        print("\n[parameter details: name | shape | numel | requires_grad]")
        for name, p in model.named_parameters():
            _kv(
                name,
                f"shape={tuple(p.shape)} | numel={p.numel()} | requires_grad={p.requires_grad}",
                k_width=44
            )


def summarize_checkpoint(ckpt: Any):
    _section("Checkpoint Summary")
    if not isinstance(ckpt, dict):
        _kv("ckpt_type", type(ckpt))
        print("Checkpoint is not a dict (maybe a raw state_dict).")
        return

    _kv("ckpt_keys", list(ckpt.keys()))

    # 常见字段（有就打印）
    for key in ["epoch", "step", "global_step", "best_acc", "best_loss", "best_metric", "time", "date"]:
        if key in ckpt:
            _kv(key, ckpt[key])

    # model/state dict 结构
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        _kv("model_state_dict.keys(sample)", list(ckpt["model"].keys())[:10])
        _kv("model_state_dict.num_keys", len(ckpt["model"]))
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        _kv("state_dict.num_keys", len(ckpt))

    # config
    if "config" in ckpt and isinstance(ckpt["config"], dict):
        _kv("config.num_keys", len(ckpt["config"]))
        _kv("config.keys(sample)", list(ckpt["config"].keys())[:20])


# =========================
# (B) Checkpoint config policy helpers
# =========================
def diff_ckpt_config_vs_local_cfg(cfg: Config, ckpt_cfg: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    """
    返回所有在 cfg 中存在、且与 ckpt_cfg 不同的字段差异：
    diffs[k] = (local_value, ckpt_value)
    """
    diffs: Dict[str, Tuple[Any, Any]] = {}
    for k, v_ckpt in ckpt_cfg.items():
        if not hasattr(cfg, k):
            continue
        v_local = getattr(cfg, k)
        if v_local != v_ckpt:
            diffs[k] = (v_local, v_ckpt)
    return diffs


def apply_ckpt_config_override(cfg: Config, ckpt_cfg: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    """
    用 ckpt["config"] 覆盖 cfg 的字段，但会尊重 cfg.CKPT_OVERRIDE_BLOCKLIST。
    仅在 CKPT_CONFIG_POLICY == "prefer_ckpt" 时调用。
    返回：overrides = {key: (old, new)}
    """
    overrides: Dict[str, Tuple[Any, Any]] = {}
    block = set(cfg.CKPT_OVERRIDE_BLOCKLIST)

    for k, v in ckpt_cfg.items():
        if not hasattr(cfg, k):
            continue
        if k in block:
            continue
        old = getattr(cfg, k)
        if old != v:
            setattr(cfg, k, v)
            overrides[k] = (old, v)
    return overrides


def handle_ckpt_config_policy(cfg: Config, ckpt: Any):
    """
    处理 ckpt["config"] 与本地 cfg 的关系：
    - prefer_local：只打印 diffs，不覆盖
    - prefer_ckpt：按 blocklist 覆盖 cfg，并打印覆盖项
    - ignore_ckpt：不对比不覆盖（仅提示）
    """
    _section("Checkpoint config vs Local CFG")
    _kv("CKPT_CONFIG_POLICY", cfg.CKPT_CONFIG_POLICY)

    if not (isinstance(ckpt, dict) and "config" in ckpt and isinstance(ckpt["config"], dict)):
        print("No ckpt['config'] found in checkpoint. Skip config compare/override.")
        return

    ckpt_cfg = ckpt["config"]

    if cfg.CKPT_CONFIG_POLICY == "ignore_ckpt":
        print("[Policy] ignore_ckpt -> Ignore ckpt['config'] completely.")
        return

    # 先对比差异（prefer_local / prefer_ckpt 都会打印差异）
    diffs = diff_ckpt_config_vs_local_cfg(cfg, ckpt_cfg)
    if len(diffs) == 0:
        print("No differences between ckpt['config'] and local CFG.")
    else:
        print("[Diffs] key : local(CFG)  !=  ckpt['config']")
        for k in sorted(diffs.keys()):
            v_local, v_ckpt = diffs[k]
            _kv(k, f"{v_local}  !=  {v_ckpt}", k_width=28)

    if cfg.CKPT_CONFIG_POLICY == "prefer_local":
        print("\n[Policy] prefer_local -> Keep local CFG. (No fields overridden)")
        return

    if cfg.CKPT_CONFIG_POLICY == "prefer_ckpt":
        _section("Apply ckpt['config'] override to CFG (prefer_ckpt)")
        _kv("override_blocklist", cfg.CKPT_OVERRIDE_BLOCKLIST)
        overrides = apply_ckpt_config_override(cfg, ckpt_cfg)
        if len(overrides) == 0:
            print("No CFG fields overridden by ckpt['config'] (after blocklist).")
        else:
            print("[Overridden fields] key : old(local) -> new(ckpt)")
            for k in sorted(overrides.keys()):
                old, new = overrides[k]
                _kv(k, f"{old}  ->  {new}", k_width=28)
        return

    # 兜底：未知策略
    print(f"[Warn] Unknown CKPT_CONFIG_POLICY={cfg.CKPT_CONFIG_POLICY!r}. Default to prefer_local behavior.")
    print("[Policy] prefer_local -> Keep local CFG. (No fields overridden)")


# =========================
# (1) Model definitions (must match training script)
# =========================
class ODEFuncConv(nn.Module):
    """
    Input/Output shape: (B, C, H, W)
    GroupNorm + Conv
    """
    def __init__(self, channels: int, hidden_channels: int, gn_groups: int):
        super().__init__()
        self.nfe = 0

        self.norm1 = nn.GroupNorm(gn_groups, channels)
        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(gn_groups, hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1)

        # same trick as training: weak initial dynamics
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
# (2) Load checkpoint
# =========================
def load_checkpoint_and_build_model(cfg: Config, device: torch.device) -> Tuple[nn.Module, Any]:
    ckpt_path = Path(cfg.CKPT_PATH)
    assert ckpt_path.exists(), f"Checkpoint not found: {cfg.CKPT_PATH}"

    ckpt = torch.load(str(ckpt_path), map_location=device)

    summarize_checkpoint(ckpt)

    # 关键：处理 ckpt config 与本地 cfg 的关系（默认 prefer_local，不覆盖）
    handle_ckpt_config_policy(cfg, ckpt)

    # 用“最终 CFG（默认就是本地 CFG）”建模
    model = NeuralODEMNIST(cfg).to(device)

    # 兼容两种保存格式：{'model': state_dict} 或直接 state_dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        _section("load_state_dict warnings")
        print("[Warn] load_state_dict not strict.")
        print("  Missing keys:", missing)
        print("  Unexpected keys:", unexpected)

    model.eval()
    return model, ckpt


# =========================
# (3) Evaluate
# =========================
@torch.no_grad()
def evaluate_full(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    nfe_meter = AverageMeter()

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[float] = []  # max prob for each sample

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        bs = x.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(accuracy_from_logits(logits, y), bs)

        # 每次 forward，ODEBlock 内部都会 reset nfe，所以这里的 nfe 是“本 batch 的 NFE”
        nfe_meter.update(float(model.odefunc.nfe), bs)

        all_preds.extend(pred.detach().cpu().tolist())
        all_targets.extend(y.detach().cpu().tolist())
        all_probs.extend(conf.detach().cpu().tolist())

    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
        "avg_nfe": nfe_meter.avg,
        "preds": np.array(all_preds, dtype=np.int64),
        "targets": np.array(all_targets, dtype=np.int64),
        "confs": np.array(all_probs, dtype=np.float32),
    }


def confusion_matrix_10(preds: np.ndarray, targets: np.ndarray, num_classes: int = 10) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[int(t), int(p)] += 1
    return cm


# =========================
# (4) Visualization
# =========================
def denorm_mnist(x: torch.Tensor) -> torch.Tensor:
    """
    x is normalized: (x - 0.1307) / 0.3081
    convert back to [0,1] approx for plotting
    """
    mean = 0.1307
    std = 0.3081
    return x * std + mean


@torch.no_grad()
def viz_random_grid(model: nn.Module, dataset, device: torch.device, save_path: str, n: int = 25):
    idxs = np.random.choice(len(dataset), size=n, replace=False).tolist()
    imgs = []
    labels = []
    for i in idxs:
        x, y = dataset[i]
        imgs.append(x)
        labels.append(int(y))

    x = torch.stack(imgs, dim=0).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)

    x_cpu = denorm_mnist(x.detach().cpu())

    side = int(np.sqrt(n))
    side = side if side * side == n else int(np.ceil(np.sqrt(n)))
    plt.figure(figsize=(10, 10))
    for i in range(n):
        ax = plt.subplot(side, side, i + 1)
        img = x_cpu[i, 0].numpy()
        ax.imshow(img, cmap="gray")
        t = labels[i]
        p = int(pred[i].item())
        c = float(conf[i].item())
        ax.set_title(f"T:{t} P:{p} ({c:.2f})", fontsize=10)
        ax.axis("off")
    plt.suptitle("Random MNIST Predictions (T=true, P=pred, conf=max softmax)", fontsize=14)
    plt.tight_layout()
    os.makedirs(str(Path(save_path).parent), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def viz_confusion_matrix(cm: np.ndarray, save_path: str):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (MNIST Test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.tight_layout()
    os.makedirs(str(Path(save_path).parent), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def viz_topk_wrong(model: nn.Module, dataset, device: torch.device, save_path: str, topk: int = 20):
    """
    遍历 test set（逐样本）收集 wrong 且置信度最高的 topk
    注意：逐样本跑会慢（每次都解ODE），建议默认关掉
    """
    wrong_samples = []  # (conf, idx, true, pred)
    for idx in range(len(dataset)):
        x, y = dataset[idx]
        x = x.unsqueeze(0).to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        conf = float(conf.item())
        pred = int(pred.item())
        true = int(y)

        if pred != true:
            wrong_samples.append((conf, idx, true, pred))

    if len(wrong_samples) == 0:
        print("[Info] No wrong samples found. Skip topk visualization.")
        return

    wrong_samples.sort(key=lambda t: t[0], reverse=True)
    wrong_samples = wrong_samples[:topk]

    n = len(wrong_samples)
    side = int(np.ceil(np.sqrt(n)))
    plt.figure(figsize=(12, 12))

    for i, (conf, idx, true, pred) in enumerate(wrong_samples):
        x, _ = dataset[idx]
        x = denorm_mnist(x).squeeze(0).numpy()

        ax = plt.subplot(side, side, i + 1)
        ax.imshow(x, cmap="gray")
        ax.set_title(f"idx {idx}\nT:{true} P:{pred} ({conf:.2f})", fontsize=10)
        ax.axis("off")

    plt.suptitle("Top-K Most Confident Wrong Predictions", fontsize=14)
    plt.tight_layout()
    os.makedirs(str(Path(save_path).parent), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


# =========================
# (5) Main
# =========================
def main():
    print_env_info()
    print_cfg(CFG, title="Config BEFORE loading checkpoint (local CFG)")

    set_seed(CFG.SEED, CFG.DETERMINISTIC)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _section("Runtime")
    _kv("device", device)
    _kv("CKPT_PATH", CFG.CKPT_PATH)
    _kv("CKPT_PATH(abs)", _safe_resolve(CFG.CKPT_PATH))
    _kv("SAVE_DIR", CFG.SAVE_DIR)
    _kv("SAVE_DIR(abs)", _safe_resolve(CFG.SAVE_DIR))
    _kv("DATA_ROOT", CFG.DATA_ROOT)
    _kv("DATA_ROOT(abs)", _safe_resolve(CFG.DATA_ROOT))

    # MNIST normalization (must match training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_set = datasets.MNIST(root=CFG.DATA_ROOT, train=False, download=True, transform=transform)

    # DataLoader 参数打印更全一点（便于你“终端输出所有参数信息”）
    pin_memory = (device.type == "cuda")
    worker_init = worker_init_fn if CFG.NUM_WORKERS > 0 else None

    test_loader = DataLoader(
        test_set,
        batch_size=CFG.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=pin_memory,
        worker_init_fn=worker_init,
    )

    _section("Data")
    _kv("test_set.size", len(test_set))
    _kv("batch_size", CFG.TEST_BATCH_SIZE)
    _kv("shuffle", False)
    _kv("num_workers", CFG.NUM_WORKERS)
    _kv("pin_memory", pin_memory)
    _kv("worker_init_fn", "worker_init_fn" if worker_init is not None else None)
    _kv("transform", transform)

    # load model (默认 prefer_local：不会用 ckpt 覆盖 CFG)
    model, ckpt = load_checkpoint_and_build_model(CFG, device)

    # 这里再打印一次最终 CFG（默认仍是本地 CFG；若你切 prefer_ckpt 才会变化）
    print_cfg(CFG, title="Config AFTER loading checkpoint (final used by THIS script)")

    print_model_params(model)

    # evaluate
    _section("Evaluation")
    metrics = evaluate_full(model, test_loader, device)
    print(f"[MNIST Test] loss = {metrics['loss']:.6f}, acc = {metrics['acc']:.6f}, avg NFE = {metrics['avg_nfe']:.2f}")

    preds = metrics["preds"]
    targets = metrics["targets"]

    # per-class accuracy
    print("[Per-class Acc]")
    for k in range(10):
        mask = (targets == k)
        acc_k = (preds[mask] == targets[mask]).mean() if mask.any() else 0.0
        print(f"  class {k}: {float(acc_k):.4f}")

    # visualization directory（严格用你本次 CFG.SAVE_DIR）
    viz_dir = Path(CFG.SAVE_DIR) / CFG.VIZ_DIRNAME
    viz_dir.mkdir(parents=True, exist_ok=True)
    _section("Visualization")
    _kv("viz_dir", str(viz_dir))
    _kv("viz_dir(abs)", str(viz_dir.resolve()))

    # (1) random grid
    grid_path = viz_dir / "random_grid_5x5.png"
    viz_random_grid(model, test_set, device, str(grid_path), n=CFG.GRID_N)
    print("[Saved]", str(grid_path))

    # (2) confusion matrix
    cm = confusion_matrix_10(preds, targets, num_classes=10)
    cm_path = viz_dir / "confusion_matrix.png"
    viz_confusion_matrix(cm, str(cm_path))
    print("[Saved]", str(cm_path))

    # (3) top-k wrong（可选，默认关）
    if CFG.ENABLE_TOPK_WRONG:
        wrong_path = viz_dir / f"top{CFG.TOPK_WRONG}_wrong.png"
        viz_topk_wrong(model, test_set, device, str(wrong_path), topk=CFG.TOPK_WRONG)
        print("[Saved]", str(wrong_path))

    print("\nDone.")


if __name__ == "__main__":
    main()
