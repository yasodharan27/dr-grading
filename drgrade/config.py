"""Reads config.yaml (or config.local.yaml) and resolves its paths."""
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = (42, 123, 2026)
ARMS = ("crop", "direct")
GRADES = ("No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR")

#: Release file names of the models (scripts/download_weights.py puts them in `weights_dir`).
WEIGHT_FILES = {
    "quality": "quality_efficientnet_b0_eyeq.keras",
    "vessels": "vessels_lwnet_wnet_drive.pth",
    "lesions": "lesions_unet_se_resnet101.pt",
    **{f"grader_{arm}_{seed}": f"grader_{arm}_seed{seed}.weights.h5" for arm in ARMS for seed in SEEDS},
    "convnext_imagenet": "convnext_tiny_imagenet_notop.h5",          # training only
}


def default_config_path():
    local = os.path.join(ROOT, "config.local.yaml")
    return local if os.path.exists(local) else os.path.join(ROOT, "config.yaml")


#: Environment variables that override config.yaml (useful in containers: no rebuild to change a path).
ENV_OVERRIDES = {"DRGRADE_WEIGHTS_DIR": "weights_dir", "DRGRADE_OUTPUT_DIR": "output_dir", "DRGRADE_DEMO_IMAGES": "demo_images",
                 "DRGRADE_DEVICE": "device", "DRGRADE_ARM": "arm", "DRGRADE_RELIABILITY": "reliability_calibration"}


def load_config(path=None):
    """The configuration as a dict with absolute paths; DRGRADE_* environment variables override the file."""
    path = os.path.abspath(path or os.environ.get("DRGRADE_CONFIG") or default_config_path())
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    base = os.path.dirname(path)

    def resolve(p):
        return p if p is None or os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

    cfg["config_path"] = path
    for variable, key in ENV_OVERRIDES.items():
        if os.environ.get(variable):
            cfg[key] = os.environ[variable]
    cfg.setdefault("reliability_calibration", "data/reliability_calibration.json")
    for key in ("weights_dir", "output_dir", "demo_images", "reliability_calibration"):
        cfg[key] = resolve(cfg.get(key))
    cfg["datasets"] = {k: resolve(v) for k, v in (cfg.get("datasets") or {}).items()}
    cfg.setdefault("device", "cpu")
    cfg.setdefault("arm", "crop")
    if cfg["arm"] not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {cfg['arm']!r}")
    return cfg


def cpu_threads():
    """Threads for the CPU stages: OMP_NUM_THREADS when set (containers often see more cores than they may use),
    otherwise every core."""
    value = os.environ.get("OMP_NUM_THREADS", "")
    return int(value) if value.isdigit() and int(value) > 0 else max(1, os.cpu_count() or 1)


def configure_threads():
    """Applies `cpu_threads()` to PyTorch and TensorFlow (call before TensorFlow runs anything)."""
    import tensorflow as tf
    import torch
    n = cpu_threads()
    torch.set_num_threads(n)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(n)
        tf.config.threading.set_inter_op_parallelism_threads(min(2, n))
    except RuntimeError:                                                   # TensorFlow already initialised: keep its setting
        pass
    return n


def weight_path(cfg, key):
    """Path of one model file; raises with a hint when it has not been downloaded."""
    path = os.path.join(cfg["weights_dir"], WEIGHT_FILES[key])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is missing: run `python scripts/download_weights.py` first")
    return path
