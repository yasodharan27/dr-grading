"""Training of Stage 4, lesion segmentation: U-Net with an ImageNet SE-ResNet-101 encoder (segmentation_models_pytorch)
for MA, HE, EX and SE, on IDRiD (44 training images, 10 validation) plus TJDR (443 training, 110 validation).

Data: the Stage-2 image of each raw file is resized to 1536 x 1536 (uint8) with its four binary class masks (resized
and re-binarised at 0.5) and cached as one .npz per image. TJDR's official TEST split is used as the validation set,
so the best checkpoint is selected on it: its numbers are selection-set numbers, not a held-out evaluation.

Protocol: 512 x 512 patches cut from the 1536 images, centred on a random lesion pixel of a random class with
probability 0.5; batches of 8 drawn half from IDRiD and half from TJDR (with replacement); random flips; partial-label
loss (per-class BCE with positive weight clip(sqrt(neg / pos), 1, 20) from the training pixels, plus a batch-pooled
soft Dice, only over the images that annotate the class); AdamW (decoder learning rate 3e-4, encoder 0.1 x, weight
decay 1e-4 on matrices only), 500 warm-up steps then cosine decay to 1 % over 12,000 iterations, gradient clipping
at 1.0, an exponential moving average of the weights (decay 0.999) that is validated and exported, mixed precision
on GPU. Validation every 1,000 steps on the full 1536 frames; selection score = mean over IDRiD-val and the two
TJDR cameras of the four-class mean pixel AUPR.

Resumable: `latest.pt` is written every 500 steps; batches are a pure function of (seed, step).
"""
import copy
import dataclasses
import json
import os
import time

import numpy as np

from . import data, lesions, preprocess
from .training import atomic_json, read_json, seed_everything

CLASSES = lesions.CLASSES
TJDR_MASK_CODES = {"MA": 3, "HE": 2, "EX": 1, "SE": 4}            # palette indices of the TJDR annotation PNGs
ENCODER_REPO, ENCODER_FILE = "smp-hub/se_resnet101.imagenet", "model.safetensors"
ENCODER_REVISION = "71fe95cc0a27f444cf83671f354de02dc741b18b"
ENCODER_BYTES = 197_790_168
SMP_VERSION = "0.5.0"
SIZE = lesions.INFERENCE_SIZE
AP_BINS = 10000
SELECTION_GROUPS = ("IDRiD-val", "TJDR-val-TRC50DX", "TJDR-val-CLARUS500")


@dataclasses.dataclass
class TrainConfig:
    seed: int = 42
    iterations: int = 12000
    warmup_steps: int = 500
    batch_size: int = 8
    patch: int = 512
    p_lesion: float = 0.5
    decoder_lr: float = 3e-4
    encoder_lr_factor: float = 0.1
    weight_decay: float = 1e-4
    min_lr_factor: float = 0.01
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    amp: bool = True
    flips: bool = True
    val_every: int = 1000
    checkpoint_every: int = 500
    num_workers: int = 2


class LesionTrainingError(RuntimeError):
    pass


# --------------------------------------------------------------------------- data cache

def resize_mask(mask, size=SIZE, threshold=0.5):
    """Binary native mask -> (size, size) uint8 {0, 1}: bilinear (anti-aliased) resize, re-binarised at `threshold`."""
    from skimage.transform import resize
    m = (np.asarray(mask) > 0).astype(np.float32)
    return (resize(m, (size, size), order=1, mode="reflect", anti_aliasing=True, preserve_range=True) >= threshold).astype(np.uint8)


def read_idrid_masks(cfg, image_id):
    """(4, H, W) uint8 {0, 1}; a missing mask file means the lesion is absent from that image."""
    from PIL import Image
    masks, shape = [], None
    for c in CLASSES:
        path = data.idrid_seg_mask_path(cfg, image_id, c)
        if os.path.exists(path):
            with Image.open(path) as im:
                m = np.asarray(im)
            m = m[..., 0] if m.ndim == 3 else m
            if not set(np.unique(m).tolist()) <= {0, 1, 255}:
                raise LesionTrainingError(f"{path}: unexpected mask values")
            masks.append((m > 0).astype(np.uint8))
            shape = m.shape
        else:
            masks.append(None)
    if shape is None:
        raise LesionTrainingError(f"{image_id}: no lesion mask files")
    return np.stack([m if m is not None else np.zeros(shape, np.uint8) for m in masks])


def read_tjdr_masks(cfg, official_split, image_id):
    """(4, H, W) uint8 {0, 1} from the palette-index PNG (read as indices, never converted to luminance)."""
    from PIL import Image
    with Image.open(data.tjdr_mask_path(cfg, official_split, image_id)) as im:
        if im.mode != "P":
            raise LesionTrainingError(f"{image_id}: mask mode {im.mode!r}, expected a palette image")
        index = np.asarray(im)
    if not set(np.unique(index).tolist()) <= {0, 1, 2, 3, 4}:
        raise LesionTrainingError(f"{image_id}: mask values outside 0-4")
    return np.stack([(index == TJDR_MASK_CODES[c]).astype(np.uint8) for c in CLASSES])


def cache_entries(cfg, limit=None):
    """[(dataset, role, official split, image id, camera)]: the IDRiD 44 / 10 split and the usable TJDR images
    (official train -> role train, official test -> role val). `limit`: first N per (dataset, role)."""
    rows = []
    ids = data.idrid_seg_ids()
    for role in ("train", "val"):
        rows += [("IDRiD", role, "a. Training Set", i, "IDRiD-Kowa") for i in ids[role][:limit]]
    t = data.tjdr_entries()
    for role in ("train", "val"):
        part = t[t["role"] == role]
        if limit:                                                     # both cameras in a smoke subset
            part = part.groupby("camera", group_keys=False).head(max(1, limit // 2))
        rows += [("TJDR", role, r.official_split, r.image_id, f"TJDR-{r.camera}") for r in part.itertuples()]
    return rows


def sample_name(dataset, role, image_id):
    return f"{dataset}__{role}__{image_id}.npz"


def build_cache(cfg, out_dir, limit=None, idrid_jpeg=True, log=print):
    """Writes one .npz per image (rgb uint8 1536^2 x 3, masks uint8 4 x 1536^2, meta) and `manifest.json`; images
    already written are kept, so an interrupted build resumes. `idrid_jpeg`: the IDRiD Stage-2 images pass through a
    JPEG round trip (quality 95), as they did when the released model was trained (the Stage-2 files were written as
    .jpg); False keeps them lossless, as at inference."""
    os.makedirs(out_dir, exist_ok=True)
    entries = cache_entries(cfg, limit)
    files = {}
    for n, (dataset, role, split, image_id, camera) in enumerate(entries, 1):
        name = sample_name(dataset, role, image_id)
        path = os.path.join(out_dir, name)
        meta = {"dataset": dataset, "role": role, "split": split, "image_id": image_id, "camera": camera, "classes": list(CLASSES)}
        files[name] = meta
        if os.path.exists(path):
            continue
        if dataset == "IDRiD":
            rgb = preprocess.stage2(data.idrid_seg_image_path(cfg, image_id), jpeg_round_trip=idrid_jpeg)
            masks = read_idrid_masks(cfg, image_id)
        else:
            rgb = preprocess.stage2(data.tjdr_image_path(cfg, split, image_id))
            masks = read_tjdr_masks(cfg, split, image_id)
        if masks.shape[1:] != rgb.shape[:2]:
            raise LesionTrainingError(f"{image_id}: mask {masks.shape[1:]} vs image {rgb.shape[:2]}")
        rgb_1536 = np.rint(lesions.resize_full_frame(rgb, SIZE) * 255.0).astype(np.uint8)
        masks_1536 = np.stack([resize_mask(m) for m in masks])
        tmp = path + ".part"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, rgb=rgb_1536, masks=masks_1536, meta=np.asarray(json.dumps(meta, sort_keys=True)))
        os.replace(tmp, path)
        if n % 25 == 0:
            log(f"  cache {n}/{len(entries)}")
    counts = {}
    for m in files.values():
        counts[f"{m['dataset']}/{m['role']}"] = counts.get(f"{m['dataset']}/{m['role']}", 0) + 1
    manifest = {"size": SIZE, "classes": list(CLASSES), "counts": counts, "files": files, "idrid_jpeg_round_trip": bool(idrid_jpeg),
                "complete": counts == {"IDRiD/train": 44, "IDRiD/val": 10, "TJDR/train": 443, "TJDR/val": 110}}
    atomic_json(os.path.join(out_dir, "manifest.json"), manifest)
    return manifest


class TrainingCache:
    def __init__(self, directory, require_complete=True):
        self.directory = directory
        path = os.path.join(directory, "manifest.json")
        if not os.path.exists(path):
            raise LesionTrainingError(f"{directory} has no manifest.json: build the cache first")
        self.manifest = read_json(path)
        if require_complete and not self.manifest["complete"]:
            raise LesionTrainingError(f"the cache is incomplete: {self.manifest['counts']}; pass --allow-partial for a smoke test")

    def names(self, role, dataset=None):
        return sorted(k for k, v in self.manifest["files"].items() if v["role"] == role and (dataset is None or v["dataset"] == dataset))

    def load(self, name):
        with np.load(os.path.join(self.directory, name), allow_pickle=False) as d:
            return d["rgb"], d["masks"], json.loads(str(d["meta"]))


# --------------------------------------------------------------------------- sampling

def fov_mask(rgb01, threshold=10.0 / 255.0):
    """Field of view: pixels whose brightest channel exceeds the black-border level."""
    return (np.asarray(rgb01, np.float32).max(axis=-1) > threshold).astype(np.uint8)


def sample_patch_origin(rng, masks, patch=512, p_lesion=0.5):
    """(y0, x0) of a patch: with probability `p_lesion` centred (clamped to the image) on a random positive pixel of a
    random class that has positives, otherwise uniform. `masks`: (K, H, W)."""
    k, h, w = masks.shape
    if h < patch or w < patch:
        raise ValueError(f"image {h}x{w} smaller than the patch {patch}")
    candidates = [c for c in range(k) if masks[c].any()]
    if candidates and rng.random() < p_lesion:
        c = int(rng.choice(candidates))
        ys, xs = np.nonzero(masks[c])
        j = int(rng.integers(len(ys)))
        return int(np.clip(ys[j] - patch // 2, 0, h - patch)), int(np.clip(xs[j] - patch // 2, 0, w - patch))
    return int(rng.integers(0, h - patch + 1)), int(rng.integers(0, w - patch + 1))


def balanced_batch_plan(names_by_dataset, batch_size, step, seed):
    """The batch for `step`: batch_size / 2 images drawn uniformly (with replacement) from each dataset, each with
    its own seed. Counter-based, so any step can be reproduced after a resume."""
    datasets = sorted(names_by_dataset)
    if batch_size % len(datasets):
        raise ValueError(f"batch_size {batch_size} is not divisible by {len(datasets)} datasets")
    rng = np.random.default_rng([int(seed), int(step)])
    plan = []
    for d in datasets:
        names = names_by_dataset[d]
        for k in rng.integers(0, len(names), batch_size // len(datasets)):
            plan.append((names[int(k)], int(rng.integers(0, 2 ** 31 - 1))))
    return plan


class PatchDataset:
    """Map-style dataset keyed by (cache file, item seed): a class-aware patch with random flips, ImageNet-normalised."""

    def __init__(self, cache, patch=512, p_lesion=0.5, flips=True):
        self.cache, self.patch, self.p_lesion, self.flips = cache, patch, p_lesion, flips

    def __getitem__(self, key):
        import torch
        name, item_seed = key
        rgb, masks, meta = self.cache.load(name)
        rng = np.random.default_rng(int(item_seed))
        y0, x0 = sample_patch_origin(rng, masks, self.patch, self.p_lesion)
        p = self.patch
        img = rgb[y0:y0 + p, x0:x0 + p].astype(np.float32) / 255.0
        m = masks[:, y0:y0 + p, x0:x0 + p]
        if self.flips:
            if rng.random() < 0.5:
                img, m = img[:, ::-1], m[:, :, ::-1]
            if rng.random() < 0.5:
                img, m = img[::-1], m[:, ::-1]
        img = np.ascontiguousarray(img)
        return {"x": lesions.to_model_input(img)[0], "y": torch.from_numpy(np.ascontiguousarray(m).astype(np.float32)),
                "fov": torch.from_numpy(fov_mask(img)[None].astype(np.float32))}

    @staticmethod
    def collate(items):
        import torch
        return {k: torch.stack([i[k] for i in items]) for k in ("x", "y", "fov")}


# --------------------------------------------------------------------------- loss, optimiser, EMA

def positive_weights(pos_pixels, neg_pixels, low=1.0, high=20.0):
    """w+_k = clip(sqrt(neg_k / pos_k), 1, 20) from field-of-view training pixel counts."""
    pos, neg = np.asarray(pos_pixels, np.float64), np.asarray(neg_pixels, np.float64)
    if np.any(pos <= 0):
        raise ValueError(f"every class needs positive training pixels, got {pos.tolist()}")
    return np.clip(np.sqrt(neg / pos), low, high).astype(np.float32)


def pixel_counts(cache, role="train"):
    pos, neg = np.zeros(len(CLASSES)), np.zeros(len(CLASSES))
    for name in cache.names(role):
        rgb, masks, _ = cache.load(name)
        fov = fov_mask(rgb.astype(np.float32) / 255.0).astype(bool)
        for k in range(len(CLASSES)):
            p = int((masks[k].astype(bool) & fov).sum())
            pos[k] += p
            neg[k] += int(fov.sum()) - p
    return pos, neg


def loss_fn(logits, targets, fov, pos_weight, eps=1.0):
    """Per class k: masked BCE (positives weighted by w+_k, averaged over the field of view of each image, then over
    the images) + soft Dice pooled over the batch; the classes are averaged. All four classes are annotated in both
    datasets. logits / targets (B, K, H, W), fov (B, 1, H, W)."""
    import torch
    import torch.nn.functional as F
    pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32, device=logits.device)
    total, per_class = logits.sum() * 0.0, {}
    for c in range(logits.shape[1]):
        z, y, f = logits[:, c].float(), targets[:, c].float(), fov[:, 0].float()
        area = f.sum(dim=(1, 2)).clamp_min(1.0)
        bce_map = -pos_weight[c] * y * F.logsigmoid(z) - (1.0 - y) * F.logsigmoid(-z)
        bce = ((f * bce_map).sum(dim=(1, 2)) / area).mean()
        s = torch.sigmoid(z)
        dice = 1.0 - (2.0 * (f * s * y).sum() + eps) / ((f * (s + y)).sum() + eps)
        total = total + bce + dice
        per_class[CLASSES[c]] = {"bce": float(bce.detach()), "dice_loss": float(dice.detach())}
    return total / logits.shape[1], per_class


def build_model(pretrained, cache_dir=None):
    """The U-Net; `pretrained=True` loads the ImageNet SE-ResNet-101 encoder (strict) from its pinned revision."""
    import segmentation_models_pytorch as smp
    if smp.__version__ != SMP_VERSION:
        raise RuntimeError(f"segmentation_models_pytorch {smp.__version__} != {SMP_VERSION}")
    model = lesions.build_model(CLASSES)
    if pretrained:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        path = hf_hub_download(ENCODER_REPO, ENCODER_FILE, revision=ENCODER_REVISION, cache_dir=cache_dir)
        if os.path.getsize(path) != ENCODER_BYTES:
            raise LesionTrainingError(f"{path} is {os.path.getsize(path)} bytes, expected {ENCODER_BYTES}")
        model.encoder.load_state_dict(load_file(path, device="cpu"))
    return model


class ModelEMA:
    """Exponential moving average of the parameters; buffers (BatchNorm statistics) are copied from the live model."""

    def __init__(self, model, decay=0.999):
        import torch
        self.decay, self.module, self._torch = float(decay), copy.deepcopy(model).eval(), torch
        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with self._torch.no_grad():
            for e, p in zip(self.module.parameters(), model.parameters()):
                e.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
            for e, b in zip(self.module.buffers(), model.buffers()):
                e.copy_(b)


def build_optimizer(model, cfg):
    import torch
    enc = list(model.encoder.parameters())
    ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in ids]
    groups = []
    for params, lr in ((enc, cfg.decoder_lr * cfg.encoder_lr_factor), (rest, cfg.decoder_lr)):
        groups += [{"params": [p for p in params if p.ndim > 1], "lr": lr, "weight_decay": cfg.weight_decay},
                   {"params": [p for p in params if p.ndim <= 1], "lr": lr, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups)


def lr_factor(step, cfg):
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    t = (step - cfg.warmup_steps) / max(1, cfg.iterations - cfg.warmup_steps)
    return cfg.min_lr_factor + (1 - cfg.min_lr_factor) * 0.5 * (1 + np.cos(np.pi * min(1.0, t)))


# --------------------------------------------------------------------------- validation

class MetricAccumulator:
    """Streaming per-class pooled soft Dice and pixel AUPR (exact-count histograms) over images."""

    def __init__(self, n_classes=len(CLASSES), bins=AP_BINS):
        self.bins, self.images = bins, 0
        self.inter, self.psum, self.ysum = np.zeros(n_classes), np.zeros(n_classes), np.zeros(n_classes)
        self.pos_hist, self.neg_hist = np.zeros((n_classes, bins), np.int64), np.zeros((n_classes, bins), np.int64)

    def update(self, probs, targets):
        self.images += 1
        for k in range(probs.shape[0]):
            p, y = probs[k].astype(np.float64).ravel(), targets[k].astype(bool).ravel()
            self.inter[k] += p[y].sum()
            self.psum[k] += p.sum()
            self.ysum[k] += y.sum()
            idx = np.minimum((p * self.bins).astype(np.int64), self.bins - 1)
            self.pos_hist[k] += np.bincount(idx[y], minlength=self.bins)
            self.neg_hist[k] += np.bincount(idx[~y], minlength=self.bins)

    @staticmethod
    def average_precision(pos_hist, neg_hist):
        tp, fp = np.cumsum(pos_hist[::-1]), np.cumsum(neg_hist[::-1])
        if tp[-1] == 0:
            return float("nan")
        precision, recall = tp / np.maximum(tp + fp, 1), tp / tp[-1]
        return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))

    def result(self):
        out = {"images": self.images}
        for k, c in enumerate(CLASSES):
            out[c] = {"dice_pooled_soft": float((2 * self.inter[k] + 1) / (self.psum[k] + self.ysum[k] + 1)),
                      "aupr": self.average_precision(self.pos_hist[k], self.neg_hist[k]), "positive_pixels": int(self.ysum[k])}
        aupr = [out[c]["aupr"] for c in CLASSES if not np.isnan(out[c]["aupr"])]
        out["mean_dice"] = float(np.mean([out[c]["dice_pooled_soft"] for c in CLASSES]))
        out["mean_aupr"] = float(np.mean(aupr)) if aupr else float("nan")
        return out


def validate(model, cache, device, amp=True):
    """Full-frame (1536^2) inference on every validation image, reported per dataset and per TJDR camera."""
    groups = {}
    for name in cache.names("val"):
        rgb, masks, meta = cache.load(name)
        probs = lesions.predict_probabilities(model, rgb.astype(np.float32) / 255.0, device, amp).transpose(2, 0, 1)
        keys = ["IDRiD-val"] if meta["dataset"] == "IDRiD" else [f"TJDR-val-{meta['camera'].replace('TJDR-', '')}", "TJDR-val-all"]
        for g in keys:
            groups.setdefault(g, MetricAccumulator()).update(probs, masks)
    report = {g: acc.result() for g, acc in sorted(groups.items())}
    scores = [report[g]["mean_aupr"] for g in SELECTION_GROUPS if g in report and not np.isnan(report[g]["mean_aupr"])]
    report["selection_score"] = float(np.mean(scores)) if scores else float("nan")
    return report


# --------------------------------------------------------------------------- the run

def train(run_dir, cache, cfg=None, device="cuda", max_steps=None, model=None, log=print):
    """Trains (or resumes) one run. `model`: a prebuilt model (tests); otherwise the U-Net with its ImageNet encoder.
    The best EMA weights are `<run_dir>/best_ema.pt`; returns {'step', 'best'}."""
    import torch
    cfg = cfg or TrainConfig()
    if cfg.batch_size % 2:
        raise ValueError("dataset-balanced batches need an even batch size")
    os.makedirs(run_dir, exist_ok=True)
    seed_everything(cfg.seed)
    model = (model or build_model(pretrained=True)).to(device)
    ema = ModelEMA(model, cfg.ema_decay)
    optimizer = build_optimizer(model, cfg)
    for g in optimizer.param_groups:
        g["base_lr"] = g["lr"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, cfg))
    use_amp = bool(cfg.amp and torch.device(device).type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    weights_path = os.path.join(run_dir, "loss_weights.json")
    if os.path.exists(weights_path):
        weights = read_json(weights_path)
    else:
        pos, neg = pixel_counts(cache)
        weights = {"pos_weight": positive_weights(pos, neg).tolist(), "pos_pixels": [int(x) for x in pos], "neg_pixels": [int(x) for x in neg]}
        atomic_json(weights_path, weights)
    latest = os.path.join(run_dir, "latest.pt")
    identity = {"config": dataclasses.asdict(cfg), "counts": cache.manifest["counts"]}
    step, history, best = 0, [], {"score": -1.0, "step": None}
    if os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        if ck["identity"] != json.loads(json.dumps(identity)):
            raise LesionTrainingError("resume refused: the configuration or the data differ from the checkpoint")
        model.load_state_dict(ck["model"])
        ema.module.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        step, history, best = ck["step"], ck["history"], ck["best"]
        log(f"resumed at step {step}")

    names = {d: cache.names("train", d) for d in ("IDRiD", "TJDR")}
    end = cfg.iterations if max_steps is None else min(cfg.iterations, step + max_steps)
    loader = torch.utils.data.DataLoader(PatchDataset(cache, cfg.patch, cfg.p_lesion, cfg.flips),
                                         batch_sampler=[balanced_batch_plan(names, cfg.batch_size, s, cfg.seed) for s in range(step, end)],
                                         num_workers=cfg.num_workers, collate_fn=PatchDataset.collate,
                                         pin_memory=torch.device(device).type == "cuda")

    def save(path):
        tmp = path + ".part"
        torch.save({"step": step, "model": model.state_dict(), "ema": ema.module.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "identity": identity, "history": history,
                    "best": best}, tmp)
        os.replace(tmp, path)

    started = time.time()
    for batch in loader:
        model.train()
        x, y, fov = (batch[k].to(device, non_blocking=True) for k in ("x", "y", "fov"))
        with torch.autocast(device_type=torch.device(device).type, enabled=use_amp):
            logits = model(x)
        loss, per_class = loss_fn(logits.float(), y, fov, weights["pos_weight"])
        if not torch.isfinite(loss):
            raise LesionTrainingError(f"step {step}: the loss is not finite")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)
        step += 1
        history_loss = float(loss.detach())
        if step % 50 == 0 or step == end or step <= 3:
            log(f"step {step}: loss {history_loss:.4f} grad {grad_norm:.3f} lr {optimizer.param_groups[0]['lr']:.2e} "
                f"({(time.time() - started) / max(1, step):.1f} s/step)")
        if step % cfg.val_every == 0 or step == cfg.iterations:
            report = validate(ema.module, cache, device, use_amp)
            history.append({"step": step, "validation": report})
            atomic_json(os.path.join(run_dir, "validation_history.json"), history)
            if report["selection_score"] > best["score"]:
                best = {"score": report["selection_score"], "step": step}
                torch.save(ema.module.state_dict(), os.path.join(run_dir, "best_ema.pt"))
            log(f"validation @ {step}: selection {report['selection_score']:.4f} (best {best['score']:.4f} @ step {best['step']})")
        if step % cfg.checkpoint_every == 0 or step == end:
            save(latest)
    return {"step": step, "best": best, "run_dir": run_dir}


def export_best(run_dir, out_path):
    """Writes the best EMA state dict as the Stage-4 model file (`lesions_unet_se_resnet101.pt`)."""
    import shutil
    src = os.path.join(run_dir, "best_ema.pt")
    if not os.path.exists(src):
        raise LesionTrainingError(f"{src} does not exist: no validation has run yet")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    shutil.copyfile(src, out_path)
    return out_path
