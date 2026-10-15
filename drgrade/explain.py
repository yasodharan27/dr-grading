"""Explanations of the grader. Every method explains the same scalar, the EXPECTED GRADE
E[grade] = sum_k P(grade > k) (or, with `cut`, one cumulative probability P(grade > cut)).

  grad_cam               feature-map channels weighted by the spatial mean of the score's gradient, rectified,
                         scaled to [0, 1] and resized to the frame (16 x 16 cells of 32 x 32 pixels)
  integrated_gradients   path integral of input gradients from a black image, midpoint rule (Sundararajan et al.)
  occlusion              score drop when a square patch is set to black (Zeiler & Fergus); model-agnostic
  superpixel_shap        Kernel SHAP over SLIC superpixels (Lundberg & Lee); costly, for figures only
  localisation           overlap of a map with an expert lesion mask: pointing game, precision of the top 10 % of
                         pixels, AUROC of the map, and the lesion share a uniform map would score

A map shows where the score is sensitive. Overlap with lesions is not evidence that the model detects lesions.

Gradients run through ONE traced function per (model, kind, cut): calling the Keras model eagerly makes it
compile a new XLA kernel for every grouped (depthwise) convolution on every call, and host memory grows until the
process dies.
"""
import numpy as np

from . import metrics as mt


# --------------------------------------------------------------------------- the explained scalar

def score(logits, cut=None):
    p = np.cumprod(1.0 / (1.0 + np.exp(-np.asarray(logits, np.float64))), axis=1)
    return p.sum(axis=1) if cut is None else p[:, int(cut)]


def _tf_score(logits, cut=None):
    import tensorflow as tf
    p = tf.math.cumprod(tf.sigmoid(tf.cast(logits, tf.float32)), axis=1)
    return tf.reduce_sum(p, axis=1) if cut is None else p[:, int(cut)]


_COMPILED = {}


def compiled_gradients(grad_model, wrt, cut=None):
    """rgb -> (features, logits, gradient), traced once per (model, wrt, cut). `wrt` = "features" (Grad-CAM: the
    gradient of each image's score) or "input" (Integrated Gradients: of the summed score w.r.t. the rgb input)."""
    import tensorflow as tf
    key = (id(grad_model), wrt, cut)
    if key not in _COMPILED:
        def run(rgb):
            with tf.GradientTape() as tape:
                tape.watch(rgb)
                features, logits = grad_model(rgb, training=False)
                s = _tf_score(logits, cut)
                if wrt == "input":
                    s = tf.reduce_sum(s)
            return features, logits, tape.gradient(s, features if wrt == "features" else rgb)
        _COMPILED[key] = (grad_model, tf.function(run, input_signature=[tf.TensorSpec([None, None, None, 3], tf.float32)]))
    return _COMPILED[key][1]


def model_fn(grader, batch_size=8):
    """`rgb (N, H, W, 3) -> logits (N, 4)` for a `grader.Grader`."""
    return lambda rgb: grader.logits(rgb, batch_size)


def grader_scores(fn, rgb_batch, cut=None):
    return score(fn(np.asarray(rgb_batch, np.float32)), cut)


# --------------------------------------------------------------------------- Grad-CAM

def grad_cam(grader, rgb, cut=None):
    """rgb (N, 512, 512, 3) in [0, 1] -> (maps (N, 512, 512) in [0, 1], logits (N, 4))."""
    import tensorflow as tf
    rgb = tf.convert_to_tensor(np.asarray(rgb, np.float32))
    features, logits, gradients = compiled_gradients(grader.grad_model, "features", cut)(rgb)
    features, gradients = tf.cast(features, tf.float32), tf.cast(gradients, tf.float32)
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * features, axis=-1))
    peak = tf.reduce_max(cam, axis=(1, 2), keepdims=True)
    cam = tf.where(peak > 0, cam / tf.maximum(peak, 1e-12), tf.zeros_like(cam))
    cam = tf.image.resize(cam[..., None], rgb.shape[1:3], method="bilinear")[..., 0]
    return np.asarray(cam, np.float32), np.asarray(logits, np.float64)


def display_scale(heat, percentile=99.0):
    """A map for DISPLAY only: values divided by their `percentile`-th percentile and clipped to [0, 1]. A pixel-level
    attribution map (Integrated Gradients) is dominated by a few extreme pixels, so scaling by its maximum leaves the
    picture nearly empty. The map itself, and every number computed from it, is unchanged."""
    heat = np.asarray(heat, np.float32)
    top = float(np.percentile(heat, percentile))
    return np.clip(heat / top, 0.0, 1.0) if top > 0 else np.zeros_like(heat)


def overlay(rgb, heat, alpha=0.45):
    """RGB uint8: the frame with the map over it (red = high)."""
    rgb = np.asarray(rgb, np.float32)
    colour = np.stack([heat, np.zeros_like(heat), 1.0 - heat], axis=-1) * heat[..., None]
    return np.clip(255.0 * ((1.0 - alpha * heat[..., None]) * rgb + alpha * colour), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- Integrated Gradients

def integrated_gradients(grader, rgb, cut=None, steps=32, baseline=None, batch=2):
    """Integrated Gradients of ONE frame (H, W, 3) from `baseline` (default black), midpoint sum of `steps` points.
    Returns (attribution (H, W, 3), completeness gap |sum(attribution) - (s(x) - s(baseline))|, map (H, W) = sum
    over channels of |attribution| scaled to [0, 1])."""
    import tensorflow as tf
    x = np.asarray(rgb, np.float32)
    if x.ndim != 3:
        raise ValueError("one image (H, W, 3) at a time")
    x0 = np.zeros_like(x) if baseline is None else np.asarray(baseline, np.float32)
    alphas = (np.arange(steps, dtype=np.float32) + 0.5) / steps
    run = compiled_gradients(grader.grad_model, "input", cut)
    total = np.zeros_like(x, dtype=np.float64)
    for s in range(0, steps, batch):
        a = alphas[s:s + batch][:, None, None, None]
        total += np.asarray(run(tf.convert_to_tensor(x0[None] + a * (x - x0)[None]))[2], np.float64).sum(axis=0)
    attribution = (x - x0).astype(np.float64) * total / steps
    s0, s1 = score(np.asarray(run(tf.convert_to_tensor(np.stack([x0, x])))[1], np.float64), cut)
    gap = float(abs(attribution.sum() - (s1 - s0)))
    heat = np.abs(attribution).sum(axis=-1)
    return attribution.astype(np.float32), gap, (heat / heat.max() if heat.max() > 0 else heat).astype(np.float32)


# --------------------------------------------------------------------------- Occlusion

def occlusion(fn, rgb, cut=None, patch=64, stride=32, fill=0.0):
    """Score drop when a patch x patch square is set to `fill`, every `stride` pixels; overlapping drops averaged.
    Returns (raw drops (H, W), map scaled to [0, 1] after clipping at zero). `fn`: rgb batch -> logits."""
    x = np.asarray(rgb, np.float32)
    h, w = x.shape[:2]
    base = grader_scores(fn, x[None], cut)[0]
    boxes = [(t, l) for t in range(0, max(h - patch, 0) + 1, stride) for l in range(0, max(w - patch, 0) + 1, stride)]
    occluded = np.repeat(x[None], len(boxes), axis=0)
    for k, (t, l) in enumerate(boxes):
        occluded[k, t:t + patch, l:l + patch] = fill
    drops = base - grader_scores(fn, occluded, cut)
    total, count = np.zeros((h, w)), np.zeros((h, w))
    for (t, l), d in zip(boxes, drops):
        total[t:t + patch, l:l + patch] += d
        count[t:t + patch, l:l + patch] += 1
    raw = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    pos = np.clip(raw, 0, None)
    return raw.astype(np.float32), (pos / pos.max() if pos.max() > 0 else pos).astype(np.float32)


# --------------------------------------------------------------------------- Kernel SHAP over superpixels

def superpixels(rgb, n_segments=50, compactness=10.0):
    from skimage.segmentation import slic
    return slic(np.asarray(rgb, np.float64), n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=-1)


def kernel_shap(value_fn, n_features, n_samples=512, seed=0):
    """Kernel SHAP with the empty and full coalitions enforced exactly. All coalitions are enumerated when they fit
    in `n_samples` (exact Shapley values); otherwise coalition sizes are drawn in proportion to their kernel weight
    and each draw is paired with its complement. `value_fn(masks (S, n) of 0/1) -> values (S,)`.
    Returns (phi (n,), value of the empty coalition)."""
    import itertools
    from math import comb
    m = n_features
    ends = value_fn(np.stack([np.zeros(m), np.ones(m)]))
    v0, total = float(ends[0]), float(ends[1] - ends[0])
    if 2 ** m - 2 <= n_samples:
        masks = np.array([[1.0 if j in c else 0.0 for j in range(m)] for r in range(1, m) for c in itertools.combinations(range(m), r)])
        weights = np.array([(m - 1) / (comb(m, int(k)) * k * (m - k)) for k in masks.sum(axis=1)])
    else:
        rng = np.random.default_rng(seed)
        sizes = np.arange(1, m)
        p = (m - 1) / (sizes * (m - sizes))
        half = max(1, n_samples // 2)
        drawn = np.zeros((half, m))
        for i, k in enumerate(rng.choice(sizes, half, p=p / p.sum())):
            drawn[i, rng.choice(m, int(k), replace=False)] = 1.0
        masks, weights = np.concatenate([drawn, 1.0 - drawn]), np.ones(2 * half)
    y = value_fn(masks) - v0
    a = (masks[:, :-1] - masks[:, -1:]) * np.sqrt(weights)[:, None]          # last feature eliminated through efficiency
    b = (y - masks[:, -1] * total) * np.sqrt(weights)
    phi_rest, *_ = np.linalg.lstsq(a, b, rcond=None)
    return np.append(phi_rest, total - phi_rest.sum()), v0


def superpixel_shap(fn, rgb, cut=None, n_segments=50, n_samples=512, fill=0.0, seed=0, batch=32):
    """Kernel SHAP of the score over SLIC superpixels of one frame: (map (H, W), segments, phi, base value)."""
    x = np.asarray(rgb, np.float32)
    segments = superpixels(x, n_segments)
    n = int(segments.max()) + 1

    def value_fn(masks):
        out = []
        for s in range(0, len(masks), batch):
            images = np.repeat(x[None], len(masks[s:s + batch]), axis=0)
            for k, mask in enumerate(masks[s:s + batch]):
                images[k][~mask.astype(bool)[segments]] = fill
            out.append(grader_scores(fn, images, cut))
        return np.concatenate(out)

    phi, base = kernel_shap(value_fn, n, n_samples, seed)
    return phi[segments].astype(np.float32), segments, phi, base


# --------------------------------------------------------------------------- localisation against expert masks

def localisation(heat, lesion_mask, top_fraction=0.1):
    """Pointing-game hit, lesion share of the map's top pixels, AUROC of the map as a lesion score, and the lesion
    share of the image (what a uniform map would get). Describes overlap only."""
    heat, mask = np.asarray(heat, np.float64), np.asarray(lesion_mask, bool)
    if heat.shape != mask.shape:
        raise ValueError("map and mask must have the same shape")
    peak = np.unravel_index(int(np.argmax(heat)), heat.shape)
    k = max(1, int(round(top_fraction * heat.size)))
    top = np.argsort(heat, axis=None, kind="stable")[-k:]
    return {"pointing_hit": bool(mask[peak]), "top_precision": float(mask.reshape(-1)[top].mean()),
            "auroc": mt.auroc(mask.reshape(-1), heat.reshape(-1)), "lesion_fraction": float(mask.mean())}


def map_on_lesions(heat, lesion_mask, top_fraction=0.1):
    """(share of the map's top `top_fraction` pixels that lie on the mask, the mask's share of the frame -- what a map
    unrelated to the lesions would give)."""
    heat, mask = np.asarray(heat, np.float64), np.asarray(lesion_mask, bool)
    k = max(1, int(round(top_fraction * heat.size)))
    top = np.argsort(heat, axis=None, kind="stable")[-k:]
    return float(mask.reshape(-1)[top].mean()), float(mask.mean())


def frame_mask(mask_native, box=None, size=512):
    """An expert mask in the grader's frame: the same cut (crop arm) and resize as the image (nearest neighbour)."""
    import cv2
    m = np.asarray(mask_native) > 0
    if box is not None:
        top, bottom, left, right = box
        m = m[top:bottom, left:right]
    return cv2.resize(m.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)


def union_mask(mask_paths):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    return np.any(np.stack([np.asarray(Image.open(p)) > 0 for p in mask_paths]), axis=0)


def localisation_study(items, arm, graders, methods=("grad_cam", "integrated_gradients", "occlusion"), occlusion_images=20,
                       ig_steps=32, progress_csv=None, log=print):
    """Each method's map (mean over the seeds' maps) against the union of an image's expert lesion masks, in the
    grader's frame for `arm`. `items`: [(name, image path, [mask paths])]. Occlusion runs on the first
    `occlusion_images` items only. With `progress_csv`, rows are appended per image and images already there for
    this arm are skipped (an interrupted run resumes). Returns the rows."""
    import csv
    import os

    from . import preprocess
    done = set()
    if progress_csv and os.path.exists(progress_csv):
        with open(progress_csv, newline="") as fh:
            done = {r["image"] for r in csv.DictReader(fh) if r["arm"] == arm}
    rows = []
    fns = [model_fn(g) for g in graders]
    for k, (name, image_path, masks) in enumerate(items):
        if name in done:
            continue
        f = preprocess.frames(image_path)
        rgb = f[arm]
        mask = frame_mask(union_mask(masks), None if arm == "direct" else f["box"])
        maps = {}
        if "grad_cam" in methods:
            maps["grad_cam"] = np.mean([grad_cam(g, rgb[None])[0][0] for g in graders], axis=0)
        if "integrated_gradients" in methods:
            maps["integrated_gradients"] = np.mean([integrated_gradients(g, rgb, steps=ig_steps)[2] for g in graders], axis=0)
        if "occlusion" in methods and k < occlusion_images:
            maps["occlusion"] = np.mean([occlusion(fn, rgb)[1] for fn in fns], axis=0)
        new = [{"image": name, "arm": arm, "method": method, **localisation(heat, mask)} for method, heat in maps.items()]
        rows += new
        if progress_csv:
            first = not os.path.exists(progress_csv)
            with open(progress_csv, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(new[0]))
                if first:
                    writer.writeheader()
                writer.writerows(new)
        if (k + 1) % 10 == 0:
            log(f"  {arm}: {k + 1}/{len(items)} images")
    if progress_csv:
        import pandas as pd
        return pd.read_csv(progress_csv).query("arm == @arm").to_dict("records")
    return rows


def localisation_summary(rows):
    """Per arm and method: images, pointing-game hit rate, mean top-10 % precision, mean AUROC, mean lesion share."""
    import pandas as pd
    frame = pd.DataFrame(rows)
    out = frame.groupby(["arm", "method"]).agg(images=("image", "count"), pointing_hit=("pointing_hit", "mean"),
                                               top10_precision=("top_precision", "mean"), auroc=("auroc", "mean"),
                                               uniform_map_baseline=("lesion_fraction", "mean")).reset_index()
    return out
