"""Uncertainty of the grader from its three-seed ensemble, and test-time augmentation.

Per image, from the seeds' CORN logits:
  ensemble prediction   the seeds' cumulative probabilities averaged, decoded at 0.5
  entropy               predictive entropy of the ensemble's class distribution
  seed_disagreement     standard deviation of the three seeds' expected grades
  threshold_margin      minus the distance of the nearest cumulative probability to 0.5

A score is tested by selective prediction: the most uncertain images are referred and agreement (QWK, accuracy)
is measured on the rest at each coverage, against random referral (200 random orders). The model has no dropout,
so this is deep-ensemble uncertainty, not Monte Carlo dropout. Test-time augmentation runs the model on the 8
flips / quarter turns of the frame (the transformations seen in training) and reports their spread.
"""
import json
import os

import numpy as np

from . import metrics as mt
from .grader import class_probabilities, cumulative

SCORES = ("entropy", "seed_disagreement", "threshold_margin")
COVERAGES = (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5)
RANDOM_REPEATS, RANDOM_SEED = 200, 20261012


def read_seed_tables(paths):
    """{seed: per-image table path} -> (ids, grades, logits (seeds, N, 4)); same images in the same order."""
    import pandas as pd
    frames = [pd.read_csv(paths[s], dtype={"image_id": str}) for s in sorted(paths)]
    ids, grades = frames[0]["image_id"].to_numpy(), frames[0]["true_grade"].to_numpy(int)
    for f in frames[1:]:
        if not np.array_equal(f["image_id"].to_numpy(), ids) or not np.array_equal(f["true_grade"].to_numpy(int), grades):
            raise ValueError("the seed tables are not the same images in the same order")
    return ids, grades, np.stack([f[[f"logit_{k}" for k in range(4)]].to_numpy(np.float64) for f in frames])


def ensemble(logits):
    """logits (seeds, N, 4) -> per image: ensemble grade, class distribution and the three scores (higher = less
    certain), plus each seed's decoded grade."""
    p_seed = np.stack([cumulative(z) for z in logits])
    p = p_seed.mean(axis=0)
    prob = np.clip(class_probabilities(p), 1e-12, 1.0)
    prob = prob / prob.sum(axis=1, keepdims=True)
    return {"pred": (p > 0.5).sum(axis=1).astype(int), "p_gt": p, "class_prob": prob,
            "entropy": -(prob * np.log(prob)).sum(axis=1),
            "seed_disagreement": p_seed.sum(axis=2).std(axis=0),
            "threshold_margin": -np.abs(p - 0.5).min(axis=1),
            "seed_pred": (p_seed > 0.5).sum(axis=2).astype(int)}


def selective(grades, pred, score, coverages=COVERAGES):
    order = np.argsort(score, kind="stable")                               # most certain first
    rows = []
    for c in coverages:
        keep = order[:max(1, int(round(c * len(order))))]
        rows.append({"coverage": c, "n": int(len(keep)), "qwk": mt.qwk(grades[keep], pred[keep]),
                     "accuracy": float((pred[keep] == grades[keep]).mean()), "referable_retained": float((grades[keep] >= 2).mean())})
    return rows


def random_baseline(grades, pred, coverages=COVERAGES, repeats=RANDOM_REPEATS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    out = {c: [] for c in coverages}
    for _ in range(repeats):
        order = rng.permutation(len(grades))
        for c in coverages:
            keep = order[:max(1, int(round(c * len(order))))]
            out[c].append(mt.qwk(grades[keep], pred[keep]))
    return [{"coverage": c, "qwk_mean": float(np.mean(v)), "qwk_low": float(np.percentile(v, 2.5)), "qwk_high": float(np.percentile(v, 97.5))}
            for c, v in out.items()]


def error_detection_auroc(grades, pred, score):
    """How well the score ranks wrong predictions above right ones (0.5 = uninformative)."""
    return mt.auroc(pred != grades, score)


def evaluate(paths, name):
    ids, grades, logits = read_seed_tables(paths)
    e = ensemble(logits)
    result = {"set": name, "images": int(len(grades)), "ensemble_qwk": mt.qwk(grades, e["pred"]),
              "per_seed_qwk": [mt.qwk(grades, e["seed_pred"][k]) for k in range(len(logits))],
              "seeds_disagree_on_grade": float((e["seed_pred"].min(axis=0) != e["seed_pred"].max(axis=0)).mean()),
              "random_referral": random_baseline(grades, e["pred"]), "scores": {}}
    for s in SCORES:
        result["scores"][s] = {"error_detection_auroc": error_detection_auroc(grades, e["pred"], e[s]),
                               "selective": selective(grades, e["pred"], e[s])}
    import pandas as pd
    table = pd.DataFrame({"image_id": ids, "true_grade": grades, "ensemble_grade": e["pred"], **{s: e[s] for s in SCORES},
                          **{f"class_prob_{k}": e["class_prob"][:, k] for k in range(5)}})
    return result, table


def run(sets, out_dir):
    """`sets`: {name: {seed: per-image table path}}. Writes uncertainty_<name>.csv and uncertainty_results.json."""
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for name, paths in sets.items():
        results[name], table = evaluate(paths, name)
        table.to_csv(os.path.join(out_dir, f"uncertainty_{name}.csv"), index=False)
    with open(os.path.join(out_dir, "uncertainty_results.json"), "w") as fh:
        json.dump(mt.jsonable(results), fh, indent=1)
    return results


# --------------------------------------------------------------------------- test-time augmentation

def dihedral(rgb, k):
    """The k-th of the 8 flips / quarter turns (k = 0 is the identity) of (..., H, W, C) arrays."""
    x = np.rot90(np.asarray(rgb), k % 4, axes=(-3, -2))
    return np.flip(x, axis=-2) if k >= 4 else x


def tta_logits(model_fn, rgb):
    """`model_fn(rgb (N, H, W, 3)) -> logits (N, 4)` on each of the 8 transformations, all views in one call: (8, N, 4)."""
    rgb = np.asarray(rgb)
    views = np.concatenate([np.ascontiguousarray(dihedral(rgb, k)) for k in range(8)])
    return np.asarray(model_fn(views), np.float64).reshape(8, len(rgb), -1)


def tta_summary(logits_tta):
    """For logits (T, N, 4): the TTA-mean grade, the SD of the expected grade across transformations, and whether
    the decoded grade changes between them."""
    p = np.stack([cumulative(z) for z in logits_tta])
    grades = (p > 0.5).sum(axis=2)
    return {"pred": (p.mean(axis=0) > 0.5).sum(axis=1).astype(int), "expected_grade_sd": p.sum(axis=2).std(axis=0),
            "grade_changes": grades.min(axis=0) != grades.max(axis=0)}
