"""Metrics on per-image prediction tables, and the paired grade-stratified bootstrap.

A per-image table (CSV) holds, per image: image_id, true_grade, predicted_grade, logit_0..3, p_gt_0..3
(cumulative probabilities P(grade > k)) and class_prob_0..4. `arrays()` turns one into the arrays used here.

  QWK        quadratic weighted kappa over grades 0-4
  AUROC      per cumulative cut: grade >= c scored by P(grade > c - 1), c = 1..4; their mean
  bootstrap  images resampled with replacement WITHIN each grade (2,000 resamples, seed 20260927); paired:
             every model is scored on the same resamples
"""
import numpy as np

N_BOOT = 2000
BOOT_SEED = 20260927
CUTS = (1, 2, 3, 4)
METRICS = ("qwk", "auroc_ge3_g4_vs_g012", "mean_cut_auroc", "grade3_recall", "false_urgent_rate", "mae")
BOOT_KEYS = ("qwk", "auroc_ge3_g4_vs_g012", "mean_cut_auroc")
HEADLINE_KEYS = ("qwk", "auroc_ge1", "auroc_ge2", "auroc_ge3", "auroc_ge4", "mean_cut_auroc", "auroc_ge3_g4_vs_g012",
                 "grade3_recall", "grade4_recall", "false_urgent_rate", "mae")
_QWK_WEIGHTS = (np.arange(5)[:, None] - np.arange(5)[None, :]) ** 2 / 16.0


# --------------------------------------------------------------------------- tables

def read_table(path):
    import pandas as pd
    return pd.read_csv(path, dtype={"image_id": str})


def arrays(frame):
    """The columns the metrics need, as arrays."""
    return {"ids": frame["image_id"].astype(str).to_numpy(), "grade": frame["true_grade"].to_numpy(int),
            "pred": frame["predicted_grade"].to_numpy(int),
            "p_gt": np.stack([frame[f"p_gt_{k}"].to_numpy(np.float64) for k in range(4)], 1)}


def subset(table, mask):
    return {k: v[mask] for k, v in table.items()}


def per_image_rows(ids, grades, logits):
    """The per-image table of one model from its logits (list of dicts, the CSV schema)."""
    from .grader import decode
    logits = np.asarray(logits, np.float64)
    grades = np.asarray(grades, int)
    d = decode(logits)
    p_gt, pred, prob = d["p_gt"].astype(np.float64), d["grade"], d["class_prob"].astype(np.float64)
    argmax = prob.argmax(axis=-1)
    margin = np.min(np.abs(p_gt - 0.5), axis=-1)
    entropy = -np.sum(prob * np.log(np.clip(prob, 1e-12, 1.0)), axis=-1)
    rows = []
    for i, image_id in enumerate(ids):
        row = {"image_id": image_id, "true_grade": int(grades[i]), "predicted_grade": int(pred[i]),
               "correct": bool(grades[i] == pred[i]), "error_distance": int(abs(grades[i] - pred[i]))}
        row.update({f"logit_{k}": float(logits[i, k]) for k in range(4)})
        row.update({f"p_gt_{k}": float(p_gt[i, k]) for k in range(4)})
        row.update({f"p_cond_{k}": float(d["p_cond"][i, k]) for k in range(4)})
        row.update({f"class_prob_{g}": float(prob[i, g]) for g in range(5)})
        row.update({"predicted_class_probability": float(prob[i, pred[i]]), "argmax_class": int(argmax[i]),
                    "argmax_matches_decode": bool(argmax[i] == pred[i]), "nearest_threshold_margin": float(margin[i]),
                    "class_entropy_nats": float(entropy[i])})
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- single metrics

def auroc(positive, score):
    from sklearn.metrics import roc_auc_score
    positive = np.asarray(positive, bool)
    if positive.all() or not positive.any():
        return float("nan")
    return float(roc_auc_score(positive, score))


def rank_auroc(positive, score):
    """AUROC by the rank-sum formula (average ranks for ties): the same value as sklearn's, faster."""
    from scipy.stats import rankdata
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((rankdata(score)[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def qwk(grades, pred):
    """Quadratic weighted kappa over the five grades (the same value as sklearn's cohen_kappa_score)."""
    observed = np.bincount(np.asarray(grades) * 5 + np.asarray(pred), minlength=25).reshape(5, 5).astype(np.float64)
    expected = np.outer(observed.sum(1), observed.sum(0)) / observed.sum()
    denominator = float((_QWK_WEIGHTS * expected).sum())
    return 1.0 - float((_QWK_WEIGHTS * observed).sum()) / denominator if denominator else float("nan")


def metrics(table, idx=None):
    """The reported quantities of one table (optionally on resample `idx`)."""
    from sklearn.metrics import cohen_kappa_score
    g, pred, p = table["grade"], table["pred"], table["p_gt"]
    if idx is not None:
        g, pred, p = g[idx], pred[idx], p[idx]
    low, sel = g <= 2, g != 3
    cuts = [auroc(g >= c, p[:, c - 1]) for c in CUTS]
    return {"qwk": float(cohen_kappa_score(g, pred, weights="quadratic", labels=list(range(5)))),
            "auroc_ge3_g4_vs_g012": auroc(g[sel] == 4, p[sel, 2]),
            "mean_cut_auroc": float(np.mean(cuts)),
            "auroc_cuts": {f"ge{c}": v for c, v in zip(CUTS, cuts)},
            "grade3_recall": float(np.mean(pred[g == 3] == 3)) if np.any(g == 3) else float("nan"),
            "false_urgent_rate": float(np.mean(pred[low] >= 3)) if low.any() else float("nan"),
            "mae": float(np.mean(np.abs(pred - g))),
            "recall_per_grade": {int(k): float(np.mean(pred[g == k] == k)) if np.any(g == k) else float("nan") for k in range(5)},
            "confusion_matrix": [[int(np.sum((g == a) & (pred == b))) for b in range(5)] for a in range(5)]}


def headline(m):
    return {"qwk": m["qwk"], "auroc_ge1": m["auroc_cuts"]["ge1"], "auroc_ge2": m["auroc_cuts"]["ge2"],
            "auroc_ge3": m["auroc_cuts"]["ge3"], "auroc_ge4": m["auroc_cuts"]["ge4"], "mean_cut_auroc": m["mean_cut_auroc"],
            "auroc_ge3_g4_vs_g012": m["auroc_ge3_g4_vs_g012"], "grade3_recall": m["grade3_recall"],
            "grade4_recall": m["recall_per_grade"][4], "false_urgent_rate": m["false_urgent_rate"], "mae": m["mae"]}


def referable(table):
    """Sensitivity and specificity for referable DR (grade >= 2) at the decoded grade."""
    g, pred = table["grade"] >= 2, table["pred"] >= 2
    return {"sensitivity": float(pred[g].mean()) if g.any() else float("nan"),
            "specificity": float((~pred[~g]).mean()) if (~g).any() else float("nan")}


def calibration(frame, bins=15):
    """Top-label ECE (equal-width bins), negative log-likelihood and multiclass Brier score from the class
    probabilities of a per-image table."""
    grade = frame["true_grade"].to_numpy(int)
    prob = np.stack([frame[f"class_prob_{k}"].to_numpy(np.float64) for k in range(5)], 1)
    confidence, top = prob.max(1), prob.argmax(1)
    correct = (top == grade).astype(np.float64)
    edges, ece = np.linspace(0.0, 1.0, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidence > lo) & (confidence <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - confidence[m].mean())
    return {"ece_top_label": float(ece), "nll": float(-np.mean(np.log(np.clip(prob[np.arange(len(grade)), grade], 1e-12, 1.0)))),
            "brier_multiclass": float(np.mean(np.sum((prob - np.eye(5)[grade]) ** 2, axis=1))),
            "mean_confidence": float(confidence.mean()), "top_label_accuracy": float(correct.mean())}


def ensemble(tables):
    """One table from several seeds' tables of the same images: cumulative probabilities averaged, decoded at 0.5."""
    first = tables[0]
    p = np.mean([t["p_gt"] for t in tables], axis=0)
    return {"ids": first["ids"], "grade": first["grade"], "p_gt": p, "pred": (p > 0.5).sum(axis=1).astype(int)}


# --------------------------------------------------------------------------- bootstrap

def bootstrap_indices(grades, n_boot=N_BOOT, seed=BOOT_SEED):
    """Resamples of row positions: images drawn with replacement within every grade."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(grades == g) for g in np.unique(grades)]
    return [np.concatenate([rng.choice(g, g.size) for g in groups]) for _ in range(n_boot)]


def bootstrap_metrics(table, idx):
    """The bootstrapped quantities on one resample (numpy only; equal to `metrics`)."""
    g, pred, p = table["grade"][idx], table["pred"][idx], table["p_gt"][idx]
    sel = g != 3
    return {"qwk": qwk(g, pred), "auroc_ge3_g4_vs_g012": rank_auroc(g[sel] == 4, p[sel, 2]),
            "mean_cut_auroc": float(np.mean([rank_auroc(g >= c, p[:, c - 1]) for c in CUTS]))}


def interval(draws):
    lo, hi = np.nanpercentile(np.asarray(draws, np.float64), [2.5, 97.5])
    return float(lo), float(hi)


def check_paired(a, b):
    if not np.array_equal(a["ids"], b["ids"]) or not np.array_equal(a["grade"], b["grade"]):
        raise ValueError("the two tables are not the same images in the same order")


def paired_delta(a, b, keys=BOOT_KEYS, indices=None):
    """a - b on the same images: point differences of every metric and percentile 95 % intervals over `indices`."""
    check_paired(a, b)
    ma, mb = metrics(a), metrics(b)
    out = {k: {"delta": ma[k] - mb[k]} for k in METRICS}
    if indices is not None:
        draws = {k: [] for k in keys}
        for idx in indices:
            da, db = bootstrap_metrics(a, idx), bootstrap_metrics(b, idx)
            for k in keys:
                draws[k].append(da[k] - db[k])
        for k in keys:
            lo, hi = np.nanpercentile(draws[k], [2.5, 97.5])
            out[k].update(ci_low=float(lo), ci_high=float(hi))
            out[k]["_draws"] = np.asarray(draws[k], np.float64)
    return out


def jsonable(obj):
    """Nested containers with numpy scalars turned into Python ones (bootstrap draws dropped)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj
