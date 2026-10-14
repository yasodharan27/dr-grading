"""External evaluation on DDR and IDRiD: predictions of the two grader arms, and every reported statistic.

Models compared (three seeds each, seed-labelled pairs 42 / 123 / 2026):
  crop     joint APTOS + EyePACS grader, fundus-cropped frames (the final model)
  direct   joint APTOS + EyePACS grader, whole-image frames
  p        APTOS-only grader (baseline; stored predictions in data/baselines/)
  p_ep     APTOS grader started from an EyePACS-adapted encoder (baseline; stored predictions)

Main contrasts: crop - p and direct - p (QWK, mean over the three seed pairs, paired grade-stratified bootstrap
interval). Populations: DDR primary 3,540 images (broader 3,725 as a sensitivity set), IDRiD primary 100 images
(all 103 as a sensitivity set); see data/README.md for their definitions.
"""
import json
import os

import numpy as np

from . import metrics as mt
from .config import SEEDS

MODELS = ("p", "p_ep", "direct", "crop")
RUN_MODELS = ("direct", "crop")
BASELINES = ("p", "p_ep")
CONTRASTS = {"direct_minus_p": ("direct", "p"), "crop_minus_p": ("crop", "p"), "direct_minus_p_ep": ("direct", "p_ep"),
             "crop_minus_p_ep": ("crop", "p_ep"), "crop_minus_direct": ("crop", "direct")}
MAIN = ("direct_minus_p", "crop_minus_p")
POPULATIONS = {"ddr": {"primary_3540": "primary", "broader_3725": None}, "idrid": {"primary_100": "primary", "all_103": None}}
SPLIT_FILES = {"ddr": "ddr_test.csv", "idrid": "idrid_test.csv"}


def reading(lo, hi):
    if lo > 0:
        return "supported: the interval lies wholly above zero"
    if hi < 0:
        return "reversed: the interval lies wholly below zero"
    return "no difference shown: the interval includes zero (no equivalence or non-inferiority statement)"


# --------------------------------------------------------------------------- statistics

def qwk_difference(tables, a, b, mask, indices):
    """QWK(a) - QWK(b) per seed-labelled pair on the masked images, their mean, and the percentile interval of
    that mean over `indices` (the same resamples applied to every table)."""
    pairs = [(mt.subset(tables[s][a], mask), mt.subset(tables[s][b], mask)) for s in SEEDS]
    for x, y in pairs:
        mt.check_paired(x, y)
    per_seed = [mt.qwk(x["grade"], x["pred"]) - mt.qwk(y["grade"], y["pred"]) for x, y in pairs]
    draws = [float(np.mean([mt.qwk(x["grade"][i], x["pred"][i]) - mt.qwk(y["grade"][i], y["pred"][i]) for x, y in pairs]))
             for i in indices]
    lo, hi = mt.interval(draws)
    positive = int(sum(d > 0 for d in per_seed))
    return {"mean": float(np.mean(per_seed)), "per_seed": {str(s): float(d) for s, d in zip(SEEDS, per_seed)},
            "ci_low": lo, "ci_high": hi, "seeds_positive": positive, "seed_level_inconsistency": positive != len(SEEDS),
            "n": int(np.sum(mask)), "resamples": len(indices)}


def summarise(tables, mask, models=MODELS, contrasts=CONTRASTS, main=MAIN, n_boot=mt.N_BOOT):
    """Per-seed metrics of every model, mean and SD over the seeds, bootstrap intervals of the seed mean, and the
    paired differences of every contrast (point difference per seed, interval of the seed mean)."""
    per_seed = {}
    for seed in SEEDS:
        per_seed[seed] = {}
        for model in models:
            m = mt.metrics(mt.subset(tables[seed][model], mask))
            per_seed[seed][model] = {**mt.headline(m), "recall_per_grade": m["recall_per_grade"], "confusion_matrix": m["confusion_matrix"]}

    def stat(values):
        return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)), "per_seed": [float(v) for v in values]}

    aggregate = {}
    for model in models:
        aggregate[model] = {k: stat([per_seed[s][model][k] for s in SEEDS]) for k in mt.HEADLINE_KEYS}
        for g in range(5):
            aggregate[model][f"recall_grade{g}"] = stat([per_seed[s][model]["recall_per_grade"][g] for s in SEEDS])
    grades = mt.subset(tables[SEEDS[0]][models[0]], mask)["grade"]
    indices = mt.bootstrap_indices(grades, n_boot)
    intervals = {}
    for model in models:
        draws = {k: [] for k in mt.BOOT_KEYS}
        for idx in indices:
            values = [mt.bootstrap_metrics(mt.subset(tables[s][model], mask), idx) for s in SEEDS]
            for k in mt.BOOT_KEYS:
                draws[k].append(np.mean([v[k] for v in values]))
        intervals[model] = {k: [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))] for k, d in draws.items()}
    differences = {}
    for name, (a, b) in contrasts.items():
        point, draws = {k: [] for k in mt.METRICS}, {k: [] for k in mt.BOOT_KEYS}
        recall = {g: [] for g in range(5)}
        for s in SEEDS:
            d = mt.paired_delta(mt.subset(tables[s][a], mask), mt.subset(tables[s][b], mask), indices=indices)
            for k in mt.METRICS:
                point[k].append(d[k]["delta"])
            for k in mt.BOOT_KEYS:
                draws[k].append(d[k]["_draws"])
            for g in range(5):
                recall[g].append(per_seed[s][a]["recall_per_grade"][g] - per_seed[s][b]["recall_per_grade"][g])
        differences[name] = {k: stat(v) for k, v in point.items()}
        for k in mt.BOOT_KEYS:
            lo, hi = np.nanpercentile(np.mean(np.stack(draws[k], 0), 0), [2.5, 97.5])
            differences[name][k].update(ci_low=float(lo), ci_high=float(hi))
        for g in range(5):
            differences[name][f"recall_grade{g}"] = stat(recall[g])
        for cut in ("auroc_ge1", "auroc_ge2", "auroc_ge3", "auroc_ge4", "grade4_recall"):
            differences[name][cut] = stat([per_seed[s][a][cut] - per_seed[s][b][cut] for s in SEEDS])
        differences[name]["role"] = "main" if name in main else "secondary, descriptive"
        differences[name]["decoded_grades_differ"] = [int((mt.subset(tables[s][a], mask)["pred"] != mt.subset(tables[s][b], mask)["pred"]).sum())
                                                       for s in SEEDS]
    return {"n": int(mask.sum()), "grade_counts": np.bincount(grades, minlength=5).tolist(), "per_seed": per_seed,
            "aggregate": aggregate, "seed_mean_interval_95": intervals, "paired_differences": differences,
            "n_boot": int(n_boot), "boot_seed": mt.BOOT_SEED, "bootstrap": "grade-stratified over the test images"}


def analyse(tables, frames, mask, families=None):
    """One population: every contrast with interval and reading, the metric block, the seed ensemble, the
    referable-DR operating point and calibration; on DDR also the main contrasts inside each file-name family."""
    grades = tables[SEEDS[0]][MODELS[0]]["grade"][mask]
    indices = mt.bootstrap_indices(grades)
    contrasts = {}
    for name, (a, b) in CONTRASTS.items():
        d = qwk_difference(tables, a, b, mask, indices)
        d["reading"] = reading(d["ci_low"], d["ci_high"])
        contrasts[name] = d
    ensembles = {m: mt.ensemble([tables[s][m] for s in SEEDS]) for m in MODELS}
    out = {"n": int(mask.sum()), "grade_counts": np.bincount(grades, minlength=5).tolist(), "contrasts_qwk": contrasts,
           "metric_block": summarise(tables, mask),
           "seed_ensemble_qwk": {m: mt.qwk(grades, ensembles[m]["pred"][mask]) for m in MODELS},
           "referable_dr": {m: {str(s): mt.referable(mt.subset(tables[s][m], mask)) for s in SEEDS} for m in MODELS},
           "calibration": {m: {str(s): mt.calibration(frames[(m, s)][mask]) for s in SEEDS} for m in MODELS}}
    if families is not None:
        out["by_family"] = {}
        for family in ("timestamp", "007"):
            inside = mask & (families == family)
            family_indices = mt.bootstrap_indices(tables[SEEDS[0]]["p"]["grade"][inside])
            out["by_family"][family] = {"n": int(inside.sum()),
                                        **{name: qwk_difference(tables, *CONTRASTS[name], inside, family_indices) for name in MAIN}}
        out["by_family"]["limit"] = "grade 0 and grade 4 are never in both families: this cannot show independence of source"
    return out


# --------------------------------------------------------------------------- inputs and outputs

def read_split(splits_dir, dataset):
    """The evaluation list in prediction order: {'ids', 'grades', 'primary', 'family' (DDR only)}."""
    import pandas as pd
    frame = pd.read_csv(os.path.join(splits_dir, SPLIT_FILES[dataset]), dtype={"image": str})
    out = {"ids": frame["image"].to_list(), "grades": frame["grade"].to_numpy(int), "primary": frame["primary"].to_numpy(bool)}
    if "family" in frame:
        out["family"] = frame["family"].astype(str).to_numpy()
    return out


def prediction_path(directory, model, seed):
    return os.path.join(directory, f"per_image_{model}_seed{seed}.csv")


def load_tables(split, predictions_dir, baselines_dir):
    """Per-seed tables of the four models, checked to be the evaluation list's images in its order."""
    tables, frames = {s: {} for s in SEEDS}, {}
    for model in MODELS:
        directory = baselines_dir if model in BASELINES else predictions_dir
        for seed in SEEDS:
            frame = mt.read_table(prediction_path(directory, model, seed))
            if frame["image_id"].to_list() != split["ids"] or not np.array_equal(frame["true_grade"].to_numpy(int), split["grades"]):
                raise ValueError(f"{prediction_path(directory, model, seed)} is not the evaluation list's images in its order")
            tables[seed][model], frames[(model, seed)] = mt.arrays(frame), frame
    return tables, frames


def evaluate(dataset, splits_dir, predictions_dir, baselines_dir, out_dir=None):
    """Every population of one dataset from per-image prediction tables. Returns {population: result} and, with
    `out_dir`, writes result_<population>.json there."""
    split = read_split(splits_dir, dataset)
    tables, frames = load_tables(split, predictions_dir, baselines_dir)
    results = {}
    for name, which in POPULATIONS[dataset].items():
        mask = split["primary"] if which == "primary" else np.ones(len(split["ids"]), bool)
        results[name] = mt.jsonable(analyse(tables, frames, mask, split.get("family")))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"result_{name}.json"), "w") as fh:
                json.dump(results[name], fh, indent=1)
    return results


def predict(cfg, image_paths, ids, grades, out_dir, arms=RUN_MODELS, batch_size=8, log=print):
    """Per-image prediction tables of both arms' three seeds for raw images (one arm's graders in memory at a
    time). Writes per_image_<arm>_seed<seed>.csv to `out_dir`."""
    import gc

    import pandas as pd

    from . import grader as gr
    from . import preprocess
    os.makedirs(out_dir, exist_ok=True)
    for arm in arms:
        graders = gr.load_arm(cfg, arm)
        logits = {s: [] for s in SEEDS}
        for start in range(0, len(image_paths), batch_size):
            rgb = np.stack([preprocess.frame(p, arm) for p in image_paths[start:start + batch_size]])
            for s in SEEDS:
                logits[s].append(graders[s].logits(rgb, batch_size))
            if (start // batch_size) % 25 == 0:
                log(f"  {arm}: {min(start + batch_size, len(image_paths))}/{len(image_paths)} images")
        for s in SEEDS:
            pd.DataFrame(mt.per_image_rows(ids, grades, np.concatenate(logits[s]))).to_csv(prediction_path(out_dir, arm, s), index=False)
        del graders
        gc.collect()
