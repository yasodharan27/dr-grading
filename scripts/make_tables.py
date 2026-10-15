"""Regenerates every results table from the released per-image prediction tables.

    python scripts/make_tables.py                  # writes results/external/, results/tables/ and results/TABLES.md
    python scripts/make_tables.py --check          # recompute and compare with the committed files; exit 1 on a difference

Inputs: results/predictions/<ddr|idrid>/ (the two graders, three seeds), data/baselines/<ddr|idrid>/ (the two
baselines), results/validation/ (development sets), results/uncertainty/uncertainty_results.json
(scripts/run_uncertainty.py), results/explanations/localisation_summary.csv (scripts/run_explanations.py), data/*.

Tables:
  dev_validation.csv       per arm and seed: APTOS / EyePACS validation QWK, the monitored mean, best epoch (0-based)
  external_models.csv      per dataset, population, model: QWK per seed, mean, SD, seed-ensemble QWK, referable DR
  external_contrasts.csv   QWK differences with paired grade-stratified bootstrap intervals (2,000 resamples)
  uncertainty_summary.csv  ensemble QWK, seed disagreement, error detection, selective prediction at 80 % coverage
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drgrade import evaluate as ev  # noqa: E402
from drgrade import metrics as mt  # noqa: E402
from drgrade.config import ARMS, ROOT, SEEDS  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
SPLITS = os.path.join(ROOT, "data", "splits")
MODEL_ORDER = ("p", "p_ep", "direct", "crop")
MODEL_NAMES = {"p": "P (APTOS only, ImageNet start)", "p_ep": "P-EP (APTOS, EyePACS-adapted start)",
               "direct": "joint, whole-image frames (J)", "crop": "joint, fundus-cropped frames (J-crop)"}


def dev_validation():
    runs = pd.read_csv(os.path.join(ROOT, "data", "training_runs.csv"))
    rows = []
    for r in runs.itertuples():
        row = {"arm": r.arm, "seed": r.seed, "best_epoch_index_0based": r.best_epoch_index_0based, "epochs_run": r.epochs_run}
        for name in ("aptos_val", "eyepacs_val"):
            t = mt.read_table(os.path.join(RESULTS, "validation", f"{r.arm}_{name}_seed{r.seed}.csv"))
            row[f"{name}_qwk"] = float(mt.qwk(t["true_grade"].to_numpy(int), t["predicted_grade"].to_numpy(int)))
        row["selection"] = 0.5 * (row["aptos_val_qwk"] + row["eyepacs_val_qwk"])
        rows.append(row)
    return pd.DataFrame(rows)


def external(results):
    models, contrasts = [], []
    for dataset, populations in results.items():
        for population, r in populations.items():
            block = r["metric_block"]
            for m in MODEL_ORDER:
                per_seed = [block["per_seed"][str(s)][m]["qwk"] for s in SEEDS]
                ref = [r["referable_dr"][m][str(s)] for s in SEEDS]
                models.append({"dataset": dataset, "population": population, "n": r["n"], "model": m,
                               **{f"qwk_seed{s}": v for s, v in zip(SEEDS, per_seed)}, "qwk_mean": float(np.mean(per_seed)),
                               "qwk_sd": float(np.std(per_seed, ddof=1)), "qwk_seed_ensemble": r["seed_ensemble_qwk"][m],
                               "mean_cut_auroc_mean": block["aggregate"][m]["mean_cut_auroc"]["mean"],
                               "referable_sensitivity_mean": float(np.mean([x["sensitivity"] for x in ref])),
                               "referable_specificity_mean": float(np.mean([x["specificity"] for x in ref]))})
            for name, c in r["contrasts_qwk"].items():
                contrasts.append({"dataset": dataset, "population": population, "n": r["n"], "contrast": name,
                                  "role": "main" if name in ev.MAIN else "descriptive", "qwk_difference_mean": c["mean"],
                                  "ci_low": c["ci_low"], "ci_high": c["ci_high"],
                                  **{f"diff_seed{s}": c["per_seed"][str(s)] for s in SEEDS}, "seeds_positive": c["seeds_positive"],
                                  "reading": c["reading"]})
    return pd.DataFrame(models), pd.DataFrame(contrasts)


def uncertainty_summary():
    with open(os.path.join(RESULTS, "uncertainty", "uncertainty_results.json"), encoding="utf-8") as fh:
        results = json.load(fh)
    rows = []
    for name, r in results.items():
        arm, which = name.split("_", 1)
        e = r["scores"]["entropy"]
        at = next(x for x in e["selective"] if x["coverage"] == 0.8)
        rnd = next(x for x in r["random_referral"] if x["coverage"] == 0.8)
        rows.append({"arm": arm, "set": which, "images": r["images"], "ensemble_qwk": r["ensemble_qwk"],
                     "seeds_disagree_on_grade": r["seeds_disagree_on_grade"], "error_detection_auroc_entropy": e["error_detection_auroc"],
                     "qwk_at_80pct_coverage_entropy": at["qwk"], "qwk_at_80pct_coverage_random_mean": rnd["qwk_mean"]})
    return pd.DataFrame(rows)


def markdown(dev, models, contrasts, unc, localisation):
    def f(v, d=3):
        return f"{v:.{d}f}"
    lines = ["# Results tables (generated by scripts/make_tables.py)", ""]
    lines += ["## Development validation sets (the monitored mean of the two QWKs selected each checkpoint)", "",
              "| arm | seed | APTOS val QWK | EyePACS val QWK | mean | best epoch (0-based) | epochs run |", "|---|---|---|---|---|---|---|"]
    for r in dev.itertuples():
        lines.append(f"| {r.arm} | {r.seed} | {f(r.aptos_val_qwk, 4)} | {f(r.eyepacs_val_qwk, 4)} | {f(r.selection, 4)} | {r.best_epoch_index_0based} | {r.epochs_run} |")
    for dataset, population in (("ddr", "primary_3540"), ("idrid", "primary_100")):
        sub = models[(models.dataset == dataset) & (models.population == population)]
        lines += ["", f"## {dataset.upper()} {population} (n = {int(sub.n.iloc[0])}): QWK", "",
                  "| model | seed 42 | seed 123 | seed 2026 | mean (SD) | seed ensemble |", "|---|---|---|---|---|---|"]
        for r in sub.itertuples():
            lines.append(f"| {MODEL_NAMES[r.model]} | {f(r.qwk_seed42)} | {f(r.qwk_seed123)} | {f(r.qwk_seed2026)} | {f(r.qwk_mean)} ({f(r.qwk_sd)}) | {f(r.qwk_seed_ensemble)} |")
        c = contrasts[(contrasts.dataset == dataset) & (contrasts.population == population)]
        lines += ["", "| contrast (QWK difference) | mean | 95 % interval | seeds positive |", "|---|---|---|---|"]
        for r in c.itertuples():
            lines.append(f"| {r.contrast.replace('_minus_', ' - ')}{' (main)' if r.role == 'main' else ''} | {r.qwk_difference_mean:+.3f} | "
                         f"{r.ci_low:+.3f} to {r.ci_high:+.3f} | {r.seeds_positive} of 3 |")
    lines += ["", "## Uncertainty (three-seed ensemble)", "",
              "| arm | set | images | ensemble QWK | seeds disagree | error-detection AUROC (entropy) | QWK at 80 % coverage | random referral |",
              "|---|---|---|---|---|---|---|---|"]
    for r in unc.itertuples():
        lines.append(f"| {r.arm} | {r.set} | {r.images} | {f(r.ensemble_qwk)} | {100 * r.seeds_disagree_on_grade:.1f} % | "
                     f"{f(r.error_detection_auroc_entropy)} | {f(r.qwk_at_80pct_coverage_entropy)} | {f(r.qwk_at_80pct_coverage_random_mean)} |")
    lines += ["", "## Localisation of explanation maps on DDR lesion masks (60 images; Occlusion 20)", "",
              "| arm | method | images | pointing game | top-10 % precision | AUROC | uniform-map baseline |", "|---|---|---|---|---|---|---|"]
    for r in localisation.itertuples():
        lines.append(f"| {r.arm} | {r.method} | {r.images} | {f(r.pointing_hit)} | {f(r.top10_precision)} | {f(r.auroc)} | {f(r.uniform_map_baseline)} |")
    return "\n".join(lines) + "\n"


def build(out_dir_external, out_dir_tables, datasets=("ddr", "idrid")):
    results = {}
    for dataset in datasets:
        results[dataset] = ev.evaluate(dataset, SPLITS, os.path.join(RESULTS, "predictions", dataset),
                                       os.path.join(ROOT, "data", "baselines", dataset), os.path.join(out_dir_external, dataset))
    dev = dev_validation()
    models, contrasts = external(results)
    unc = uncertainty_summary()
    localisation = pd.read_csv(os.path.join(RESULTS, "explanations", "localisation_summary.csv"))
    os.makedirs(out_dir_tables, exist_ok=True)
    tables = {"dev_validation.csv": dev, "external_models.csv": models, "external_contrasts.csv": contrasts, "uncertainty_summary.csv": unc}
    for name, t in tables.items():
        t.to_csv(os.path.join(out_dir_tables, name), index=False, lineterminator="\n", float_format="%.6f")
    with open(os.path.join(out_dir_tables, "TABLES.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(markdown(dev, models, contrasts, unc, localisation))
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="recompute into a temporary folder and compare with results/")
    parser.add_argument("--datasets", nargs="+", default=["ddr", "idrid"], choices=("ddr", "idrid"),
                        help="with --check: only these datasets' external results (DDR's bootstrap takes about 15 minutes)")
    args = parser.parse_args(argv)
    if not args.check:
        build(os.path.join(RESULTS, "external"), os.path.join(RESULTS, "tables"), tuple(args.datasets))
        print("written results/external/ and results/tables/")
        return
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        build(os.path.join(tmp, "external"), os.path.join(tmp, "tables"), tuple(args.datasets))
        bad = []
        if set(args.datasets) == {"ddr", "idrid"}:                      # the tables span both datasets
            for name in sorted(os.listdir(os.path.join(tmp, "tables"))):
                a, b = os.path.join(tmp, "tables", name), os.path.join(RESULTS, "tables", name)
                if not os.path.exists(b) or open(a, "rb").read() != open(b, "rb").read():
                    bad.append(name)
        for dataset in args.datasets:
            for name in sorted(os.listdir(os.path.join(tmp, "external", dataset))):
                a, b = os.path.join(tmp, "external", dataset, name), os.path.join(RESULTS, "external", dataset, name)
                if not os.path.exists(b) or json.load(open(a)) != json.load(open(b)):
                    bad.append(f"external/{dataset}/{name}")
    print("all tables identical to the committed ones" if not bad else f"DIFFERENT: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
