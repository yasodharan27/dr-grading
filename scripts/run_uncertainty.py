"""Uncertainty of the three-seed ensembles: error detection, selective prediction against random referral, seed
disagreement. Computed from per-image prediction tables (no model is run).

    python scripts/run_uncertainty.py                                   # the released tables in results/
    python scripts/run_uncertainty.py --predictions DIR --validation DIR2

Sets, for each grader arm (crop, direct): the APTOS and EyePACS validation sets (results/validation/), and the DDR
broader list (3,725 images) and all 103 IDRiD test images (results/predictions/). The validation sets also selected
the checkpoints, so their numbers are optimistic.

Output: <out>/uncertainty_results.json and one uncertainty_<arm>_<set>.csv per set (image id, grade, ensemble grade,
the three scores, class probabilities).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drgrade import uncertainty  # noqa: E402
from drgrade.config import ARMS, ROOT, SEEDS  # noqa: E402

SETS = ("aptos_val", "eyepacs_val", "ddr", "idrid")


def sets(predictions, validation):
    out = {}
    for arm in ARMS:
        for name in SETS:
            if name.endswith("_val"):
                out[f"{arm}_{name}"] = {s: os.path.join(validation, f"{arm}_{name}_seed{s}.csv") for s in SEEDS}
            else:
                out[f"{arm}_{name}"] = {s: os.path.join(predictions, name, f"per_image_{arm}_seed{s}.csv") for s in SEEDS}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", default=os.path.join(ROOT, "results", "predictions"))
    parser.add_argument("--validation", default=os.path.join(ROOT, "results", "validation"))
    parser.add_argument("--out", default=os.path.join(ROOT, "results", "uncertainty"))
    args = parser.parse_args(argv)
    wanted = sets(args.predictions, args.validation)
    missing = [p for paths in wanted.values() for p in paths.values() if not os.path.exists(p)]
    if missing:
        sys.exit(f"{len(missing)} prediction tables are missing, e.g. {missing[0]}")
    results = uncertainty.run(wanted, args.out)
    for name, r in results.items():
        e = r["scores"]["entropy"]
        at80 = next(x for x in e["selective"] if x["coverage"] == 0.8)
        rnd = next(x for x in r["random_referral"] if x["coverage"] == 0.8)
        print(f"{name:<18} n={r['images']:>5}  ensemble QWK {r['ensemble_qwk']:.4f}  seeds disagree on {100 * r['seeds_disagree_on_grade']:.1f} %  "
              f"error-detection AUROC {e['error_detection_auroc']:.3f}  QWK at 80 % coverage {at80['qwk']:.4f} (random {rnd['qwk_mean']:.4f})")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
