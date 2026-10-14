"""External evaluation on DDR or IDRiD.

    # statistics from per-image prediction tables (the released ones in results/predictions/, or your own)
    python scripts/evaluate_external.py ddr --predictions results/predictions/ddr

    # predict first: both grader arms, three seeds each, on the raw test images (CPU: about 1 s per image per arm)
    python scripts/evaluate_external.py idrid --predict

The baselines p and p_ep are read from data/baselines/<dataset>/. Results go to <output_dir>/external/<dataset>/:
result_<population>.json per population (DDR: primary_3540, broader_3725; IDRiD: primary_100, all_103).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from drgrade import evaluate as ev  # noqa: E402
from drgrade.config import ROOT, load_config  # noqa: E402

SPLITS = os.path.join(ROOT, "data", "splits")


def image_paths(cfg, dataset, ids):
    root = cfg["datasets"][dataset]
    if dataset == "ddr":
        folder = os.path.join(root, "DR_grading", "test")
        return [os.path.join(folder, i) for i in ids]
    folder = os.path.join(root, "B. Disease Grading", "1. Original Images", "b. Testing Set")
    return [os.path.join(folder, f"{i}.jpg") for i in ids]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=("ddr", "idrid"))
    parser.add_argument("--config")
    parser.add_argument("--predictions", help="folder with per_image_<crop|direct>_seed<seed>.csv")
    parser.add_argument("--predict", action="store_true", help="run both grader arms on the raw images first")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    out = args.out or os.path.join(cfg["output_dir"], "external", args.dataset)
    predictions = args.predictions
    if args.predict:
        split = ev.read_split(SPLITS, args.dataset)
        paths = image_paths(cfg, args.dataset, split["ids"])
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            sys.exit(f"{len(missing)} test images are missing, e.g. {missing[0]}; see data/README.md")
        predictions = os.path.join(out, "predictions")
        ev.predict(cfg, paths, split["ids"], split["grades"], predictions)
    if not predictions:
        sys.exit("give --predictions DIR or --predict")
    results = ev.evaluate(args.dataset, SPLITS, predictions, os.path.join(ROOT, "data", "baselines", args.dataset), out)
    for population, r in results.items():
        print(f"{args.dataset} {population} (n = {r['n']}):")
        for name in ev.MAIN:
            c = r["contrasts_qwk"][name]
            print(f"  {name:<16} QWK difference {c['mean']:+.3f} (95 % interval {c['ci_low']:+.3f} to {c['ci_high']:+.3f}); {c['reading']}")
        print("  seed-ensemble QWK: " + ", ".join(f"{m} {v:.3f}" for m, v in r["seed_ensemble_qwk"].items()))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
