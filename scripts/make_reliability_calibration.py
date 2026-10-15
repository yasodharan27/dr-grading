"""Rebuilds data/reliability_calibration.json (the demo's reliability wording) from the per-image ensemble tables of
the two validation sets, APTOS 2019 validation (730 images) and EyePACS validation (3,512), for each grader arm:

    python scripts/make_reliability_calibration.py

Inputs: results/uncertainty/uncertainty_<arm>_{aptos,eyepacs}_val.csv (made by scripts/run_uncertainty.py; the columns
image_id, true_grade, ensemble_grade and entropy are used).
These validation sets also selected the checkpoints, so the rates are optimistic for new images.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drgrade.config import ARMS, ROOT  # noqa: E402
from drgrade.uncertainty import reliability_calibration  # noqa: E402

TABLES = os.path.join(ROOT, "results", "uncertainty")
SOURCE = "three-seed ensemble on the APTOS 2019 validation split (730) and the EyePACS validation split (3,512), pooled"
NOTE = "these validation sets also selected the checkpoints, so the rates are optimistic for new images"


def build(tables_dir=TABLES):
    return {"source": SOURCE, "note": NOTE,
            "arms": {arm: reliability_calibration([os.path.join(tables_dir, f"uncertainty_{arm}_{s}_val.csv") for s in ("aptos", "eyepacs")])
                     for arm in ARMS}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", default=TABLES)
    parser.add_argument("--out", default=os.path.join(ROOT, "data", "reliability_calibration.json"))
    args = parser.parse_args(argv)
    calibration = build(args.tables)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(calibration, fh, indent=1)
        fh.write("\n")
    for arm, c in calibration["arms"].items():
        print(f"{arm}: entropy threshold {c['entropy_threshold']:.3f}; certain {c['certain']['images']} images, exact "
              f"{100 * c['certain']['exact']:.1f} % / within one {100 * c['certain']['within_one']:.1f} %; uncertain "
              f"{c['uncertain']['images']} images, {100 * c['uncertain']['exact']:.1f} % / {100 * c['uncertain']['within_one']:.1f} %")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
