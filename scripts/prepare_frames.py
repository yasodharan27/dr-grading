"""Builds the 512 x 512 grader frames (Stage 2, then whole image or fundus box, then resize) of the training images.

    python scripts/prepare_frames.py --dataset aptos   --arm crop
    python scripts/prepare_frames.py --dataset eyepacs --arm crop --workers 4
    python scripts/prepare_frames.py --dataset aptos   --arm direct --limit 50     # first 50 images of each split
    python scripts/prepare_frames.py --dataset aptos   --arm crop --ids 000c1434d8d7 001639a390f0

Output: <output_dir>/frames/<dataset>_<arm>/ (shards of float16 frames, about 1.5 MB per frame: APTOS 3,651 frames
5.5 GB, EyePACS 35,122 frames 53 GB). The build is resumable: shards that already exist are kept. This is the only
place frames are written; nothing is written outside the configured output folder.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drgrade import data, frames  # noqa: E402
from drgrade.config import load_config  # noqa: E402

DATASETS = {"aptos": (data.aptos_entries, data.aptos_path), "eyepacs": (data.eyepacs_entries, data.eyepacs_path)}


def select(entries, limit=None, ids=None, limit_val=None):
    """Applies --ids (keep these) and --limit / --limit-val (first N of the train / val split) to {split: [(id, grade)]}."""
    out = {}
    for split, rows in entries.items():
        if ids:
            rows = [r for r in rows if r[0] in ids]
        n = (limit_val if limit_val is not None else limit) if split == "val" else limit
        out[split] = rows[:n] if n else rows
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--arm", required=True, choices=("crop", "direct"))
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=("train", "val"))
    parser.add_argument("--limit", type=int, help="only the first N images of the train split (of each split without --limit-val)")
    parser.add_argument("--limit-val", type=int, help="only the first N images of the val split")
    parser.add_argument("--ids", nargs="+", help="only these image ids (file names for EyePACS)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--config")
    parser.add_argument("--out", help="override <output_dir>/frames/<dataset>_<arm>")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    read_entries, path_of = DATASETS[args.dataset]
    chosen = select({s: read_entries()[s] for s in args.splits}, args.limit, set(args.ids) if args.ids else None, args.limit_val)
    items = {s: [(i, g, path_of(cfg, i)) for i, g in rows] for s, rows in chosen.items()}
    missing = [p for rows in items.values() for _, _, p in rows if not os.path.exists(p)]
    if missing:
        sys.exit(f"{len(missing)} raw images are missing, e.g. {missing[0]}; check datasets.{args.dataset} in the config")
    out = args.out or frames.cache_dir(cfg, args.dataset, args.arm)
    print(f"{args.dataset} ({args.arm}): " + ", ".join(f"{s} {len(r)}" for s, r in items.items()) + f" -> {out}")
    frames.build(items, out, args.arm, args.workers)
    print(frames.json_summary(out))


if __name__ == "__main__":
    main()
