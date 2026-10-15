"""Trains Stage 4 (lesion segmentation: MA, HE, EX, SE) on IDRiD (44 / 10) and TJDR (443 / 110).

    python scripts/train_lesions.py                      # build the cache if needed, then train (12,000 iterations)
    python scripts/train_lesions.py --cache-only         # only build the 1536 x 1536 training cache

The cache (<output_dir>/lesion_cache, about 3.6 GB) and the run (<output_dir>/training/lesions) are resumable: rerun
the same command after a crash. The best EMA weights are copied to <weights_dir>/lesions_unet_se_resnet101.pt when
training ends, unless that file exists (then they stay in the run folder as best_ema.pt).

Data note: the IDRiD Stage-2 images pass through a JPEG round trip (quality 95) before the 1536 resize, because the
released model was trained on Stage-2 files written as .jpg; this reproduces its training images exactly
(--no-idrid-jpeg keeps them lossless). TJDR images were lossless PNG.

Selection note: the best checkpoint is chosen on TJDR's official TEST split, used here as validation (together with
the 10 IDRiD validation images). The Stage-4 numbers it prints are selection-set numbers, not a held-out evaluation.
The ImageNet SE-ResNet-101 encoder is downloaded from Hugging Face (smp-hub/se_resnet101.imagenet, 198 MB).
"""
import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drgrade import train_lesions as tl  # noqa: E402
from drgrade.config import WEIGHT_FILES, load_config  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--cache-dir", help="default: <output_dir>/lesion_cache")
    parser.add_argument("--run-dir", help="default: <output_dir>/training/lesions")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--limit", type=int, help="first N images per dataset and role in the cache (smoke tests)")
    parser.add_argument("--no-idrid-jpeg", action="store_true",
                        help="keep the IDRiD Stage-2 images lossless (the released model's were JPEG quality 95)")
    parser.add_argument("--allow-partial", action="store_true", help="train on a cache that holds fewer than the 607 images")
    parser.add_argument("--iterations", type=int, default=tl.TrainConfig.iterations)
    parser.add_argument("--max-steps", type=int, help="stop after this many steps of this invocation (smoke tests)")
    parser.add_argument("--batch-size", type=int, default=tl.TrainConfig.batch_size)
    parser.add_argument("--patch", type=int, default=tl.TrainConfig.patch, help="patch size; the protocol's 512 unless a small machine needs less")
    parser.add_argument("--val-every", type=int, default=tl.TrainConfig.val_every)
    parser.add_argument("--checkpoint-every", type=int, default=tl.TrainConfig.checkpoint_every)
    parser.add_argument("--warmup-steps", type=int, default=tl.TrainConfig.warmup_steps)
    parser.add_argument("--num-workers", type=int, default=tl.TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=tl.TrainConfig.seed)
    parser.add_argument("--device", help="cuda or cpu (default: cuda when available)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    cache_dir = args.cache_dir or os.path.join(cfg["output_dir"], "lesion_cache")
    manifest = tl.build_cache(cfg, cache_dir, args.limit, idrid_jpeg=not args.no_idrid_jpeg)
    print(f"cache {cache_dir}: {manifest['counts']} (complete: {manifest['complete']})")
    if args.cache_only:
        return
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache = tl.TrainingCache(cache_dir, require_complete=not args.allow_partial)
    config = dataclasses.replace(tl.TrainConfig(), iterations=args.iterations, batch_size=args.batch_size, patch=args.patch,
                                 val_every=args.val_every, checkpoint_every=args.checkpoint_every, warmup_steps=args.warmup_steps,
                                 num_workers=args.num_workers, seed=args.seed)
    run_dir = args.run_dir or os.path.join(cfg["output_dir"], "training", "lesions")
    result = tl.train(run_dir, cache, config, device, args.max_steps)
    print(f"step {result['step']}: best selection score {result['best']['score']:.4f} at step {result['best']['step']}")
    if result["best"]["step"] is not None:
        release = os.path.join(cfg["weights_dir"], WEIGHT_FILES["lesions"])
        if not os.path.exists(release) and not args.limit:
            print("exported:", tl.export_best(run_dir, release))
        else:
            print("best EMA weights stay in", os.path.join(run_dir, "best_ema.pt"))


if __name__ == "__main__":
    main()
