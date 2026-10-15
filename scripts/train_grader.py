"""Trains the grader on joint APTOS 2019 + EyePACS frames (see drgrade/train_grader.py for the protocol).

    python scripts/prepare_frames.py --dataset aptos   --arm crop
    python scripts/prepare_frames.py --dataset eyepacs --arm crop
    python scripts/train_grader.py --frame crop --seed 42

`--frame crop|direct` selects the frames (both caches must be built for it), `--seed` one of 42 / 123 / 2026 (any
integer works). The run folder is <output_dir>/training/grader_<frame>_seed<seed>; rerunning the same command after
a crash resumes from the last finished epoch. The best checkpoint is copied to
<weights_dir>/grader_<frame>_seed<seed>.weights.h5 when training ends (use --no-export to skip).

Needs the ImageNet ConvNeXt-Tiny file convnext_tiny_imagenet_notop.h5 in the weights folder
(python scripts/download_weights.py). Training from scratch is not bit-identical to the released models (GPU
nondeterminism); expect validation QWK within a few thousandths.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from drgrade import data, frames, train_grader as tg  # noqa: E402
from drgrade.config import WEIGHT_FILES, load_config  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame", required=True, choices=("crop", "direct"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config")
    parser.add_argument("--aptos-frames", help="default: <output_dir>/frames/aptos_<frame>")
    parser.add_argument("--eyepacs-frames", help="default: <output_dir>/frames/eyepacs_<frame>")
    parser.add_argument("--run-dir")
    parser.add_argument("--max-epochs", type=int, help="stop after this many epochs (default 50)")
    parser.add_argument("--steps-per-epoch", type=int, help="truncate every epoch to N steps (smoke tests)")
    parser.add_argument("--batch-size", type=int, help="override the protocol's batch size of 16 (smoke tests on small machines only)")
    parser.add_argument("--allow-partial", action="store_true", help="accept frame caches that hold fewer images than the full splits")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    aptos_cache = frames.FrameCache(args.aptos_frames or frames.cache_dir(cfg, "aptos", args.frame))
    eyepacs_cache = frames.FrameCache(args.eyepacs_frames or frames.cache_dir(cfg, "eyepacs", args.frame))
    for cache in (aptos_cache, eyepacs_cache):
        if cache.arm != args.frame:
            sys.exit(f"{cache.directory} holds {cache.arm} frames, not {args.frame}")
    aptos = {s: aptos_cache.entries(s) for s in ("train", "val")}
    eyepacs = {s: eyepacs_cache.entries(s) for s in ("train", "val")}
    sizes = {"aptos_train": len(aptos["train"]), "eyepacs_train": len(eyepacs["train"]), "aptos_val": len(aptos["val"]),
             "eyepacs_val": len(eyepacs["val"])}
    if sizes != tg.FULL_COUNTS and not args.allow_partial:
        sys.exit(f"the frame caches hold {sizes}, the full splits are {tg.FULL_COUNTS}; build them with prepare_frames.py "
                 f"or pass --allow-partial for a smoke test")
    run_dir = args.run_dir or os.path.join(cfg["output_dir"], "training", f"grader_{args.frame}_seed{args.seed}")
    imagenet = os.path.join(cfg["weights_dir"], WEIGHT_FILES["convnext_imagenet"])
    if not os.path.exists(imagenet):
        sys.exit(f"{imagenet} is missing: run `python scripts/download_weights.py` first")
    state = tg.train(run_dir, tg.JointFrames(aptos_cache, eyepacs_cache), aptos, eyepacs, args.seed, args.frame,
                     imagenet_weights=imagenet, protocol=dict(tg.PROTOCOL, **({"batch_size": args.batch_size} if args.batch_size else {})),
                     mixed_precision=not args.no_mixed_precision, max_epochs=args.max_epochs,
                     steps_per_epoch=args.steps_per_epoch, workers=args.workers)
    best = state["best"]
    print(f"finished: {state['stopped']}; best epoch index {best['epoch_index']} (0-based): APTOS val QWK "
          f"{best['aptos_val_qwk']:.4f}, EyePACS val QWK {best['eyepacs_val_qwk']:.4f}, selection {best['selection']:.4f}")
    if not args.no_export:
        out = os.path.join(cfg["weights_dir"], f"grader_{args.frame}_seed{args.seed}.weights.h5")
        if os.path.exists(out):
            out = os.path.join(run_dir, "best_export.weights.h5")      # never overwrite weights already in the folder
        print("best checkpoint:", tg.export_best(run_dir, out))


if __name__ == "__main__":
    main()
