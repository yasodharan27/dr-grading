"""Trains Stage 1 (image quality, EfficientNet-B0 on EyeQ) and evaluates it on the EyeQ test split.

    python scripts/train_quality.py                      # train, then evaluate the best model on the test split
    python scripts/train_quality.py --eval-only MODEL    # evaluate a saved .keras model only

The run folder is <output_dir>/training/quality; rerunning after a crash resumes from the last finished epoch. The
best model (lowest validation loss) is copied to <weights_dir>/quality_efficientnet_b0_eyeq.keras unless that file
exists (then it stays in the run folder as best_model.keras). Needs the ImageNet EfficientNet-B0 weights, which
Keras downloads on first use (16 MB).
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from drgrade import train_quality as tq  # noqa: E402
from drgrade.config import WEIGHT_FILES, load_config  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--run-dir")
    parser.add_argument("--eval-only", metavar="MODEL", help="evaluate this .keras model on the EyeQ test split and exit")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batch-size", type=int, help="override the protocol's 32 (smoke tests on small machines)")
    parser.add_argument("--limit", type=int, help="use only N random training images (smoke tests)")
    parser.add_argument("--limit-test", type=int, help="evaluate on N random test images (smoke tests)")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_dir = args.run_dir or os.path.join(cfg["output_dir"], "training", "quality")
    if args.eval_only:
        tq.evaluate_test(args.eval_only, cfg, args.limit_test, out_path=os.path.join(run_dir, "test_evaluation.json"))
        return
    protocol = dict(tq.PROTOCOL, **({"batch_size": args.batch_size} if args.batch_size else {}))
    state = tq.train(run_dir, cfg, protocol=protocol, limit=args.limit, max_epochs=args.max_epochs)
    best = state["best"]
    print(f"finished: {state['stopped'] or 'stopped at --max-epochs'}; best epoch index {best['epoch_index']} (0-based): "
          f"val loss {best['val_loss']:.4f}, val accuracy {best['val_accuracy']:.4f}")
    model_path = os.path.join(run_dir, "best_model.keras")
    release = os.path.join(cfg["weights_dir"], WEIGHT_FILES["quality"])
    if not os.path.exists(release) and not args.limit:
        os.makedirs(cfg["weights_dir"], exist_ok=True)
        shutil.copyfile(model_path, release)
        print("copied to", release)
    if not args.no_eval:
        tq.evaluate_test(model_path, cfg, args.limit_test, out_path=os.path.join(run_dir, "test_evaluation.json"))


if __name__ == "__main__":
    main()
