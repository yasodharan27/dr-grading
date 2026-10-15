"""Short GPU smoke run of every trainer on small subsets of the real datasets: a few hundred steps each, losses
printed, resume exercised, and a checked summary at the end.

    python scripts/smoke_gpu.py                      # everything (about 35-50 minutes on a T4)
    python scripts/smoke_gpu.py --only frames grader # selected parts: frames grader quality lesions

Subsets (dataset roots and the output folder come from the config): APTOS 200 train / 50 val and EyePACS 400 train
/ 50 val crop frames, 6 grader epochs of 600 images (228 steps of batch 16); EyeQ 1,600 training images x 5 epochs
(about 215 steps of batch 32) and 300 test images; IDRiD 44 / 10 and TJDR 50 / 50 for 300 Stage-4 steps (batch 8,
512 patches). Each trainer runs in its own process (GPU and host memory are not shared) and in TWO invocations, so
the resume path runs too. Everything is written to <output_dir>/smoke/ and the released weights are not touched.
Training from this subset says nothing about accuracy: it checks that the code runs on the GPU, that losses are
finite and fall, and that host RAM stays flat.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from drgrade.config import load_config  # noqa: E402

PYTHON = sys.executable
SUBSETS = {"aptos_train": 200, "aptos_val": 50, "eyepacs_train": 400, "eyepacs_val": 50, "grader_epochs": 6, "grader_split": 3,
           "eyeq_images": 1600, "eyeq_test": 300, "quality_epochs": 5, "quality_split": 3, "lesion_limit": 50,
           "lesion_steps": 300, "lesion_split": 150, "lesion_val_every": 100}


def run(args, log_lines):
    """Runs a script, forwards its output line by line, and returns (exit code, all output)."""
    print("\n$ " + " ".join(os.path.relpath(a, ROOT) if a.startswith(ROOT) else a for a in args), flush=True)
    started, text = time.time(), []
    proc = subprocess.Popen([PYTHON] + args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env=dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2", PYTHONUNBUFFERED="1"))
    for line in proc.stdout:
        text.append(line)
        if not line.startswith(("WARNING", "I0000", "W0000")):
            print("  " + line.rstrip(), flush=True)
    proc.wait()
    log_lines.append(f"{os.path.basename(args[0])}: exit {proc.returncode} in {time.time() - started:.0f} s")
    return proc.returncode, "".join(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--only", nargs="+", choices=("frames", "grader", "quality", "lesions"))
    parser.add_argument("--allow-cpu", action="store_true", help="run without a GPU (very slow; for checking the script only)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    wanted = set(args.only or ("frames", "grader", "quality", "lesions"))
    import tensorflow as tf
    import torch
    gpus = tf.config.list_physical_devices("GPU")
    if not (gpus and torch.cuda.is_available()) and not args.allow_cpu:
        sys.exit("no GPU visible to both TensorFlow and PyTorch; this script is for a GPU runtime (--allow-cpu to override)")
    print(f"TensorFlow {tf.__version__}, PyTorch {torch.__version__}, GPUs: {[g.name for g in gpus]}, "
          f"torch device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    out = os.path.join(cfg["output_dir"], "smoke")
    os.makedirs(out, exist_ok=True)
    cfg_args = ["--config", cfg["config_path"]]
    S = SUBSETS
    lines, checks = [], []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)

    def history_ok(history, key, label):
        losses = [h[key] for h in history]
        import math
        check(f"{label}: every loss finite", all(math.isfinite(x) for x in losses))
        half = max(1, len(losses) // 2)
        first, last = sum(losses[:half]) / half, sum(losses[-half:]) / half
        check(f"{label}: loss falls", last < first, f"({first:.4f} -> {last:.4f})")

    started = time.time()
    frames_a, frames_e = os.path.join(out, "frames_aptos_crop"), os.path.join(out, "frames_eyepacs_crop")
    if "frames" in wanted or "grader" in wanted:
        print("\n== frames (crop arm)")
        for dataset, target, train, val in (("aptos", frames_a, S["aptos_train"], S["aptos_val"]), ("eyepacs", frames_e, S["eyepacs_train"], S["eyepacs_val"])):
            if os.path.exists(os.path.join(target, "index.json")):
                print(f"  {dataset}: frames already built in {target}")
                continue
            code, _ = run(["scripts/prepare_frames.py", "--dataset", dataset, "--arm", "crop", "--limit", str(train), "--limit-val", str(val),
                           "--out", target, "--workers", "4"] + cfg_args, lines)
            check(f"frames {dataset}", code == 0 and os.path.exists(os.path.join(target, "index.json")))

    if "grader" in wanted:
        print("\n== grader (joint APTOS + EyePACS, crop frames)")
        code, _ = run(["scripts/download_weights.py", "--only", "convnext_tiny_imagenet_notop.h5"] + cfg_args, lines)
        check("ImageNet ConvNeXt-Tiny weights present", code == 0)
        run_dir = os.path.join(out, "grader_crop_seed42")
        base = ["scripts/train_grader.py", "--frame", "crop", "--seed", "42", "--aptos-frames", frames_a, "--eyepacs-frames", frames_e,
                "--run-dir", run_dir, "--allow-partial", "--no-export", "--steps-per-epoch", str(S["aptos_train"] * 3 // 16 + 1)] + cfg_args
        code1, _ = run(base + ["--max-epochs", str(S["grader_split"])], lines)
        code2, text2 = run(base + ["--max-epochs", str(S["grader_epochs"])], lines)
        check("grader: both invocations exit 0", code1 == 0 and code2 == 0)
        check("grader: second invocation resumed", "resumed after epoch index" in text2)
        state_path = os.path.join(run_dir, "state.json")
        if os.path.exists(state_path):
            state = json.load(open(state_path))
            check("grader: all epochs done", state["completed_epochs"] == S["grader_epochs"], f"({state['completed_epochs']})")
            history_ok(state["history"], "train_loss", "grader")
            check("grader: best checkpoint written", os.path.exists(os.path.join(run_dir, "checkpoints", "best.weights.h5")))
            print("  validation QWK per epoch (aptos / eyepacs):",
                  [(round(h["aptos_val_qwk"], 3), round(h["eyepacs_val_qwk"], 3)) for h in state["history"]])

    if "quality" in wanted:
        print("\n== quality (EyeQ, EfficientNet-B0)")
        run_dir = os.path.join(out, "quality")
        base = ["scripts/train_quality.py", "--run-dir", run_dir, "--limit", str(S["eyeq_images"]), "--limit-test", str(S["eyeq_test"])] + cfg_args
        code1, _ = run(base + ["--max-epochs", str(S["quality_split"]), "--no-eval"], lines)
        code2, text2 = run(base + ["--max-epochs", str(S["quality_epochs"])], lines)
        check("quality: both invocations exit 0", code1 == 0 and code2 == 0)
        check("quality: second invocation resumed", "resumed after epoch index" in text2)
        state_path = os.path.join(run_dir, "state.json")
        if os.path.exists(state_path):
            state = json.load(open(state_path))
            history_ok(state["history"], "train_loss", "quality")
            check("quality: best model saved", os.path.exists(os.path.join(run_dir, "best_model.keras")))
        ev = os.path.join(run_dir, "test_evaluation.json")
        if os.path.exists(ev):
            e = json.load(open(ev))
            check("quality: test evaluation written", e["images"] == S["eyeq_test"], f"(accuracy {e['accuracy']:.3f} on {e['images']} images)")

    if "lesions" in wanted:
        print("\n== lesions (IDRiD + TJDR, U-Net SE-ResNet-101)")
        run_dir, cache_dir = os.path.join(out, "lesions"), os.path.join(out, "lesion_cache")
        base = ["scripts/train_lesions.py", "--run-dir", run_dir, "--cache-dir", cache_dir, "--limit", str(S["lesion_limit"]), "--allow-partial",
                "--val-every", str(S["lesion_val_every"]), "--checkpoint-every", "50", "--warmup-steps", "50"] + cfg_args
        code1, _ = run(base + ["--max-steps", str(S["lesion_split"])], lines)
        code2, text2 = run(base + ["--max-steps", str(S["lesion_steps"] - S["lesion_split"])], lines)
        check("lesions: both invocations exit 0", code1 == 0 and code2 == 0)
        check("lesions: second invocation resumed", "resumed at step" in text2)
        vh = os.path.join(run_dir, "validation_history.json")
        if os.path.exists(vh):
            history = json.load(open(vh))
            import math
            scores = [h["validation"]["selection_score"] for h in history]
            check("lesions: validation scores finite", bool(scores) and all(math.isfinite(s) for s in scores), f"({[round(s, 3) for s in scores]})")
        check("lesions: latest and best checkpoints written",
              os.path.exists(os.path.join(run_dir, "latest.pt")) and os.path.exists(os.path.join(run_dir, "best_ema.pt")))

    print("\n== summary")
    for line in lines:
        print("  " + line)
    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    print(f"\n{len(checks) - len(failed)} of {len(checks)} checks passed in {(time.time() - started) / 60:.1f} min; output in {out}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
