"""Checks that this installation reproduces a reference set of pipeline outputs on the same images: Stage-1
quality probabilities, Stage-2 frames, the Stage-3 vessel map, the Stage-4 maps, every grader's logits and decoded
grades (both arms, three seeds) and, optionally, the Grad-CAM maps.

    python scripts/parity_check.py --images DIR --reference FILE.npz            # compare
    python scripts/parity_check.py --images DIR --write FILE.npz [--gradcam] [--compact]   # write a reference with this code

The reference holds, for the images of DIR in sorted order: names, quality (N, 3), box (N, 4), frame_direct and
frame_crop (N, 512, 512, 3), vessel (N, 512, 512), lesion_maps (N, 512, 512, 8) uint8, logits_<arm>_<seed> (N, 4)
and optionally gradcam_<arm>_<seed> (N, 512, 512). Run on CPU: GPU kernels differ in the last bits.
"""
import argparse
import gc
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from drgrade import explain, lesions, preprocess, quality, vessels  # noqa: E402
from drgrade import grader as gr  # noqa: E402
from drgrade.config import ARMS, SEEDS, configure_threads, load_config, weight_path  # noqa: E402

TOLERANCE = {"quality": 1e-5, "frame_direct": 1e-6, "frame_crop": 1e-6, "vessel": 1e-5, "lesion_maps": 0, "logits": 1e-5, "gradcam": 1e-4}
IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def compute(cfg, paths, with_gradcam=False, log=print):
    configure_threads()
    out = {"names": np.array([os.path.basename(p) for p in paths])}
    q_model = quality.load_model(weight_path(cfg, "quality"))
    v_model = vessels.load_model(weight_path(cfg, "vessels"), "cpu")
    l_model = lesions.load_model(weight_path(cfg, "lesions"), "cpu")
    rows = {k: [] for k in ("quality", "box", "frame_direct", "frame_crop", "vessel", "lesion_maps")}
    for k, path in enumerate(paths):
        q = quality.predict(q_model, path)
        f = preprocess.frames(path)
        rows["quality"].append([q["probabilities"][c] for c in quality.CLASSES])
        rows["box"].append(f["box"])
        rows["frame_direct"].append(f["direct"])
        rows["frame_crop"].append(f["crop"])
        rows["vessel"].append(vessels.vessel_frame(v_model, f["stage2"]))
        rows["lesion_maps"].append(lesions.frame_maps(l_model, f["stage2"], device="cpu"))
        log(f"  stages {k + 1}/{len(paths)} {os.path.basename(path)}")
    out.update({k: np.asarray(v) for k, v in rows.items()})
    del q_model, v_model, l_model
    for arm in ARMS:
        frames = out[f"frame_{arm}"]
        for seed, g in gr.load_arm(cfg, arm).items():
            out[f"logits_{arm}_{seed}"] = np.concatenate([g.logits(frames[i:i + 1]) for i in range(len(frames))])
            if with_gradcam:
                out[f"gradcam_{arm}_{seed}"] = np.concatenate([explain.grad_cam(g, frames[i:i + 1])[0] for i in range(len(frames))])
        gc.collect()
        log(f"  graders {arm} done")
    return out


def compare(new, ref):
    """Rows of (quantity, max abs difference, tolerance, pass) and the overall verdict. Only the quantities the
    reference holds are compared (a compact reference has no frames or maps)."""
    if list(new["names"]) != list(ref["names"]):
        raise ValueError("the images differ from the reference's")
    rows = []

    def add(name, a, b, tol):
        diff = float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)))) if np.size(a) else 0.0
        rows.append((name, diff, tol, diff <= tol))

    if "quality" in ref:
        add("quality probabilities", new["quality"], ref["quality"], TOLERANCE["quality"])
        rows.append(("quality labels identical", 0.0, 0, bool(np.array_equal(new["quality"].argmax(1), ref["quality"].argmax(1)))))
    if "box" in ref:
        rows.append(("fundus boxes identical", 0.0, 0, bool(np.array_equal(new["box"], ref["box"]))))
    for key in ("frame_direct", "frame_crop", "vessel", "lesion_maps"):
        if key in ref:
            add(key, new[key], ref[key], TOLERANCE[key])
    for arm in ARMS:
        for seed in SEEDS:
            key = f"logits_{arm}_{seed}"
            if key not in ref:
                continue
            add(key, new[key], ref[key], TOLERANCE["logits"])
            same = np.array_equal(gr.decode(new[key])["grade"], gr.decode(ref[key])["grade"])
            rows.append((f"grades_{arm}_{seed} identical", 0.0, 0, bool(same)))
            if f"gradcam_{arm}_{seed}" in ref and f"gradcam_{arm}_{seed}" in new:
                add(f"gradcam_{arm}_{seed}", new[f"gradcam_{arm}_{seed}"], ref[f"gradcam_{arm}_{seed}"], TOLERANCE["gradcam"])
    return rows, all(r[3] for r in rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--images", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reference")
    group.add_argument("--write")
    parser.add_argument("--gradcam", action="store_true", help="also compute (and compare) the Grad-CAM maps")
    parser.add_argument("--save-new", help="also save this run's outputs to this .npz")
    parser.add_argument("--compact", action="store_true",
                        help="with --write: keep only image names, quality probabilities, fundus boxes and grader logits (a small file)")
    parser.add_argument("--tolerance", action="append", default=[], metavar="KEY=VALUE",
                        help=f"override a tolerance, e.g. logits=1e-4 (keys: {', '.join(TOLERANCE)}); for another platform's reference")
    args = parser.parse_args(argv)
    for item in args.tolerance:
        key, value = item.split("=", 1)
        if key not in TOLERANCE:
            sys.exit(f"unknown tolerance {key!r}; keys: {list(TOLERANCE)}")
        TOLERANCE[key] = float(value)
    cfg = load_config(args.config)
    paths = [os.path.join(args.images, n) for n in sorted(os.listdir(args.images)) if n.lower().endswith(IMAGE_TYPES)]
    if not paths:
        sys.exit(f"no images in {args.images}")
    if args.reference:
        ref = dict(np.load(args.reference))
        with_gradcam = args.gradcam or any(k.startswith("gradcam_") for k in ref)
        new = compute(cfg, paths, with_gradcam)
        if args.save_new:
            np.savez(args.save_new, **new)
        rows, ok = compare(new, ref)
        width = max(len(r[0]) for r in rows)
        print(f"{'quantity':<{width}}  {'max |diff|':>12}  {'tolerance':>9}  result")
        for name, diff, tol, passed in rows:
            print(f"{name:<{width}}  {diff:12.3e}  {tol:9.0e}  {'PASS' if passed else 'FAIL'}")
        print(f"PARITY {'PASS' if ok else 'FAIL'} on {len(paths)} images")
        sys.exit(0 if ok else 1)
    out = compute(cfg, paths, args.gradcam)
    if args.compact:
        out = {k: v for k, v in out.items() if k in ("names", "quality", "box") or k.startswith("logits_")}
    np.savez_compressed(args.write, **out)
    print(f"reference written to {args.write}")


if __name__ == "__main__":
    main()
