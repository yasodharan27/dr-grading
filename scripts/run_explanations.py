"""Explanation experiments: the localisation check against expert lesion masks, and Kernel SHAP figures.

    python scripts/run_explanations.py localisation                    # 60 DDR lesion-segmentation training images, both arms
    python scripts/run_explanations.py localisation --limit 5 --arms crop
    python scripts/run_explanations.py shap data/demo_images/idrid_grade3_IDRiD_001.jpg --arm crop

localisation: for each arm, the maps of Grad-CAM, Integrated Gradients (32 steps) and Occlusion (64-pixel patches,
stride 32; first 20 images only) are averaged over the three seeds and compared with the union of the image's expert
lesion masks (pointing game, share of the top 10 % of the map on lesions, AUROC of the map as a lesion score, and
the lesion share of the frame, which is what an unrelated map would score). The sample is fixed in
data/splits/ddr_lesion_localisation_sample.csv. Rows are appended to localisation_rows.csv as they are produced, so
an interrupted run resumes. Output folder: <output_dir>/explanations.

Overlap with lesions describes where a map is high; it is not evidence that the model detects lesions.

shap: superpixel Kernel SHAP (50 SLIC segments, 512 coalitions) of the expected grade for one image and one arm, the
mean over the three seeds, drawn as a figure (red raised the expected grade, blue lowered it). Costly: 512 forward
passes per seed.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from drgrade import data, explain, preprocess  # noqa: E402
from drgrade import grader as gr  # noqa: E402
from drgrade.config import ARMS, load_config  # noqa: E402


def localisation(args, cfg, out):
    sample = data.ddr_lesion_sample()[:args.limit]
    items = []
    for name, _ in sample:
        image, masks = data.ddr_lesion_paths(cfg, name)
        items.append((name, image, list(masks.values())))
    progress = os.path.join(out, "localisation_rows.csv")
    rows = []
    occlusion_images = min(20, len(items))
    for arm in args.arms:
        graders = list(gr.load_arm(cfg, arm).values())
        print(f"{arm}: {len(items)} images")
        rows += explain.localisation_study(items, arm, graders, methods=tuple(args.methods), occlusion_images=occlusion_images,
                                           progress_csv=progress)
        del graders
    import pandas as pd
    per_image = pd.read_csv(progress)
    per_image.to_csv(os.path.join(out, "localisation_per_image.csv"), index=False)
    summary = explain.localisation_summary(per_image.to_dict("records")).round(3)
    summary.to_csv(os.path.join(out, "localisation_summary.csv"), index=False)
    print(summary.to_string(index=False))
    print("uniform-map baseline: pointing game and top-10 % precision = the lesion share of the frame; AUROC = 0.5")


def shap(args, cfg, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = preprocess.frame(args.image, args.arm)
    graders = gr.load_arm(cfg, args.arm)
    heat = np.mean([explain.superpixel_shap(explain.model_fn(g, 8), rgb, n_segments=50, n_samples=512)[0] for g in graders.values()], axis=0)
    limit = float(np.abs(heat).max()) or 1.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), layout="constrained")
    axes[0].imshow(rgb)
    axes[0].set_title(f"{os.path.basename(args.image)} ({args.arm} frame)")
    axes[1].imshow(heat, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[1].set_title("Superpixel SHAP of the expected grade, 3-seed mean")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Red raised the expected grade, blue lowered it. A sensitivity map, not evidence of lesion detection.", fontsize=10)
    path = os.path.join(out, f"shap_{os.path.splitext(os.path.basename(args.image))[0]}_{args.arm}.png")
    fig.savefig(path, dpi=110)
    print("written", path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("localisation")
    a.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    a.add_argument("--limit", type=int, help="only the first N images of the sample")
    a.add_argument("--methods", nargs="+", default=["grad_cam", "integrated_gradients", "occlusion"],
                   choices=("grad_cam", "integrated_gradients", "occlusion"),
                   help="Integrated Gradients (32 steps x 3 seeds) and Occlusion take minutes per image on a CPU")
    b = sub.add_parser("shap")
    b.add_argument("image")
    b.add_argument("--arm", default="crop", choices=ARMS)
    for p in (a, b):
        p.add_argument("--config")
        p.add_argument("--out")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    out = args.out or os.path.join(cfg["output_dir"], "explanations")
    os.makedirs(out, exist_ok=True)
    {"localisation": localisation, "shap": shap}[args.command](args, cfg, out)


if __name__ == "__main__":
    main()
