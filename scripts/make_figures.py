"""Regenerates the repository's own figures.

    python scripts/make_figures.py                        # pipeline diagram, selective-prediction curves, confusion matrices
    python scripts/make_figures.py --examples data/demo_images/idrid_grade3_IDRiD_001.jpg ...   # + per-image stage panels

Output: figures/pipeline.png, figures/selective_prediction.png, figures/confusion_matrices.png and, with --examples,
figures/example_<image>.png (needs the weights; Grad-CAM, Integrated Gradients and Occlusion on CPU take several
minutes per image).

Colours: the crop arm is blue and the direct arm orange (validated for colour-vision deficiency; the arms also differ
in line style), grey is the random-referral reference.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drgrade import metrics as mt  # noqa: E402
from drgrade.config import ARMS, ROOT, SEEDS  # noqa: E402

FIGURES = os.path.join(ROOT, "figures")
RESULTS = os.path.join(ROOT, "results")
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
COLOUR = {"crop": "#2a78d6", "direct": "#eb6834", "random": "#8a8984"}
STYLE = {"crop": "-", "direct": "--"}
LABEL = {"crop": "fundus-cropped frames (J-crop)", "direct": "whole-image frames (J)"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    return plt


def pipeline_diagram(path):
    plt = _plt()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    def box(x, y, w, h, title, body, face="#ffffff", edge="#2a78d6"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor=face, edgecolor=edge, linewidth=1.6))
        ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=8.6, color=INK2, linespacing=1.35)

    def arrow(a, b, rad=0.0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=INK2, connectionstyle=f"arc3,rad={rad}"))

    box(0.15, 2.0, 1.7, 1.7, "Fundus image", "raw colour\nphotograph", edge="#8a8984")
    box(2.35, 3.75, 2.5, 1.5, "Stage 1  Quality", "EfficientNet-B0 (EyeQ)\nGood / Usable / Reject")
    box(2.35, 1.85, 2.5, 1.6, "Stage 2  Preprocess", "gamma 1.2, CLAHE;\nfundus box (crop arm);\nresize to 512 x 512")
    box(5.4, 3.75, 2.3, 1.5, "Stage 3  Vessels", "LWNet (pretrained,\ninference only)")
    box(5.4, 1.85, 2.3, 1.6, "Stage 4  Lesions", "U-Net, SE-ResNet-101:\nMA, HE, EX, SE")
    box(8.25, 1.85, 2.2, 1.6, "Grader x 3 seeds", "ConvNeXt-Tiny + CORN\nRGB frame only", edge="#eb6834")
    box(10.9, 3.75, 2.0, 1.5, "Uncertainty", "ensemble entropy,\nseed disagreement,\nflip / rotation check")
    box(10.9, 1.85, 2.0, 1.6, "Explanations", "Grad-CAM, Integrated\nGradients, Occlusion,\nSHAP (figures)")
    box(7.0, 0.1, 5.9, 1.35, "Grade, summary and reliability flag",
        "ICDR grade with its probability; lesions found by Stage 4;\nmap overlap with them; 'review recommended' when checks disagree", edge="#eb6834")
    arrow((1.85, 2.85), (2.35, 2.65))
    arrow((1.85, 3.2), (2.35, 4.4), 0.1)
    arrow((4.85, 2.65), (5.4, 2.65))                                         # Stage 2 -> Stage 4
    arrow((4.85, 3.0), (5.4, 4.4), -0.15)                                    # Stage 2 -> Stage 3
    # the 512 x 512 frame goes to the grader (not through Stage 4): a line under Stage 4
    ax.plot([3.6, 3.6, 8.0], [1.85, 1.55, 1.55], color=INK2, linewidth=1.4)
    arrow((8.0, 1.55), (8.45, 1.85), -0.2)
    ax.text(5.8, 1.62, "512 x 512 RGB frame", ha="center", fontsize=8.2, color=INK2)
    arrow((10.45, 2.65), (10.9, 2.65))
    arrow((9.35, 3.45), (10.9, 4.3), 0.15)
    arrow((9.35, 1.85), (9.35, 1.45))
    arrow((11.9, 1.85), (11.5, 1.45))
    ax.add_patch(FancyArrowPatch((6.55, 1.85), (7.3, 1.45), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color=INK2, linestyle=":"))
    ax.text(3.6, 0.75, "The grade comes from the RGB grader alone;\nvessel and lesion maps are supporting views.", ha="center", fontsize=8.8, color=INK2, style="italic")
    ax.text(0.15, 5.4, "Research prototype - not a medical device; the output is not a diagnosis.", fontsize=9, color=INK2)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def selective_prediction(path):
    plt = _plt()
    with open(os.path.join(RESULTS, "uncertainty", "uncertainty_results.json"), encoding="utf-8") as fh:
        results = json.load(fh)
    panels = [("aptos_val", "APTOS validation (730)"), ("eyepacs_val", "EyePACS validation (3,512)"),
              ("ddr", "DDR broader list (3,725)"), ("idrid", "IDRiD test (103)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharex=True)
    for ax, (name, title) in zip(axes, panels):
        for arm in ARMS:
            r = results[f"{arm}_{name}"]
            sel = r["scores"]["entropy"]["selective"]
            ax.plot([s["coverage"] for s in sel], [s["qwk"] for s in sel], STYLE[arm], color=COLOUR[arm], linewidth=2, marker="o", markersize=5,
                    label=f"{LABEL[arm]}, referred by entropy")
        rnd = results[f"crop_{name}"]["random_referral"]
        ax.plot([x["coverage"] for x in rnd], [x["qwk_mean"] for x in rnd], ":", color=COLOUR["random"], linewidth=2, label="random referral (crop)")
        ax.fill_between([x["coverage"] for x in rnd], [x["qwk_low"] for x in rnd], [x["qwk_high"] for x in rnd], color=COLOUR["random"], alpha=0.2, linewidth=0)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("coverage (share of images kept)")
        ax.invert_xaxis()
        ax.grid(color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("QWK on the images kept")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Selective prediction: referring the least certain images raises agreement on the rest "
                 "(validation sets also selected the checkpoints)", fontsize=11, y=1.03)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def confusion_matrices(path):
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#9ec5f4", "#2a78d6", "#0d366b"])
    import pandas as pd
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, (dataset, split_file, title) in zip(axes, (("ddr", "ddr_test.csv", "DDR primary (3,540 images)"), ("idrid", "idrid_test.csv", "IDRiD primary (100 images)"))):
        split = pd.read_csv(os.path.join(ROOT, "data", "splits", split_file))
        mask = split["primary"].to_numpy(bool)
        tables = [mt.arrays(mt.read_table(os.path.join(RESULTS, "predictions", dataset, f"per_image_crop_seed{s}.csv"))) for s in SEEDS]
        ens = mt.ensemble(tables)
        g, p = ens["grade"][mask], ens["pred"][mask]
        cm = np.array([[int(np.sum((g == a) & (p == b))) for b in range(5)] for a in range(5)])
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=10, color="#ffffff" if norm[i, j] > 0.5 else INK)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xlabel("predicted grade (three-seed ensemble)")
        ax.set_ylabel("reference grade")
        ax.set_title(f"{title}\nQWK {mt.qwk(g, p):.3f}", fontsize=10.5)
    fig.suptitle("J-crop confusion matrices (counts; colour = share of the reference grade)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def examples(images, out_dir, cfg_path=None):
    from drgrade import pipeline
    from drgrade.config import load_config
    cfg = load_config(cfg_path)
    pipe = pipeline.Pipeline(cfg, "crop")
    for path in images:
        result, views = pipe.run(path, explanations=("grad_cam", "integrated_gradients", "occlusion"), settings=pipeline.DEMO_SETTINGS)
        stem = os.path.splitext(os.path.basename(path))[0]
        pipeline.figure(result, views, os.path.join(out_dir, f"example_{stem}.png"))
        print(f"{stem}: grade {result['grade']}, reference in the file name; written")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", nargs="*", help="image files to run through the pipeline for per-image panels")
    parser.add_argument("--only", nargs="+", choices=("pipeline", "selective", "confusion"))
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    os.makedirs(FIGURES, exist_ok=True)
    wanted = set(args.only or ("pipeline", "selective", "confusion"))
    if "pipeline" in wanted:
        pipeline_diagram(os.path.join(FIGURES, "pipeline.png"))
    if "selective" in wanted:
        selective_prediction(os.path.join(FIGURES, "selective_prediction.png"))
    if "confusion" in wanted:
        confusion_matrices(os.path.join(FIGURES, "confusion_matrices.png"))
    if args.examples:
        examples(args.examples, FIGURES, args.config)
    print("figures in", FIGURES)


if __name__ == "__main__":
    main()
