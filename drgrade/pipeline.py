"""One fundus image through every stage.

    raw image -> Stage 1 quality (EfficientNet-B0) -> Stage 2 preprocessing (gamma, CLAHE) -> grader frame (crop or
    direct) -> Stage 3 vessels (LWNet) -> Stage 4 lesions (U-Net) -> grade from the three-seed grader ensemble ->
    uncertainty (ensemble and test-time augmentation) -> explanation maps

The grade comes from the RGB grader alone; the vessel and lesion maps are supporting views. Research prototype:
not a medical device, and its output is not a diagnosis.

    python -m drgrade.pipeline IMAGE [IMAGE ...] [--arm crop|direct] [--explain grad_cam integrated_gradients occlusion]
"""
import argparse
import gc
import json
import os
import time

import numpy as np

from . import explain, lesions, preprocess, quality, summary, uncertainty, vessels
from . import grader as gr
from .config import GRADES, SEEDS, configure_threads, load_config, weight_path

NOTICE = "Research prototype - not a medical device; its output is not a diagnosis."
EXPLANATIONS = ("grad_cam", "integrated_gradients", "occlusion")
#: Settings of the explanation maps. The interactive demo uses faster settings (fewer steps, no overlap) than the
#: localisation study (Integrated Gradients 32 steps, Occlusion stride 32); they are labelled where shown.
STUDY_SETTINGS = {"ig_steps": 32, "occlusion_stride": 32}
DEMO_SETTINGS = {"ig_steps": 16, "occlusion_stride": 64}


class Pipeline:
    """Loads the frozen models once. Only one arm's three graders are held in memory at a time."""

    def __init__(self, cfg, arm=None, log=print):
        self.cfg, self.device = cfg, cfg["device"]
        log(f"CPU threads: {configure_threads()}")
        log("loading Stage 1 (image quality) ...")
        self.quality = quality.load_model(weight_path(cfg, "quality"))
        log("loading Stage 3 (LWNet) and Stage 4 (lesion U-Net) ...")
        self.vessels = vessels.load_model(weight_path(cfg, "vessels"), self.device)
        self.lesions = lesions.load_model(weight_path(cfg, "lesions"), self.device)
        self.calibration = summary.load_calibration(cfg["reliability_calibration"])
        self.arm, self.graders = None, {}
        self.load_arm(arm or cfg["arm"], log)

    def load_arm(self, arm, log=print):
        if arm == self.arm:
            return
        self.graders = {}
        gc.collect()
        log(f"loading the three {arm} graders ...")
        self.graders = gr.load_arm(self.cfg, arm)
        self.arm = arm

    def stages(self, image_path, arm=None, explanations=("grad_cam",), settings=None):
        """Yields one dict per stage as soon as it is done: {'stage', 'seconds', ...data}."""
        settings = settings or STUDY_SETTINGS
        arm = arm or self.arm
        self.load_arm(arm)
        t = time.time()
        q = quality.predict(self.quality, image_path)
        yield {"stage": "quality", "seconds": time.time() - t, "quality": q}

        t = time.time()
        f = preprocess.frames(image_path)
        yield {"stage": "preprocessing", "seconds": time.time() - t, "stage2": f["stage2"], "box": f["box"], "frame": f[arm], "arm": arm}

        t = time.time()
        vessel = vessels.vessel_frame(self.vessels, f["stage2"])
        yield {"stage": "vessels", "seconds": time.time() - t, "vessel": vessel, "vessel_fraction": float((vessel > 0.5).mean())}

        t = time.time()
        maps = lesions.frame_maps(self.lesions, f["stage2"], device=self.device)
        lesion_max = lesions.max_maps(maps)
        area = {c: float((lesion_max[..., i] > 0.5).mean()) for i, c in enumerate(lesions.CLASSES)}
        yield {"stage": "lesions", "seconds": time.time() - t, "maps": maps, "lesion_max": lesion_max, "lesion_area": area,
               "frame_direct": f["direct"]}

        t = time.time()
        frame = f[arm]
        graders = [self.graders[s] for s in SEEDS]
        fns = [explain.model_fn(g, batch_size=2) for g in graders]          # batch 2: faster and leaner than 8 on CPU
        logits = np.stack([fn(frame[None]) for fn in fns])                       # (3, 1, 4)
        e = uncertainty.ensemble(logits)
        grade = int(e["pred"][0])
        yield {"stage": "grade", "seconds": time.time() - t, "grade": grade, "grade_name": GRADES[grade],
               "referable": grade >= 2, "probabilities": [float(p) for p in e["class_prob"][0]],
               "per_seed": {str(s): int(g) for s, g in zip(SEEDS, e["seed_pred"][:, 0])},
               "expected_grade": float(e["p_gt"][0].sum()), "logits": logits[:, 0]}

        t = time.time()
        tta_summaries = [uncertainty.tta_summary(uncertainty.tta_logits(fn, frame[None])) for fn in fns]
        seeds_pred = e["seed_pred"][:, 0].tolist()
        entropy = float(e["entropy"][0])
        tta_change = bool(any(x["grade_changes"][0] for x in tta_summaries))
        yield {"stage": "uncertainty", "seconds": time.time() - t, "entropy": entropy,
               "seed_disagreement": float(e["seed_disagreement"][0]), "seeds_agree": len(set(seeds_pred)) == 1,
               "tta_expected_grade_sd": float(np.mean([x["expected_grade_sd"][0] for x in tta_summaries])),
               "tta_grade_changes": tta_change}

        # Stage-4 lesions (any class above 0.5) in the grader's frame: the whole-image maps are brought back to the
        # native size, then given the same cut and resize as the image
        import cv2
        native_h, native_w = f["stage2"].shape[:2]
        lesions_native = cv2.resize((lesion_max > 0.5).any(axis=-1).astype(np.uint8), (native_w, native_h),
                                    interpolation=cv2.INTER_NEAREST)
        lesions_frame = explain.frame_mask(lesions_native, None if arm == "direct" else f["box"])
        overlaps = {}
        for name in explanations:
            t = time.time()
            if name == "grad_cam":
                heat = np.mean([explain.grad_cam(g, frame[None])[0][0] for g in graders], axis=0)
                label = "Grad-CAM, mean of the 3 seeds"
            elif name == "integrated_gradients":
                heat = explain.integrated_gradients(graders[0], frame, steps=settings["ig_steps"])[2]
                label = f"Integrated Gradients, seed {SEEDS[0]}, {settings['ig_steps']} steps"
            elif name == "occlusion":
                heat = explain.occlusion(fns[0], frame, patch=64, stride=settings["occlusion_stride"])[1]
                label = f"Occlusion, seed {SEEDS[0]}, 64-pixel patches, stride {settings['occlusion_stride']}"
            else:
                raise ValueError(f"unknown explanation {name!r}; choose from {EXPLANATIONS}")
            overlaps[summary.METHOD_NAMES[name]] = explain.map_on_lesions(heat, lesions_frame)
            yield {"stage": "explanation", "method": name, "label": label, "seconds": time.time() - t, "map": heat,
                   "overlay": explain.overlay(frame, heat)}

        t = time.time()
        review, reasons, rates = summary.reliability(q["label"], seeds_pred, tta_change, entropy, self.calibration["arms"][arm])
        lines = summary.written_summary(grade, [float(p) for p in e["class_prob"][0]], area, overlaps, review, reasons, rates,
                                        self.calibration["note"])
        yield {"stage": "summary", "seconds": time.time() - t, "lines": lines, "review_recommended": review, "reasons": reasons,
               "validation_rates": rates, "map_on_lesions": overlaps}

    def run(self, image_path, arm=None, explanations=("grad_cam",), settings=None):
        """(result: JSON-ready summary, views: the arrays for figures) of one image."""
        result, views = {"image": os.path.basename(image_path), "notice": NOTICE, "explanations": {}, "seconds": {}}, {"explanations": {}}
        for s in self.stages(image_path, arm, explanations, settings):
            kind = s["stage"]
            result["seconds"][s.get("method", kind)] = round(float(s["seconds"]), 1)
            if kind == "quality":
                result["quality"] = s["quality"]
            elif kind == "preprocessing":
                result["arm"], result["fundus_box"] = s["arm"], list(s["box"])
                views["frame"] = s["frame"]
            elif kind == "vessels":
                result["vessel_fraction"] = s["vessel_fraction"]
                views["vessel"] = s["vessel"]
            elif kind == "lesions":
                result["lesion_area"] = s["lesion_area"]
                views["lesion_max"] = s["lesion_max"]
            elif kind == "grade":
                result.update({k: s[k] for k in ("grade", "grade_name", "referable", "probabilities", "per_seed", "expected_grade")})
            elif kind == "uncertainty":
                result["uncertainty"] = {k: s[k] for k in ("entropy", "seed_disagreement", "seeds_agree", "tta_expected_grade_sd",
                                                           "tta_grade_changes")}
            elif kind == "explanation":
                result["explanations"][s["method"]] = s["label"]
                views["explanations"][s["method"]] = s["overlay"]
            else:
                result["summary"] = {k: s[k] for k in ("lines", "review_recommended", "reasons", "validation_rates", "map_on_lesions")}
        return result, views


def figure(result, views, path):
    """A summary picture: frame, vessel map, the four lesion maps, the explanation overlays and the reading."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("Grader frame", views["frame"], None), ("Vessels (Stage 3)", views["vessel"], "gray")]
    panels += [(f"{lesions.NAMES[c]} (Stage 4)", views["lesion_max"][..., i], "magma") for i, c in enumerate(lesions.CLASSES)]
    panels += [(result["explanations"][m], picture, None) for m, picture in views["explanations"].items()]
    rows = int(np.ceil(len(panels) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(18, 4.7 * rows + 1.8), layout="constrained")
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (title, image, cmap) in zip(np.ravel(axes), panels):
        ax.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        ax.set_title(title, fontsize=10)
    u = result["uncertainty"]
    probabilities = "  ".join(f"{g.split()[0]} {p:.2f}" for g, p in zip(GRADES, result["probabilities"]))
    fig.suptitle(f"{result['image']}  |  quality: {result['quality']['label']}  |  grade {result['grade']}: {result['grade_name']}  |  "
                 f"referable: {'yes' if result['referable'] else 'no'}\n{probabilities}\nensemble entropy {u['entropy']:.2f}  |  "
                 f"seed grades {list(result['per_seed'].values())}  |  TTA spread {u['tta_expected_grade_sd']:.2f}\n"
                 f"Explanation maps show where the score is sensitive, not that lesions were detected. {NOTICE}", fontsize=11)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Grade fundus images with the full pipeline")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--config")
    parser.add_argument("--arm", choices=("crop", "direct"))
    parser.add_argument("--explain", nargs="*", default=["grad_cam"], choices=EXPLANATIONS)
    parser.add_argument("--out", help="folder for one JSON and one PNG per image (default: <output_dir>/pipeline)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    out = args.out or os.path.join(cfg["output_dir"], "pipeline")
    os.makedirs(out, exist_ok=True)
    pipe = Pipeline(cfg, args.arm)
    for path in args.images:
        result, views = pipe.run(path, explanations=args.explain)
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(out, f"{stem}.json"), "w") as fh:
            json.dump(result, fh, indent=1)
        figure(result, views, os.path.join(out, f"{stem}.png"))
        print(f"{result['image']}: grade {result['grade']} ({result['grade_name']}), quality {result['quality']['label']}, "
              f"seeds {list(result['per_seed'].values())}, entropy {result['uncertainty']['entropy']:.2f}")
        for line in result["summary"]["lines"]:
            print("   " + line.replace("**", ""))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
