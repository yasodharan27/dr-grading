"""Local web demo of the full pipeline: upload a fundus image (or pick a demo image with a known grade) and watch
the stages appear one after another.

    python app/server.py                       # http://localhost:8501
    python app/server.py --host 0.0.0.0        # inside Docker (HOST and PORT environment variables work too)

Only Python's standard library serves the page. An uploaded image is written to a temporary file for the run and
deleted as soon as the run ends; nothing uploaded is kept. One image runs at a time. Research prototype: not a
medical device, and its output is not a diagnosis.
"""
import argparse
import base64
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

GRADES = ("No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR")
LESIONS = ("MA", "HE", "EX", "SE")
LESION_NAMES = {"MA": "microaneurysms", "HE": "haemorrhages", "EX": "hard exudates", "SE": "soft exudates"}
LESION_COLOURS = {"MA": (1.0, 1.0, 0.0), "HE": (1.0, 0.0, 0.0), "EX": (0.0, 1.0, 1.0), "SE": (1.0, 0.0, 1.0)}
NOTICE = "Research prototype - not a medical device; its output is not a diagnosis."
EXPLANATIONS = ("grad_cam", "integrated_gradients", "occlusion")
MAX_UPLOAD_BYTES = 40 * 1024 * 1024


def png(array, cmap=None, size=None):
    """Base64 PNG of an RGB image (H, W, 3; [0, 1] or uint8) or of a map (H, W) in [0, 1] through a colormap."""
    from PIL import Image
    a = np.asarray(array)
    if a.ndim == 2:
        import matplotlib
        a = matplotlib.colormaps[cmap or "inferno"](np.clip(a.astype(np.float64), 0, 1))[..., :3]
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(a)
    if size:
        im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def lesion_overlay(rgb, lesion_max):
    """The frame with each lesion class tinted where its map exceeds 0.5."""
    out = np.asarray(rgb, np.float32).copy()
    for i, name in enumerate(LESIONS):
        m = lesion_max[..., i] > 0.5
        out[m] = 0.35 * out[m] + 0.65 * np.asarray(LESION_COLOURS[name])
    return out


def cards(image_path, s, reference):
    """The page card of one pipeline stage (title, text, images, data)."""
    def img(caption, array, cmap=None, size=420):
        return {"caption": caption, "png": png(array, cmap, size)}

    kind = s["stage"]
    if kind == "quality":
        from PIL import Image
        q = s["quality"]
        with Image.open(image_path) as im:
            raw = np.asarray(im.convert("RGB"))
        text = (f"Quality: {q['label']} (confidence {q['confidence']:.2f}). " + ", ".join(f"{k} {v:.2f}" for k, v in q["probabilities"].items())
                + (" - a Reject image would normally be retaken; the remaining stages run anyway." if q["label"] == "Reject" else ""))
        return {"title": "Stage 1 - Image quality (EfficientNet-B0, EyeQ)", "text": text, "images": [img("Uploaded image", raw)], "data": {"quality": q}}
    if kind == "preprocessing":
        top, bottom, left, right = s["box"]
        how = ("the whole Stage-2 image resized to 512 x 512" if s["arm"] == "direct" else
               f"the Stage-2 image cut to the fundus box (rows {top}-{bottom}, columns {left}-{right}) and resized to 512 x 512")
        return {"title": "Stage 2 - Preprocessing (gamma 1.2, then CLAHE on the L channel) and the grader frame",
                "text": f"Native size {s['stage2'].shape[1]} x {s['stage2'].shape[0]}. Grader frame ({s['arm']} arm): {how}.",
                "images": [img("After Stage 2", s["stage2"]), img("Grader frame, 512 x 512", s["frame"])], "data": {}}
    if kind == "vessels":
        return {"title": "Stage 3 - Vessel segmentation (pretrained LWNet)",
                "text": f"Vessel pixels above 0.5: {100 * s['vessel_fraction']:.1f} % of the frame. A supporting view: the grade does not use it.",
                "images": [img("Vessel probability", s["vessel"], "gray")], "data": {}}
    if kind == "lesions":
        area = s["lesion_area"]
        return {"title": "Stage 4 - Lesion segmentation (SE-ResNet-101 U-Net: MA, HE, EX, SE)",
                "text": "Area above 0.5: " + ", ".join(f"{LESION_NAMES[k]} {100 * v:.2f} %" for k, v in area.items())
                        + ". A supporting view: the grade does not use it.",
                "images": [img("Lesions over the frame (yellow MA, red HE, cyan EX, magenta SE)", lesion_overlay(s["frame_direct"], s["lesion_max"]))]
                          + [img(LESION_NAMES[n], s["lesion_max"][..., i], "magma", 200) for i, n in enumerate(LESIONS)],
                "data": {"lesion_area": area}}
    if kind == "grade":
        ref = "" if reference is None else (f" Reference grade: {reference['reference_grade']} ({GRADES[int(reference['reference_grade'])]})"
                                            f" - {reference['source']}.")
        seeds = list(s["per_seed"].values())
        return {"title": "Grade - three-seed grader ensemble (ConvNeXt-Tiny + CORN)",
                "text": f"Predicted grade {s['grade']}: {s['grade_name']}. Referable (grade 2 or worse): {'yes' if s['referable'] else 'no'}. "
                        f"Per-seed grades {seeds}.{ref}", "images": [],
                "data": {"grade": s["grade"], "grade_name": s["grade_name"], "probabilities": s["probabilities"], "per_seed": seeds,
                         "expected_grade": s["expected_grade"], "reference": reference}}
    if kind == "uncertainty":
        return {"title": "Uncertainty - seed ensemble and test-time augmentation",
                "text": f"Predictive entropy {s['entropy']:.2f} (0 = certain, {np.log(5):.2f} = uniform). Seed disagreement (SD of the expected grade) "
                        f"{s['seed_disagreement']:.2f}; the three seeds {'agree' if s['seeds_agree'] else 'DISAGREE'}. Under 8 flips / rotations the "
                        f"expected grade varies by SD {s['tta_expected_grade_sd']:.2f}"
                        f"{' and the decoded grade CHANGES for at least one seed' if s['tta_grade_changes'] else '; the decoded grade does not change'}.",
                "images": [], "data": {k: s[k] for k in ("entropy", "seed_disagreement", "tta_expected_grade_sd", "tta_grade_changes")}}
    if kind == "summary":
        return {"title": "Summary - written explanation and reliability", "text": "", "images": [],
                "data": {"summary_lines": s["lines"], "review_recommended": s["review_recommended"], "reasons": s["reasons"],
                         "validation_rates": s["validation_rates"], "map_on_lesions": s["map_on_lesions"]}}
    return {"title": f"Why this grade - {s['label']}",
            "text": ("Where the model's expected grade is most sensitive (red = high). This shows sensitivity, not that the model detected lesions."
                     + (" Colours are scaled to the 99th percentile of the map for display." if "integrated" in s["label"].lower() else "")),
            "images": [img(s["label"], s["overlay"]), img("map", s.get("display_map", s["map"]), "inferno", 200)], "data": {}}


# --------------------------------------------------------------------------- the server

class Demo:
    """Holds the pipeline, the demo list and the jobs. `engine` needs `stages(path, arm, explanations, settings)`."""

    def __init__(self, engine, demo_dir=None):
        self.engine, self.demo_dir = engine, demo_dir
        self.jobs, self.lock = {}, threading.Lock()

    def demo_list(self):
        path = os.path.join(self.demo_dir or "", "labels.csv")
        if not self.demo_dir or not os.path.exists(path):
            return []
        with open(path, newline="", encoding="utf-8") as fh:
            return [{k: (int(v) if k == "reference_grade" else v) for k, v in r.items()} for r in csv.DictReader(fh)]

    def start(self, body):
        """Starts one run; returns {'id'} or {'error'}."""
        if not self.lock.acquire(blocking=False):
            return {"error": "another image is still running; wait for it to finish"}
        try:
            arm = body.get("arm") if body.get("arm") in ("direct", "crop") else None
            explanations = [x for x in body.get("explanations", []) if x in EXPLANATIONS]
            temporary, reference = None, None
            if body.get("demo"):
                entry = next((d for d in self.demo_list() if d["file"] == body["demo"]), None)
                if entry is None:
                    raise ValueError("unknown demo image")
                path, reference = os.path.join(self.demo_dir, entry["file"]), entry
            elif body.get("upload"):
                data = base64.b64decode(body["upload"])
                if len(data) > MAX_UPLOAD_BYTES:
                    raise ValueError("the image is larger than 40 MB")
                suffix = os.path.splitext(os.path.basename(body.get("filename", "upload.png")))[1].lower() or ".png"
                fd, temporary = tempfile.mkstemp(suffix=suffix if suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp") else ".png")
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                path = temporary
                digest = hashlib.sha256(data).hexdigest()                   # an uploaded copy of a demo image keeps its grade
                reference = next((d for d in self.demo_list() if d.get("sha256") == digest), None)
            else:
                raise ValueError("choose a demo image or upload one")
        except Exception as error:  # noqa: BLE001 -- reported to the page
            self.lock.release()
            return {"error": str(error)}
        job_id = uuid.uuid4().hex[:12]
        self.jobs[job_id] = {"stages": [], "done": False, "error": None}
        threading.Thread(target=self._work, args=(job_id, path, temporary, arm, explanations, reference), daemon=True).start()
        return {"id": job_id}

    def _work(self, job_id, path, temporary, arm, explanations, reference):
        from drgrade.pipeline import DEMO_SETTINGS
        job = self.jobs[job_id]
        try:
            for s in self.engine.stages(path, arm, explanations, DEMO_SETTINGS):
                card = cards(path, s, reference)
                job["stages"].append({**card, "seconds": round(float(s["seconds"]), 1)})
            job["stages"].append({"title": "Done", "text": NOTICE, "images": [], "data": {}, "seconds": None})
        except Exception as error:  # noqa: BLE001 -- shown on the page and printed in the terminal
            traceback.print_exc()
            job["error"] = f"{type(error).__name__}: {error}"
        finally:
            if temporary and os.path.exists(temporary):
                os.remove(temporary)                                        # no uploaded image is kept
            job["done"] = True
            self.lock.release()


def make_handler(demo, page):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload, kind="application/json"):
            data = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/":
                return self._send(200, page, "text/html; charset=utf-8")
            if self.path == "/demo-list":
                return self._send(200, [{k: d[k] for k in ("file", "source", "reference_grade")} for d in demo.demo_list()])
            if self.path.startswith("/job/"):
                job = demo.jobs.get(self.path.split("/")[-1])
                return self._send(200, job if job else {"error": "unknown job", "stages": [], "done": True})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/run":
                return self._send(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length", 0))
            if length > 2 * MAX_UPLOAD_BYTES:
                return self._send(413, {"error": "request too large"})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": "the request is not JSON"})
            return self._send(200, demo.start(body))

    return Handler


def read_page():
    with open(os.path.join(HERE, "page.html"), encoding="utf-8") as fh:
        return fh.read()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Web demo of the DR grading pipeline")
    parser.add_argument("--config")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"), help="default: $HOST or 127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8501")), help="default: $PORT or 8501")
    args = parser.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    from drgrade.config import load_config
    from drgrade.pipeline import Pipeline
    cfg = load_config(args.config)
    demo = Demo(Pipeline(cfg), cfg.get("demo_images"))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(demo, read_page()))
    print(f"open http://localhost:{args.port}   (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
