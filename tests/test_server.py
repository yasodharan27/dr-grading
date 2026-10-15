import base64
import io
import json
import os
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
import server  # noqa: E402


class FakeEngine:
    """Yields every stage the page knows, with small arrays; records the path it was given."""

    def __init__(self):
        self.paths = []

    def stages(self, path, arm, explanations, settings):
        self.paths.append(path)
        assert os.path.exists(path)
        frame = np.zeros((32, 32, 3), np.float32)
        yield {"stage": "quality", "seconds": 0.1, "quality": {"label": "Good", "confidence": 0.9, "class_index": 0,
                                                               "probabilities": {"Good": 0.9, "Usable": 0.08, "Reject": 0.02}}}
        yield {"stage": "preprocessing", "seconds": 0.1, "stage2": np.zeros((40, 50, 3), np.uint8), "box": (0, 40, 0, 50), "frame": frame,
               "arm": arm or "crop"}
        yield {"stage": "vessels", "seconds": 0.1, "vessel": np.zeros((32, 32), np.float32), "vessel_fraction": 0.1}
        yield {"stage": "lesions", "seconds": 0.1, "lesion_max": np.zeros((32, 32, 4), np.float32), "frame_direct": frame,
               "lesion_area": {"MA": 0.0, "HE": 0.0, "EX": 0.0, "SE": 0.0}}
        yield {"stage": "grade", "seconds": 0.1, "grade": 2, "grade_name": "Moderate NPDR", "referable": True,
               "probabilities": [0.1, 0.2, 0.5, 0.1, 0.1], "per_seed": {"42": 2, "123": 2, "2026": 1}, "expected_grade": 1.9}
        yield {"stage": "uncertainty", "seconds": 0.1, "entropy": 1.2, "seed_disagreement": 0.3, "seeds_agree": False,
               "tta_expected_grade_sd": 0.1, "tta_grade_changes": False}
        for name in explanations:
            yield {"stage": "explanation", "method": name, "label": name, "seconds": 0.1, "map": np.zeros((32, 32)),
                   "overlay": np.zeros((32, 32, 3), np.uint8)}
        yield {"stage": "summary", "seconds": 0.0, "lines": ["**Grade 2 - Moderate NPDR** <script>x</script>"], "review_recommended": True,
               "reasons": ["r"], "validation_rates": {"exact": 0.6, "within_one": 0.9, "images": 3}, "map_on_lesions": {"Grad-CAM": (0.1, 0.01)}}


@pytest.fixture()
def running(tmp_path):
    engine = FakeEngine()
    demo = server.Demo(engine, str(tmp_path))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(demo, "<html>page</html>"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", engine
    httpd.shutdown()


def _post(url, body):
    request = urllib.request.Request(url + "/run", data=json.dumps(body).encode(), method="POST")
    return json.loads(urllib.request.urlopen(request, timeout=10).read())


def _wait(url, job_id):
    for _ in range(100):
        job = json.loads(urllib.request.urlopen(f"{url}/job/{job_id}", timeout=10).read())
        if job["done"]:
            return job
        time.sleep(0.05)
    raise AssertionError("the job did not finish")


def test_upload_runs_every_stage_and_the_upload_is_deleted(running):
    url, engine = running
    buf = io.BytesIO()
    Image.fromarray(np.zeros((40, 50, 3), np.uint8)).save(buf, format="PNG")
    job = _post(url, {"upload": base64.b64encode(buf.getvalue()).decode(), "filename": "eye.png", "arm": "crop",
                      "explanations": ["grad_cam", "not_a_method"]})
    result = _wait(url, job["id"])
    assert result["error"] is None
    titles = [s["title"] for s in result["stages"]]
    assert len(titles) == 9 and titles[-1] == "Done" and titles[-2].startswith("Summary") and any("Grad-CAM" in t or "grad_cam" in t for t in titles)
    assert result["stages"][4]["data"]["grade"] == 2
    assert result["stages"][-2]["data"]["summary_lines"][0].startswith("**Grade 2") and result["stages"][-2]["data"]["review_recommended"]
    assert engine.paths and not os.path.exists(engine.paths[0])          # the uploaded file is gone


def test_page_demo_list_and_errors(running):
    url, _ = running
    assert urllib.request.urlopen(url + "/", timeout=10).read() == b"<html>page</html>"
    assert json.loads(urllib.request.urlopen(url + "/demo-list", timeout=10).read()) == []
    assert "error" in _post(url, {})
    assert "error" in _post(url, {"demo": "missing.png"})
