import os

import numpy as np

from drgrade import explain, summary
from drgrade.config import ROOT
from drgrade.lesions import CLASSES

CAL = {"coverage": 0.8, "entropy_threshold": 0.7, "certain": {"exact": 0.92, "within_one": 0.98, "images": 10},
       "uncertain": {"exact": 0.62, "within_one": 0.91, "images": 3}}


def test_reliability_flags_each_failed_check_and_uses_the_validation_band():
    review, reasons, rates = summary.reliability("Good", [2, 2, 2], False, 0.3, CAL)
    assert not review and reasons == [] and rates["exact"] == 0.92
    review, reasons, rates = summary.reliability("Reject", [2, 4, 2], True, 0.9, CAL)
    assert review and len(reasons) == 4 and rates["exact"] == 0.62
    assert summary.reliability("Usable", [3, 3, 3], False, 0.71, CAL)[0]           # entropy alone


def test_map_on_lesions_is_the_top_decile_share_beside_the_chance_share():
    heat = np.zeros((10, 10))
    heat[0, :] = 1.0                                                      # the top 10 % = the first row
    mask = np.zeros((10, 10), bool)
    mask[0, :5] = True
    assert explain.map_on_lesions(heat, mask) == (0.5, 0.05)


def test_the_written_summary_states_only_what_the_outputs_say():
    rates = {"exact": 0.92, "within_one": 0.98, "images": 3393}
    lines = summary.written_summary(2, [0.0, 0.0, 0.76, 0.0, 0.24], {"MA": 0.0, "HE": 0.003, "EX": 0.018, "SE": 0.0},
                                    {"Grad-CAM": (0.35, 0.02)}, False, [], rates, "optimistic")
    text = " ".join(lines)
    assert "Grade 2 - Moderate NPDR" in text and "76 %" in text
    assert text.index("hard exudates") < text.index("haemorrhages")              # largest area first
    assert "microaneurysms (" not in text                                      # not found -> not listed
    assert "35 % of its most influential area" in text
    assert "No reliability flag" in text and "exact in 92 %" in text
    lines = summary.written_summary(3, [0, 0, 0.1, 0.9, 0], {k: 0.0 for k in CLASSES}, {}, True,
                                    ["the three models give different grades [3, 2, 3]"], rates, "x")
    text = " ".join(lines)
    assert "Stage 4 marks no lesions" in text and "worth a closer look" in text and "Review recommended" in text


def test_the_reliability_calibration_file_matches_the_validation_tables_it_came_from():
    import importlib.util
    cal = summary.load_calibration(os.path.join(ROOT, "data", "reliability_calibration.json"))
    for arm in ("crop", "direct"):
        c = cal["arms"][arm]
        assert (c["images"], c["certain"]["images"] + c["uncertain"]["images"]) == (4242, 4242)
        assert c["certain"]["exact"] > c["uncertain"]["exact"]
    spec = importlib.util.spec_from_file_location("make_cal", os.path.join(ROOT, "scripts", "make_reliability_calibration.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.build() == cal
