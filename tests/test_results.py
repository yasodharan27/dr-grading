"""The committed results are internally consistent and reproduce from the committed per-image tables."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from drgrade import evaluate as ev
from drgrade import grader as gr
from drgrade.config import ARMS, ROOT, SEEDS

R = os.path.join(ROOT, "results")
SPLITS = os.path.join(ROOT, "data", "splits")


@pytest.mark.parametrize("path", [os.path.join(R, "predictions", d, f"per_image_{a}_seed{s}.csv") for d in ("ddr", "idrid") for a in ARMS for s in SEEDS]
                         + [os.path.join(R, "validation", f"{a}_{n}_seed{s}.csv") for a in ARMS for n in ("aptos_val", "eyepacs_val") for s in SEEDS])
def test_decoded_grades_and_probabilities_follow_from_the_logits(path):
    t = pd.read_csv(path)
    d = gr.decode(t[[f"logit_{k}" for k in range(4)]].to_numpy(np.float64))
    assert np.array_equal(d["grade"], t["predicted_grade"].to_numpy(int))
    assert np.allclose(d["p_gt"], t[[f"p_gt_{k}" for k in range(4)]].to_numpy(), atol=1e-6)
    assert np.allclose(d["class_prob"], t[[f"class_prob_{k}" for k in range(5)]].to_numpy(), atol=1e-6)


@pytest.mark.parametrize("dataset", ["ddr", "idrid"])
def test_every_table_holds_the_evaluation_list_in_order(dataset):
    split = pd.read_csv(os.path.join(SPLITS, f"{dataset}_test.csv"), dtype={"image": str})
    folders = [os.path.join(R, "predictions", dataset), os.path.join(ROOT, "data", "baselines", dataset)]
    for folder in folders:
        for name in sorted(os.listdir(folder)):
            t = pd.read_csv(os.path.join(folder, name), dtype={"image_id": str})
            assert t["image_id"].tolist() == split["image"].tolist(), name
            assert t["true_grade"].tolist() == split["grade"].tolist(), name


def test_headline_contrasts():
    c = pd.read_csv(os.path.join(R, "tables", "external_contrasts.csv"))
    row = lambda d, p, n: c[(c.dataset == d) & (c.population == p) & (c.contrast == n)].iloc[0]       # noqa: E731
    ddr = row("ddr", "primary_3540", "crop_minus_p")
    assert ddr.qwk_difference_mean == pytest.approx(0.3013, abs=5e-4) and ddr.ci_low > 0 and ddr.seeds_positive == 3
    idrid = row("idrid", "primary_100", "crop_minus_p")
    assert idrid.qwk_difference_mean == pytest.approx(0.146, abs=1e-3) and idrid.ci_low > 0
    assert (row("ddr", "primary_3540", "direct_minus_p").qwk_difference_mean) == pytest.approx(0.290, abs=1e-3)


def test_idrid_results_recompute_exactly(tmp_path):
    got = ev.evaluate("idrid", SPLITS, os.path.join(R, "predictions", "idrid"), os.path.join(ROOT, "data", "baselines", "idrid"), str(tmp_path))
    for population in got:
        with open(os.path.join(R, "external", "idrid", f"result_{population}.json"), encoding="utf-8") as fh:
            assert json.load(fh) == json.loads(json.dumps(got[population]))
