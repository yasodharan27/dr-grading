import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest

from drgrade import data
from drgrade import grader as gr
from drgrade import train_grader as tg

SIZE = 64


class FakeFrames:
    """Random frames with a grade-dependent brightness, so the loss has something to learn."""

    def __init__(self, grades):
        self.grades = grades

    def frame(self, image):
        rng = np.random.default_rng(abs(hash(("frame", image))) % 2 ** 32)
        return np.clip(rng.random((SIZE, SIZE, 3)) * 0.4 + 0.1 * self.grades[image], 0, 1).astype(np.float32)


def _parts():
    rng = np.random.default_rng(0)
    make = lambda prefix, n: [(f"{prefix}{i}", int(rng.integers(0, 5))) for i in range(n)]       # noqa: E731
    aptos = {"train": make("a", 6), "val": make("av", 4)}
    eyepacs = {"train": make("e", 14), "val": make("ev", 4)}
    grades = {i: g for part in (aptos, eyepacs) for rows in part.values() for i, g in rows}
    return aptos, eyepacs, FakeFrames(grades)


def _run(run_dir, max_epochs):
    aptos, eyepacs, frames = _parts()
    return tg.train(str(run_dir), frames, aptos, eyepacs, 42, "crop", protocol=dict(tg.PROTOCOL, batch_size=4), image_size=SIZE,
                    mixed_precision=False, max_epochs=max_epochs, workers=1, log=lambda *_: None,
                    model_factory=lambda: gr.build_model(42, None, SIZE))


def test_epoch_has_the_joint_composition_and_wraps_the_rotation():
    aptos, eyepacs, _ = _parts()
    rotation = tg.eyepacs_rotation(eyepacs["train"], 42)
    first, wrapped = tg.epoch_entries(aptos["train"], rotation, 0), tg.epoch_entries(aptos["train"], rotation, 2)
    assert len(first) == 6 + 12 and [e for e in first[:6]] == [(i, g) for i, g in aptos["train"]]
    assert set(first[6:]) <= set(rotation) and len(set(first[6:])) == 12
    assert len(wrapped) == 18                                  # 2 x 12 = 24 > 14: the slice wraps round
    assert sorted(tg.epoch_order(first, 42, 0)) == sorted(first) and tg.epoch_order(first, 42, 0) != tg.epoch_order(first, 42, 1)


def test_the_full_split_files_give_the_approved_class_weights():
    aptos, eyepacs = data.aptos_entries(), data.eyepacs_entries()
    assert (len(aptos["train"]), len(aptos["val"]), len(eyepacs["train"]), len(eyepacs["val"])) == (2921, 730, 31610, 3512)
    weights, _ = tg.class_weights(aptos["train"], eyepacs["train"], seed=42, epochs=50)
    assert np.allclose(weights, tg.APPROVED_CLASS_WEIGHTS, atol=5e-5)


def test_weighted_corn_loss_matches_a_direct_computation():
    logits = np.array([[1.0, -0.5, 0.2, 0.0], [-2.0, 1.0, 1.0, 1.0]], np.float32)
    grades = np.array([2, 0], np.int32)
    w = [1.0, 2.0, 3.0, 4.0, 5.0]
    loss = float(tg.weighted_corn_loss(w)(grades, logits))
    bce = lambda z, y: max(z, 0) - z * y + np.log1p(np.exp(-abs(z)))                              # noqa: E731
    num = sum(w[2] * bce(logits[0, k], float(2 > k)) for k in range(3)) + w[0] * bce(logits[1, 0], 0.0)
    assert loss == pytest.approx(num / 4, rel=1e-5)             # thresholds k <= grade: 3 pairs + 1 pair


def test_training_is_resumable_and_equals_an_uninterrupted_run(tmp_path):
    whole = _run(tmp_path / "whole", 3)
    _run(tmp_path / "split", 2)                                # interrupted after two epochs (not a terminal stop)
    resumed = _run(tmp_path / "split", 3)
    assert whole["completed_epochs"] == resumed["completed_epochs"] == 3
    for a, b in zip(whole["history"], resumed["history"]):
        assert a["train_loss"] == pytest.approx(b["train_loss"], rel=1e-3, abs=1e-4)
        assert a["selection"] == pytest.approx(b["selection"], abs=1e-9)
    assert np.isfinite([h["train_loss"] for h in whole["history"]]).all()
    assert os.path.exists(tmp_path / "split" / "checkpoints" / "best.weights.h5")
    assert sorted(f for f in os.listdir(tmp_path / "split" / "checkpoints") if f.startswith("last_")) == \
        ["last_0002.weights.h5", "last_0002_optimizer.npz"]    # only the newest generation is kept
