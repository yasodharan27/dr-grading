import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest

from drgrade import explain, uncertainty


class TinyGrader:
    """A small differentiable stand-in with the Grader interface: grad_model rgb -> (feature map, logits)."""

    def __init__(self, size=32):
        import keras
        keras.utils.set_random_seed(0)
        rgb = keras.Input((size, size, 3))
        f = keras.layers.Conv2D(8, 3, strides=4, padding="same", activation="tanh")(rgb)
        logits = keras.layers.Dense(4)(keras.layers.GlobalAveragePooling2D()(f))
        self.grad_model = keras.Model(rgb, [f, logits])

    def logits(self, frames, batch_size=8):
        return np.asarray(self.grad_model.predict_on_batch(np.asarray(frames, np.float32))[1], np.float64)


@pytest.fixture(scope="module")
def tiny():
    return TinyGrader()


def test_integrated_gradients_completeness(tiny):
    x = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    attribution, gap, heat = explain.integrated_gradients(tiny, x, steps=64)
    change = explain.score(tiny.logits(x[None]))[0] - explain.score(tiny.logits(np.zeros_like(x)[None]))[0]
    assert attribution.shape == x.shape and heat.shape == (32, 32) and heat.max() == pytest.approx(1.0)
    assert gap < 0.02 * abs(change) + 1e-4                                # sum of attributions = score change


def test_grad_cam_shape_and_range(tiny):
    x = np.random.default_rng(1).random((2, 32, 32, 3)).astype(np.float32)
    cam, logits = explain.grad_cam(tiny, x)
    assert cam.shape == (2, 32, 32) and logits.shape == (2, 4)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6
    assert np.allclose(logits, tiny.logits(x), atol=1e-5)


def test_occlusion_finds_the_region_that_raises_the_score():
    def fn(rgb):                                                           # logits rise with the brightness of the top-left corner
        corner = np.asarray(rgb)[:, :16, :16].mean(axis=(1, 2, 3))
        return np.stack([4 * corner - 1] * 4, axis=1)
    x = np.ones((64, 64, 3), np.float32)
    raw, heat = explain.occlusion(fn, x, patch=16, stride=16)
    assert heat[:16, :16].min() == pytest.approx(1.0) and heat[32:, 32:].max() == 0.0


def test_kernel_shap_is_exact_for_an_additive_game():
    weights = np.array([0.5, -1.0, 2.0, 0.25])
    phi, base = explain.kernel_shap(lambda masks: masks @ weights + 3.0, 4, n_samples=512)
    assert base == pytest.approx(3.0) and np.allclose(phi, weights, atol=1e-9)


def test_localisation_measures():
    heat = np.zeros((10, 10)); heat[2, 3] = 1.0
    mask = np.zeros((10, 10), bool); mask[2, 3] = True
    m = explain.localisation(heat, mask, top_fraction=0.01)
    assert m["pointing_hit"] and m["top_precision"] == 1.0 and m["lesion_fraction"] == pytest.approx(0.01) and m["auroc"] == 1.0


def test_dihedral_views_are_the_eight_distinct_symmetries():
    x = np.arange(2 * 3 * 3 * 1).reshape(2, 3, 3, 1)
    views = [uncertainty.dihedral(x, k) for k in range(8)]
    assert np.array_equal(views[0], x)
    assert len({v.tobytes() for v in views}) == 8


def test_tta_logits_runs_every_view_in_one_call():
    calls = []

    def fn(rgb):                                                           # logits = mean of the top row: orientation-dependent
        calls.append(len(rgb))
        return np.repeat(np.asarray(rgb)[:, 0].mean(axis=(1, 2))[:, None], 4, axis=1)
    x = np.random.default_rng(4).random((2, 6, 6, 3))
    out = uncertainty.tta_logits(fn, x)
    assert calls == [16] and out.shape == (8, 2, 4)
    for k in range(8):
        assert np.allclose(out[k], fn(uncertainty.dihedral(x, k)))


def test_ensemble_scores():
    certain = np.full((3, 1, 4), 12.0)                                    # every seed: grade 4 with certainty
    split = np.stack([np.full((1, 4), 12.0), np.full((1, 4), -12.0), np.full((1, 4), 12.0)])
    e1, e2 = uncertainty.ensemble(certain), uncertainty.ensemble(split)
    assert e1["pred"][0] == 4 and e1["entropy"][0] < 1e-3 and e1["seed_disagreement"][0] < 1e-3
    assert e2["seed_pred"][:, 0].tolist() == [4, 0, 4] and e2["seed_disagreement"][0] > 1.5 and e2["entropy"][0] > 0.5


def test_display_scale_changes_the_picture_not_the_map():
    heat = np.zeros((50, 50), np.float32)
    heat[10, 10] = 100.0                                                   # one extreme pixel
    heat[20:30, 20:30] = 1.0
    before = heat.copy()
    shown = explain.display_scale(heat)
    assert np.array_equal(heat, before)                                    # the map itself is untouched
    assert shown.max() == 1.0 and shown[25, 25] == pytest.approx(1.0) and shown.min() == 0.0
    assert explain.display_scale(np.zeros((4, 4))).max() == 0.0
