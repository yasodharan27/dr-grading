import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

from drgrade import explain
from drgrade import grader as gr


def test_checkpoint_round_trip_and_explainer_graph(tmp_path):
    """A saved grader loads into `Grader`, whose prediction graph and explainer graph give the same logits."""
    import keras
    keras.mixed_precision.set_global_policy("float32")
    model = gr.build_model(seed=7, image_size=64)
    model.get_layer("corn").get_layer("corn_logits").bias.assign(np.array([0.5, -0.2, 0.1, -1.0], np.float32))
    path = str(tmp_path / "g.weights.h5")
    model.save_weights(path)
    x = np.random.default_rng(0).random((2, 64, 64, 3)).astype(np.float32)
    expected = np.asarray(model.predict_on_batch(x), np.float64)
    g = gr.Grader(path, image_size=64)
    assert np.allclose(g.logits(x), expected, atol=1e-5)
    features, logits = g.grad_model.predict_on_batch(x)
    assert features.shape == (2, 2, 2, gr.FEATURE_DIM)
    assert np.allclose(logits, expected, atol=1e-5)
    cam, cam_logits = explain.grad_cam(g, x)
    assert cam.shape == (2, 64, 64) and np.allclose(cam_logits, expected, atol=1e-5)


def test_head_initialisation_depends_on_the_seed_only():
    import keras
    keras.mixed_precision.set_global_policy("float32")
    a = gr.build_model(seed=42, image_size=32).get_layer("corn").get_weights()
    b = gr.build_model(seed=42, image_size=32).get_layer("corn").get_weights()
    c = gr.build_model(seed=123, image_size=32).get_layer("corn").get_weights()
    assert np.array_equal(a[0], b[0]) and not np.array_equal(a[0], c[0]) and not a[1].any()
