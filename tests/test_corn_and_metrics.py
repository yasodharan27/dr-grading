import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from drgrade import grader as gr
from drgrade import metrics as mt


def test_corn_decode_counts_cumulative_probabilities_above_half():
    # sigmoid(0) = 0.5 exactly: a cumulative probability of 0.5 is NOT above 0.5
    logits = np.array([[10.0, 10.0, 10.0, 10.0], [-10.0, 10.0, 10.0, 10.0], [10.0, 0.0, 10.0, 10.0], [10.0, 10.0, 10.0, -10.0]])
    d = gr.decode(logits)
    assert d["grade"].tolist() == [4, 0, 1, 3]
    assert np.allclose(d["class_prob"].sum(axis=1), 1.0, atol=1e-6)
    assert np.all(np.diff(d["p_gt"], axis=1) <= 1e-7)                   # cumulative probabilities never increase


def test_class_probabilities_and_expected_grade():
    p_gt = np.array([[0.9, 0.6, 0.3, 0.1]])
    assert np.allclose(gr.class_probabilities(p_gt), [[0.1, 0.3, 0.3, 0.2, 0.1]])
    logits = np.log(np.array([[0.9, 0.6 / 0.9, 0.3 / 0.6, 0.1 / 0.3]]) / (1 - np.array([[0.9, 0.6 / 0.9, 0.3 / 0.6, 0.1 / 0.3]])))
    assert gr.expected_grade(logits)[0] == pytest.approx(1.9)


def test_qwk_matches_sklearn_and_known_values():
    rng = np.random.default_rng(0)
    g, p = rng.integers(0, 5, 300), rng.integers(0, 5, 300)
    assert mt.qwk(g, p) == pytest.approx(cohen_kappa_score(g, p, weights="quadratic", labels=list(range(5))), abs=1e-12)
    assert mt.qwk(g, g) == pytest.approx(1.0)
    assert mt.qwk(np.array([0, 4]), np.array([4, 0])) == pytest.approx(-1.0)


def test_rank_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    y = rng.random(500) < 0.3
    s = np.round(rng.random(500), 1)                                    # many ties
    assert mt.rank_auroc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert np.isnan(mt.rank_auroc(np.zeros(5, bool), s[:5]))


def _table(grades, logits, ids=None):
    d = gr.decode(logits)
    return {"ids": np.asarray(ids if ids is not None else [str(i) for i in range(len(grades))]), "grade": np.asarray(grades),
            "pred": d["grade"], "p_gt": d["p_gt"].astype(np.float64)}


def test_bootstrap_is_grade_stratified_and_reproducible():
    grades = np.repeat(np.arange(5), [50, 10, 30, 5, 8])
    a, b = mt.bootstrap_indices(grades, 20), mt.bootstrap_indices(grades, 20)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    for idx in a:
        assert np.array_equal(np.bincount(grades[idx], minlength=5), np.bincount(grades, minlength=5))


def test_bootstrap_metrics_equal_full_metrics_on_identity_resample():
    rng = np.random.default_rng(2)
    grades = rng.integers(0, 5, 200)
    table = _table(grades, rng.normal(size=(200, 4)) + grades[:, None] - 1.5)
    full, boot = mt.metrics(table), mt.bootstrap_metrics(table, np.arange(200))
    for k in mt.BOOT_KEYS:
        assert boot[k] == pytest.approx(full[k], abs=1e-12)


def test_paired_delta_refuses_different_images():
    rng = np.random.default_rng(3)
    a = _table(np.arange(5), rng.normal(size=(5, 4)))
    b = _table(np.arange(5), rng.normal(size=(5, 4)), ids=list("abcde"))
    with pytest.raises(ValueError):
        mt.paired_delta(a, b)


def test_seed_ensemble_averages_cumulative_probabilities():
    t1 = {"ids": np.array(["x"]), "grade": np.array([2]), "pred": np.array([2]), "p_gt": np.array([[0.9, 0.7, 0.2, 0.1]])}
    t2 = {"ids": np.array(["x"]), "grade": np.array([2]), "pred": np.array([1]), "p_gt": np.array([[0.9, 0.2, 0.1, 0.0]])}
    e = mt.ensemble([t1, t2])
    assert np.allclose(e["p_gt"], [[0.9, 0.45, 0.15, 0.05]]) and e["pred"].tolist() == [1]


def test_per_image_rows_schema():
    rows = mt.per_image_rows(["a", "b"], [0, 3], np.array([[-2.0, 0, 0, 0], [3.0, 3.0, 3.0, -3.0]]))
    assert rows[0]["predicted_grade"] == 0 and rows[1]["predicted_grade"] == 3
    assert {"logit_3", "p_gt_0", "p_cond_2", "class_prob_4", "nearest_threshold_margin", "class_entropy_nats"} <= set(rows[0])
