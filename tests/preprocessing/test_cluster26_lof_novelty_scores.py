"""Cluster 26 regression: LOF with novelty=True scores the training rows instead of crashing (audit 2026-11-20, finding 26-F16)."""

import numpy as np

from mlframe.preprocessing.outlier_detector_zoo import _fit_anomaly_score


def test_lof_novelty_mode_scores_without_error():
    """novelty=True no longer reaches the unavailable fit_predict (26-F16)."""
    X = np.random.default_rng(0).normal(size=(60, 3))
    scores = _fit_anomaly_score("lof", X, random_state=0, detector_kwargs={"novelty": True, "n_neighbors": 5})
    assert scores.shape == (60,)
    assert np.all(np.isfinite(scores))


def test_lof_default_mode_still_scores():
    """Control: the default (novelty=False) path still returns one score per row (26-F16)."""
    X = np.random.default_rng(1).normal(size=(60, 3))
    scores = _fit_anomaly_score("lof", X, random_state=0, detector_kwargs={"n_neighbors": 5})
    assert scores.shape == (60,)


def test_contamination_cutoff_is_deterministic_under_ties():
    """All-equal scores flag exactly the requested count, the same rows on every call (26-F16)."""
    from mlframe.preprocessing.outlier_detector_zoo import select_outlier_threshold

    scores = np.zeros(20)
    first = select_outlier_threshold(scores, method="contamination", contamination=0.25)
    second = select_outlier_threshold(scores, method="contamination", contamination=0.25)
    assert int(first.sum()) == 5
    assert np.array_equal(first, second)
