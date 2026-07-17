"""biz_value test for ``preprocessing.outlier_detector_zoo.select_outlier_threshold``.

Synthetic: 950 inlier rows drawn tightly around the origin plus 50 outlier rows placed far away
(known contamination rate = 5%). IsolationForest anomaly scores are computed once, then each threshold
method converts those scores into a boolean flag. The test proves ``method="contamination"`` recovers the
planted outlier COUNT almost exactly and does so with near-perfect precision/recall against the known
ground-truth labels -- the value the helper adds over "each caller hand-rolls a percentile cutoff."
"""

from __future__ import annotations

import numpy as np

from mlframe.preprocessing.outlier_detector_zoo import make_outlier_detector, select_outlier_threshold


def _make_known_contamination_dataset(seed: int = 0, n_inliers: int = 950, n_outliers: int = 50):
    rng = np.random.default_rng(seed)
    inliers = rng.normal(loc=0.0, scale=1.0, size=(n_inliers, 5))
    outliers = rng.normal(loc=25.0, scale=1.0, size=(n_outliers, 5))
    X = np.vstack([inliers, outliers])
    y_true = np.zeros(X.shape[0], dtype=bool)
    y_true[n_inliers:] = True
    return X, y_true


def _anomaly_scores(X: np.ndarray) -> np.ndarray:
    detector = make_outlier_detector("isolation_forest", n_estimators=200, random_state=0)
    detector.fit(X)
    return -detector.decision_function(X)  # negate to higher = more anomalous


def test_biz_val_select_outlier_threshold_contamination_recovers_known_rate():
    X, y_true = _make_known_contamination_dataset()
    scores = _anomaly_scores(X)
    true_rate = y_true.mean()  # 0.05

    flags = select_outlier_threshold(scores, method="contamination", contamination=true_rate)

    assert flags.sum() == round(true_rate * X.shape[0]), f"expected exactly {round(true_rate * X.shape[0])} rows flagged, got {flags.sum()}"

    precision = (flags & y_true).sum() / flags.sum()
    recall = (flags & y_true).sum() / y_true.sum()
    assert precision >= 0.95, f"expected near-perfect precision recovering a well-separated planted contamination rate, got {precision:.4f}"
    assert recall >= 0.95, f"expected near-perfect recall recovering a well-separated planted contamination rate, got {recall:.4f}"


def test_biz_val_select_outlier_threshold_percentile_matches_contamination_on_clean_split():
    X, y_true = _make_known_contamination_dataset()
    scores = _anomaly_scores(X)

    flags = select_outlier_threshold(scores, method="percentile", percentile=95.0)
    precision = (flags & y_true).sum() / max(flags.sum(), 1)
    recall = (flags & y_true).sum() / y_true.sum()
    assert precision >= 0.9, f"expected percentile-95 cutoff to recover the well-separated 5% contamination with high precision, got {precision:.4f}"
    assert recall >= 0.9, f"expected percentile-95 cutoff to recover the well-separated 5% contamination with high recall, got {recall:.4f}"


def test_biz_val_select_outlier_threshold_iqr_flags_far_fewer_on_gaussian_noise():
    # Pure Gaussian noise, no planted outliers: Tukey's fence (1.5*IQR) should flag only a small tail,
    # far below a naive top-10%-by-contamination cutoff -- proving "iqr" answers a genuinely different
    # question (distribution-driven cutoff) than "contamination" (fixed-rate cutoff).
    rng = np.random.default_rng(1)
    scores = rng.normal(size=5000)

    iqr_flags = select_outlier_threshold(scores, method="iqr", iqr_multiplier=1.5)
    contamination_flags = select_outlier_threshold(scores, method="contamination", contamination=0.10)

    assert iqr_flags.sum() < contamination_flags.sum() * 0.5, (
        f"expected IQR fence on pure Gaussian noise to flag far fewer rows than a fixed 10% contamination "
        f"cutoff, got iqr={iqr_flags.sum()} vs contamination={contamination_flags.sum()}"
    )


def test_select_outlier_threshold_rejects_unknown_method():
    import pytest

    with pytest.raises(ValueError):
        select_outlier_threshold(np.array([1.0, 2.0, 3.0]), method="bogus")
