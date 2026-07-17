"""Unit + biz_value tests for Inductive Venn-Abers probability calibration on
``CompositeClassificationEstimator`` (binary).

Validity contract: on a deliberately miscalibrated (over-confident) inner the
Venn-Abers calibrated probabilities have a LOWER Expected Calibration Error than
the raw ``predict_proba``, the ``[p0, p1]`` interval brackets the point estimate and
the empirical frequency, and the calibrated probability is monotone in the score.
"""

from __future__ import annotations

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from mlframe.training.composite.classification import CompositeClassificationEstimator
from mlframe.training.composite.venn_abers import _isotonic_envelopes


def _overconfident_estimator():
    # A deep, near-unregularised forest over-fits the train split => its train-time
    # predict_proba is over-confident (probabilities pushed toward 0/1).
    return CompositeClassificationEstimator(
        base_estimator=lgb.LGBMClassifier(n_estimators=300, num_leaves=63, min_child_samples=2, learning_rate=0.2, verbose=-1),
    )


def _fit_binary(n=3000, seed=0, sep=1.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    centers = np.array([[-sep] * 6, [sep] * 6])
    X = centers[y] + rng.normal(0, 2.0, size=(n, 6))
    n_tr = n // 3
    n_cal = n // 3
    sl = (slice(0, n_tr), slice(n_tr, n_tr + n_cal), slice(n_tr + n_cal, n))
    est = _overconfident_estimator().fit(X[sl[0]], y[sl[0]])
    return est, X, y, sl[1], sl[2]


def _ece(p_pos, y_true, n_bins=10):
    """Expected Calibration Error of P(y=1) predictions against 0/1 labels."""
    p = np.asarray(p_pos, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return ece


# -- unit: isotonic envelopes ordering --------------------------------------
def test_envelopes_p0_le_p1():
    rng = np.random.default_rng(3)
    s = np.sort(rng.uniform(0, 1, 200))
    y = (rng.uniform(0, 1, 200) < s).astype(float)  # label prob ~ score
    _grid, p0, p1 = _isotonic_envelopes(s, y)
    assert np.all(p0 <= p1 + 1e-12), "Venn-Abers p0 must be <= p1 on every grid point"
    assert np.all((p0 >= 0) & (p1 <= 1))


# -- unit: before-fit / before-calibration raises ---------------------------
def test_calibrate_before_fit_raises():
    from sklearn.exceptions import NotFittedError

    est = _overconfident_estimator()
    with pytest.raises(NotFittedError):
        est.calibrate_venn_abers(np.zeros((5, 6)), np.zeros(5))


def test_predict_interval_without_calibration_raises():
    est, X, _y, _sl_cal, sl_te = _fit_binary()
    with pytest.raises(RuntimeError):
        est.predict_proba_interval(X[sl_te])


def test_multiclass_rejected():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=900)
    X = rng.normal(0, 1, size=(900, 6)) + y[:, None]
    est = _overconfident_estimator().fit(X[:300], y[:300])
    with pytest.raises(ValueError):
        est.calibrate_venn_abers(X[300:600], y[300:600])


# -- unit: interval brackets the point estimate -----------------------------
def test_interval_contains_point_estimate():
    est, X, y, sl_cal, sl_te = _fit_binary()
    est.calibrate_venn_abers(X[sl_cal], y[sl_cal])
    lo, hi = est.predict_proba_interval(X[sl_te])
    p = est.predict_proba_venn_abers(X[sl_te])[:, 1]
    assert np.all(lo <= hi + 1e-12)
    assert np.all((p >= lo - 1e-9) & (p <= hi + 1e-9)), "point estimate must lie in [p0, p1]"
    assert np.all((p >= 0) & (p <= 1))


# -- unit: monotone in score ------------------------------------------------
def test_calibrated_prob_monotone_in_score():
    est, X, y, sl_cal, _sl_te = _fit_binary()
    est.calibrate_venn_abers(X[sl_cal], y[sl_cal])
    grid = np.linspace(0.0, 1.0, 50)
    from mlframe.training.composite.venn_abers import _lookup_interval

    lo, hi = _lookup_interval(est, grid)
    denom = 1.0 - lo + hi
    p = hi / denom
    assert np.all(np.diff(p) >= -1e-9), "calibrated probability must be non-decreasing in score"


# -- biz_value: ECE improves vs raw + interval brackets empirical frequency --
def test_biz_val_venn_abers_lowers_ece_vs_raw():
    est, X, y, sl_cal, sl_te = _fit_binary(seed=7)
    raw = est.predict_proba(X[sl_te])[:, 1]
    est.calibrate_venn_abers(X[sl_cal], y[sl_cal])
    cal = est.predict_proba_venn_abers(X[sl_te])[:, 1]
    ece_raw = _ece(raw, y[sl_te])
    ece_cal = _ece(cal, y[sl_te])
    # The over-confident inner has a clearly worse ECE; Venn-Abers must reduce it.
    assert ece_cal <= ece_raw - 0.01, f"Venn-Abers ECE {ece_cal:.4f} did not beat raw {ece_raw:.4f}"


def test_biz_val_interval_brackets_empirical_frequency():
    est, X, y, sl_cal, sl_te = _fit_binary(seed=11)
    est.calibrate_venn_abers(X[sl_cal], y[sl_cal])
    lo, hi = est.predict_proba_interval(X[sl_te])
    yt = y[sl_te]
    # Bin by point estimate; in each populated bin the empirical positive frequency
    # should fall inside [mean(p0), mean(p1)] (allow a small finite-sample slack).
    p = est.predict_proba_venn_abers(X[sl_te])[:, 1]
    edges = np.linspace(0.0, 1.0, 6)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, 4)
    brackets = 0
    populated = 0
    for b in range(5):
        m = idx == b
        if m.sum() < 30:
            continue
        populated += 1
        freq = yt[m].mean()
        if lo[m].mean() - 0.06 <= freq <= hi[m].mean() + 0.06:
            brackets += 1
    assert populated >= 2
    assert brackets >= populated - 1, f"interval bracketed empirical freq in only {brackets}/{populated} bins"
