"""Unit tests for ``CompositeQRFEstimator`` (quantile-regression-forest composite).

Covers the contract that distinguishes the QRF composite from the dense-quantile
estimator: a SINGLE fitted forest serves ANY query quantile (no per-level refit),
the returned quantiles are monotone (non-crossing) per row, per-quantile coverage is
near-nominal on held-out data, the CDF is a valid monotone step function, and the
CRPS is finite. Pure sklearn backend -- no optional inner lib required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.qrf import (
    CompositeQRFEstimator,
    _LeafResidualForest,
    _weighted_quantiles,
)


def _hetero_xy(n: int = 3000, seed: int = 0):
    """y = 2*base + eps(base) with noise std growing in |base| (heteroscedastic)."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(-3.0, 3.0, n)
    x1 = rng.normal(0.0, 1.0, n)
    sigma = 0.3 + 0.9 * np.abs(base)
    y = 2.0 * base + 0.5 * x1 + sigma * rng.standard_normal(n)
    return pd.DataFrame({"base": base, "x1": x1}), y


def _fit_split(seed: int = 0, n: int = 3000):
    X, y = _hetero_xy(n=n, seed=seed)
    ntr = (n * 3) // 4
    Xtr, ytr = X.iloc[:ntr], y[:ntr]
    Xte, yte = X.iloc[ntr:], y[ntr:]
    est = CompositeQRFEstimator(
        base_column="base",
        n_estimators=120,
        min_samples_leaf=8,
        random_state=0,
    ).fit(Xtr, ytr)
    return est, Xte, yte


# ----------------------------------------------------------------------
def test_single_fit_serves_many_quantiles():
    """ONE fitted forest answers many different quantile sets, all (n, n_q) shaped.

    The whole point of the QRF composite: no per-level refit. We request three
    DIFFERENT level sets from the same fitted model and each returns the right shape
    with finite values.
    """
    est, Xte, _ = _fit_split()
    for levels in ([0.5], [0.1, 0.9], [0.05, 0.25, 0.5, 0.75, 0.95]):
        Q = est.predict_quantile(Xte, quantiles=levels)
        assert Q.shape == (len(Xte), len(levels))
        assert np.all(np.isfinite(Q))


def test_quantiles_are_monotone_non_crossing():
    """Per-row predicted quantiles are ascending (q_low <= ... <= q_high)."""
    est, Xte, _ = _fit_split()
    Q = est.predict_quantile(Xte, quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    diffs = np.diff(Q, axis=1)
    assert np.all(diffs >= -1e-9), "predicted quantiles must not cross within a row"


def test_per_quantile_coverage_near_nominal():
    """Held-out P(y <= Q(q)) ~= q for several q from the single fit.

    For a calibrated conditional distribution the empirical fraction of held-out y
    below the predicted q-quantile should track q. We allow a 0.06 tolerance band
    (finite-sample + forest smoothing).
    """
    est, Xte, yte = _fit_split(n=4000)
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        below = float(np.mean(yte <= est.predict_quantile(Xte, quantiles=[q])[:, 0]))
        assert abs(below - q) <= 0.06, f"q={q}: empirical below={below:.3f}"


def test_central_band_coverage_near_nominal():
    """An 80% central band (q=0.1..0.9) covers ~80% of held-out y."""
    est, Xte, yte = _fit_split(n=4000)
    Q = est.predict_quantile(Xte, quantiles=[0.1, 0.9])
    cov = float(np.mean((yte >= Q[:, 0]) & (yte <= Q[:, 1])))
    assert 0.74 <= cov <= 0.86, f"80%% band coverage was {cov:.3f}"


def test_crps_is_finite_and_per_row_shape():
    """CRPS returns a finite scalar (mean) and a finite (n,) vector (none)."""
    est, Xte, yte = _fit_split()
    scalar = est.crps(Xte, yte)
    assert np.isfinite(scalar) and scalar > 0.0
    per_row = est.crps(Xte, yte, reduce="none")
    assert per_row.shape == (len(Xte),)
    assert np.all(np.isfinite(per_row))


def test_predict_cdf_is_monotone_valid():
    """predict_cdf rows are non-decreasing in the y-grid and bounded in [0, 1]."""
    est, Xte, _ = _fit_split()
    grid = np.linspace(-8.0, 8.0, 9)
    cdf = est.predict_cdf(Xte.iloc[:50], grid)
    assert cdf.shape == (50, len(grid))
    assert np.all(np.diff(cdf, axis=1) >= -1e-9), "CDF must be non-decreasing"
    assert cdf.min() >= 0.0 and cdf.max() <= 1.0


def test_predict_is_median():
    """predict() equals the 0.5-quantile column."""
    est, Xte, _ = _fit_split()
    med = est.predict(Xte)
    q50 = est.predict_quantile(Xte, quantiles=[0.5])[:, 0]
    assert np.allclose(med, q50)


def test_invalid_levels_raise():
    """Out-of-(0,1) and empty level requests raise."""
    est, Xte, _ = _fit_split()
    with pytest.raises(ValueError):
        est.predict_quantile(Xte, quantiles=[0.0, 0.5])
    with pytest.raises(ValueError):
        est.predict_quantile(Xte, quantiles=[1.0])
    with pytest.raises(ValueError):
        est.predict_quantile(Xte, quantiles=[])


def test_predict_before_fit_raises():
    """Calling predict surface before fit raises NotFittedError."""
    from sklearn.exceptions import NotFittedError

    est = CompositeQRFEstimator(base_column="base")
    with pytest.raises(NotFittedError):
        est.predict_quantile(pd.DataFrame({"base": [1.0]}), quantiles=[0.5])


def test_crps_length_mismatch_raises():
    """CRPS with mismatched y_true length raises."""
    est, Xte, _ = _fit_split()
    with pytest.raises(ValueError):
        est.crps(Xte, np.zeros(len(Xte) + 1))


# ----------------------------------------------------------------------
# Backend / kernel-level checks
# ----------------------------------------------------------------------
def test_weighted_quantiles_basic():
    """Equal-weight weighted quantiles match unweighted percentile behaviour."""
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w = np.ones(5)
    out = _weighted_quantiles(v, w, np.array([0.5]))
    assert abs(out[0] - 3.0) <= 1e-9
    # Empty / zero-weight -> NaN.
    assert np.isnan(_weighted_quantiles(np.array([]), np.array([]), np.array([0.5]))[0])
    assert np.isnan(_weighted_quantiles(v, np.zeros(5), np.array([0.5]))[0])


def test_weighted_quantiles_respect_weights():
    """A heavily-weighted value pulls the median toward it."""
    v = np.array([0.0, 10.0])
    w = np.array([9.0, 1.0])
    med = _weighted_quantiles(v, w, np.array([0.5]))[0]
    assert med < 5.0, "median should sit near the heavily-weighted low value"


def test_leaf_residual_forest_njit_matches_numpy_fallback():
    """The njit leaf-weight kernel is numerically identical to the numpy fallback.

    Toggles the module-level numba flag to force the numpy path and asserts the two
    weight matrices (hence the resulting quantiles) agree to float tolerance.
    """
    import mlframe.training.composite.qrf as qrf_mod

    rng = np.random.default_rng(3)
    Xtr = rng.normal(0, 1, (400, 3))
    ytr = Xtr[:, 0] + 0.3 * rng.normal(0, 1, 400)
    Xq = rng.normal(0, 1, (60, 3))

    f = _LeafResidualForest(n_estimators=40, min_samples_leaf=5, random_state=1).fit(Xtr, ytr)
    orig = qrf_mod._HAS_NUMBA
    try:
        w_njit = f._leaf_weights(Xq) if orig else None
        qrf_mod._HAS_NUMBA = False
        w_numpy = f._leaf_weights(Xq)
    finally:
        qrf_mod._HAS_NUMBA = orig
    # Each row's weights sum to 1 (proper membership distribution).
    assert np.allclose(w_numpy.sum(axis=1), 1.0, atol=1e-9)
    if w_njit is not None:
        assert np.allclose(w_njit, w_numpy, atol=1e-9)


def test_batched_predict_matches_unbatched():
    """Query-row batching does not change the result (batch boundary safety)."""
    import mlframe.training.composite.qrf as qrf_mod

    est, Xte, _ = _fit_split()
    levels = [0.1, 0.5, 0.9]
    full = est.predict_quantile(Xte, quantiles=levels)
    orig = qrf_mod._PREDICT_BATCH
    try:
        qrf_mod._PREDICT_BATCH = 37  # awkward size to force partial last batch
        batched = est.predict_quantile(Xte, quantiles=levels)
    finally:
        qrf_mod._PREDICT_BATCH = orig
    assert np.allclose(full, batched, atol=1e-9)
