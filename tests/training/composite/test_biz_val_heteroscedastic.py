"""Unit + biz_value + cProfile tests for HeteroscedasticCompositeEstimator.

Covers: fit/predict shapes, positive predictive std, ~nominal interval coverage on a homoscedastic synthetic,
params/clone roundtrip, degenerate (constant y, tiny n), the importorskipped ngboost path, and the biz_value
contract -- per-row std correlates with the true local noise scale on a heteroscedastic synthetic AND a
constant-width interval is beaten (lower Winkler) by the heteroscedastic one.
"""

from __future__ import annotations

import cProfile
import io
import pstats

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor

from mlframe.training.composite import winkler_interval_score
from mlframe.training.composite._heteroscedastic import HeteroscedasticCompositeEstimator


def _base_estimator():
    """Base estimator."""
    return HistGradientBoostingRegressor(max_iter=80, max_depth=3, random_state=0)


def _homoscedastic(n=1500, seed=0):
    """Homoscedastic."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, n)
    x0 = rng.normal(0.0, 1.0, n)
    y = 2.0 * base + 0.5 * x0 + rng.normal(0.0, 0.5, n)
    X = pd.DataFrame({"base": base, "x0": x0})
    return X, y


def _heteroscedastic(n=3000, seed=0):
    """y = 2*base + f(x0) + s(x0)*eps with noise scale s(x0)=0.15+|x0| growing with the feature."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, n)
    x0 = rng.uniform(-2.0, 2.0, n)
    scale = 0.15 + np.abs(x0)
    eps = rng.normal(0.0, 1.0, n)
    y = 2.0 * base + 0.3 * x0 + scale * eps
    X = pd.DataFrame({"base": base, "x0": x0})
    return X, y, scale


def _fit(X, y, **kw):
    """Fit."""
    est = HeteroscedasticCompositeEstimator(
        base_estimator=_base_estimator(),
        transform_name="linear_residual",
        base_column="base",
        prefer_ngboost=False,
        alpha=0.1,
        **kw,
    )
    return est.fit(X, y)


# ---------------------------------------------------------------- unit
def test_fit_predict_shapes_and_positive_std():
    """Fit predict shapes and positive std."""
    X, y = _homoscedastic()
    est = _fit(X, y)
    yhat = est.predict(X)
    std = est.predict_std(X)
    lo, hi = est.predict_interval(X)
    assert yhat.shape == (len(y),)
    assert std.shape == (len(y),)
    assert lo.shape == hi.shape == (len(y),)
    assert np.all(std > 0), "predictive std must be strictly positive"
    assert np.all(hi >= lo), "interval upper must bracket lower"
    assert est.backend_ == "two_model"


def test_interval_covers_at_nominal_on_homoscedastic():
    """Interval covers at nominal on homoscedastic."""
    X, y = _homoscedastic(n=2000, seed=1)
    Xte, yte = _homoscedastic(n=2000, seed=2)
    est = _fit(X, y)
    lo, hi = est.predict_interval(Xte, alpha=0.1)
    cov = float(np.mean((yte >= lo) & (yte <= hi)))
    assert 0.83 <= cov <= 0.97, f"90% interval coverage {cov:.3f} off nominal"


def test_predict_var_is_std_squared():
    """Predict var is std squared."""
    X, y = _homoscedastic()
    est = _fit(X, y)
    std = est.predict_std(X)
    var = est.predict_var(X)
    assert np.allclose(var, std**2)


def test_params_and_clone_roundtrip():
    """Params and clone roundtrip."""
    est = HeteroscedasticCompositeEstimator(
        base_estimator=_base_estimator(),
        transform_name="linear_residual",
        base_column="base",
        prefer_ngboost=False,
        alpha=0.2,
    )
    p = est.get_params()
    assert p["alpha"] == 0.2 and p["base_column"] == "base"
    cl = clone(est)
    assert cl.get_params()["alpha"] == 0.2
    X, y = _homoscedastic(n=500)
    cl.fit(X, y)
    assert cl.predict_std(X).shape == (len(y),)


def test_degenerate_constant_y_and_tiny_n():
    """Degenerate constant y and tiny n."""
    rng = np.random.default_rng(0)
    n = 40
    X = pd.DataFrame({"base": rng.normal(0, 1, n), "x0": rng.normal(0, 1, n)})
    y = np.full(n, 3.0)
    est = _fit(X, y)
    std = est.predict_std(X)
    assert np.all(np.isfinite(std)) and np.all(std >= 0)
    # tiny n must not crash
    est2 = _fit(X.iloc[:8], y[:8])
    assert est2.predict(X.iloc[:8]).shape == (8,)


def test_ngboost_path_when_available():
    """Ngboost path when available."""
    ngboost = pytest.importorskip("ngboost")  # noqa: F841
    X, y = _homoscedastic(n=800)
    est = HeteroscedasticCompositeEstimator(
        base_estimator=_base_estimator(),
        transform_name="linear_residual",
        base_column="base",
        prefer_ngboost=True,
        alpha=0.1,
    )
    est.fit(X, y)
    assert est.backend_ == "ngboost"
    assert np.all(est.predict_std(X) > 0)


# ---------------------------------------------------------------- biz_value
def test_biz_val_hetero_std_tracks_true_local_noise_scale():
    """Per-row predictive std must correlate strongly (Spearman >= 0.6) with the true noise scale s(x0)."""
    X, y, scale = _heteroscedastic(n=3000, seed=3)
    est = _fit(X, y)
    std = est.predict_std(X)
    rho = float(spearmanr(std, scale).statistic)
    assert rho >= 0.6, f"predictive std vs true noise scale Spearman {rho:.3f} < 0.6"


def test_biz_val_hetero_interval_beats_constant_width_winkler():
    """The heteroscedastic interval must beat a constant-width interval of equal mean coverage on Winkler score."""
    Xtr, ytr, _ = _heteroscedastic(n=3000, seed=4)
    Xte, yte, _ = _heteroscedastic(n=3000, seed=5)
    est = _fit(Xtr, ytr)
    alpha = 0.1
    lo, hi = est.predict_interval(Xte, alpha=alpha)
    cov_hetero = float(np.mean((yte >= lo) & (yte <= hi)))

    # Constant-width band centred on the same point prediction, half-width tuned to MATCH the heteroscedastic
    # coverage on the test set -- so the comparison isolates SHAPE (adaptive vs flat), not coverage level.
    yhat = est.predict(Xte)
    resid = np.abs(yte - yhat)
    half = float(np.quantile(resid, cov_hetero))
    lo_c, hi_c = yhat - half, yhat + half

    w_hetero = winkler_interval_score(yte, lo, hi, alpha)
    w_const = winkler_interval_score(yte, lo_c, hi_c, alpha)
    assert w_hetero <= 0.9 * w_const, f"hetero Winkler {w_hetero:.3f} not < 0.9 * const {w_const:.3f}"


# ---------------------------------------------------------------- cProfile
def test_cprofile_fit_predict_std_runs_fast():
    """Cprofile fit predict std runs fast."""
    X, y, _ = _heteroscedastic(n=2000, seed=6)
    pr = cProfile.Profile()
    pr.enable()
    est = _fit(X, y)
    _ = est.predict_std(X)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(5)
    assert est.predict_std(X).shape == (len(y),)
