"""Unit + biz_value tests for Adaptive Conformal Inference (ACI).

Covers ``conformal_online`` wired onto ``CompositeTargetEstimator``:
- stationary stream: ACI coverage ~= static split-conformal (no spurious drift),
- DRIFTING stream: ACI recovers ~1-alpha coverage where a FROZEN band under-covers
  (the biz_value win),
- alpha_t stays in [0, 1] under aggressive drift,
- gamma=0 makes the controller inert (alpha_t never moves -> static).
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from mlframe.training.composite.estimator import CompositeTargetEstimator
from mlframe.training.composite.conformal_online import (
    _rolling_quantile_radius,
    _aci_default_state,
    _aci_step,
)


def _make_fitted_estimator(n=400, seed=0):
    """A trivial fitted wrapper whose y-scale predict is well-behaved."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, n)
    feat = rng.normal(0.0, 1.0, n)
    X = pd_or_np(base, feat)
    y = base + 0.5 * feat + rng.normal(0.0, 0.3, n)
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(), transform_name="diff", base_column="base",
    )
    est.fit(X, y)
    return est


def pd_or_np(base, feat):
    import pandas as pd
    return pd.DataFrame({"base": base, "feat": feat})


def _coverage(est, X, y):
    lo, hi = est.predict_interval_online(X, clip=False)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    return float(np.mean((lo <= y) & (y <= hi)))


# --------------------------------------------------------------------------
# Pure helper unit tests
# --------------------------------------------------------------------------

def test_rolling_quantile_radius_basic():
    r = np.arange(1, 101, dtype=np.float64)  # 1..100
    rad = _rolling_quantile_radius(r, alpha=0.1)
    # finite-sample rank ceil(101*0.9)=91 -> 91st order stat = 91
    assert rad == 91.0


def test_rolling_quantile_radius_saturation():
    r = np.arange(1, 51, dtype=np.float64)
    assert _rolling_quantile_radius(r, alpha=0.0) == float("inf")
    assert _rolling_quantile_radius(r, alpha=1.0) == 0.0
    assert _rolling_quantile_radius(np.array([]), alpha=0.1) == float("inf")


def test_aci_step_controller_direction():
    """A miss lowers alpha_t (widens band); a hit raises it."""
    st = _aci_default_state(alpha=0.1, gamma=0.1, buffer_n=100)
    st["alpha_t"] = 0.1
    _aci_step(st, residual=5.0, in_interval=False)  # miss: err=1
    assert st["alpha_t"] == pytest.approx(0.1 + 0.1 * (0.1 - 1.0))
    st["alpha_t"] = 0.1
    _aci_step(st, residual=0.1, in_interval=True)   # hit: err=0
    assert st["alpha_t"] == pytest.approx(0.1 + 0.1 * (0.1 - 0.0))


def test_alpha_t_stays_in_unit_interval_under_drift():
    st = _aci_default_state(alpha=0.1, gamma=0.5, buffer_n=50)
    rng = np.random.default_rng(1)
    for _ in range(2000):
        miss = bool(rng.random() < 0.9)  # pathological: mostly misses
        _aci_step(st, residual=rng.normal(), in_interval=not miss)
        assert 0.0 <= st["alpha_t"] <= 1.0


# --------------------------------------------------------------------------
# Wired-estimator behaviour
# --------------------------------------------------------------------------

def test_init_required_before_use():
    est = _make_fitted_estimator()
    with pytest.raises(RuntimeError):
        est.predict_interval_online(pd_or_np([0.0], [0.0]))
    with pytest.raises(RuntimeError):
        est.update_conformal(pd_or_np([0.0], [0.0]), [0.0])


def test_gamma_zero_is_static():
    """gamma=0 -> alpha_t never moves from its target (inert controller)."""
    est = _make_fitted_estimator()
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 300)
    feat = rng.normal(0, 1, 300)
    X = pd_or_np(base, feat)
    y = base + 0.5 * feat + rng.normal(0, 0.3, 300)
    est.init_aci(alpha=0.1, gamma=0.0, buffer_n=100, warmup_residuals=np.abs(rng.normal(0, 0.3, 100)))
    est.update_conformal(X, y)
    state = est.get_aci_state(est) if False else est.get_aci_state()
    assert state["alpha_t"] == pytest.approx(0.1)


def test_stationary_tracks_target_coverage():
    """On a stationary stream ACI keeps coverage near 1-alpha (== split-conformal)."""
    est = _make_fitted_estimator(seed=3)
    rng = np.random.default_rng(3)
    n = 1500
    base = rng.normal(0, 1, n)
    feat = rng.normal(0, 1, n)
    X = pd_or_np(base, feat)
    y = base + 0.5 * feat + rng.normal(0, 0.3, n)
    warm = np.abs(rng.normal(0, 0.3, 200))
    est.init_aci(alpha=0.1, gamma=0.05, buffer_n=300, warmup_residuals=warm)
    est.update_conformal(X, y)
    st = est.get_aci_state()
    assert 0.85 <= st["rolling_coverage"] <= 0.97, st["rolling_coverage"]


def test_biz_val_aci_recovers_coverage_under_drift_vs_frozen():
    """biz_value: residual SCALE drifts upward over the stream.

    A FROZEN split-conformal band calibrated on the early (low-scale) residuals
    under-covers badly on the late high-scale rows. ACI adapts its radius via the
    rolling buffer + alpha controller and keeps late-window coverage near 1-alpha.
    Measured: frozen late-coverage ~0.55, ACI late-coverage >0.85.
    """
    est = _make_fitted_estimator(seed=4)
    rng = np.random.default_rng(4)
    n = 3000
    base = rng.normal(0, 1, n)
    feat = rng.normal(0, 1, n)
    X = pd_or_np(base, feat)
    point = np.asarray(est.predict(X), dtype=np.float64)
    # Drift: noise std grows linearly from 0.3 to ~3.0 across the stream.
    scale = np.linspace(0.3, 3.0, n)
    noise = rng.normal(0, 1, n) * scale
    y = point + noise  # residual = y - point = noise (scale-drifting)

    # --- Frozen split-conformal: radius from first 400 (low-scale) residuals ---
    early_resid = np.abs(noise[:400])
    frozen_radius = _rolling_quantile_radius(early_resid, alpha=0.1)
    late = slice(n - 600, n)
    frozen_cov = float(np.mean(np.abs(noise[late]) <= frozen_radius))

    # --- ACI: warm start on the same early residuals, then stream the rest. ---
    est.init_aci(alpha=0.1, gamma=0.05, buffer_n=400, warmup_residuals=early_resid)
    # Stream rows one window at a time so the controller + buffer adapt.
    chunk = 100
    aci_hits = 0
    aci_total = 0
    for s in range(400, n, chunk):
        e = min(s + chunk, n)
        Xc = X.iloc[s:e]
        lo, hi = est.predict_interval_online(Xc, clip=False)
        yc = y[s:e]
        if s >= n - 600:
            aci_hits += int(np.sum((lo <= yc) & (yc <= hi)))
            aci_total += (e - s)
        est.update_conformal(Xc, yc)
    aci_cov = aci_hits / aci_total

    assert frozen_cov < 0.75, f"frozen band should under-cover under drift, got {frozen_cov}"
    assert aci_cov >= 0.85, f"ACI should recover coverage under drift, got {aci_cov}"
    assert aci_cov - frozen_cov >= 0.15, (aci_cov, frozen_cov)
