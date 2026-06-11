"""Unit + biz_value tests for ``CompositeDriftMonitor`` (composite/monitoring.py).

Unit: no-drift quiet, base-shift alert, residual-shift alert, missing-y path,
sketch reuse + missing-sketch guard, prediction PSI.

biz_value: on a deliberately drifted base / residual stream the monitor raises
``alert`` while a stationary continuation of the train distribution does not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlframe.training.composite.estimator import CompositeTargetEstimator
from mlframe.training.composite.monitoring import (
    CompositeDriftMonitor,
    _bin_fractions,
    _psi,
    _quantile_knots,
    _ks_statistic,
)


def _make_fitted_estimator(seed: int = 0, n: int = 2000):
    """Fit a diff-transform composite estimator; return (est, X_train, y_train)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, n)
    feat = rng.normal(0.0, 1.0, n)
    # y = base (carried by diff transform) + residual driven by feat.
    y = base + 0.5 * feat + rng.normal(0.0, 0.3, n)
    X = pd.DataFrame({"lag": base, "feat": feat})
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(), transform_name="diff", base_column="lag",
    )
    est.fit(X, y)
    return est, X, y


def _stationary_batch(seed: int, n: int = 1000):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, n)
    feat = rng.normal(0.0, 1.0, n)
    y = base + 0.5 * feat + rng.normal(0.0, 0.3, n)
    return pd.DataFrame({"lag": base, "feat": feat}), y


# ---------------------------------------------------------------------------
# Pure helper unit tests
# ---------------------------------------------------------------------------


def test_psi_zero_on_identical_distribution() -> None:
    rng = np.random.default_rng(1)
    v = rng.normal(size=5000)
    knots = _quantile_knots(v, 10)
    frac = _bin_fractions(v, knots)
    assert _psi(frac, frac) == pytest.approx(0.0, abs=1e-9)


def test_psi_large_on_shifted_distribution() -> None:
    rng = np.random.default_rng(2)
    ref = rng.normal(0.0, 1.0, 5000)
    new = rng.normal(4.0, 1.0, 5000)
    knots = _quantile_knots(ref, 10)
    psi = _psi(_bin_fractions(ref, knots), _bin_fractions(new, knots))
    assert psi > 0.25, f"shifted PSI should be large, got {psi}"


def test_ks_zero_on_same_sample_large() -> None:
    rng = np.random.default_rng(3)
    ref = rng.normal(size=20000)
    knots = _quantile_knots(ref, 10)
    # A fresh draw from the same law: KS vs the decile sketch stays below the
    # default base_ks_threshold (the sketch CDF resolves only to ~1/(K+1) steps,
    # so an in-distribution KS floors near 0.1, well under the 0.2 alert gate).
    new = rng.normal(size=20000)
    assert _ks_statistic(knots, new) < 0.2


# ---------------------------------------------------------------------------
# Monitor unit tests
# ---------------------------------------------------------------------------


def test_no_drift_quiet() -> None:
    est, X_tr, y_tr = _make_fitted_estimator(seed=0)
    mon = CompositeDriftMonitor(est)
    mon.ensure_sketch(reference=X_tr, y_reference=y_tr)
    X_new, y_new = _stationary_batch(seed=99)
    rep = mon.monitor(X_new, y_new)
    assert rep["has_y"] is True
    assert rep["alert"] is False, f"stationary stream must not alert: {rep['signals']}"
    assert rep["recommend_update"] is False


def test_base_shift_alert() -> None:
    est, X_tr, y_tr = _make_fitted_estimator(seed=0)
    mon = CompositeDriftMonitor(est)
    mon.ensure_sketch(reference=X_tr, y_reference=y_tr)
    rng = np.random.default_rng(7)
    n = 1000
    # Base mean shifted by +5 sigma: PSI + KS on the base column must alert.
    base = rng.normal(5.0, 1.0, n)
    feat = rng.normal(0.0, 1.0, n)
    X_new = pd.DataFrame({"lag": base, "feat": feat})
    rep = mon.monitor(X_new)  # missing-y path
    assert rep["has_y"] is False
    assert rep["signals"]["base_psi[lag]"]["alert"] is True
    assert rep["alert"] is True
    assert rep["recommend_update"] is True  # base family in recommend_update_on


def test_residual_shift_alert() -> None:
    est, X_tr, y_tr = _make_fitted_estimator(seed=0)
    mon = CompositeDriftMonitor(est)
    mon.ensure_sketch(reference=X_tr, y_reference=y_tr)
    X_new, y_new = _stationary_batch(seed=5)
    # Inject a large additive bias into y -> residual mean shift.
    y_drift = np.asarray(y_new) + 3.0
    rep = mon.monitor(X_new, y_drift)
    assert rep["signals"]["residual_mean_shift"]["alert"] is True
    assert rep["alert"] is True
    assert rep["recommend_update"] is True


def test_residual_scale_shift_alert() -> None:
    est, X_tr, y_tr = _make_fitted_estimator(seed=0)
    mon = CompositeDriftMonitor(est)
    mon.ensure_sketch(reference=X_tr, y_reference=y_tr)
    X_new, y_new = _stationary_batch(seed=8)
    rng = np.random.default_rng(11)
    # Triple the residual scale: log-ratio ~ ln(3) ~ 1.1 > 0.405 threshold.
    y_drift = np.asarray(y_new) + rng.normal(0.0, 1.0, len(y_new))
    rep = mon.monitor(X_new, y_drift)
    assert rep["signals"]["residual_scale_shift"]["alert"] is True


def test_missing_sketch_requires_reference() -> None:
    est, X_tr, y_tr = _make_fitted_estimator(seed=0)
    mon = CompositeDriftMonitor(est)
    X_new, _ = _stationary_batch(seed=1)
    with pytest.raises(ValueError, match="no stored drift"):
        mon.monitor(X_new)


def test_sketch_memoised_on_estimator() -> None:
    est, X_tr, y_tr = _make_fitted_estimator(seed=0)
    mon = CompositeDriftMonitor(est)
    mon.ensure_sketch(reference=X_tr, y_reference=y_tr)
    assert "_drift_sketch" in est.fitted_params_
    # A second monitor reuses the stored sketch without a reference batch.
    mon2 = CompositeDriftMonitor(est)
    X_new, y_new = _stationary_batch(seed=2)
    rep = mon2.monitor(X_new, y_new)  # no reference needed now
    assert rep["alert"] is False


def test_rolling_rmse_tracks_history() -> None:
    est, X_tr, y_tr = _make_fitted_estimator(seed=0)
    mon = CompositeDriftMonitor(est, rolling_rmse_window=3)
    mon.ensure_sketch(reference=X_tr, y_reference=y_tr)
    for s in range(5):
        X_new, y_new = _stationary_batch(seed=20 + s)
        mon.monitor(X_new, y_new)
    assert len(mon.rolling_rmse) == 3  # FIFO capped at window


def test_unfitted_estimator_rejected() -> None:
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(), transform_name="diff", base_column="lag",
    )
    with pytest.raises(ValueError, match="not fitted"):
        CompositeDriftMonitor(est)


# ---------------------------------------------------------------------------
# biz_value: drifted stream alerts, stationary does not (the monitor's job)
# ---------------------------------------------------------------------------


def test_biz_val_monitor_drift_alerts_stationary_quiet() -> None:
    """A deliberately base+residual-drifted stream MUST alert while a stationary
    continuation of the train law MUST stay quiet -- the monitor's core value.

    Pins both sides: a monitor that silently never alerts (broken PSI/KS) fails
    the drifted assertion; one that alerts on everything (bad thresholds) fails
    the stationary assertion.
    """
    est, X_tr, y_tr = _make_fitted_estimator(seed=0)
    mon = CompositeDriftMonitor(est)
    mon.ensure_sketch(reference=X_tr, y_reference=y_tr)

    # Stationary stream: quiet.
    X_ok, y_ok = _stationary_batch(seed=123)
    rep_ok = mon.monitor(X_ok, y_ok)
    assert rep_ok["alert"] is False, f"stationary stream alerted: {rep_ok['signals']}"
    assert rep_ok["recommend_update"] is False

    # Drifted stream: base mean + scale shift AND residual bias -> alert + recommend.
    rng = np.random.default_rng(321)
    n = 1000
    base = rng.normal(4.0, 2.0, n)
    feat = rng.normal(0.0, 1.0, n)
    y_drift = base + 0.5 * feat + rng.normal(0.0, 0.3, n) + 2.5
    X_bad = pd.DataFrame({"lag": base, "feat": feat})
    rep_bad = mon.monitor(X_bad, y_drift)
    assert rep_bad["alert"] is True, f"drifted stream stayed quiet: {rep_bad['signals']}"
    assert rep_bad["recommend_update"] is True
