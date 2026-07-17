"""Unit + biz_value for the per-row OOD-lag router.

After the val cross-check deploys the trained model (it wins overall), unseen groups whose target level is OUTSIDE the
train range still make the model extrapolate/clamp catastrophically while lag_predict (the row's own previous value) is
exact there (prod TVT: 21/71 groups, group 17 raw RMSE 22.4 vs lag 2.58). The router routes such rows -- lag outside the
train target range -- to lag, transferably (a fixed train-range rule, not per-group-id), and only when it improves the
honest val RMSE. biz_value asserts the router beats BOTH all-raw and all-lag on a held-out mix of in-range and
out-of-range groups.
"""

from __future__ import annotations

import math

import numpy as np

from mlframe.training.core._ood_lag_router import OODLagRouter, build_ood_lag_router


class _Const:
    """Component whose predict returns a fixed vector (positionally aligned to the passed frame length)."""

    def __init__(self, vec):
        self._v = np.asarray(vec, dtype=np.float64)

    def predict(self, _frame):
        """Predict."""
        return self._v


class _Lag:
    """Lag component: predict returns the lag column of the passed frame (a plain ndarray here)."""

    def predict(self, frame):
        """Predict."""
        return np.asarray(frame, dtype=np.float64)


class _Cfg:
    """Groups tests covering cfg."""
    def __init__(self, on=True, margin=0.0):
        self.ood_lag_routing_enabled = on
        self.ood_lag_router_margin_frac = margin


class TestOODLagRouterUnit:
    """Groups tests covering o o d lag router unit."""
    def test_routes_ood_rows_to_lag_keeps_raw_in_range(self):
        # lag values: [5, 50, 200]; train range [0,100] -> row 2 (lag=200) is OOD -> gets lag; others keep raw.
        """Routes ood rows to lag keeps raw in range."""
        raw = _Const([1.0, 2.0, 3.0])
        lag = _Lag()
        r = OODLagRouter(raw, lag, lo=0.0, hi=100.0)
        out = r.predict(np.array([5.0, 50.0, 200.0]))
        assert list(out) == [1.0, 2.0, 200.0]

    def test_nonfinite_lag_never_overwrites_raw(self):
        """Nonfinite lag never overwrites raw."""
        raw = _Const([1.0, 2.0])
        r = OODLagRouter(raw, _Lag(), lo=0.0, hi=100.0)
        out = r.predict(np.array([np.nan, 500.0]))
        assert out[0] == 1.0 and out[1] == 500.0

    def test_builder_returns_trained_when_disabled(self):
        """Builder returns trained when disabled."""
        raw = _Const([1.0] * 100)
        assert build_ood_lag_router(raw, _Lag(), np.zeros(100), np.zeros(100), np.zeros(100), _Cfg(on=False)) is raw

    def test_builder_returns_trained_when_no_ood_rows(self):
        # All val lag values inside train range -> no routing.
        """Builder returns trained when no ood rows."""
        raw = _Const(np.full(200, 5.0))
        y_train = np.linspace(0.0, 100.0, 500)
        val_lag = np.full(200, 50.0)  # all in range
        y_val = np.full(200, 50.0)
        got = build_ood_lag_router(raw, _Lag(), y_train, val_lag, y_val, _Cfg())
        assert got is raw

    def test_builder_returns_trained_when_routing_does_not_help(self):
        # OOD rows exist but the trained model is ALREADY good there -> routing to lag would not improve val -> keep raw.
        """Builder returns trained when routing does not help."""
        n = 400
        y_val = np.concatenate([np.full(n // 2, 50.0), np.full(n // 2, 500.0)])  # half in-range, half OOD-level
        val_lag = y_val.copy()  # lag == truth everywhere (so lag is perfect, but...)
        raw = _Const(y_val.copy())  # ...the trained model is ALSO perfect -> routing cannot improve
        y_train = np.linspace(0.0, 100.0, 800)
        got = build_ood_lag_router(raw, _Lag(), y_train, val_lag, y_val, _Cfg())
        assert isinstance(got, _Const)  # unchanged (a plain trained component, not a router)


class TestOODLagRouterBizValue:
    """Groups tests covering o o d lag router biz value."""
    def test_biz_val_router_beats_both_all_raw_and_all_lag(self):
        """Held-out val: in-range groups where the trained model is accurate and lag is noisy, PLUS out-of-range groups
        where the trained model extrapolates badly and lag is exact. The router (raw in-range, lag out-of-range) must
        beat BOTH all-raw and all-lag on pooled RMSE."""
        rng = np.random.default_rng(0)
        lo, hi = 0.0, 100.0
        y_train = rng.uniform(lo, hi, 2000)

        n = 4000
        in_range = rng.uniform(lo, hi, n // 2)
        out_range = rng.uniform(hi + 20.0, hi + 120.0, n // 2)  # levels the model never saw
        y_val = np.concatenate([in_range, out_range])
        # lag is a WEAK-ish baseline everywhere (noisier than the trained model in range) but is the ONLY thing that
        # tracks the out-of-range level at all -- so all-lag alone is mediocre, yet lag is what saves the OOD rows.
        lag_val = y_val + rng.normal(0.0, 2.5, n)

        # Trained model: accurate on in-range rows, but CLAMPS to hi on out-of-range rows (classic tree extrapolation).
        raw_val = np.concatenate(
            [
                in_range + rng.normal(0.0, 1.0, n // 2),  # good in range
                np.full(n // 2, hi) + rng.normal(0.0, 1.0, n // 2),  # clamped -> far from the true out-range level
            ]
        )

        rmse = lambda p: float(np.sqrt(np.mean((p - y_val) ** 2)))
        rmse_raw = rmse(raw_val)
        rmse_lag = rmse(lag_val)

        router = build_ood_lag_router(_Const(raw_val), _Lag(), y_train, lag_val, y_val, _Cfg())
        assert isinstance(router, OODLagRouter), "routing should deploy on this clearly-OOD-helped val"
        routed = router.predict(lag_val)  # _Lag.predict returns lag_val; raw is the fixed vector
        rmse_router = rmse(routed)

        assert rmse_router < rmse_raw * 0.9, f"router must beat all-raw by >10%; router={rmse_router:.3f} raw={rmse_raw:.3f}"
        assert rmse_router < rmse_lag * 0.9, f"router must beat all-lag by >10%; router={rmse_router:.3f} lag={rmse_lag:.3f}"
        assert math.isfinite(rmse_router)
