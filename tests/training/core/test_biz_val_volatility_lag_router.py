"""Unit + biz_value for the per-row VOLATILITY-lag router (MD-ordered local smoothness).

On a strong-AR target the groups where lag beats the trained model are IN-range but locally SMOOTH: within such a well
consecutive target values barely move, so lag_predict (the previous value) is near-perfect. The router routes rows whose
MD-local target volatility is low to lag. Ordering is EXPLICIT by an order column (MD) via lexsort -- it does NOT assume
the frame is row-sorted (prod frames are not). biz_value asserts the router beats BOTH all-raw and all-lag on a mix of
smooth (lag-friendly) and rough (model-friendly) wells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.core._volatility_lag_router import (
    VolatilityLagRouter, _group_local_volatility, build_volatility_lag_router,
)


class _Const:
    def __init__(self, vec):
        self._v = np.asarray(vec, dtype=np.float64)

    def predict(self, _frame):
        return self._v


class _LagCol:
    """Lag component: returns the ``lag`` column of the passed frame (mirrors _LagPredictDeployableModel reading X)."""

    def predict(self, frame):
        return np.asarray(frame["lag"].to_numpy(), dtype=np.float64)


class _Cfg:
    def __init__(self, on=True):
        self.volatility_lag_routing_enabled = on


class TestGroupLocalVolatility:
    def test_forward_diff_within_group_by_order_key(self):
        # group A rows shuffled in row order; MD gives the true order. lag = [10, 12, 15] at MD [1,2,3] -> steps |2|,|3|,nan.
        g = np.array(["A", "A", "A"])
        lag = np.array([15.0, 10.0, 12.0])   # rows in MD order 3,1,2
        md = np.array([3.0, 1.0, 2.0])
        vol = _group_local_volatility(g, lag, md)
        # MD-sorted: (MD1,lag10)->(MD2,lag12) step2; (MD2,lag12)->(MD3,lag15) step3; last (MD3) -> nan.
        # Map back to original row order [MD3, MD1, MD2] = [nan, 2, 3].
        assert np.isnan(vol[0])
        assert vol[1] == 2.0 and vol[2] == 3.0

    def test_groups_isolated(self):
        g = np.array(["A", "B", "A", "B"])
        lag = np.array([1.0, 100.0, 2.0, 130.0])
        md = np.array([1.0, 1.0, 2.0, 2.0])
        vol = _group_local_volatility(g, lag, md)
        # A: 1->2 step1 (row0), last A (row2) nan. B: 100->130 step30 (row1), last B (row3) nan.
        assert vol[0] == 1.0 and np.isnan(vol[2])
        assert vol[1] == 30.0 and np.isnan(vol[3])


def _frame(well, md, lag):
    return pd.DataFrame({"well": well, "MD": md, "lag": lag})


class TestRouterPredict:
    def test_routes_low_vol_rows_to_lag(self):
        # 2 rows smooth (vol ~0 -> route to lag), the third is last-in-group (nan vol -> keep raw).
        X = _frame(["A", "A", "A"], [1.0, 2.0, 3.0], [10.0, 10.1, 10.2])
        r = VolatilityLagRouter(_Const([1.0, 2.0, 3.0]), _LagCol(), "well", "MD", threshold=0.5)
        out = r.predict(X)
        assert out[0] == 10.0 and out[1] == 10.1   # low vol -> lag
        assert out[2] == 3.0                        # last-in-group (nan vol) -> raw

    def test_no_op_when_order_column_missing(self):
        X = pd.DataFrame({"well": ["A", "A"], "lag": [1.0, 1.0]})  # no MD
        r = VolatilityLagRouter(_Const([9.0, 9.0]), _LagCol(), "well", "MD", threshold=1.0)
        assert list(r.predict(X)) == [9.0, 9.0]  # cannot order -> plain trained


class TestBuilderValGate:
    def test_builder_no_op_without_order_column(self):
        X = pd.DataFrame({"well": ["A"] * 200, "lag": np.zeros(200)})  # no MD
        raw = _Const(np.zeros(200))
        assert build_volatility_lag_router(raw, _LagCol(), X, np.zeros(200), "well", "MD", _Cfg()) is raw

    def test_biz_val_router_beats_both_all_raw_and_all_lag(self):
        """Two well types on val: SMOOTH wells (target barely moves -> lag near-perfect, model noisy) and ROUGH wells
        (target jumps -> model good, lag lags badly). The volatility router (lag on smooth, model on rough) must beat
        BOTH all-raw and all-lag on pooled val RMSE."""
        rng = np.random.default_rng(0)
        rows = []
        raw_pred, lag_pred, y = [], [], []
        for w in range(40):
            n = 60
            md = np.arange(n, dtype=float)
            smooth = (w % 2 == 0)
            if smooth:
                truth = 100.0 + 0.02 * md + rng.normal(0, 0.05, n)      # barely moves
            else:
                truth = 100.0 + rng.normal(0, 8.0, n).cumsum() * 0.3    # jumps around
            lag = np.empty(n); lag[0] = truth[0]; lag[1:] = truth[:-1]  # exact previous value
            # model: good on rough wells (tracks jumps), noisy on smooth wells (adds error the flat series doesn't have)
            model = truth + (rng.normal(0, 3.0, n) if smooth else rng.normal(0, 1.0, n))
            for i in range(n):
                rows.append((w, md[i], lag[i]))
                raw_pred.append(model[i]); lag_pred.append(lag[i]); y.append(truth[i])
        df = pd.DataFrame(rows, columns=["well", "MD", "lag"])
        raw_pred = np.array(raw_pred); lag_pred = np.array(lag_pred); y = np.array(y)

        rmse = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))
        rmse_raw, rmse_lag = rmse(raw_pred), rmse(lag_pred)

        router = build_volatility_lag_router(_Const(raw_pred), _LagCol(), df, y, "well", "MD", _Cfg())
        assert isinstance(router, VolatilityLagRouter), "routing should deploy when smooth-well rows clearly help"
        routed = router.predict(df)
        rmse_router = rmse(routed)
        assert rmse_router < rmse_raw * 0.9, f"router must beat all-raw by >10%; router={rmse_router:.3f} raw={rmse_raw:.3f}"
        assert rmse_router < rmse_lag * 0.9, f"router must beat all-lag by >10%; router={rmse_router:.3f} lag={rmse_lag:.3f}"
