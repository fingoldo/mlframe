"""Tests for the Tweedie/signed-power unary transform and the all-NaN-base domain guard.

Two concerns live together here because both touch the transform registry's domain contract:

* ``signed_power_y`` -- ``T = sign(y) * |y|^p`` with ``p`` fitted at fit-time to minimise ``|skew(T)|``.  The biz_value test asserts a right-skewed target becomes markedly more symmetric (skew drops >=50%) AND a downstream linear model RMSE improves vs raw y.
* NaN-only base guard -- for every registry transform an all-non-finite base column must route to the fallback (all-False mask for base-reading transforms) or be genuinely ignored (base-free transforms), so no NaN ever escapes the predict path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.transforms import TRANSFORMS_REGISTRY
from mlframe.training.composite.transforms.unary import (
    signed_power_y_domain,
    signed_power_y_fit,
    signed_power_y_forward,
    signed_power_y_inverse,
)


# ----------------------------------------------------------------------
# signed_power_y -- unit + biz_value
# ----------------------------------------------------------------------


class TestSignedPowerYUnit:
    @pytest.mark.parametrize("y", [
        np.array([0.0, 1.0, -1.0, 8.0, -8.0, 1e6, -1e6]),
        np.linspace(-1000.0, 1000.0, 200),
        np.exp(np.random.default_rng(3).standard_normal(500)),
    ])
    def test_round_trip(self, y: np.ndarray) -> None:
        params = signed_power_y_fit(y)
        t = signed_power_y_forward(y, params)
        y_back = signed_power_y_inverse(t, params)
        np.testing.assert_allclose(y_back, y, rtol=1e-9, atol=1e-7)

    def test_fitted_p_in_grid_range(self) -> None:
        y = np.exp(np.random.default_rng(0).standard_normal(2000))
        p = float(signed_power_y_fit(y)["p"])
        assert 0.1 <= p <= 0.9

    def test_fewer_than_three_finite_rows_falls_back_to_identity(self) -> None:
        # All-NaN and too-short inputs must not crash; p=1 (identity) is the safe fallback.
        assert signed_power_y_fit(np.array([np.nan, np.nan]))["p"] == 1.0
        assert signed_power_y_fit(np.array([5.0]))["p"] == 1.0

    def test_preserves_sign(self) -> None:
        y = np.array([-9.0, -1.0, 0.0, 1.0, 9.0])
        t = signed_power_y_forward(y, {"p": 0.3})
        np.testing.assert_array_equal(np.sign(t), np.sign(y))

    def test_domain_rejects_non_finite(self) -> None:
        y = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
        mask = signed_power_y_domain(y)
        np.testing.assert_array_equal(mask, [True, False, False, False, True])


def _downstream_rmse(y_train, y_test, x_train, x_test, transform_fns):
    """Fit a linear model on the (transformed) target, predict, invert, return test RMSE."""
    fit_fn, fwd_fn, inv_fn = transform_fns
    if fit_fn is None:  # raw-y baseline
        m = LinearRegression().fit(x_train, y_train)
        return float(np.sqrt(np.mean((m.predict(x_test) - y_test) ** 2)))
    p = fit_fn(y_train)
    t_train = fwd_fn(y_train, p)
    m = LinearRegression().fit(x_train, t_train)
    y_hat = inv_fn(m.predict(x_test), p)
    return float(np.sqrt(np.mean((y_hat - y_test) ** 2)))


class TestSignedPowerYBizValue:
    def test_biz_val_signed_power_y_symmetrises_and_improves_rmse(self) -> None:
        """A right-skewed lognormal target becomes markedly more symmetric under signed_power_y (skew drops >=50%) AND a downstream linear model's RMSE improves vs raw y.

        Measured (seed=42, n=2000): raw skew ~6.0 -> T skew ~0.35 (94% drop); RMSE raw ~1.13 -> ~0.68 (~40% improvement).  Floors are set well below the measured win so seed noise does not trip them.
        """
        rng = np.random.default_rng(42)
        n = 2000
        z = rng.standard_normal(n)
        y = np.exp(z)  # right-skewed lognormal

        params = signed_power_y_fit(y)
        t = signed_power_y_forward(y, params)
        skew_raw = abs(float(stats.skew(y)))
        skew_t = abs(float(stats.skew(t)))
        assert skew_t <= 0.5 * skew_raw, (
            f"signed_power_y did not halve skew: raw={skew_raw:.3f} T={skew_t:.3f}"
        )

        x = (z + rng.standard_normal(n) * 0.3).reshape(-1, 1)
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.5, random_state=0)
        rmse_raw = _downstream_rmse(y_tr, y_te, x_tr, x_te, (None, None, None))
        rmse_t = _downstream_rmse(
            y_tr, y_te, x_tr, x_te,
            (signed_power_y_fit, signed_power_y_forward, signed_power_y_inverse),
        )
        assert rmse_t <= 0.85 * rmse_raw, (
            f"signed_power_y downstream RMSE did not improve: raw={rmse_raw:.3f} T={rmse_t:.3f}"
        )

    def test_biz_val_signed_power_y_beats_fixed_cbrt_on_strong_skew(self) -> None:
        """The fitted exponent adapts strength to the target: on a very heavy right skew it reaches a more symmetric T than the fixed-p=1/3 cbrt_y, so |skew| is no worse and typically better."""
        rng = np.random.default_rng(7)
        y = np.exp(2.0 * rng.standard_normal(3000))  # heavier skew than unit lognormal
        t_sp = signed_power_y_forward(y, signed_power_y_fit(y))
        t_cbrt = np.sign(y) * np.abs(y) ** (1.0 / 3.0)
        assert abs(float(stats.skew(t_sp))) <= abs(float(stats.skew(t_cbrt))) + 1e-6


# ----------------------------------------------------------------------
# All-NaN base guard: no NaN escapes predict for ANY registry transform.
# ----------------------------------------------------------------------

_RNG = np.random.default_rng(0)
_N = 300
_BASE = np.linspace(1.0, 10.0, _N)
_Y = 0.5 * _BASE + 1.0 + _RNG.standard_normal(_N) * 0.1
_GROUPS = (np.arange(_N) // 75).astype(np.int64)


def _call_fit(t, y, base):
    if not t.requires_base:
        return t.fit(y, None)
    if t.requires_groups:
        return t.fit(y, base, groups=_GROUPS[: len(y)])
    return t.fit(y, base)


def _call_forward(t, y, base, params):
    if not t.requires_base:
        return t.forward(y, None, params)
    if t.requires_groups:
        return t.forward(y, base, params, groups=_GROUPS[: len(y)])
    return t.forward(y, base, params)


def _call_inverse(t, t_hat, base, params):
    if not t.requires_base:
        return t.inverse(t_hat, None, params)
    if t.requires_groups:
        return t.inverse(t_hat, base, params, groups=_GROUPS[: len(t_hat)])
    return t.inverse(t_hat, base, params)


@pytest.mark.parametrize("name", sorted(TRANSFORMS_REGISTRY))
def test_all_nan_base_predict_domain_routes_to_fallback(name: str) -> None:
    """For every registry transform a base column that is entirely non-finite must NOT let NaN escape the inverse.

    Two valid outcomes:
    * base-reading transforms (``requires_base=True``): the predict-side ``domain_check(None, all_nan_base)`` returns an all-False mask, so the wrapper routes every row to the train-median fallback (the wrapper never calls ``inverse`` on a flagged row).
    * base-free transforms (``requires_base=False``): they genuinely ignore ``base``; the inverse runs on a placeholder base and must produce finite y.  Forcing an all-False mask there would wrongly discard valid rows.

    The contract this pins: ``domain_check`` must never crash on an all-NaN base, and any row the wrapper would actually invert must yield finite y.
    """
    t = TRANSFORMS_REGISTRY[name]
    nan_base = np.full(_N, np.nan)

    # Predict-side domain (y is None). Must not raise and must be a 1-D bool mask.
    if t.requires_base:
        pred_mask = np.asarray(t.domain_check(None, nan_base))
    else:
        # Base-free transforms: the wrapper passes ``base=None`` at predict.
        pred_mask = np.asarray(t.domain_check(None, None))
    assert pred_mask.dtype == bool
    assert pred_mask.ndim == 1

    if t.requires_base:
        # All-NaN base -> no row may be invertible (all flagged -> fallback).
        assert not pred_mask.any(), (
            f"transform={name!r} kept rows on an all-NaN base; NaN inverse would escape"
        )
        return

    # Base-free transform: fit on real y, run a forward then inverse with a
    # placeholder (zeros) base sized to t_hat, and assert finite output --
    # exactly mirroring the wrapper's ``base_arr = np.zeros_like(t_hat)`` path.
    params = _call_fit(t, _Y, None)
    t_train = _call_forward(t, _Y, None, params)
    assert np.all(np.isfinite(t_train)), f"{name}: forward produced non-finite T on finite y"
    y_back = _call_inverse(t, t_train, None, params)
    assert np.all(np.isfinite(y_back)), f"{name}: inverse produced non-finite y on a placeholder base"


@pytest.mark.parametrize("transform_name,base_column", [
    ("diff", "b"),
    ("linear_residual", "b"),
    ("ratio", "b"),
])
def test_estimator_predict_finite_when_base_all_nan(transform_name, base_column) -> None:
    """End-to-end: a base-reading composite estimator predicting on rows whose base column is entirely NaN must return finite y (the train-median fallback), never NaN.

    The inner is a NaN-tolerant ``HistGradientBoostingRegressor`` because the base column is also a feature seen by the inner; this isolates the test to the composite transform's inverse-side NaN-base guard (the subject under test) rather than the inner estimator's own NaN handling.
    """
    rng = np.random.default_rng(0)
    n = 600
    b = rng.normal(2.0, 1.0, n)
    f = rng.normal(0.0, 1.0, n)
    y = b + 0.5 * f + rng.normal(0.0, 0.5, n)
    X = pd.DataFrame({"b": b, "feat": f})
    est = CompositeTargetEstimator(
        base_estimator=HistGradientBoostingRegressor(max_iter=30, random_state=0),
        transform_name=transform_name, base_column=base_column,
    )
    est.fit(X.iloc[:400], y[:400])

    X_pred = X.iloc[400:].copy()
    X_pred["b"] = np.nan  # entire base column non-finite at predict
    y_hat = est.predict(X_pred)
    assert np.all(np.isfinite(y_hat)), (
        f"{transform_name}: NaN base at predict leaked non-finite y_hat"
    )
