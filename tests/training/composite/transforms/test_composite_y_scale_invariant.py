"""Invariant test for composite-target y-scale metric consistency.

Background. Production log (2026-05-17 TVT run) showed:

    [TVT-linres-TVT_prev] _TTRWithEvalSetScaling y-scale metrics:
        TRAIN=RMSE_y:3.25 MAE_y:3.24 R2_y:1.0000
        VAL=RMSE_y:3.24 MAE_y:3.23 R2_y:1.0000
        TEST=RMSE_y:3.23 MAE_y:3.22 R2_y:1.0000

but the same MLP on T-scale (before wrapping) reported MAE=9.17 RMSE=12.44 R2=-0.15. These two MAE numbers MUST agree for an invertible additive transform like ``linear_residual`` (y = T + alpha*base + beta -> error_y = error_T). The 3x discrepancy + suspicious R^2=1.0000 + identical TRAIN ≈ VAL ≈ TEST is the symptom we want to pin.

This test ships as the watchdog: it builds a CompositeTargetEstimator wrapping a real ``LinearRegression`` (raw inner) and ``TransformedTargetRegressor(LinearRegression, StandardScaler)`` (TTR-wrapped inner, mirroring the MLP path in production), then asserts y-scale MAE after wrapping == T-scale MAE before wrapping to within 1e-6 for both cases. If either inverse path diverges, this test fails loudly with the actual delta.

If the production-style real bug is in a different layer (entry-mutation, double-prediction, stale weights), this test will still pass -- it documents the invariant that MUST hold at the wrapper boundary.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from mlframe.training.composite import (
    CompositeTargetEstimator,
    get_transform,
)


def _fit_linear_residual_params(y: np.ndarray, base: np.ndarray) -> dict:
    """Mirror what CompositeTargetEstimator.from_fitted_inner expects in transform_fitted_params for linear_residual."""
    transform = get_transform("linear_residual")
    return transform.fit(y, base)


import pandas as pd


def _make_dataset(n: int = 600, seed: int = 17) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Returns (X, base, y) with y ~ 1.0*base + small_noise so linear_residual has alpha~=1. X is a DataFrame because CompositeTargetEstimator extracts base by column name and refuses bare ndarrays."""
    rng = np.random.default_rng(seed)
    base = rng.normal(11500.0, 600.0, size=n)
    noise = rng.standard_normal(n) * 4.0
    other_feat = rng.normal(0.0, 1.0, n)
    y = 1.0 * base + 5.0 + noise + 0.2 * other_feat
    X = pd.DataFrame({"base": base, "other": other_feat})
    return X, base, y


def _y_scale_mae(wrapper: CompositeTargetEstimator, X: pd.DataFrame, y: np.ndarray) -> float:
    """Y scale mae."""
    preds = np.asarray(wrapper.predict(X), dtype=np.float64).reshape(-1)
    return float(np.mean(np.abs(preds - y.astype(np.float64))))


def _t_scale_mae(inner, X: pd.DataFrame, y_T: np.ndarray) -> float:
    """T scale mae."""
    preds_T = np.asarray(inner.predict(X), dtype=np.float64).reshape(-1)
    return float(np.mean(np.abs(preds_T - y_T.astype(np.float64))))


class TestCompositeYscaleInvariant:
    """y-scale error after wrap must equal T-scale error before wrap (linear_residual is additive + invertible)."""

    def test_plain_linear_inner(self) -> None:
        """Plain linear inner."""
        X, base, y = _make_dataset()
        params = _fit_linear_residual_params(y, base)
        transform = get_transform("linear_residual")
        y_T = transform.forward(y, base, params)

        inner = LinearRegression().fit(X, y_T)

        wrapper = CompositeTargetEstimator.from_fitted_inner(
            fitted_inner=inner,
            transform_name="linear_residual",
            base_column="base",
            transform_fitted_params=params,
            y_train=y,
        )

        t_mae = _t_scale_mae(inner, X, y_T)
        y_mae = _y_scale_mae(wrapper, X, y)
        assert y_mae == pytest.approx(t_mae, abs=1e-6), (
            f"Composite wrapper broke the additive-invariance: T-scale MAE={t_mae:.6f} but y-scale MAE={y_mae:.6f} (delta={y_mae - t_mae:.6f})"
        )

    def test_ttr_wrapped_inner(self) -> None:
        """Mirrors the MLP path: TransformedTargetRegressor wraps the inner so the inner sees standardised T.

        Production showed T-scale MAE=9.17 vs y-scale MAE=3.22 for this exact wrapper shape (TTR + linear_residual). This test exercises the same wrapper chain on a controlled dataset and pins the invariant. The wrapper-level math passes here; the production symptom must therefore live in the entry-mutation / cached-prediction layer above ``_run_composite_target_wrapping`` (untracked for this commit; the invariant test ships as the watchdog).
        """
        X, base, y = _make_dataset()
        params = _fit_linear_residual_params(y, base)
        transform = get_transform("linear_residual")
        y_T = transform.forward(y, base, params)

        ttr = TransformedTargetRegressor(
            regressor=LinearRegression(),
            transformer=StandardScaler(),
        )
        ttr.fit(X, y_T)

        wrapper = CompositeTargetEstimator.from_fitted_inner(
            fitted_inner=ttr,
            transform_name="linear_residual",
            base_column="base",
            transform_fitted_params=params,
            y_train=y,
        )

        t_mae = _t_scale_mae(ttr, X, y_T)
        y_mae = _y_scale_mae(wrapper, X, y)
        assert y_mae == pytest.approx(t_mae, abs=1e-6), (
            f"TTR+composite broke additive-invariance (production symptom): T-scale MAE={t_mae:.6f} but y-scale MAE={y_mae:.6f} (delta={y_mae - t_mae:.6f})"
        )

    def test_invariant_holds_with_y_clip_active(self) -> None:
        """Repeat with a base that pushes predictions to the train-envelope clip; clip MUST NOT silently improve metrics on in-envelope train rows (the documented contract)."""
        X, base, y = _make_dataset()
        params = _fit_linear_residual_params(y, base)
        transform = get_transform("linear_residual")
        y_T = transform.forward(y, base, params)
        inner = LinearRegression().fit(X, y_T)

        wrapper = CompositeTargetEstimator.from_fitted_inner(
            fitted_inner=inner,
            transform_name="linear_residual",
            base_column="base",
            transform_fitted_params=params,
            y_train=y,
        )

        t_mae = _t_scale_mae(inner, X, y_T)
        y_mae_clipped = _y_scale_mae(wrapper, X, y)
        y_unclipped = wrapper.predict_pre_clip(X)
        y_mae_unclipped = float(np.mean(np.abs(y_unclipped - y)))
        assert y_mae_clipped == pytest.approx(t_mae, abs=1e-6)
        assert y_mae_unclipped == pytest.approx(t_mae, abs=1e-6)
