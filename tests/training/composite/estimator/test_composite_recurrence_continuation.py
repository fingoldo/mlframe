"""T24: opt-in recurrence-continuation seeding for the left-recurrent transforms
(ewma_residual / frac_diff). Default OFF keeps predict stateless + bit-identical;
ON seeds the inverse from the train tail so a continuation batch is not biased on
its first ~k rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import CompositeTargetEstimator


def _trending(n=2000, seed=0):
    t = np.arange(n)
    base = 0.01 * t + np.sin(t / 20.0)
    y = base + 0.5 * np.random.default_rng(seed).normal(0, 0.2, n)
    return pd.DataFrame({"base": base}), y


@pytest.mark.parametrize("transform", ["ewma_residual", "frac_diff"])
def test_continuation_reduces_cold_start_bias(transform) -> None:
    X, y = _trending()
    Xtr, Xte, ytr, yte = X.iloc[:1500], X.iloc[1500:], y[:1500], y[1500:]
    k = 12

    def first_k_rmse(flag):
        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name=transform,
            base_column="base",
            recurrence_continuation=flag,
        ).fit(Xtr, ytr)
        p = est.predict(Xte)
        return float(np.sqrt(np.mean((p[:k] - yte[:k]) ** 2)))

    assert first_k_rmse(True) < first_k_rmse(False), "continuation seeding must lower the first-k cold-start error"


@pytest.mark.parametrize("transform", ["ewma_residual", "frac_diff"])
def test_default_off_is_bit_identical(transform) -> None:
    X, y = _trending(seed=1)
    Xtr, Xte = X.iloc[:1500], X.iloc[1500:]
    base = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name=transform,
        base_column="base",
    ).fit(Xtr, y[:1500])
    explicit_off = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name=transform,
        base_column="base",
        recurrence_continuation=False,
    ).fit(Xtr, y[:1500])
    np.testing.assert_array_equal(base.predict(Xte), explicit_off.predict(Xte))
    # The persisted flag is absent by default (stateless predict).
    assert "recurrence_continuation" not in base.fitted_params_


def test_clone_preserves_flag() -> None:
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="ewma_residual",
        base_column="base",
        recurrence_continuation=True,
    )
    assert clone(est).recurrence_continuation is True


def test_tail_anchor_stored_in_fit() -> None:
    X, y = _trending(seed=2)
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="ewma_residual",
        base_column="base",
    ).fit(X, y)
    assert "tail_anchor" in est.fitted_params_
    assert "anchor" in est.fitted_params_
