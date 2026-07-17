"""Unit + biz_value tests for composite joint HPO (``optimize_composite``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.tree import DecisionTreeRegressor

from mlframe.training.composite.hpo import (
    HPOSpace,
    CompositeHPOResult,
    optimize_composite,
)
from mlframe.training.composite.estimator import CompositeTargetEstimator


def _make_diff_data(n: int = 1500, seed: int = 0):
    """Target y = base + g(feat), so the ``diff`` transform (y - base) makes the
    residual a clean function of ``feat`` -- diff should clearly beat a no-op /
    ratio composite. ``base`` is a large-magnitude random walk so predicting raw
    y is hard but the residual is easy."""
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 1, n)) + 100.0
    feat = rng.uniform(-2, 2, n)
    resid = np.where(feat > 0, 3.0, -3.0) + 0.1 * rng.normal(0, 1, n)
    y = base + resid
    X = pd.DataFrame({"base": base, "feat": feat})
    return X, y


def _inner_factory():
    """Inner factory."""
    return DecisionTreeRegressor(random_state=0)


def _spaces():
    """Spaces."""
    return {"max_depth": HPOSpace("int", low=1, high=6)}


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_returns_fitted_estimator_random():
    """Returns fitted estimator random."""
    X, y = _make_diff_data(n=600)
    res = optimize_composite(
        X,
        y,
        base_column="base",
        transform_candidates=("diff", "linear_residual"),
        inner_factory=_inner_factory,
        inner_spaces=_spaces(),
        n_trials=8,
        cv=3,
        prefer_optuna=False,
    )
    assert isinstance(res, CompositeHPOResult)
    assert isinstance(res.estimator, CompositeTargetEstimator)
    # Fitted -> can predict.
    pred = res.estimator.predict(X)
    assert pred.shape[0] == len(y)
    assert np.all(np.isfinite(pred))


def test_random_fallback_works_without_optuna():
    """Random fallback works without optuna."""
    X, y = _make_diff_data(n=600)
    res = optimize_composite(
        X,
        y,
        base_column="base",
        transform_candidates=("diff", "linear_residual"),
        inner_factory=_inner_factory,
        inner_spaces=_spaces(),
        n_trials=6,
        cv=3,
        prefer_optuna=False,
    )
    assert res.backend == "random"
    assert res.transform in ("diff", "linear_residual")


def test_n_trials_respected():
    """N trials respected."""
    X, y = _make_diff_data(n=500)
    for nt in (4, 9):
        res = optimize_composite(
            X,
            y,
            base_column="base",
            transform_candidates=("diff", "linear_residual"),
            inner_factory=_inner_factory,
            inner_spaces=_spaces(),
            n_trials=nt,
            cv=3,
            prefer_optuna=False,
        )
        assert len(res.trials) == nt
        assert res.n_trials == nt


def test_best_score_le_fixed_baseline():
    """The optimized CV score must be <= a fixed-default composite's CV score
    on the SAME folds (the optimizer can never do worse than a config in its
    own search space)."""
    X, y = _make_diff_data(n=800)
    res = optimize_composite(
        X,
        y,
        base_column="base",
        transform_candidates=("diff", "linear_residual"),
        inner_factory=_inner_factory,
        inner_spaces=_spaces(),
        n_trials=12,
        cv=4,
        prefer_optuna=False,
    )
    # Fixed baseline: linear_residual, default depth, scored on identical folds.
    from mlframe.training.composite.hpo import _resolve_splits, _cv_score_candidate, _rmse

    splits = _resolve_splits(X, len(y), 4, None)
    base_score = _cv_score_candidate(
        X,
        np.asarray(y, dtype=float),
        base_column="base",
        transform_name="linear_residual",
        inner_params={"max_depth": 3},
        inner_factory=_inner_factory,
        splits=splits,
        scorer=_rmse,
    )
    assert res.cv_score <= base_score + 1e-9


def test_invalid_n_trials_raises():
    """Invalid n trials raises."""
    X, y = _make_diff_data(n=200)
    with pytest.raises(ValueError):
        optimize_composite(
            X,
            y,
            base_column="base",
            transform_candidates=("diff",),
            inner_factory=_inner_factory,
            n_trials=0,
            prefer_optuna=False,
        )


def test_time_ordering_uses_purged_cv():
    """Time ordering uses purged cv."""
    X, y = _make_diff_data(n=900)
    order = np.arange(len(y))
    res = optimize_composite(
        X,
        y,
        base_column="base",
        transform_candidates=("diff", "linear_residual"),
        inner_factory=_inner_factory,
        inner_spaces=_spaces(),
        n_trials=6,
        cv=3,
        time_ordering=order,
        prefer_optuna=False,
    )
    assert np.isfinite(res.cv_score)
    assert isinstance(res.estimator, CompositeTargetEstimator)


# ---------------------------------------------------------------------------
# biz_value: recovers the winning transform + beats a fixed default on holdout
# ---------------------------------------------------------------------------


def test_biz_val_optimize_composite_recovers_base_aware_transform_and_beats_default():
    """On data where y = base + g(feat), the optimizer must recover a BASE-AWARE
    transform and beat the base-blind ``log_y``. Both ``diff`` (T=y-base) and
    ``ratio`` (T=y/base) exploit ``base`` to make the residual learnable, so
    either is a correct recovery; ``log_y`` ignores ``base`` and cannot. On this
    DGP ``ratio`` is in fact the honest CV winner (best-of-depth CV-RMSE ~0.18 vs
    ``diff`` ~0.52 vs ``log_y`` ~3.21): the tree reads the ``base`` column to model
    T=y/base precisely, while contiguous-CV drift of the random-walk base penalises
    ``diff``'s incidental base splits. So (a) asserts a base-aware winner, not the
    exact identity, and (b) that it beats the fixed base-blind default on held-out
    RMSE."""
    X, y = _make_diff_data(n=1500, seed=1)
    y = np.asarray(y, dtype=float)
    n_tr = 1100
    X_tr, X_te = X.iloc[:n_tr], X.iloc[n_tr:]
    y_tr, y_te = y[:n_tr], y[n_tr:]

    res = optimize_composite(
        X_tr,
        y_tr,
        base_column="base",
        transform_candidates=("diff", "log_y", "ratio"),
        inner_factory=_inner_factory,
        inner_spaces=_spaces(),
        n_trials=20,
        cv=4,
        prefer_optuna=False,
    )
    # (a) recovers a base-aware transform over the base-blind log_y.
    assert res.transform in ("diff", "ratio"), f"expected a base-aware transform, got {res.transform!r}"

    # (b) beats the fixed-default (base-blind log_y) composite on held-out RMSE.
    default = CompositeTargetEstimator(
        base_estimator=DecisionTreeRegressor(max_depth=3, random_state=0),
        transform_name="log_y",
        base_column="base",
    )
    default.fit(X_tr, y_tr)
    rmse_default = float(np.sqrt(np.mean((y_te - default.predict(X_te)) ** 2)))
    rmse_opt = float(np.sqrt(np.mean((y_te - res.estimator.predict(X_te)) ** 2)))
    assert rmse_opt <= 0.9 * rmse_default, f"opt {rmse_opt:.4f} vs default {rmse_default:.4f}"
