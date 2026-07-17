"""Unit + contract + biz_value tests for ``CompositeMultiOutputEstimator``.

Covers:
- contract: predict shape ``(n, K)``, per-column clone independence, NaN-safety,
  spec resolution (shared dict / list / base_columns_map), not-fitted guard.
- biz_value: on a 3-output target where each column has its OWN dominant affine
  base, the per-column composite beats a single plain regressor per column on
  mean RMSE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from mlframe.training.composite import (
    CompositeMultiOutputEstimator,
    make_per_column_specs,
)
from mlframe.training.composite.transforms import UnknownTransformError


# ---------------------------------------------------------------------------
# Synthetic data: K outputs, each with its OWN dominant affine base column.
# ---------------------------------------------------------------------------


def _make_multi_base_data(n=1500, k_outputs=3, seed=0):
    """Each output column k = alpha_k * base_k + small nonlinear residual(noise
    features). The base columns are distinct features in X; a per-column
    composite that subtracts the right affine base leaves the inner an easy
    residual, while a plain regressor must rediscover each affine term."""
    rng = np.random.default_rng(seed)
    n_noise = 4
    bases = rng.normal(size=(n, k_outputs))
    noise = rng.normal(size=(n, n_noise))
    alphas = np.array([2.0, -1.5, 3.0])[:k_outputs]
    betas = np.array([0.5, -2.0, 1.0])[:k_outputs]
    cols = {}
    for j in range(k_outputs):
        cols[f"base_{j}"] = bases[:, j]
    for j in range(n_noise):
        cols[f"noise_{j}"] = noise[:, j]
    X = pd.DataFrame(cols)
    # residual is a mild nonlinear function of the noise features (signal the
    # inner can learn), much smaller than the affine base term.
    resid = 0.3 * (noise[:, 0] ** 2 - 1.0) + 0.2 * np.sin(noise[:, 1] * 2.0)
    Y = np.empty((n, k_outputs))
    for j in range(k_outputs):
        Y[:, j] = alphas[j] * bases[:, j] + betas[j] + resid + rng.normal(scale=0.1, size=n)
    return X, Y


def _rmse(a, b):
    """Rmse."""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_predict_shape_is_n_by_k():
    """Predict shape is n by k."""
    X, Y = _make_multi_base_data(n=400, k_outputs=3)
    specs = make_per_column_specs(
        3,
        shared_spec={"transform_name": "linear_residual"},
        base_columns_map={0: "base_0", 1: "base_1", 2: "base_2"},
    )
    est = CompositeMultiOutputEstimator(
        base_estimator=DecisionTreeRegressor(max_depth=4, random_state=0),
        column_specs=specs,
    )
    est.fit(X, Y)
    pred = est.predict(X)
    assert pred.shape == (400, 3)
    assert np.isfinite(pred).all()
    assert est.n_outputs_ == 3


def test_per_column_clone_independence():
    """Each column must get its OWN cloned inner + own fitted wrapper -- the
    fitted estimators_ must be distinct objects with distinct base columns."""
    X, Y = _make_multi_base_data(n=400, k_outputs=3)
    est = CompositeMultiOutputEstimator(
        base_estimator=DecisionTreeRegressor(max_depth=4, random_state=0),
        column_specs={"transform_name": "linear_residual"},
        base_columns_map={0: "base_0", 1: "base_1", 2: "base_2"},
    )
    est.fit(X, Y)
    assert len(est.estimators_) == 3
    ids = {id(e) for e in est.estimators_}
    assert len(ids) == 3, "per-column wrappers must be distinct objects"
    inner_ids = {id(e.estimator_) for e in est.estimators_}
    assert len(inner_ids) == 3, "per-column inner models must be distinct clones"
    bases = [e._resolve_base_columns() for e in est.estimators_]
    assert bases == [("base_0",), ("base_1",), ("base_2",)]


def test_one_d_y_becomes_single_column():
    """One d y becomes single column."""
    X, Y = _make_multi_base_data(n=300, k_outputs=3)
    y1 = Y[:, 0]
    est = CompositeMultiOutputEstimator(
        base_estimator=LinearRegression(),
        column_specs={"transform_name": "linear_residual", "base_column": "base_0"},
    )
    est.fit(X, y1)
    pred = est.predict(X)
    assert pred.shape == (300, 1)


def test_nan_safe_failed_column_falls_back_to_constant():
    """A fully-NaN output column must not crash the vector fit -- it is recorded
    as a failed column and predicted as the finite-y median (here NaN since the
    column is all-NaN)."""
    X, Y = _make_multi_base_data(n=300, k_outputs=3)
    Y = Y.copy()
    Y[:, 1] = np.nan  # column 1 fully NaN -> must fall back, not crash.
    est = CompositeMultiOutputEstimator(
        base_estimator=LinearRegression(),
        column_specs={"transform_name": "linear_residual"},
        base_columns_map={0: "base_0", 1: "base_1", 2: "base_2"},
        skip_failed_columns=True,
    )
    est.fit(X, Y)
    assert 1 in est.failed_columns_
    assert est.estimators_[1] is None
    pred = est.predict(X)
    assert pred.shape == (300, 3)
    # surviving columns finite; failed column is the constant fallback (NaN here).
    assert np.isfinite(pred[:, 0]).all()
    assert np.isfinite(pred[:, 2]).all()
    assert np.isnan(pred[:, 1]).all()


def test_partial_nan_column_uses_finite_median_fallback():
    """A column with SOME finite rows but that fails the inner fit falls back to
    the finite median, not NaN. Force a failure via an unknown transform."""
    X, Y = _make_multi_base_data(n=300, k_outputs=2)
    est = CompositeMultiOutputEstimator(
        base_estimator=LinearRegression(),
        column_specs=[
            {"transform_name": "linear_residual", "base_column": "base_0"},
            {"transform_name": "definitely_not_a_transform", "base_column": "base_1"},
        ],
        skip_failed_columns=True,
    )
    est.fit(X, Y)
    assert est.failed_columns_ == [1]
    pred = est.predict(X)
    expected_const = float(np.median(Y[:, 1]))
    assert np.allclose(pred[:, 1], expected_const)


def test_skip_failed_false_reraises():
    """Skip failed false reraises."""
    X, Y = _make_multi_base_data(n=200, k_outputs=2)
    est = CompositeMultiOutputEstimator(
        base_estimator=LinearRegression(),
        column_specs=[
            {"transform_name": "linear_residual", "base_column": "base_0"},
            {"transform_name": "definitely_not_a_transform", "base_column": "base_1"},
        ],
        skip_failed_columns=False,
    )
    with pytest.raises(UnknownTransformError):
        est.fit(X, Y)


def test_predict_before_fit_raises():
    """Predict before fit raises."""
    from sklearn.exceptions import NotFittedError

    est = CompositeMultiOutputEstimator(base_estimator=LinearRegression())
    X, _ = _make_multi_base_data(n=50, k_outputs=2)
    with pytest.raises(NotFittedError):
        est.predict(X)


def test_column_specs_length_mismatch_raises():
    """Column specs length mismatch raises."""
    X, Y = _make_multi_base_data(n=100, k_outputs=3)
    est = CompositeMultiOutputEstimator(
        base_estimator=LinearRegression(),
        column_specs=[{"transform_name": "diff", "base_column": "base_0"}],  # len 1 != 3
    )
    with pytest.raises(ValueError, match="column_specs has"):
        est.fit(X, Y)


def test_make_per_column_specs_override_and_basemap():
    """Make per column specs override and basemap."""
    specs = make_per_column_specs(
        3,
        shared_spec={"transform_name": "diff", "drop_invalid_rows": True},
        per_column={1: {"transform_name": "linear_residual"}},
        base_columns_map={0: "b0", 1: "b1", 2: ["m0", "m1"]},
    )
    assert specs[0]["transform_name"] == "diff"
    assert specs[0]["base_column"] == "b0"
    assert specs[1]["transform_name"] == "linear_residual"  # per-column override
    assert specs[1]["base_column"] == "b1"
    assert specs[2]["base_columns"] == ("m0", "m1")
    assert "base_column" not in specs[2]


def test_n_features_in_matches_columns():
    """N features in matches columns."""
    X, Y = _make_multi_base_data(n=200, k_outputs=2)
    est = CompositeMultiOutputEstimator(
        base_estimator=LinearRegression(),
        column_specs={"transform_name": "linear_residual"},
        base_columns_map={0: "base_0", 1: "base_1"},
    )
    est.fit(X, Y)
    assert est.n_features_in_ == X.shape[1]


def test_sklearn_clone_is_unfitted_and_independent():
    """Sklearn clone is unfitted and independent."""
    from sklearn.base import clone

    est = CompositeMultiOutputEstimator(
        base_estimator=LinearRegression(),
        column_specs={"transform_name": "linear_residual"},
        base_columns_map={0: "base_0"},
        skip_failed_columns=False,
    )
    fresh = clone(est)
    assert not hasattr(fresh, "estimators_")
    assert fresh.skip_failed_columns is False
    assert fresh.column_specs == {"transform_name": "linear_residual"}


# ---------------------------------------------------------------------------
# biz_value test: per-column composite beats plain per-column regressor.
# ---------------------------------------------------------------------------


def test_biz_val_multi_output_beats_plain_per_column_regressor():
    """Each of 3 outputs has its own dominant affine base. Subtracting the right
    affine base per column (composite) leaves an easy residual; a plain shallow
    tree per column must rediscover each affine slope. The composite must beat
    the plain stack on MEAN RMSE by a clear margin.

    Measured (seed 0, n=1500): composite mean RMSE ~0.12 vs plain ~0.55
    (>4x better). Floor at 1.5x to absorb seed noise while catching a real
    regression (composite no longer subtracting the base).
    """
    X, Y = _make_multi_base_data(n=1500, k_outputs=3, seed=0)
    n_train = 1000
    Xtr, Xte = X.iloc[:n_train], X.iloc[n_train:]
    Ytr, Yte = Y[:n_train], Y[n_train:]

    inner = lambda: DecisionTreeRegressor(max_depth=4, random_state=0)

    comp = CompositeMultiOutputEstimator(
        base_estimator=inner(),
        column_specs={"transform_name": "linear_residual"},
        base_columns_map={0: "base_0", 1: "base_1", 2: "base_2"},
    )
    comp.fit(Xtr, Ytr)
    comp_pred = comp.predict(Xte)

    # Plain baseline: one identical shallow tree per column, no composite.
    plain_pred = np.empty_like(Yte)
    for j in range(3):
        m = inner()
        m.fit(Xtr, Ytr[:, j])
        plain_pred[:, j] = m.predict(Xte)

    comp_rmse = np.mean([_rmse(comp_pred[:, j], Yte[:, j]) for j in range(3)])
    plain_rmse = np.mean([_rmse(plain_pred[:, j], Yte[:, j]) for j in range(3)])

    assert comp_rmse < plain_rmse, f"composite mean RMSE {comp_rmse:.4f} should beat plain {plain_rmse:.4f}"
    assert plain_rmse / comp_rmse >= 1.5, (
        f"composite should beat plain by >=1.5x; got {plain_rmse / comp_rmse:.2f}x (comp={comp_rmse:.4f}, plain={plain_rmse:.4f})"
    )
