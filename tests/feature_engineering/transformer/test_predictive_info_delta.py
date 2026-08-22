"""Unit coverage for ``compute_predictive_info_delta_features`` (H(y) - H(y|baseline_pred_bin) per
row). No dedicated unit test existed -- only the (skipped-by-default, ``--run-biz-transformer``-gated)
real-datasets biz_val suite touched it.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytest.importorskip("lightgbm")

from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer.predictive_info_delta import compute_predictive_info_delta_features

_EXPECTED_COLS = ["pinfo_delta", "pinfo_baseline_pred", "pinfo_bin", "pinfo_H_given_bin", "pinfo_H_marginal"]


def _make_regression_data(n=200, p=5, seed=0):
    """Synthetic regression data: a two-informative-feature linear target plus noise columns."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)
    return X, y


def _make_binary_data(n=200, p=5, seed=0):
    """Synthetic binary-classification data derived from the same linear score, thresholded at 0."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)
    return X, y


def test_mode_a_query_regression_shape_and_finite():
    """Mode A (X_query given): output has the 5 documented columns, one row per query, all finite."""
    X_train, y_train = _make_regression_data(n=150, seed=0)
    X_query, _ = _make_regression_data(n=20, seed=1)

    out = compute_predictive_info_delta_features(X_train, y_train, X_query, seed=42, task="regression", n_bins=10)

    assert isinstance(out, pl.DataFrame)
    assert out.columns == _EXPECTED_COLS
    assert out.height == X_query.shape[0]
    for col in _EXPECTED_COLS:
        assert np.all(np.isfinite(out[col].to_numpy()))


def test_bin_index_within_range():
    """``pinfo_bin`` must fall in [0, n_bins), the quantile-bucket contract."""
    n_bins = 6
    X_train, y_train = _make_regression_data(n=150, seed=0)
    X_query, _ = _make_regression_data(n=30, seed=2)

    out = compute_predictive_info_delta_features(X_train, y_train, X_query, seed=1, task="regression", n_bins=n_bins)
    bins = out["pinfo_bin"].to_numpy()
    assert np.all(bins >= 0) and np.all(bins < n_bins)


def test_delta_equals_marginal_minus_conditional():
    """pred_info_delta = H_marginal - H_given_bin by construction -- must hold exactly (up to
    float32 rounding) for every row, not just on average."""
    X_train, y_train = _make_regression_data(n=120, seed=0)
    X_query, _ = _make_regression_data(n=25, seed=3)

    out = compute_predictive_info_delta_features(X_train, y_train, X_query, seed=5, task="regression", n_bins=8)
    delta = out["pinfo_delta"].to_numpy()
    h_marg = out["pinfo_H_marginal"].to_numpy()
    h_bin = out["pinfo_H_given_bin"].to_numpy()
    np.testing.assert_allclose(delta, h_marg - h_bin, atol=1e-4)


def test_marginal_entropy_constant_across_rows_within_a_fold():
    """H_marginal is a single per-fold scalar broadcast to every query row (it does not depend on
    the row's own bin) -- verify it is genuinely constant, not silently varying."""
    X_train, y_train = _make_regression_data(n=100, seed=0)
    X_query, _ = _make_regression_data(n=15, seed=4)

    out = compute_predictive_info_delta_features(X_train, y_train, X_query, seed=2, task="regression", n_bins=5)
    h_marg = out["pinfo_H_marginal"].to_numpy()
    assert np.allclose(h_marg, h_marg[0])


def test_binary_task_bernoulli_entropy_bounds():
    """Binary task: H_marginal/H_given_bin are Bernoulli entropies, bounded in [0, log(2)]."""
    X_train, y_train = _make_binary_data(n=150, seed=0)
    X_query, _ = _make_binary_data(n=20, seed=1)

    out = compute_predictive_info_delta_features(X_train, y_train, X_query, seed=9, task="binary", n_bins=8)
    for col in ("pinfo_H_marginal", "pinfo_H_given_bin"):
        vals = out[col].to_numpy()
        assert np.all(vals >= -1e-4)
        assert np.all(vals <= np.log(2) + 1e-3)


def test_mode_b_splitter_covers_every_train_row():
    """Mode B (X_query=None, splitter given): every train row gets an OOF assignment."""
    X_train, y_train = _make_regression_data(n=100, seed=0)
    splitter = KFold(n_splits=4, shuffle=True, random_state=0)

    out = compute_predictive_info_delta_features(X_train, y_train, X_query=None, splitter=splitter, seed=7, task="regression", n_bins=5)
    assert out.height == X_train.shape[0]
    assert np.all(np.isfinite(out["pinfo_delta"].to_numpy()))


def test_mode_b_without_splitter_raises():
    """X_query=None with no splitter is documented as invalid -- must raise."""
    X_train, y_train = _make_regression_data(n=50, seed=0)
    with pytest.raises(ValueError, match="splitter"):
        compute_predictive_info_delta_features(X_train, y_train, X_query=None, splitter=None, seed=1)


def test_deterministic_across_repeated_calls():
    """Same inputs + same seed -> byte-identical output across repeated calls."""
    X_train, y_train = _make_regression_data(n=100, seed=0)
    X_query, _ = _make_regression_data(n=15, seed=2)

    out1 = compute_predictive_info_delta_features(X_train, y_train, X_query, seed=13, task="regression", n_bins=6)
    out2 = compute_predictive_info_delta_features(X_train, y_train, X_query, seed=13, task="regression", n_bins=6)
    for col in _EXPECTED_COLS:
        np.testing.assert_array_equal(out1[col].to_numpy(), out2[col].to_numpy())
