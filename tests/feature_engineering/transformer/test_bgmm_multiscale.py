"""Unit coverage for ``compute_bgmm_multiscale_features`` (multi-scale BGM virtual sampling: fit
BayesianGaussianMixture at several component counts, expose distance + log-gap features from
each). No dedicated unit test existed for this module.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer.bgmm_multiscale import compute_bgmm_multiscale_features

_K_SCALES = (1, 3, 5, 10)
_COMPONENT_COUNTS = (3, 5, 8)


def _make_binary_data(n=300, p=4, seed=0):
    """Synthetic binary-classification data derived from a linear score, thresholded at 0."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)
    return X, y


def _make_regression_data(n=300, p=4, seed=0):
    """Synthetic regression data: a single-informative-feature linear target plus noise columns."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)
    return X, y


def test_default_column_count_and_names():
    """Default component_counts=(3,5,8) -> 3 x 2 x 4 = 24 columns, named K{n}_pos_k{k}/K{n}_loggap_k{k}."""
    X_train, y_train = _make_binary_data(n=200, seed=0)
    X_query, _ = _make_binary_data(n=15, seed=1)

    out = compute_bgmm_multiscale_features(X_train, y_train, X_query, seed=42, task="binary")

    assert isinstance(out, pl.DataFrame)
    expected = set()
    for n_comp in _COMPONENT_COUNTS:
        for k in _K_SCALES:
            expected.add(f"bgmms_K{n_comp}_pos_k{k}")
            expected.add(f"bgmms_K{n_comp}_loggap_k{k}")
    assert set(out.columns) == expected
    assert out.height == X_query.shape[0]
    for col in out.columns:
        assert np.all(np.isfinite(out[col].to_numpy()))


def test_pos_distance_columns_are_non_negative():
    """Every pos_k* column is a genuine distance -- must be >= 0 (loggap columns are unbounded log-ratios and excluded)."""
    X_train, y_train = _make_binary_data(n=200, seed=0)
    X_query, _ = _make_binary_data(n=15, seed=2)

    out = compute_bgmm_multiscale_features(X_train, y_train, X_query, seed=3, task="binary", component_counts=(4,))
    for k in _K_SCALES:
        assert np.all(out[f"bgmms_K4_pos_k{k}"].to_numpy() >= 0.0)


def test_pos_distance_monotonically_nondecreasing_within_a_scale():
    """Within one scale, k1 <= k3 <= k5 <= k10 (kNN distances are sorted by construction)."""
    X_train, y_train = _make_binary_data(n=200, seed=0)
    X_query, _ = _make_binary_data(n=15, seed=3)

    out = compute_bgmm_multiscale_features(X_train, y_train, X_query, seed=4, task="binary", component_counts=(4,))
    d = out.select([f"bgmms_K4_pos_k{k}" for k in _K_SCALES]).to_numpy()
    assert np.all(d[:, 0] <= d[:, 1] + 1e-4)
    assert np.all(d[:, 1] <= d[:, 2] + 1e-4)
    assert np.all(d[:, 2] <= d[:, 3] + 1e-4)


def test_regression_task_runs_via_quantile_slice():
    """task='regression' uses class_or_quantile_slice's high-quantile split instead of y>0.5 --
    exercised end-to-end for finiteness/shape."""
    X_train, y_train = _make_regression_data(n=200, seed=0)
    X_query, _ = _make_regression_data(n=15, seed=1)

    out = compute_bgmm_multiscale_features(X_train, y_train, X_query, seed=5, task="regression", component_counts=(4,), q_high=0.8)
    assert out.height == X_query.shape[0]
    for col in out.columns:
        assert np.all(np.isfinite(out[col].to_numpy()))


def test_custom_component_counts_control_column_count():
    """A single-element component_counts tuple emits exactly 1 x 2 x len(K_SCALES) columns."""
    X_train, y_train = _make_binary_data(n=200, seed=0)
    X_query, _ = _make_binary_data(n=15, seed=1)

    out = compute_bgmm_multiscale_features(X_train, y_train, X_query, seed=6, task="binary", component_counts=(3,))
    assert out.width == 1 * 2 * len(_K_SCALES)


def test_mode_a_oof_covers_every_train_row():
    """Mode A (X_query=None, splitter given): every train row gets an OOF assignment."""
    X_train, y_train = _make_binary_data(n=200, seed=0)
    splitter = KFold(n_splits=4, shuffle=True, random_state=0)

    out = compute_bgmm_multiscale_features(X_train, y_train, None, splitter=splitter, seed=7, task="binary", component_counts=(3,))
    assert out.height == X_train.shape[0]
    for col in out.columns:
        assert np.all(np.isfinite(out[col].to_numpy()))


def test_mode_b_without_splitter_raises():
    """X_query=None with no splitter is documented as invalid -- must raise."""
    X_train, y_train = _make_binary_data(n=50, seed=0)
    with pytest.raises(ValueError, match="splitter"):
        compute_bgmm_multiscale_features(X_train, y_train, X_query=None, splitter=None, seed=1)


def test_deterministic_across_repeated_calls():
    """Same inputs + same seed -> byte-identical output across repeated calls."""
    X_train, y_train = _make_binary_data(n=150, seed=0)
    X_query, _ = _make_binary_data(n=10, seed=4)

    out1 = compute_bgmm_multiscale_features(X_train, y_train, X_query, seed=11, task="binary", component_counts=(3,))
    out2 = compute_bgmm_multiscale_features(X_train, y_train, X_query, seed=11, task="binary", component_counts=(3,))
    for col in out1.columns:
        np.testing.assert_array_equal(out1[col].to_numpy(), out2[col].to_numpy())
