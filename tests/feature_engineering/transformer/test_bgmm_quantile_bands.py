"""Unit coverage for ``compute_bgmm_quantile_bands_features`` (per-y-quantile-band BGM virtual
samples; per-row kNN distance features against each band's real+virtual point cloud). No
dedicated unit test existed for this module.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer.bgmm_quantile_bands import compute_bgmm_quantile_bands_features

_K_SCALES = (1, 3, 5, 10)


def _make_regression_data(n=300, p=4, seed=0):
    """Synthetic regression data: a single-informative-feature linear target plus noise columns."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)
    return X, y


def _make_binary_data(n=300, p=4, seed=0):
    """Synthetic binary-classification data derived from the same linear score, thresholded at 0."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)
    return X, y


def test_regression_task_emits_n_bands_times_k_scales_columns():
    """Regression: n_bands x len(K_SCALES) columns, named Q1..Qn per band."""
    n_bands = 5
    X_train, y_train = _make_regression_data(n=200, seed=0)
    X_query, _ = _make_regression_data(n=15, seed=1)

    out = compute_bgmm_quantile_bands_features(X_train, y_train, X_query, seed=42, task="regression", n_bands=n_bands)

    assert isinstance(out, pl.DataFrame)
    expected = {f"bqb_Q{b + 1}_k{k}" for b in range(n_bands) for k in _K_SCALES}
    assert set(out.columns) == expected
    assert out.height == X_query.shape[0]
    for col in out.columns:
        assert np.all(np.isfinite(out[col].to_numpy()))


def test_binary_task_emits_pos_neg_columns():
    """Binary: 2 bands (pos/neg) x len(K_SCALES) columns."""
    X_train, y_train = _make_binary_data(n=200, seed=0)
    X_query, _ = _make_binary_data(n=15, seed=1)

    out = compute_bgmm_quantile_bands_features(X_train, y_train, X_query, seed=1, task="binary")
    expected = {f"bqb_{tag}_k{k}" for tag in ("pos", "neg") for k in _K_SCALES}
    assert set(out.columns) == expected


def test_distances_are_non_negative():
    """Every k-distance column is a genuine distance -- must be >= 0."""
    X_train, y_train = _make_regression_data(n=200, seed=0)
    X_query, _ = _make_regression_data(n=15, seed=2)

    out = compute_bgmm_quantile_bands_features(X_train, y_train, X_query, seed=3, task="regression", n_bands=3)
    for col in out.columns:
        assert np.all(out[col].to_numpy() >= 0.0)


def test_k_distances_monotonically_nondecreasing_within_a_band():
    """Within one band, k1 <= k3 <= k5 <= k10 (kNN distances are sorted by construction)."""
    X_train, y_train = _make_regression_data(n=200, seed=0)
    X_query, _ = _make_regression_data(n=15, seed=3)

    out = compute_bgmm_quantile_bands_features(X_train, y_train, X_query, seed=4, task="regression", n_bands=3)
    d = out.select(["bqb_Q1_k1", "bqb_Q1_k3", "bqb_Q1_k5", "bqb_Q1_k10"]).to_numpy()
    assert np.all(d[:, 0] <= d[:, 1] + 1e-4)
    assert np.all(d[:, 1] <= d[:, 2] + 1e-4)
    assert np.all(d[:, 2] <= d[:, 3] + 1e-4)


def test_mode_a_oof_covers_every_train_row():
    """Mode A (X_query=None, splitter given): every train row gets an OOF assignment."""
    X_train, y_train = _make_regression_data(n=150, seed=0)
    splitter = KFold(n_splits=4, shuffle=True, random_state=0)

    out = compute_bgmm_quantile_bands_features(X_train, y_train, None, splitter=splitter, seed=7, task="regression", n_bands=3)
    assert out.height == X_train.shape[0]
    for col in out.columns:
        assert np.all(np.isfinite(out[col].to_numpy()))


def test_mode_b_without_splitter_raises():
    """X_query=None with no splitter is documented as invalid -- must raise."""
    X_train, y_train = _make_regression_data(n=50, seed=0)
    with pytest.raises(ValueError, match="splitter"):
        compute_bgmm_quantile_bands_features(X_train, y_train, X_query=None, splitter=None, seed=1)


def test_deterministic_across_repeated_calls():
    """Same inputs + same seed -> byte-identical output across repeated calls."""
    X_train, y_train = _make_regression_data(n=150, seed=0)
    X_query, _ = _make_regression_data(n=10, seed=4)

    out1 = compute_bgmm_quantile_bands_features(X_train, y_train, X_query, seed=11, task="regression", n_bands=3)
    out2 = compute_bgmm_quantile_bands_features(X_train, y_train, X_query, seed=11, task="regression", n_bands=3)
    for col in out1.columns:
        np.testing.assert_array_equal(out1[col].to_numpy(), out2[col].to_numpy())
