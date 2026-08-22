"""Unit coverage for ``compute_residual_stratified_distance_features`` (iter103: distances to
nearest "easy" vs "hard" (baseline-residual-stratified) training rows). No dedicated unit test
existed for this module.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytest.importorskip("lightgbm")

from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer.residual_stratified_distance import compute_residual_stratified_distance_features

_EXPECTED_COLS = [
    "rsd_d_easy_k1", "rsd_d_easy_k3", "rsd_d_easy_k5",
    "rsd_d_hard_k1", "rsd_d_hard_k3", "rsd_d_hard_k5",
    "rsd_logratio_k1", "rsd_logratio_k3", "rsd_logratio_k5",
    "rsd_mean_r_easy_k5", "rsd_mean_r_hard_k5",
]


def _make_regression_data(n=200, p=5, seed=0):
    """Synthetic regression data: a two-informative-feature linear target plus noise columns."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)
    return X, y


def test_mode_a_query_shape_columns_and_finite():
    """Mode A (X_query given): the documented 11 columns, one row per query, finite values."""
    X_train, y_train = _make_regression_data(n=150, seed=0)
    X_query, _ = _make_regression_data(n=20, seed=1)

    out = compute_residual_stratified_distance_features(X_train, y_train, X_query, seed=42, task="regression")

    assert isinstance(out, pl.DataFrame)
    assert out.columns == _EXPECTED_COLS
    assert out.height == X_query.shape[0]
    for col in _EXPECTED_COLS:
        assert np.all(np.isfinite(out[col].to_numpy()))


def test_distances_are_non_negative():
    """Every d_easy_k*/d_hard_k* column is a genuine distance -- must be >= 0."""
    X_train, y_train = _make_regression_data(n=150, seed=0)
    X_query, _ = _make_regression_data(n=15, seed=2)

    out = compute_residual_stratified_distance_features(X_train, y_train, X_query, seed=1, task="regression")
    for col in ("rsd_d_easy_k1", "rsd_d_easy_k3", "rsd_d_easy_k5", "rsd_d_hard_k1", "rsd_d_hard_k3", "rsd_d_hard_k5"):
        assert np.all(out[col].to_numpy() >= 0.0)


def test_k_distances_are_monotonically_nondecreasing():
    """Within a band, distance to the k=1 nearest neighbour must be <= k=3's <= k=5's (kNN distances
    are sorted by construction, and k1/k3/k5 quantiles of that sorted list preserve the order)."""
    X_train, y_train = _make_regression_data(n=150, seed=0)
    X_query, _ = _make_regression_data(n=15, seed=3)

    out = compute_residual_stratified_distance_features(X_train, y_train, X_query, seed=4, task="regression")
    d_easy = out.select(["rsd_d_easy_k1", "rsd_d_easy_k3", "rsd_d_easy_k5"]).to_numpy()
    d_hard = out.select(["rsd_d_hard_k1", "rsd_d_hard_k3", "rsd_d_hard_k5"]).to_numpy()
    assert np.all(d_easy[:, 0] <= d_easy[:, 1] + 1e-5)
    assert np.all(d_easy[:, 1] <= d_easy[:, 2] + 1e-5)
    assert np.all(d_hard[:, 0] <= d_hard[:, 1] + 1e-5)
    assert np.all(d_hard[:, 1] <= d_hard[:, 2] + 1e-5)


def test_logratio_equals_log_of_distance_ratio():
    """logratio_k* = log((d_hard + eps) / (d_easy + eps)) by construction -- must hold per row."""
    X_train, y_train = _make_regression_data(n=150, seed=0)
    X_query, _ = _make_regression_data(n=20, seed=5)

    out = compute_residual_stratified_distance_features(X_train, y_train, X_query, seed=6, task="regression")
    for suffix, k in (("k1", "k1"), ("k3", "k3"), ("k5", "k5")):
        d_easy = out[f"rsd_d_easy_{k}"].to_numpy()
        d_hard = out[f"rsd_d_hard_{k}"].to_numpy()
        expected = np.log((d_hard + 1e-6) / (d_easy + 1e-6))
        np.testing.assert_allclose(out[f"rsd_logratio_{suffix}"].to_numpy(), expected, atol=1e-3)


def test_mode_b_splitter_covers_every_train_row():
    """Mode B (X_query=None, splitter given): every train row gets an OOF assignment."""
    X_train, y_train = _make_regression_data(n=100, seed=0)
    splitter = KFold(n_splits=4, shuffle=True, random_state=0)

    out = compute_residual_stratified_distance_features(X_train, y_train, X_query=None, splitter=splitter, seed=7, task="regression")
    assert out.height == X_train.shape[0]
    for col in _EXPECTED_COLS:
        assert np.all(np.isfinite(out[col].to_numpy()))


def test_mode_b_without_splitter_raises():
    """X_query=None with no splitter is documented as invalid -- must raise."""
    X_train, y_train = _make_regression_data(n=50, seed=0)
    with pytest.raises(ValueError, match="splitter"):
        compute_residual_stratified_distance_features(X_train, y_train, X_query=None, splitter=None, seed=1)


def test_deterministic_across_repeated_calls():
    """Same inputs + same seed -> byte-identical output across repeated calls."""
    X_train, y_train = _make_regression_data(n=100, seed=0)
    X_query, _ = _make_regression_data(n=15, seed=4)

    out1 = compute_residual_stratified_distance_features(X_train, y_train, X_query, seed=13, task="regression")
    out2 = compute_residual_stratified_distance_features(X_train, y_train, X_query, seed=13, task="regression")
    for col in _EXPECTED_COLS:
        np.testing.assert_array_equal(out1[col].to_numpy(), out2[col].to_numpy())
