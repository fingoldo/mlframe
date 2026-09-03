"""Unit coverage for ``compute_ks_shift_features`` (local KS-test + moment-shift distributional
features, njit KS/Wasserstein-1 kernel). No dedicated unit test existed for this module.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer.ks_shift import compute_ks_shift_features


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


def test_regression_task_emits_4_columns_all_finite():
    """Regression: the documented 4 columns (ks, w1, mean_shift, log_var_ratio), one row per
    query, all finite."""
    X_train, y_train = _make_regression_data(n=150, seed=0)
    X_query, _ = _make_regression_data(n=20, seed=1)

    out = compute_ks_shift_features(X_train, y_train, X_query, seed=42, task="regression", k=16)

    assert isinstance(out, pl.DataFrame)
    assert set(out.columns) == {"ksshift_mean_shift", "ksshift_log_var_ratio", "ksshift_ks", "ksshift_w1"}
    assert out.height == X_query.shape[0]
    for col in out.columns:
        assert np.all(np.isfinite(out[col].to_numpy()))


def test_binary_task_emits_only_2_columns():
    """Binary: KS/Wasserstein collapse to mean-shift for {0,1}-valued y, so only 2 columns are
    emitted (mean_shift, log_var_ratio) -- no ks/w1."""
    X_train, y_train = _make_binary_data(n=150, seed=0)
    X_query, _ = _make_binary_data(n=20, seed=1)

    out = compute_ks_shift_features(X_train, y_train, X_query, seed=1, task="binary", k=16)
    assert set(out.columns) == {"ksshift_mean_shift", "ksshift_log_var_ratio"}


def test_ks_statistic_bounded_in_unit_interval():
    """KS = sup|F_local - F_global| is a distance between two CDFs -- bounded in [0, 1]."""
    X_train, y_train = _make_regression_data(n=200, seed=0)
    X_query, _ = _make_regression_data(n=30, seed=2)

    out = compute_ks_shift_features(X_train, y_train, X_query, seed=3, task="regression", k=20)
    ks = out["ksshift_ks"].to_numpy()
    assert np.all(ks >= -1e-5) and np.all(ks <= 1.0 + 1e-5)


def test_wasserstein_is_non_negative():
    """W1 = integral of |F_local - F_global| dt is a non-negative earth-mover distance."""
    X_train, y_train = _make_regression_data(n=200, seed=0)
    X_query, _ = _make_regression_data(n=30, seed=2)

    out = compute_ks_shift_features(X_train, y_train, X_query, seed=3, task="regression", k=20)
    assert np.all(out["ksshift_w1"].to_numpy() >= -1e-5)


def test_ks_near_zero_when_query_neighborhood_matches_global_distribution():
    """A query whose k nearest neighbors are drawn i.i.d. from the SAME distribution as the full
    train set should show a small KS statistic (no real distributional shift) -- checked via a
    query batch placed at the data's own centroid, which pulls a broad, representative
    neighborhood rather than a distributional outlier region."""
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((500, 3)).astype(np.float32)
    y_train = rng.standard_normal(500).astype(np.float32)  # y independent of X: every neighborhood is a random subsample
    X_query = np.zeros((10, 3), dtype=np.float32)  # centroid query: neighborhood is an unbiased random subsample of y

    out = compute_ks_shift_features(X_train, y_train, X_query, seed=5, task="regression", k=64)
    # unbiased subsample vs the full distribution -> small (not necessarily 0) KS/W1.
    assert out["ksshift_ks"].to_numpy().mean() < 0.3


def test_k_below_minimum_raises():
    """``k < 4`` is documented as invalid (too few neighbors for a meaningful CDF comparison)."""
    X_train, y_train = _make_regression_data(n=50, seed=0)
    X_query, _ = _make_regression_data(n=5, seed=1)
    with pytest.raises(ValueError, match="k must be"):
        compute_ks_shift_features(X_train, y_train, X_query, seed=1, k=2)


def test_invalid_task_raises():
    """An unrecognized ``task`` value must raise, not silently fall through."""
    X_train, y_train = _make_regression_data(n=50, seed=0)
    X_query, _ = _make_regression_data(n=5, seed=1)
    with pytest.raises(ValueError, match="task"):
        compute_ks_shift_features(X_train, y_train, X_query, seed=1, task="bogus")  # type: ignore[arg-type]


def test_mode_a_oof_covers_every_train_row():
    """Mode A (X_query=None, splitter given): every train row gets an OOF assignment."""
    X_train, y_train = _make_regression_data(n=100, seed=0)
    splitter = KFold(n_splits=4, shuffle=True, random_state=0)

    out = compute_ks_shift_features(X_train, y_train, None, splitter=splitter, seed=7, task="regression", k=16)
    assert out.height == X_train.shape[0]
    for col in out.columns:
        assert np.all(np.isfinite(out[col].to_numpy()))


def test_mode_b_without_splitter_raises():
    """X_query=None with no splitter is documented as invalid -- must raise."""
    X_train, y_train = _make_regression_data(n=50, seed=0)
    with pytest.raises(ValueError, match="splitter"):
        compute_ks_shift_features(X_train, y_train, X_query=None, splitter=None, seed=1)


def test_deterministic_across_repeated_calls():
    """Same inputs + same seed -> byte-identical output across repeated calls."""
    X_train, y_train = _make_regression_data(n=120, seed=0)
    X_query, _ = _make_regression_data(n=15, seed=4)

    out1 = compute_ks_shift_features(X_train, y_train, X_query, seed=11, task="regression", k=16)
    out2 = compute_ks_shift_features(X_train, y_train, X_query, seed=11, task="regression", k=16)
    for col in out1.columns:
        np.testing.assert_array_equal(out1[col].to_numpy(), out2[col].to_numpy())
