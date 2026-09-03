"""Unit coverage for ``compute_class_conditional_anchor_attention`` (K-means anchors fit
separately on positive/negative rows; per-row softmax-similarity features). No dedicated unit
test existed for this module.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer.class_conditional_anchor import compute_class_conditional_anchor_attention


def _make_binary_data(n=300, p=5, pos_rate=0.3, seed=0):
    """Synthetic binary-classification data with a mixture-of-Gaussians class structure."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(np.float32)
    X = rng.standard_normal((n, p)).astype(np.float32)
    X[:, 0] += y * 2.0  # feature 0 shifts with class -> a real, learnable class-conditional structure
    return X, y


def test_mode_b_query_shape_and_simplex_similarities():
    """Mode B: (2K+1) columns, sim_pos/sim_neg rows are each a softmax simplex (sum to 1)."""
    K = 4
    X_train, y_train = _make_binary_data(n=200, seed=0)
    X_query, _ = _make_binary_data(n=20, seed=1)

    out = compute_class_conditional_anchor_attention(X_train, y_train, X_query, seed=42, n_anchors_per_class=K)

    assert isinstance(out, pl.DataFrame)
    assert out.height == X_query.shape[0]
    pos_cols = [f"ccanchor_pos_a{j}" for j in range(K)]
    neg_cols = [f"ccanchor_neg_a{j}" for j in range(K)]
    assert set(out.columns) == set(pos_cols) | set(neg_cols) | {"ccanchor_mass_pos"}

    sim_pos = out.select(pos_cols).to_numpy()
    sim_neg = out.select(neg_cols).to_numpy()
    assert np.allclose(sim_pos.sum(axis=1), 1.0, atol=1e-4)
    assert np.allclose(sim_neg.sum(axis=1), 1.0, atol=1e-4)
    assert np.all(sim_pos >= 0.0) and np.all(sim_neg >= 0.0)


def test_mass_pos_bounded_in_unit_interval():
    """mass_pos is a fraction of the unified 2K-anchor softmax mass -- bounded in [0, 1]."""
    X_train, y_train = _make_binary_data(n=200, seed=0)
    X_query, _ = _make_binary_data(n=20, seed=2)

    out = compute_class_conditional_anchor_attention(X_train, y_train, X_query, seed=1, n_anchors_per_class=4)
    mass = out["ccanchor_mass_pos"].to_numpy()
    assert np.all(mass >= -1e-5) and np.all(mass <= 1.0 + 1e-5)


def test_mass_pos_separates_query_rows_by_true_class():
    """Sanity: queries drawn from the POSITIVE class distribution should show, on average, higher
    mass_pos than queries drawn from the NEGATIVE class distribution -- confirms the anchors
    actually encode real class-conditional structure, not noise."""
    X_train, y_train = _make_binary_data(n=400, seed=0)
    X_query, y_query = _make_binary_data(n=100, seed=3)

    out = compute_class_conditional_anchor_attention(X_train, y_train, X_query, seed=5, n_anchors_per_class=6)
    mass = out["ccanchor_mass_pos"].to_numpy()
    mass_true_pos = mass[y_query > 0.5]
    mass_true_neg = mass[y_query <= 0.5]
    assert mass_true_pos.mean() > mass_true_neg.mean()


def test_mode_a_oof_covers_every_train_row():
    """Mode A (X_query=None, splitter given): every train row gets an OOF assignment."""
    X_train, y_train = _make_binary_data(n=300, seed=0)
    splitter = KFold(n_splits=4, shuffle=True, random_state=0)

    out = compute_class_conditional_anchor_attention(X_train, y_train, None, splitter=splitter, seed=7, n_anchors_per_class=4)
    assert out.height == X_train.shape[0]
    for col in out.columns:
        assert np.all(np.isfinite(out[col].to_numpy()))


def test_mode_b_without_splitter_raises():
    """X_query=None with no splitter is documented as invalid -- must raise."""
    X_train, y_train = _make_binary_data(n=50, seed=0)
    with pytest.raises(ValueError, match="splitter"):
        compute_class_conditional_anchor_attention(X_train, y_train, X_query=None, splitter=None, seed=1)


def test_regression_task_not_implemented_raises():
    """Only task='binary' is implemented; task='regression' must raise NotImplementedError."""
    X_train, y_train = _make_binary_data(n=50, seed=0)
    X_query, _ = _make_binary_data(n=10, seed=1)
    with pytest.raises(NotImplementedError):
        compute_class_conditional_anchor_attention(X_train, y_train, X_query, seed=1, task="regression")


def test_n_anchors_below_minimum_raises():
    """``n_anchors_per_class < 2`` is documented as invalid."""
    X_train, y_train = _make_binary_data(n=50, seed=0)
    X_query, _ = _make_binary_data(n=10, seed=1)
    with pytest.raises(ValueError, match="n_anchors_per_class"):
        compute_class_conditional_anchor_attention(X_train, y_train, X_query, seed=1, n_anchors_per_class=1)


def test_non_positive_softmax_temp_raises():
    """``softmax_temp <= 0`` is documented as invalid."""
    X_train, y_train = _make_binary_data(n=50, seed=0)
    X_query, _ = _make_binary_data(n=10, seed=1)
    with pytest.raises(ValueError, match="softmax_temp"):
        compute_class_conditional_anchor_attention(X_train, y_train, X_query, seed=1, softmax_temp=0.0)


def test_too_few_rows_per_class_raises():
    """A degenerate fixture with <2 rows in a class must raise a clear error, not silently misbehave."""
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((20, 3)).astype(np.float32)
    y_train = np.zeros(20, dtype=np.float32)
    y_train[0] = 1.0  # only 1 positive row
    X_query = rng.standard_normal((5, 3)).astype(np.float32)
    with pytest.raises(ValueError, match="need >=2 rows per class"):
        compute_class_conditional_anchor_attention(X_train, y_train, X_query, seed=1, n_anchors_per_class=4)


def test_deterministic_across_repeated_calls():
    """Same inputs + same seed -> byte-identical output across repeated calls."""
    X_train, y_train = _make_binary_data(n=150, seed=0)
    X_query, _ = _make_binary_data(n=15, seed=4)

    out1 = compute_class_conditional_anchor_attention(X_train, y_train, X_query, seed=11, n_anchors_per_class=4)
    out2 = compute_class_conditional_anchor_attention(X_train, y_train, X_query, seed=11, n_anchors_per_class=4)
    for col in out1.columns:
        np.testing.assert_array_equal(out1[col].to_numpy(), out2[col].to_numpy())
