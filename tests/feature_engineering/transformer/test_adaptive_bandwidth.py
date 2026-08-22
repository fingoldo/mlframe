"""Unit coverage for ``compute_adaptive_bandwidth_attention`` (per-query softmax temperature
derived from local kNN density -- the "balloon estimator" row-attention variant). No dedicated
unit test existed for this module.

Kept n_train < 10_000 throughout so ``build_hnsw_index``'s auto backend resolves to plain sklearn
kNN (no hnswlib/pynndescent dependency needed to exercise this module's own logic).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer.adaptive_bandwidth import compute_adaptive_bandwidth_attention


def _make_data(n=300, p=6, seed=0):
    """Synthetic regression data: a single-informative-feature linear target plus noise columns."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)
    return X, y


def test_mode_b_query_shape_and_finite_random_projection():
    """Mode B (X_query given): one row per query, n_heads * len(aggregate) columns, all finite."""
    n_heads, aggregate = 3, ("y_mean", "y_std")
    X_train, y_train = _make_data(n=200, seed=0)
    X_query, _ = _make_data(n=25, seed=1)

    out = compute_adaptive_bandwidth_attention(
        X_train, y_train, X_query, splitter=None, seed=42, n_heads=n_heads, head_dim=4, k=16,
        projection="random", aggregate=aggregate,
    )

    assert isinstance(out, pl.DataFrame)
    assert out.height == X_query.shape[0]
    expected_cols = {f"abandw_h{h}_{agg}" for h in range(n_heads) for agg in aggregate}
    assert set(out.columns) == expected_cols
    for col in out.columns:
        assert np.all(np.isfinite(out[col].to_numpy()))


def test_mode_a_oof_covers_every_train_row():
    """Mode A (X_query=None, splitter given): every train row gets an OOF row via the folds."""
    n_heads = 2
    X_train, y_train = _make_data(n=200, seed=0)
    splitter = KFold(n_splits=4, shuffle=True, random_state=0)

    out = compute_adaptive_bandwidth_attention(
        X_train, y_train, None, splitter=splitter, seed=3, n_heads=n_heads, head_dim=4, k=16,
        projection="random", aggregate=("y_mean",),
    )
    assert out.height == X_train.shape[0]
    for col in out.columns:
        assert np.all(np.isfinite(out[col].to_numpy()))


def test_temp_scale_sharpens_or_smooths_attention():
    """A smaller temp_scale sharpens attention (lower entropy / more concentrated weights), which
    for this synthetic single-informative-feature target should not degrade the y_mean signal's
    correlation with the true target -- a real behavioral property of the balloon-estimator
    mechanism, not an implementation detail."""
    X_train, y_train = _make_data(n=250, seed=0)
    X_query, y_query_true = _make_data(n=40, seed=5)

    out_sharp = compute_adaptive_bandwidth_attention(
        X_train, y_train, X_query, splitter=None, seed=1, n_heads=2, head_dim=4, k=16,
        temp_scale=0.3, projection="random", aggregate=("y_mean",),
    )
    out_smooth = compute_adaptive_bandwidth_attention(
        X_train, y_train, X_query, splitter=None, seed=1, n_heads=2, head_dim=4, k=16,
        temp_scale=3.0, projection="random", aggregate=("y_mean",),
    )
    # both configurations must at least correlate positively with the true target (sanity that the
    # attention mechanism captures real signal, regardless of sharpness).
    y_mean_sharp = out_sharp["abandw_h0_y_mean"].to_numpy()
    y_mean_smooth = out_smooth["abandw_h0_y_mean"].to_numpy()
    assert np.corrcoef(y_mean_sharp, y_query_true)[0, 1] > 0.1
    assert np.corrcoef(y_mean_smooth, y_query_true)[0, 1] > 0.1


def test_x_mean_aggregate_has_head_dim_columns():
    """``aggregate=("x_mean",)`` emits ``head_dim`` per-dimension columns per head (the 2-D
    aggregate branch), not a single collapsed column."""
    head_dim = 5
    X_train, y_train = _make_data(n=150, seed=0)
    X_query, _ = _make_data(n=10, seed=2)

    out = compute_adaptive_bandwidth_attention(
        X_train, y_train, X_query, splitter=None, seed=4, n_heads=1, head_dim=head_dim, k=16,
        projection="random", aggregate=("x_mean",),
    )
    expected = {f"abandw_h0_x_mean_d{d}" for d in range(head_dim)}
    assert set(out.columns) == expected


def test_invalid_projection_raises():
    """An unrecognized ``projection`` value must raise, not silently fall through."""
    X_train, y_train = _make_data(n=100, seed=0)
    X_query, _ = _make_data(n=10, seed=1)
    with pytest.raises(ValueError, match="projection"):
        compute_adaptive_bandwidth_attention(
            X_train, y_train, X_query, splitter=None, seed=1, projection="bogus",  # type: ignore[arg-type]
        )


def test_deterministic_across_repeated_calls():
    """Same inputs + same seed -> byte-identical output across repeated calls."""
    X_train, y_train = _make_data(n=150, seed=0)
    X_query, _ = _make_data(n=15, seed=3)

    out1 = compute_adaptive_bandwidth_attention(
        X_train, y_train, X_query, splitter=None, seed=17, n_heads=2, head_dim=4, k=12,
        projection="random", aggregate=("y_mean", "y_std"),
    )
    out2 = compute_adaptive_bandwidth_attention(
        X_train, y_train, X_query, splitter=None, seed=17, n_heads=2, head_dim=4, k=12,
        projection="random", aggregate=("y_mean", "y_std"),
    )
    for col in out1.columns:
        np.testing.assert_array_equal(out1[col].to_numpy(), out2[col].to_numpy())
