"""Unit coverage for ``compute_hard_row_attention_features`` (top-K hardest-residual training
rows as softmax attention anchors). No dedicated unit test existed -- only the
(skipped-by-default, ``--run-biz-transformer``-gated) real-datasets biz_val suite touched it.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytest.importorskip("lightgbm")

from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer.hard_row_attention import compute_hard_row_attention_features


def _make_regression_data(n=200, p=5, seed=0):
    """Synthetic regression data: a two-informative-feature linear target plus noise columns."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)
    return X, y


def _expected_cols(n_hard, prefix="hrattn"):
    """The documented n_hard + 6 output column names in their emitted order."""
    cols = [f"{prefix}_w_h{a}" for a in range(n_hard)]
    cols += [f"{prefix}_entropy", f"{prefix}_y_agg", f"{prefix}_abs_resid_agg", f"{prefix}_signed_resid_agg", f"{prefix}_best_hard", f"{prefix}_min_dist"]
    return cols


def test_mode_a_query_shape_columns_and_weight_simplex():
    """Mode A: output has n_hard + 6 columns in the documented order, one row per query row, and
    the n_hard softmax weight columns sum to ~1 per row (a genuine attention simplex, not raw scores)."""
    n_hard = 8
    X_train, y_train = _make_regression_data(n=100, seed=0)
    X_query, _ = _make_regression_data(n=15, seed=1)

    out = compute_hard_row_attention_features(X_train, y_train, X_query, seed=42, task="regression", n_hard=n_hard)

    assert isinstance(out, pl.DataFrame)
    assert out.columns == _expected_cols(n_hard)
    assert out.height == X_query.shape[0]

    w_cols = [f"hrattn_w_h{a}" for a in range(n_hard)]
    weights = out.select(w_cols).to_numpy()
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-4)
    assert np.all(weights >= 0.0)


def test_entropy_bounded_by_log_n_hard():
    """Softmax entropy over n_hard anchors is bounded in [0, log(n_hard)] -- a real invariant of a
    valid probability simplex, not an implementation detail."""
    n_hard = 8
    X_train, y_train = _make_regression_data(n=100, seed=0)
    X_query, _ = _make_regression_data(n=20, seed=2)

    out = compute_hard_row_attention_features(X_train, y_train, X_query, seed=1, task="regression", n_hard=n_hard)
    entropy = out["hrattn_entropy"].to_numpy()
    assert np.all(entropy >= -1e-4)
    assert np.all(entropy <= np.log(n_hard) + 1e-3)


def test_best_hard_index_matches_argmax_weight():
    """``best_hard`` must equal argmax over the row's own weight columns (internal consistency
    between the reported best-anchor index and the weights actually emitted)."""
    n_hard = 6
    X_train, y_train = _make_regression_data(n=80, seed=0)
    X_query, _ = _make_regression_data(n=12, seed=3)

    out = compute_hard_row_attention_features(X_train, y_train, X_query, seed=5, task="regression", n_hard=n_hard)
    w_cols = [f"hrattn_w_h{a}" for a in range(n_hard)]
    weights = out.select(w_cols).to_numpy()
    expected_best = weights.argmax(axis=1).astype(np.float32)
    np.testing.assert_array_equal(out["hrattn_best_hard"].to_numpy(), expected_best)


def test_n_hard_exceeds_fold_row_count_pads_without_crashing():
    """When a fold has fewer training rows than ``n_hard``, the module pads the anchor set with a
    repeated row rather than crashing or shrinking the output width -- verify padding produces a
    valid (still-normalized) output at the documented width."""
    n_hard = 16
    X_train, y_train = _make_regression_data(n=10, seed=0)  # fewer rows than n_hard
    X_query, _ = _make_regression_data(n=5, seed=1)

    out = compute_hard_row_attention_features(X_train, y_train, X_query, seed=7, task="regression", n_hard=n_hard)
    assert out.columns == _expected_cols(n_hard)
    w_cols = [f"hrattn_w_h{a}" for a in range(n_hard)]
    weights = out.select(w_cols).to_numpy()
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-4)


def test_mode_b_splitter_covers_every_train_row():
    """Mode B (X_query=None, splitter given): every train row gets an OOF assignment."""
    n_hard = 5
    X_train, y_train = _make_regression_data(n=100, seed=0)
    splitter = KFold(n_splits=4, shuffle=True, random_state=0)

    out = compute_hard_row_attention_features(X_train, y_train, X_query=None, splitter=splitter, seed=3, task="regression", n_hard=n_hard)
    assert out.height == X_train.shape[0]
    w_cols = [f"hrattn_w_h{a}" for a in range(n_hard)]
    weights = out.select(w_cols).to_numpy()
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-4)


def test_mode_b_without_splitter_raises():
    """X_query=None with no splitter is documented as invalid -- must raise."""
    X_train, y_train = _make_regression_data(n=50, seed=0)
    with pytest.raises(ValueError, match="splitter"):
        compute_hard_row_attention_features(X_train, y_train, X_query=None, splitter=None, seed=1)


def test_deterministic_across_repeated_calls():
    """Same inputs + same seed -> byte-identical output (lexsort tiebreak + seeded LGBM)."""
    n_hard = 6
    X_train, y_train = _make_regression_data(n=90, seed=0)
    X_query, _ = _make_regression_data(n=10, seed=4)

    out1 = compute_hard_row_attention_features(X_train, y_train, X_query, seed=11, task="regression", n_hard=n_hard)
    out2 = compute_hard_row_attention_features(X_train, y_train, X_query, seed=11, task="regression", n_hard=n_hard)
    for col in out1.columns:
        np.testing.assert_array_equal(out1[col].to_numpy(), out2[col].to_numpy())


def test_binary_task_runs_and_produces_finite_output():
    """The binary-classification branch of the shared baseline fitter is exercised end-to-end."""
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((100, 5)).astype(np.float32)
    y_train = (X_train[:, 0] + 0.5 * X_train[:, 1] > 0).astype(np.float32)
    X_query = rng.standard_normal((10, 5)).astype(np.float32)

    out = compute_hard_row_attention_features(X_train, y_train, X_query, seed=9, task="binary", n_hard=6)
    for col in out.columns:
        assert np.all(np.isfinite(out[col].to_numpy()))
