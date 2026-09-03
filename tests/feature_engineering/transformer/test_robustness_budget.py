"""Unit coverage for ``compute_robustness_budget_features`` (per-row prediction stability under
Gaussian noise injection). This module had no dedicated unit test -- only the (skipped-by-default,
``--run-biz-transformer``-gated) real-datasets biz_val suite touched it -- leaving its core mechanics,
error paths, and both task branches (binary/regression) unexercised in ordinary CI.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytest.importorskip("lightgbm")

from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer.robustness_budget import compute_robustness_budget_features

_EXPECTED_COLS = ["robust_pred_orig", "robust_pred_mean", "robust_pred_std", "robust_pred_range", "robust_flip_rate"]


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


def test_mode_a_query_regression_shape_and_dtype():
    """Mode A (X_query given): output is a polars frame with the 5 documented columns, one row per
    query row, finite values, and the requested dtype."""
    X_train, y_train = _make_regression_data(n=150, seed=0)
    X_query, _ = _make_regression_data(n=20, seed=1)

    out = compute_robustness_budget_features(X_train, y_train, X_query, seed=42, task="regression", n_perturbations=8)

    assert isinstance(out, pl.DataFrame)
    assert out.columns == _EXPECTED_COLS
    assert out.height == X_query.shape[0]
    for col in _EXPECTED_COLS:
        arr = out[col].to_numpy()
        assert arr.dtype == np.float32
        assert np.all(np.isfinite(arr))
    # regression path never flips a class -> flip_rate is identically 0.
    assert np.all(out["robust_flip_rate"].to_numpy() == 0.0)
    # a genuine noise-response signal: std/range must be positive somewhere (the model is not constant).
    assert out["robust_pred_std"].to_numpy().sum() > 0.0


def test_mode_a_query_binary_flip_rate_in_unit_interval():
    """Binary task: flip_rate is a fraction in [0, 1], and pred_orig/pred_mean are valid probabilities."""
    X_train, y_train = _make_binary_data(n=150, seed=0)
    X_query, _ = _make_binary_data(n=20, seed=1)

    out = compute_robustness_budget_features(X_train, y_train, X_query, seed=42, task="binary", n_perturbations=8)

    flip = out["robust_flip_rate"].to_numpy()
    assert np.all(flip >= 0.0) and np.all(flip <= 1.0)
    for col in ("robust_pred_orig", "robust_pred_mean"):
        vals = out[col].to_numpy()
        assert np.all(vals >= 0.0) and np.all(vals <= 1.0)


def test_mode_b_splitter_produces_oof_coverage_for_every_row():
    """Mode B (X_query=None, splitter given): every train row gets exactly one OOF assignment via
    the splitter's folds, so the output has one row per TRAIN row with no leftover zeros from an
    unassigned fold."""
    X_train, y_train = _make_regression_data(n=100, seed=0)
    splitter = KFold(n_splits=4, shuffle=True, random_state=0)

    out = compute_robustness_budget_features(X_train, y_train, X_query=None, splitter=splitter, seed=7, task="regression", n_perturbations=4)

    assert out.height == X_train.shape[0]
    # every row's OOF pred_std must be finite and non-negative (the +1e-9 floor rules out an
    # exact-zero row only ever occurring from being genuinely skipped by every fold).
    std = out["robust_pred_std"].to_numpy()
    assert np.all(np.isfinite(std))
    assert np.all(std > 0.0)


def test_mode_b_without_splitter_raises():
    """X_query=None with no splitter is documented as invalid (Mode A requires X_query, Mode B
    requires a splitter) -- must raise, not silently return an empty/degenerate frame."""
    X_train, y_train = _make_regression_data(n=50, seed=0)
    with pytest.raises(ValueError, match="splitter"):
        compute_robustness_budget_features(X_train, y_train, X_query=None, splitter=None, seed=1)


def test_seed_validation_rejects_non_int():
    """require_seed's contract (no None, no derived/non-int seeds) must actually be enforced at
    this call site, not silently bypassed."""
    X_train, y_train = _make_regression_data(n=50, seed=0)
    X_query, _ = _make_regression_data(n=10, seed=1)
    with pytest.raises(TypeError):
        compute_robustness_budget_features(X_train, y_train, X_query, seed=None)  # type: ignore[arg-type]


def test_deterministic_across_repeated_calls():
    """Same inputs + same seed -> byte-identical output (the perturbation RNG is seeded, not
    system-random), so results are reproducible across repeated calls / process restarts."""
    X_train, y_train = _make_regression_data(n=120, seed=0)
    X_query, _ = _make_regression_data(n=15, seed=2)

    out1 = compute_robustness_budget_features(X_train, y_train, X_query, seed=99, task="regression", n_perturbations=6)
    out2 = compute_robustness_budget_features(X_train, y_train, X_query, seed=99, task="regression", n_perturbations=6)

    for col in _EXPECTED_COLS:
        np.testing.assert_array_equal(out1[col].to_numpy(), out2[col].to_numpy())


def test_standardize_false_skips_scaler_but_still_finite():
    """``standardize=False`` bypasses the RobustScaler branch entirely -- must still produce a
    valid, finite output on the raw (unscaled) features."""
    X_train, y_train = _make_regression_data(n=100, seed=0)
    X_query, _ = _make_regression_data(n=10, seed=3)

    out = compute_robustness_budget_features(X_train, y_train, X_query, seed=5, task="regression", n_perturbations=4, standardize=False)
    for col in _EXPECTED_COLS:
        assert np.all(np.isfinite(out[col].to_numpy()))


def test_column_prefix_is_applied():
    """``column_prefix`` renames every output column -- the caller-facing naming contract."""
    X_train, y_train = _make_regression_data(n=60, seed=0)
    X_query, _ = _make_regression_data(n=8, seed=4)

    out = compute_robustness_budget_features(X_train, y_train, X_query, seed=3, task="regression", n_perturbations=4, column_prefix="rb2")
    assert out.columns == [f"rb2_{suffix}" for suffix in ("pred_orig", "pred_mean", "pred_std", "pred_range", "flip_rate")]
