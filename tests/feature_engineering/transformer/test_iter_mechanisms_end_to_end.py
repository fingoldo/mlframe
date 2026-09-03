"""End-to-end smoke coverage for a batch of "iter" feature-engineering mechanisms that previously had
no test exercising their public ``compute_*`` entry point (only narrow internal-helper tests existed, if
any). Each mechanism follows the same two-mode contract: Mode A (``X_query=None``, OOF via ``splitter``)
and Mode B (``X_query`` given, fit-on-full-train then score new rows). These tests exercise both modes
with small synthetic data and assert finite, correctly-shaped, non-degenerate output -- not selection
quality (that's the province of a dedicated biz_val test where the mechanism's own claim warrants one).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

pytestmark = pytest.mark.fast


def _make_regression_data(n=200, d=6, seed=0):
    """Small synthetic regression dataset with a genuine X->y relationship."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.1 * rng.standard_normal(n)).astype(np.float32)
    return X, y


def _make_binary_data(n=200, d=6, seed=0):
    """Small synthetic binary-classification dataset with a genuine X->y relationship."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    logits = X[:, 0] + 0.5 * X[:, 1]
    y = (logits + 0.3 * rng.standard_normal(n) > 0).astype(np.float32)
    return X, y


def _assert_finite_shaped(df, n_rows, n_cols=None):
    """Shared assertion: output is a polars DataFrame with the expected row count, all-finite values."""
    assert df.shape[0] == n_rows
    if n_cols is not None:
        assert df.shape[1] == n_cols
    arr = df.to_numpy()
    assert np.all(np.isfinite(arr)), "non-finite value in output"


def test_multi_threshold_ordinal_both_modes():
    """compute_multi_threshold_ordinal_features: Mode A (OOF) and Mode B (X_query) both regression + binary."""
    pytest.importorskip("lightgbm")
    from mlframe.feature_engineering.transformer.multi_threshold_ordinal import (
        compute_multi_threshold_ordinal_features,
    )

    X, y = _make_regression_data(n=180, seed=1)
    spl = KFold(n_splits=3, shuffle=True, random_state=0)
    out_a = compute_multi_threshold_ordinal_features(X, y, None, splitter=spl, seed=42, task="regression")
    _assert_finite_shaped(out_a, 180, 5)

    X_q, _ = _make_regression_data(n=40, seed=2)
    out_b = compute_multi_threshold_ordinal_features(X, y, X_q, seed=42, task="regression")
    _assert_finite_shaped(out_b, 40, 5)

    Xb, yb = _make_binary_data(n=180, seed=3)
    out_bin = compute_multi_threshold_ordinal_features(Xb, yb, None, splitter=spl, seed=42, task="binary")
    _assert_finite_shaped(out_bin, 180, 5)


def test_target_kmeans_codebook_both_modes():
    """compute_target_kmeans_codebook_features: Mode A (OOF) and Mode B (X_query), regression + binary."""
    pytest.importorskip("lightgbm")
    from mlframe.feature_engineering.transformer.target_kmeans_codebook import (
        compute_target_kmeans_codebook_features,
    )

    X, y = _make_regression_data(n=180, seed=4)
    spl = KFold(n_splits=3, shuffle=True, random_state=0)
    out_a = compute_target_kmeans_codebook_features(X, y, None, splitter=spl, seed=42, task="regression")
    _assert_finite_shaped(out_a, 180, 5)

    X_q, _ = _make_regression_data(n=40, seed=5)
    out_b = compute_target_kmeans_codebook_features(X, y, X_q, seed=42, task="regression")
    _assert_finite_shaped(out_b, 40, 5)

    Xb, yb = _make_binary_data(n=180, seed=6)
    out_bin = compute_target_kmeans_codebook_features(Xb, yb, None, splitter=spl, seed=42, task="binary")
    _assert_finite_shaped(out_bin, 180, 5)


def test_variance_baseline_both_modes():
    """compute_variance_baseline_features: Mode A (OOF) and Mode B (X_query), regression + binary."""
    pytest.importorskip("lightgbm")
    from mlframe.feature_engineering.transformer.variance_baseline import (
        compute_variance_baseline_features,
    )

    X, y = _make_regression_data(n=180, seed=7)
    spl = KFold(n_splits=3, shuffle=True, random_state=0)
    out_a = compute_variance_baseline_features(X, y, None, splitter=spl, seed=42, task="regression")
    _assert_finite_shaped(out_a, 180, 5)

    X_q, _ = _make_regression_data(n=40, seed=8)
    out_b = compute_variance_baseline_features(X, y, X_q, seed=42, task="regression")
    _assert_finite_shaped(out_b, 40, 5)

    Xb, yb = _make_binary_data(n=180, seed=9)
    out_bin = compute_variance_baseline_features(Xb, yb, None, splitter=spl, seed=42, task="binary")
    _assert_finite_shaped(out_bin, 180, 5)


def test_anchor_attention_both_modes():
    """compute_anchor_attention: Mode A (OOF) and Mode B (X_query)."""
    from mlframe.feature_engineering.transformer.anchor_attention import compute_anchor_attention

    X, y = _make_regression_data(n=150, d=5, seed=10)
    spl = KFold(n_splits=3, shuffle=True, random_state=0)
    n_anchors = 8
    n_cols = n_anchors + 2  # n_anchors similarity columns + 2 pooled aggregate columns (y_mean, y_std)
    out_a = compute_anchor_attention(X, y, None, splitter=spl, seed=42, n_anchors=n_anchors)
    _assert_finite_shaped(out_a, 150, n_cols)

    X_q, _ = _make_regression_data(n=30, d=5, seed=11)
    out_b = compute_anchor_attention(X, y, X_q, seed=42, n_anchors=n_anchors)
    _assert_finite_shaped(out_b, 30, n_cols)

    with pytest.raises(ValueError):
        compute_anchor_attention(X, y, X_q, seed=42, n_anchors=1)
    with pytest.raises(ValueError):
        compute_anchor_attention(X, y, X_q, seed=42, aggregate=())


def test_spectral_attention_both_modes():
    """compute_spectral_attention: Mode A (OOF) and Mode B (X_query)."""
    from mlframe.feature_engineering.transformer.spectral_attention import compute_spectral_attention

    X, y = _make_regression_data(n=150, d=5, seed=12)
    spl = KFold(n_splits=3, shuffle=True, random_state=0)
    n_eigvecs = 4
    out_a = compute_spectral_attention(X, y, None, splitter=spl, seed=42, n_eigvecs=n_eigvecs, k_graph=8)
    _assert_finite_shaped(out_a, 150, n_eigvecs)

    X_q, _ = _make_regression_data(n=30, d=5, seed=13)
    out_b = compute_spectral_attention(X, y, X_q, seed=42, n_eigvecs=n_eigvecs, k_graph=8)
    _assert_finite_shaped(out_b, 30, n_eigvecs)


def test_baseline_disagreement_smote_both_modes():
    """compute_baseline_disagreement_smote_features: Mode A (OOF) and Mode B (X_query), regression + binary."""
    pytest.importorskip("lightgbm")
    from mlframe.feature_engineering.transformer.baseline_disagreement_smote import (
        compute_baseline_disagreement_smote_features,
    )

    X, y = _make_regression_data(n=180, seed=14)
    spl = KFold(n_splits=3, shuffle=True, random_state=0)
    out_a = compute_baseline_disagreement_smote_features(X, y, None, splitter=spl, seed=42, task="regression")
    _assert_finite_shaped(out_a, 180)

    X_q, _ = _make_regression_data(n=40, seed=15)
    out_b = compute_baseline_disagreement_smote_features(X, y, X_q, seed=42, task="regression")
    _assert_finite_shaped(out_b, 40)

    Xb, yb = _make_binary_data(n=180, seed=16)
    out_bin = compute_baseline_disagreement_smote_features(Xb, yb, None, splitter=spl, seed=42, task="binary")
    _assert_finite_shaped(out_bin, 180)


def test_baseline_disagreement_balanced_both_modes():
    """compute_baseline_disagreement_balanced_features: Mode A (OOF) and Mode B (X_query), regression + binary."""
    pytest.importorskip("lightgbm")
    from mlframe.feature_engineering.transformer.baseline_disagreement_balanced import (
        compute_baseline_disagreement_balanced_features,
    )

    X, y = _make_regression_data(n=180, seed=17)
    spl = KFold(n_splits=3, shuffle=True, random_state=0)
    out_a = compute_baseline_disagreement_balanced_features(X, y, None, splitter=spl, seed=42, task="regression")
    _assert_finite_shaped(out_a, 180)

    X_q, _ = _make_regression_data(n=40, seed=18)
    out_b = compute_baseline_disagreement_balanced_features(X, y, X_q, seed=42, task="regression")
    _assert_finite_shaped(out_b, 40)

    Xb, yb = _make_binary_data(n=180, seed=19)
    out_bin = compute_baseline_disagreement_balanced_features(Xb, yb, None, splitter=spl, seed=42, task="binary")
    _assert_finite_shaped(out_bin, 180)


def test_pure_pos_smote_both_modes():
    """compute_pure_pos_smote_features: Mode A (OOF) and Mode B (X_query), default binary task."""
    pytest.importorskip("lightgbm")
    from mlframe.feature_engineering.transformer.pure_pos_smote import (
        compute_pure_pos_smote_features,
    )

    # Needs a real minority class for SMOTE to have something to oversample.
    rng = np.random.default_rng(20)
    n, d = 300, 6
    X = rng.standard_normal((n, d)).astype(np.float32)
    logits = X[:, 0] + 0.5 * X[:, 1]
    y = (logits > np.quantile(logits, 0.8)).astype(np.float32)  # ~20% positive
    spl = KFold(n_splits=3, shuffle=True, random_state=0)

    out_a = compute_pure_pos_smote_features(X, y, None, splitter=spl, seed=42, task="binary")
    _assert_finite_shaped(out_a, n)

    X_q = rng.standard_normal((30, d)).astype(np.float32)
    out_b = compute_pure_pos_smote_features(X, y, X_q, seed=42, task="binary")
    _assert_finite_shaped(out_b, 30)
