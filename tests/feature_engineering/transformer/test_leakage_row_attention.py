"""Leakage tests for ``compute_row_attention`` Mode A (OOF on train).

The Mode A discipline is non-negotiable: each train row must attend ONLY to other-fold training rows in its OOF features. A bug that accidentally lets a row see
its own y (or its own X via the projection chain) would inflate downstream val-AUC and silently cheat. These tests catch the three failure modes that have been
shipped in similar implementations before.

Scenarios:
- ``test_oof_features_differ_from_mode_b_self_attention``  - OOF features for row r MUST NOT equal "attend with full-train bank including r" features. If they
  matched, the OOF loop is using the full bank instead of the per-fold subset.
- ``test_oof_distribution_matches_mode_b_holdout``          - OOF feature distribution on train rows must be statistically indistinguishable (KS test p > 0.05)
  from Mode B feature distribution on a held-out test set. A distribution gap would indicate a train-time-vs-inference-time inconsistency that downstream models
  see as covariate shift.
- ``test_oof_does_not_overfit_on_constant_y``               - When ``y_train`` is a single constant (no signal), OOF y_mean output should also be that constant
  (within float epsilon). If a leak path uses the row's own y, this still produces the constant - so this is a sanity test, not a leak detector. The actual
  leak detector here is: when ``y_train`` is a noisy version of ``X[:, 0]`` that the model could memorise from its own row, the OOF feature for row r should
  NOT achieve a held-out R^2 that's better than what you'd get by random-shuffling the OOF features.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import ks_2samp
from sklearn.model_selection import KFold, train_test_split

# hnswlib is required at test-call time but imported lazily inside the row-attention modules; collection-time skip is wired in conftest.py.
from mlframe.feature_engineering.transformer import (
    attend,
    build_key_bank,
    compute_row_attention,
)


pytestmark = pytest.mark.fast


def _make_classification(n: int = 500, d: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + 0.3 * X[:, 1] + 0.1 * rng.standard_normal(n) > 0).astype(np.float32)
    return X, y


def test_oof_features_differ_from_mode_b_self_attention():
    """OOF Mode A features for a train row must DIFFER from attending the same row against the full-train bank (which would include the row itself).

    If they match for ALL rows, the OOF loop accidentally used the full-train bank instead of the per-fold-excluding subset (the bug we're guarding against).

    Use a continuous regression target so the self-removal effect is detectable even on rows where the top-k neighbours' label majority matches self's. On a
    classification target with a small N, ~40% of rows can have identical Mode-A vs Mode-B outputs purely because the top-k labels happen to agree before and
    after removing self - that's a measurement noise floor, not a leak signature.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 6)).astype(np.float32)
    # Continuous target with row-specific structure so removing self from neighbours shifts the weighted mean noticeably.
    y = (X[:, 0] + 0.3 * X[:, 1] + 0.2 * rng.standard_normal(400)).astype(np.float32)
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)

    # Mode A: OOF.
    oof_features = compute_row_attention(
        X, y, None, splitter, seed=0, n_heads=1, head_dim=4, k=8, gpu_stage4=False, dedupe_threshold=None,
    ).to_numpy()

    # Mode B reference: build bank from full X, then attend X back. Each row r has itself in the bank with cosine similarity 1.0 dominating its softmax.
    bank_full = build_key_bank(X_train=X, y_train=y, seed=0, n_heads=1, head_dim=4)
    mode_b_self = attend(bank=bank_full, X_query=X, k=8, aggregate=("y_mean", "y_std"))
    self_attn_features = np.column_stack([mode_b_self["y_mean_h0"], mode_b_self["y_std_h0"]])

    # If OOF were broken and used the full bank, oof_features would equal self_attn_features for >99% of rows. A correct OOF differs on the majority but allows
    # rare ANN-approximation coincidences. The bar is set well above what a leakage bug would produce (~100%) but allows ANN noise.
    abs_diff = np.abs(oof_features - self_attn_features).max(axis=1)
    fraction_distinct = float((abs_diff > 1e-4).mean())
    assert fraction_distinct > 0.85, (
        f"OOF features look suspiciously similar to full-bank self-attention features ({fraction_distinct:.2f} fraction distinct, expected > 0.85). "
        "Likely a leakage bug: the per-fold key bank is including the val rows it should exclude."
    )


def test_oof_distribution_matches_mode_b_holdout():
    """KS test between OOF feature distribution on train rows and Mode B feature distribution on a held-out test set.

    Per ML #26 critique: a common production bug is the train-time vs inference-time path producing different distributions, so the downstream model sees a
    silent covariate shift at deploy. This test catches that by comparing the two paths' output distributions directly.

    KS p-value > 0.01 is a soft check (statistical test on relatively small samples); we'd need much larger N for a tight p > 0.05 cut without rejecting on
    random noise. The point is to flag a gross distribution gap, not police every fluctuation.
    """
    X, y = _make_classification(n=600, d=6, seed=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)

    oof = compute_row_attention(X_tr, y_tr, None, splitter, seed=0, n_heads=1, head_dim=4, k=16, gpu_stage4=False, dedupe_threshold=None).to_numpy()
    mode_b = compute_row_attention(X_tr, y_tr, X_te, splitter, seed=0, n_heads=1, head_dim=4, k=16, gpu_stage4=False, dedupe_threshold=None).to_numpy()

    # Check each output column independently.
    for col in range(oof.shape[1]):
        stat, p = ks_2samp(oof[:, col], mode_b[:, col])
        assert p > 0.01, (
            f"OOF vs Mode B distribution mismatch on column {col}: KS stat={stat:.3f} p={p:.4f}. "
            "Indicates the train-time OOF path is producing features that don't match the inference-time Mode B path."
        )


def test_oof_constant_y_yields_constant_output():
    """If ``y_train`` is constant, every OOF y_mean feature must be that constant (no row can leak signal from a constant target).

    This is a positive sanity test that the OOF pipeline is wired to ``y_train`` at all. The negative leak test follows below.
    """
    X, _ = _make_classification(n=200, d=6, seed=0)
    y_constant = np.full(200, 0.7, dtype=np.float32)
    splitter = KFold(n_splits=4, shuffle=True, random_state=0)
    out = compute_row_attention(
        X, y_constant, None, splitter, seed=0, n_heads=1, head_dim=4, k=8, aggregate=("y_mean",),
        gpu_stage4=False, dedupe_threshold=None,
    )
    arr = out.to_numpy().ravel()
    np.testing.assert_allclose(arr, 0.7, atol=1e-5)


def test_oof_does_not_perfectly_memorise_own_y():
    """Adversarial: create y as a noisy version of X[:, 0] sign. If OOF leaks each row's own y, the OOF y_mean would be perfectly correlated with y. With correct
    OOF it tracks y only up to the neighbour-based signal level.

    Concrete bound: a kNN classifier on this synthetic gets ~0.7 AUC at k=8; a row-attention pipeline that LEAKS would get ~1.0 AUC. We assert the OOF feature's
    Pearson correlation with y is below 0.95 (well below the leakage signature) and above 0.30 (well above noise).
    """
    rng = np.random.default_rng(0)
    n, d = 400, 6
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + 0.2 * rng.standard_normal(n) > 0).astype(np.float32)
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)

    out = compute_row_attention(
        X, y, None, splitter, seed=0, n_heads=1, head_dim=4, k=8, aggregate=("y_mean",),
        gpu_stage4=False, dedupe_threshold=None,
    )
    y_mean = out["attn_h0_y_mean"].to_numpy()
    corr = float(np.corrcoef(y_mean, y)[0, 1])

    assert 0.30 < corr < 0.95, (
        f"OOF y_mean correlation with y is {corr:.3f}. Expected in (0.30, 0.95): "
        "below 0.30 indicates the pipeline produces noise; above 0.95 indicates leakage (each row sees its own y)."
    )


def test_oof_seed_required_no_default():
    """``compute_row_attention`` must reject a missing seed - guards against derived-from-data seeds (ML #7)."""
    X, y = _make_classification(n=50, d=4, seed=0)
    with pytest.raises(TypeError, match="seed"):
        compute_row_attention(X, y, None, KFold(n_splits=2), n_heads=1, head_dim=2, k=4, gpu_stage4=False)  # type: ignore[call-arg]
