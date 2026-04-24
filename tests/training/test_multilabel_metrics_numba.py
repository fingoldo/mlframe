"""Numba multilabel metrics — correctness vs sklearn ground truth + edge cases.

Phase C tests. Cover:
- hamming_loss / subset_accuracy / jaccard_score_multilabel match sklearn
- Parallel variant equivalence (when N*K > 1M)
- Empty-union Jaccard returns 1.0 (well-defined choice)
- Public wrappers reject mismatched shapes BEFORE numba call (Tier 1 #6)
- Auto-reshape from (N,) to (N, 1)
- Dtype coercion (bool, int8, int32, float → uint8)
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics import (
    hamming_loss as fast_hamming_loss,
    subset_accuracy as fast_subset_accuracy,
    jaccard_score_multilabel as fast_jaccard_score,
    _fast_hamming_loss_seq,
    _fast_hamming_loss_par,
    _fast_subset_accuracy_seq,
    _fast_jaccard_score_seq,
)


# ---------------------------------------------------------------------------
# Correctness: numba vs sklearn ground truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed,N,K", [
    (0, 10, 3),
    (1, 100, 5),
    (2, 500, 8),
    (3, 50, 20),
])
def test_hamming_loss_matches_sklearn(seed, N, K):
    from sklearn.metrics import hamming_loss as sk_hamming
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=(N, K)).astype(np.uint8)
    y_pred = rng.integers(0, 2, size=(N, K)).astype(np.uint8)
    fast = fast_hamming_loss(y_true, y_pred)
    sk = sk_hamming(y_true, y_pred)
    assert abs(fast - sk) < 1e-12


@pytest.mark.parametrize("seed,N,K", [
    (0, 10, 3),
    (1, 100, 5),
    (2, 500, 8),
])
def test_subset_accuracy_matches_sklearn(seed, N, K):
    from sklearn.metrics import accuracy_score
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=(N, K)).astype(np.uint8)
    y_pred = rng.integers(0, 2, size=(N, K)).astype(np.uint8)
    fast = fast_subset_accuracy(y_true, y_pred)
    sk = accuracy_score(y_true, y_pred)
    assert abs(fast - sk) < 1e-12


@pytest.mark.parametrize("seed,N,K", [
    (0, 10, 3),
    (1, 100, 5),
    (2, 500, 8),
])
def test_jaccard_matches_sklearn_samples(seed, N, K):
    from sklearn.metrics import jaccard_score as sk_jaccard
    import warnings
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=(N, K)).astype(np.uint8)
    y_pred = rng.integers(0, 2, size=(N, K)).astype(np.uint8)
    # Avoid empty-union rows for direct comparison (sklearn raises on them).
    # Add a 1 in column 0 to guarantee non-empty rows.
    y_true[:, 0] = 1
    y_pred[:, 0] = 1
    fast = fast_jaccard_score(y_true, y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sk = sk_jaccard(y_true, y_pred, average="samples")
    assert abs(fast - sk) < 1e-12


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_jaccard_empty_union_returns_one():
    """Both rows empty → Jaccard 1.0 (defined as 'both empty = perfect')."""
    y_true = np.zeros((1, 3), dtype=np.uint8)
    y_pred = np.zeros((1, 3), dtype=np.uint8)
    assert fast_jaccard_score(y_true, y_pred) == 1.0


def test_jaccard_perfect_match():
    y = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    assert fast_jaccard_score(y, y) == 1.0


def test_subset_accuracy_perfect_match_one_mismatch():
    y_true = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    y_pred = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.uint8)  # last row mismatch
    assert abs(fast_subset_accuracy(y_true, y_pred) - 2 / 3) < 1e-12


def test_hamming_loss_known_value():
    # 2 rows, 3 labels each. Mismatches: row0 col2 (1 mismatch); row1 cols 0,1 (2 mismatches).
    # Total 3 mismatches / 6 elements = 0.5
    y_true = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    y_pred = np.array([[1, 0, 0], [1, 0, 1]], dtype=np.uint8)
    assert abs(fast_hamming_loss(y_true, y_pred) - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# Wrapper input validation (Review Tier 1 #6 — must validate BEFORE numba)
# ---------------------------------------------------------------------------


def test_hamming_loss_shape_mismatch_raises():
    y_true = np.zeros((10, 3), dtype=np.uint8)
    y_pred = np.zeros((10, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="!="):
        fast_hamming_loss(y_true, y_pred)


def test_subset_accuracy_shape_mismatch_raises():
    y_true = np.zeros((10, 3), dtype=np.uint8)
    y_pred = np.zeros((5, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="!="):
        fast_subset_accuracy(y_true, y_pred)


def test_jaccard_3d_input_raises():
    y_true = np.zeros((10, 3, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        fast_jaccard_score(y_true, y_true)


# ---------------------------------------------------------------------------
# 1-D auto-reshape, dtype coercion
# ---------------------------------------------------------------------------


def test_hamming_loss_1d_auto_reshape():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])
    # 1 mismatch / 4 = 0.25
    assert abs(fast_hamming_loss(y_true, y_pred) - 0.25) < 1e-12


@pytest.mark.parametrize("dtype", [np.bool_, np.int8, np.int32, np.float32, np.float64])
def test_dtype_coercion(dtype):
    y_true = np.array([[1, 0], [0, 1]], dtype=dtype)
    y_pred = np.array([[1, 0], [0, 0]], dtype=dtype)
    # 1 mismatch / 4 = 0.25
    assert abs(fast_hamming_loss(y_true, y_pred) - 0.25) < 1e-12


# ---------------------------------------------------------------------------
# Parallel variant equivalence (large frame)
# ---------------------------------------------------------------------------


def test_hamming_par_equivalence_large():
    """At N*K > 1M, parallel auto-selected. Verify same numeric result as seq."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=(50_000, 25)).astype(np.uint8)  # N*K = 1.25M
    y_pred = rng.integers(0, 2, size=(50_000, 25)).astype(np.uint8)
    seq = _fast_hamming_loss_seq(y_true, y_pred)
    par = _fast_hamming_loss_par(y_true, y_pred)
    # Welford-style aggregation: par computes mean of per-row means, seq
    # is total mismatches / total cells. These match exactly when all rows
    # have the same K (which they do here).
    assert abs(seq - par) < 1e-12
