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

from mlframe.metrics.core import (
    hamming_loss as fast_hamming_loss,
    subset_accuracy as fast_subset_accuracy,
    jaccard_score_multilabel as fast_jaccard_score,
    _fast_hamming_loss_seq,
    _fast_hamming_loss_par,
    _fast_subset_accuracy_seq,
    _fast_subset_accuracy_par,
    _fast_jaccard_score_seq,
    _fast_jaccard_score_par,
    _pack_for_bitmap_numpy,
    _pack_for_bitmap_kernel_seq,
    _pack_for_bitmap_kernel_par,
)

# ---------------------------------------------------------------------------
# Bitmap packer: fused njit kernels must be bit-identical to the numpy
# reference (np.packbits-pad64-view) across byte-aligned + non-aligned K,
# all-zero / all-one rows, and the parallel-dispatch threshold.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("K", [16, 17, 23, 24, 31, 32, 33, 40, 63, 64])
def test_pack_for_bitmap_njit_bit_identical_to_numpy(K):
    """Pack for bitmap njit bit identical to numpy."""
    rng = np.random.default_rng(K)
    for arr in (
        (rng.random((1000, K)) < 0.3).astype(np.uint8),
        np.zeros((50, K), dtype=np.uint8),
        np.ones((50, K), dtype=np.uint8),
    ):
        ref = _pack_for_bitmap_numpy(arr)
        assert np.array_equal(_pack_for_bitmap_kernel_seq(arr), ref), f"seq mismatch K={K}"
        assert np.array_equal(_pack_for_bitmap_kernel_par(arr), ref), f"par mismatch K={K}"


def test_pack_for_bitmap_dispatches_to_parallel_njit_above_threshold(monkeypatch):
    """Pack for bitmap dispatches to parallel njit above threshold."""
    from mlframe.metrics import _multilabel_metrics as mlm

    thr = mlm._PARALLEL_MULTILABEL_THRESHOLD
    seen = {"par": 0, "seq": 0}
    orig_par, orig_seq = mlm._pack_for_bitmap_kernel_par, mlm._pack_for_bitmap_kernel_seq

    def spy_par(arr):
        """Spy par."""
        seen["par"] += 1
        return orig_par(arr)

    def spy_seq(arr):
        """Spy seq."""
        seen["seq"] += 1
        return orig_seq(arr)

    monkeypatch.setattr(mlm, "_pack_for_bitmap_kernel_par", spy_par)
    monkeypatch.setattr(mlm, "_pack_for_bitmap_kernel_seq", spy_seq)

    rng = np.random.default_rng(0)
    big = (rng.random((thr, 32)) < 0.2).astype(np.uint8)
    small = (rng.random((10, 32)) < 0.2).astype(np.uint8)
    out_big = mlm._pack_for_bitmap(big)
    out_small = mlm._pack_for_bitmap(small)

    assert seen["par"] == 1 and seen["seq"] == 1
    assert np.array_equal(out_big, _pack_for_bitmap_numpy(big))
    assert np.array_equal(out_small, _pack_for_bitmap_numpy(small))


# ---------------------------------------------------------------------------
# Correctness: numba vs sklearn ground truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seed,N,K",
    [
        (0, 10, 3),
        (1, 100, 5),
        (2, 500, 8),
        (3, 50, 20),
    ],
)
def test_hamming_loss_matches_sklearn(seed, N, K):
    """Hamming loss matches sklearn."""
    from sklearn.metrics import hamming_loss as sk_hamming

    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=(N, K)).astype(np.uint8)
    y_pred = rng.integers(0, 2, size=(N, K)).astype(np.uint8)
    fast = fast_hamming_loss(y_true, y_pred)
    sk = sk_hamming(y_true, y_pred)
    assert abs(fast - sk) < 1e-12


@pytest.mark.parametrize(
    "seed,N,K",
    [
        (0, 10, 3),
        (1, 100, 5),
        (2, 500, 8),
    ],
)
def test_subset_accuracy_matches_sklearn(seed, N, K):
    """Subset accuracy matches sklearn."""
    from sklearn.metrics import accuracy_score

    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=(N, K)).astype(np.uint8)
    y_pred = rng.integers(0, 2, size=(N, K)).astype(np.uint8)
    fast = fast_subset_accuracy(y_true, y_pred)
    sk = accuracy_score(y_true, y_pred)
    assert abs(fast - sk) < 1e-12


@pytest.mark.parametrize(
    "seed,N,K",
    [
        (0, 10, 3),
        (1, 100, 5),
        (2, 500, 8),
    ],
)
def test_jaccard_matches_sklearn_samples(seed, N, K):
    """Jaccard matches sklearn samples."""
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
    """Jaccard perfect match."""
    y = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    assert fast_jaccard_score(y, y) == 1.0


def test_subset_accuracy_perfect_match_one_mismatch():
    """Subset accuracy perfect match one mismatch."""
    y_true = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    y_pred = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.uint8)  # last row mismatch
    assert abs(fast_subset_accuracy(y_true, y_pred) - 2 / 3) < 1e-12


def test_hamming_loss_known_value():
    # 2 rows, 3 labels each. Mismatches: row0 col2 (1 mismatch); row1 cols 0,1 (2 mismatches).
    # Total 3 mismatches / 6 elements = 0.5
    """Hamming loss known value."""
    y_true = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    y_pred = np.array([[1, 0, 0], [1, 0, 1]], dtype=np.uint8)
    assert abs(fast_hamming_loss(y_true, y_pred) - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# Wrapper input validation (Review Tier 1 #6 — must validate BEFORE numba)
# ---------------------------------------------------------------------------


def test_hamming_loss_shape_mismatch_raises():
    """Hamming loss shape mismatch raises."""
    y_true = np.zeros((10, 3), dtype=np.uint8)
    y_pred = np.zeros((10, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="!="):
        fast_hamming_loss(y_true, y_pred)


def test_subset_accuracy_shape_mismatch_raises():
    """Subset accuracy shape mismatch raises."""
    y_true = np.zeros((10, 3), dtype=np.uint8)
    y_pred = np.zeros((5, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="!="):
        fast_subset_accuracy(y_true, y_pred)


def test_jaccard_3d_input_raises():
    """Jaccard 3d input raises."""
    y_true = np.zeros((10, 3, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        fast_jaccard_score(y_true, y_true)


# ---------------------------------------------------------------------------
# 1-D auto-reshape, dtype coercion
# ---------------------------------------------------------------------------


def test_hamming_loss_1d_auto_reshape():
    """Hamming loss 1d auto reshape."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])
    # 1 mismatch / 4 = 0.25
    assert abs(fast_hamming_loss(y_true, y_pred) - 0.25) < 1e-12


@pytest.mark.parametrize("dtype", [np.bool_, np.int8, np.int32, np.float32, np.float64])
def test_dtype_coercion(dtype):
    """Dtype coercion."""
    y_true = np.array([[1, 0], [0, 1]], dtype=dtype)
    y_pred = np.array([[1, 0], [0, 0]], dtype=dtype)
    # 1 mismatch / 4 = 0.25
    assert abs(fast_hamming_loss(y_true, y_pred) - 0.25) < 1e-12


# ---------------------------------------------------------------------------
# Parallel variant equivalence (large frame)
# ---------------------------------------------------------------------------


def test_hamming_par_equivalence_large():
    """At N*K > 1M, parallel auto-selected. Verify same numeric result as seq.

    Regression: the parallel kernels must COMPILE on numba 0.63.1. The pre-fix bodies
    (inner-loop accumulator read after the loop, then combined with a parfor reduction)
    aborted compilation with "unexpected cycle in lookup()" before this ever ran.
    """
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=(50_000, 25)).astype(np.uint8)  # N*K = 1.25M
    y_pred = rng.integers(0, 2, size=(50_000, 25)).astype(np.uint8)
    seq = _fast_hamming_loss_seq(y_true, y_pred)
    par = _fast_hamming_loss_par(y_true, y_pred)
    # Welford-style aggregation: par computes mean of per-row means, seq
    # is total mismatches / total cells. These match exactly when all rows
    # have the same K (which they do here).
    assert abs(seq - par) < 1e-12


def test_jaccard_par_equivalence_large():
    """Parallel Jaccard kernel must compile (numba 0.63.1 parfor cyclic-lookup bug) and
    match the sequential reference. Avoid empty-union rows so seq/par agree exactly."""
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, size=(60_000, 20)).astype(np.uint8)
    y_pred = rng.integers(0, 2, size=(60_000, 20)).astype(np.uint8)
    y_true[:, 0] = 1  # guarantee non-empty union per row
    seq = _fast_jaccard_score_seq(y_true, y_pred)
    par = _fast_jaccard_score_par(y_true, y_pred)
    assert abs(seq - par) < 1e-12


def test_subset_accuracy_par_equivalence_large():
    """Parallel subset-accuracy kernel compiles and matches the sequential reference."""
    rng = np.random.default_rng(11)
    y_true = rng.integers(0, 2, size=(60_000, 8)).astype(np.uint8)
    y_pred = rng.integers(0, 2, size=(60_000, 8)).astype(np.uint8)
    seq = _fast_subset_accuracy_seq(y_true, y_pred)
    par = _fast_subset_accuracy_par(y_true, y_pred)
    assert abs(seq - par) < 1e-12
