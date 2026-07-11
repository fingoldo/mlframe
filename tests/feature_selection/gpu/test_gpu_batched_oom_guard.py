"""Unit test for the GPU-batched permutation OOM guard's per-row byte cost (2026-07-09 fix).

``mi_direct_gpu_batched`` / ``_streamed`` cap their batch_size to a fraction of free VRAM using
``bytes_per_perm``. Before this fix that constant was ``n * 4`` (only the ``perms_y`` int32 array),
undercounting the true simultaneously-live working set (``rand`` float64 + ``perm_idx`` int64 +
``perms_y`` int32 = 20 bytes/row) by ~5x -- a batch sized "safe" by the old formula could actually need
5x more VRAM than budgeted. Pure arithmetic, no cupy/GPU required.
"""
from __future__ import annotations

from mlframe.feature_selection.filters.gpu import _gpu_batched_bytes_per_perm


def test_bytes_per_perm_reflects_true_20_bytes_per_row():
    assert _gpu_batched_bytes_per_perm(1000) == 20_000
    assert _gpu_batched_bytes_per_perm(1) == 20


def test_bytes_per_perm_no_longer_undercounts_5x():
    # The pre-fix formula was n * 4 (only perms_y int32); the fix is n * 20 (rand f64 + perm_idx i64 +
    # perms_y i32). Pin the ratio directly so a future accidental revert is caught.
    n = 12_345
    old_undercounting_formula = n * 4
    fixed = _gpu_batched_bytes_per_perm(n)
    assert fixed == n * 20
    assert fixed == 5 * old_undercounting_formula


def test_safe_batch_shrinks_5x_under_fixed_formula():
    """End-to-end of the OOM-guard arithmetic itself (the exact expression both call sites use)."""
    free_bytes = 2 * 1024 * 1024 * 1024  # 2 GiB
    n = 100_000

    old_bytes_per_perm = n * 4
    old_safe_batch = max(1, int(free_bytes // 2 // old_bytes_per_perm))

    new_bytes_per_perm = _gpu_batched_bytes_per_perm(n)
    new_safe_batch = max(1, int(free_bytes // 2 // new_bytes_per_perm))

    assert new_safe_batch < old_safe_batch
    ratio = old_safe_batch / new_safe_batch
    assert 4.9 <= ratio <= 5.1, f"expected ~5x reduction in safe_batch, got {ratio:.2f}x ({old_safe_batch} -> {new_safe_batch})"
