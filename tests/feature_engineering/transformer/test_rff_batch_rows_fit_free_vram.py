"""The RFF streaming batch is clamped so its double-buffered device arrays fit in half of the free VRAM."""

import mlframe.feature_engineering.transformer._kernels_cupy as kc


def test_batch_rows_shrink_to_fit_half_the_free_vram():
    """With 1 MiB free and 800 bytes per row, the batch is at most (1 MiB / 2) / 800 rows."""
    per_row = 2 * (100 + 2 * 50) * 4
    assert kc._clamp_rff_batch_rows(100_000, 100, 50, 4, 1 << 20) == (1 << 20) // 2 // per_row


def test_batch_rows_unchanged_when_plenty_of_vram_and_never_below_one():
    """A large free VRAM keeps the requested batch; no free VRAM still allows one row."""
    assert kc._clamp_rff_batch_rows(1000, 10, 5, 4, 1 << 40) == 1000
    assert kc._clamp_rff_batch_rows(1000, 10, 5, 4, 0) == 1000
    assert kc._clamp_rff_batch_rows(1000, 10000, 5000, 8, 1 << 10) == 1
