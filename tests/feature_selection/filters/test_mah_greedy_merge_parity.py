"""_greedy_merge_bins (merge only) must return the same table as _greedy_merge_with_history (merge plus step log)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._mah import _greedy_merge_bins, _greedy_merge_with_history


def _table(seed: int, kx: int, ky: int, structured: bool) -> np.ndarray:
    """Joint count table: dependent (block-diagonal ridge) or independent noise."""
    rng = np.random.default_rng(seed)
    if structured:
        base = rng.poisson(2.0, size=(kx, ky)).astype(np.float64)
        base[np.arange(min(kx, ky)), np.arange(min(kx, ky))] += 40.0
        return base
    return rng.poisson(6.0, size=(kx, ky)).astype(np.float64)


@pytest.mark.parametrize("structured", [True, False])
@pytest.mark.parametrize("shape", [(6, 6), (12, 8), (16, 16), (3, 9)])
def test_merge_only_equals_merge_with_history(shape, structured):
    """Identical merged table for dependent and independent joints."""
    joint = _table(shape[0] * 31 + shape[1], shape[0], shape[1], structured)
    merged, row_merges, col_merges = _greedy_merge_with_history(joint)
    np.testing.assert_array_equal(_greedy_merge_bins(joint), merged)
    assert merged.shape[0] == joint.shape[0] - len(row_merges)
    assert merged.shape[1] == joint.shape[1] - len(col_merges)


def test_merge_shrinks_noise_and_preserves_mass():
    """Negative control: the merge is not a no-op on independent noise, and it never changes the total count."""
    joint = _table(1, 16, 16, False)
    merged = _greedy_merge_bins(joint)
    assert merged.size < joint.size
    assert merged.sum() == joint.sum()
