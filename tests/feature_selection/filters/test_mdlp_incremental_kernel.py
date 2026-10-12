"""The incremental MDLP best-split kernel searches the same candidates as the reference kernel and finds the same gain."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.supervised_binning import _mdlp_best_split_inc_njit, _mdlp_best_split_njit


@pytest.mark.parametrize("n,k", [(20, 2), (300, 3), (1000, 16), (5000, 8)])
@pytest.mark.parametrize("mode", ["continuous_random_labels", "discrete_x", "signal"])
def test_incremental_kernel_matches_the_reference(n, k, mode):
    """Gain to rounding; the best index is the same unless two candidates tie in gain."""
    rng = np.random.default_rng(n * 7 + k)
    mismatches = 0
    for _ in range(40):
        x = np.sort(rng.standard_normal(n)) if mode != "discrete_x" else np.sort(rng.integers(0, 30, n).astype(float))
        y = rng.integers(0, k, n).astype(np.int64) if mode != "signal" else np.clip((x * 2 + rng.standard_normal(n)).astype(np.int64) % k, 0, k - 1)
        ref = _mdlp_best_split_njit(x, y, k, 5)
        inc = _mdlp_best_split_inc_njit(x, y, k, 5)
        assert abs(ref[1] - inc[1]) < 1e-10 or (ref[1] < 0 and inc[1] < 0)
        assert ref[2] == inc[2]  # h_full is computed the same way
        if ref[0] != inc[0]:
            mismatches += 1
            assert abs(ref[1] - inc[1]) < 1e-10  # a different index is only allowed where the gains tie
    assert mismatches <= 2


def test_incremental_kernel_degenerate_inputs():
    """Too few rows, a constant label and a constant x return the same sentinel as the reference."""
    x = np.arange(6, dtype=np.float64)
    assert _mdlp_best_split_inc_njit(x, np.zeros(6, dtype=np.int64), 1, 5)[0] == -1
    assert _mdlp_best_split_inc_njit(np.arange(8, dtype=np.float64), np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64), 2, 5)[0] in (3, -1)
    assert _mdlp_best_split_inc_njit(np.ones(40), (np.arange(40) % 2).astype(np.int64), 2, 5)[0] == -1
