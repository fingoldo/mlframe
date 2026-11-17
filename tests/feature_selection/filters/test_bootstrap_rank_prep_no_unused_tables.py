"""The resample tables of the rank-MI bootstrap carry only what the kernels read (no per-row draw-position tables)."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._bootstrap_rank_mi_njit import _resample_tables, bootstrap_rank_mi


def test_resample_tables_hold_no_unused_draw_position_tables():
    """``rowptr`` / ``poslist`` were built per resample (an argsort each) and never read."""
    rng = np.random.default_rng(0)
    reps = [rng.integers(0, 200, 160).astype(np.int64) for _ in range(3)]
    prep = _resample_tables(reps, rng.integers(0, 2, 200), 200, 10, True)
    assert "rowptr" not in prep and "poslist" not in prep
    assert set(prep) >= {"counts_t", "cum", "idx", "binpos"}


def test_bootstrap_rank_mi_tied_float32_matches_float64_kernel():
    """The sorted walk with tied values matches the generic float64 kernel (same ranks, same tie rule)."""
    rng = np.random.default_rng(1)
    arr = np.round(rng.normal(size=(300, 4)), 1).astype(np.float32)
    y = rng.integers(0, 2, 300)
    reps = [np.sort(rng.integers(0, 300, 240)).astype(np.int64) for _ in range(4)]
    mi32, ok32 = bootstrap_rank_mi(arr, y, reps, 5, try_tied_device=False)
    mi64, ok64 = bootstrap_rank_mi(arr.astype(np.float64), y, reps, 5)
    assert ok32.all() and ok64.all()
    np.testing.assert_allclose(mi32, mi64, atol=1e-12)
