"""The cached run starts of the strata equal a fresh count, a new view of the same layout hits the cache, a different layout does not (MRMR-STRATA-RUNS, 2026-11-19)."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no device: nothing to compare on
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters._cmi_null_strata_gpu import _strata_runs


def _layout(n: int, nrank: int, seed: int):
    """A z-sorted dense stratum rank column ``(n, 1)`` as the layout cache holds it."""
    rng = np.random.default_rng(seed)
    z = np.sort(rng.integers(0, nrank, n))
    _, z = np.unique(z, return_inverse=True)
    return cp.asarray(z.astype(np.float64))[:, None], int(z.max()) + 1


def test_runs_equal_a_fresh_count_and_new_views_hit_the_cache():
    """``starts`` is the cumulative stratum size; the view ``z[:, 0]`` taken twice returns the same arrays."""
    z, S = _layout(20000, 3000, 1)
    a_rank, a_starts = _strata_runs(cp, z[:, 0], S)
    b_rank, b_starts = _strata_runs(cp, z[:, 0], S)
    assert a_starts is b_starts and a_rank is b_rank, "a second view of the same layout missed the cache"
    sizes = np.bincount(cp.asnumpy(z[:, 0]).astype(np.int64), minlength=S)
    expected = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)
    np.testing.assert_array_equal(cp.asnumpy(a_starts), expected)
    np.testing.assert_array_equal(cp.asnumpy(a_rank), cp.asnumpy(z[:, 0]).astype(np.int32))


def test_a_different_layout_does_not_reuse_the_entry():
    """Another stratum layout gets its own starts."""
    z1, s1 = _layout(20000, 3000, 2)
    z2, s2 = _layout(20000, 1500, 3)
    _, st1 = _strata_runs(cp, z1[:, 0], s1)
    _, st2 = _strata_runs(cp, z2[:, 0], s2)
    assert st1 is not st2
    assert int(st1.shape[0]) == s1 + 1 and int(st2.shape[0]) == s2 + 1
