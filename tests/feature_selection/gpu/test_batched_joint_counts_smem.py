"""The shared-memory joint-histogram kernel gives the global-atomics kernel's counts."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _fe_batched_mi_cmi as M


@pytest.mark.parametrize("n,k,kx,kb", [(5000, 7, 10, 4), (20011, 33, 6, 2), (3000, 70, 12, 9)])
def test_shared_counts_equal_global_counts(n, k, kx, kb, monkeypatch):
    """Counts agree with the bincount reference and with the global-atomics kernel, including a ragged last column tile and row chunk."""
    r = np.random.default_rng(0)
    X = r.integers(0, kx, (n, k)).astype(np.int64)
    b = r.integers(0, kb, n).astype(np.int64)
    ref = np.zeros((k, kx * kb), dtype=np.int64)
    for c in range(k):
        ref[c] = np.bincount(X[:, c] * kb + b, minlength=kx * kb)
    monkeypatch.setenv("MLFRAME_HIST_SMEM", "1")
    assert M._hist_smem_tile(kx * kb, k) > 0
    shared = M._batched_joint_counts2(cp.asarray(X), cp.asarray(b), kx, kb).get()
    monkeypatch.setenv("MLFRAME_HIST_SMEM", "0")
    assert M._hist_smem_tile(kx * kb, k) == 0
    plain = M._batched_joint_counts2(cp.asarray(X), cp.asarray(b), kx, kb).get()
    np.testing.assert_array_equal(shared, ref)
    np.testing.assert_array_equal(plain, ref)


def test_many_cells_stay_on_the_global_kernel():
    """Above the measured crossover the dispatch keeps the global-atomics kernel."""
    assert M._hist_smem_tile(M._HIST_SMEM_MAX_CELLS + 1, 50) == 0
