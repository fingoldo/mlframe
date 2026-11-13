"""``pack_cls`` builds the per-permutation class bit-sets with one warp per 64-row word (two ballots); it must match a numpy packing for any row count and class count."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_DEVICE = False
pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")


def _reference(y_all, n_y, words):
    """Little-endian bit r of word w is row 64 w + r; the count is the popcount of the class row."""
    n_perm, n = y_all.shape
    cls = np.zeros((n_perm, n_y, words), dtype=np.uint64)
    for p in range(n_perm):
        for c in range(n_y):
            bits = np.zeros(words * 64, dtype=np.uint8)
            bits[:n] = y_all[p] == c
            cls[p, c] = np.packbits(bits, bitorder="little").view(np.uint64)
    return cls, (y_all[:, :, None] == np.arange(n_y)).sum(axis=1)


@pytest.mark.parametrize("n,n_y,n_perm", [(1, 2, 3), (63, 2, 2), (64, 2, 2), (65, 3, 4), (4097, 2, 5), (10_001, 5, 3)])
def test_pack_cls_matches_numpy(n, n_y, n_perm):
    """Row counts around the word and warp boundaries, with two to five classes, give the same bit-sets and class counts."""
    from mlframe.feature_selection.filters._batch_mi_noise_gate_bits import _pack_module
    from pyutilz.performance.gpu_hw_launch import block_size_for

    rng = np.random.default_rng(n * 7 + n_y)
    y_all = rng.integers(0, n_y, (n_perm, n)).astype(np.int32)
    words = (n + 63) // 64
    mod = _pack_module(cp, 4, n_y, "signed char")
    kern = mod.get_function("pack_cls")
    cls = cp.empty((n_perm, n_y, words), dtype=cp.uint64)
    cnt = cp.zeros((n_perm, n_y), dtype=cp.int64)
    block = block_size_for(kern, work_items=words * 32)
    kern(((words * 32 + block - 1) // block, n_perm), (block,), (cp.asarray(y_all), np.int64(n), np.int32(words), np.int32(n_y), cls, cnt))
    want_cls, want_cnt = _reference(y_all, n_y, words)
    np.testing.assert_array_equal(cls.get(), want_cls)
    np.testing.assert_array_equal(cnt.get(), want_cnt)
