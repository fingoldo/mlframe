"""The rank-binned resident MI is scored in column blocks sized from the thread's pool room, with the values of one call."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _gpu_resident_rank_bin as R


def test_blocks_equal_one_call():
    """Under a 4 MB slot pool the 4000 x 40 matrix is scored in several blocks and equals the call with room."""
    rng = np.random.default_rng(0)
    X = cp.asarray(rng.standard_normal((4000, 40)))
    y = cp.asarray((rng.standard_normal(4000) > 0).astype(np.int64))
    whole = R.plugin_mi_classif_batch_rank_cuda_resident(X, y, 10)
    if whole is None:
        pytest.skip("rank resident MI is not usable on this host")
    calls = []
    real = R._plugin_mi_classif_batch_rank_block

    def spy(Xc, *a, **kw):
        """Record the block width."""
        calls.append(int(Xc.shape[1]))
        return real(Xc, *a, **kw)

    R._plugin_mi_classif_batch_rank_block = spy
    try:
        slot = cp.cuda.MemoryPool()
        slot.set_limit(size=int(X.nbytes) + 6 * 1024 * 1024)
        with cp.cuda.using_allocator(slot.malloc):
            out = R.plugin_mi_classif_batch_rank_cuda_resident(X, y, 10)
    finally:
        R._plugin_mi_classif_batch_rank_block = real
    assert sum(calls) == 40 and len(calls) > 1
    assert np.array_equal(np.asarray(out), np.asarray(whole))
