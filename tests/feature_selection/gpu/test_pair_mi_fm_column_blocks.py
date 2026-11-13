"""The feature-major pair-MI backend streams the pairs in column blocks when the card cannot take the referenced columns whole."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from mlframe.feature_selection.filters import _batch_pair_mi_cuda_fm as F

pytestmark = pytest.mark.gpu


def _case(n=3001, ncol=30, npairs=80, nb=5, ky=3, seed=0):
    """Byte codes of ``ncol`` columns (an odd row count, so the padding path runs), random pairs and labels."""
    rng = np.random.default_rng(seed)
    fd = rng.integers(0, nb, (n, ncol)).astype(np.uint8)
    y = rng.integers(0, ky, n).astype(np.uint8)
    a = rng.integers(0, ncol, npairs).astype(np.int64)
    b = rng.integers(0, ncol, npairs).astype(np.int64)
    return fd, a, b, np.full(ncol, nb, np.int32), y, np.bincount(y, minlength=ky) / n


def test_blocked_values_equal_the_single_upload(monkeypatch):
    """With a refused cushion and a block limit of 8 columns the MI vector equals the single-upload one, and only a chunk's columns are uploaded."""
    from mlframe.feature_selection.filters import _fe_gpu_vram, _pair_operand_blocks

    args = _case()
    whole = F.batch_pair_mi_cuda_feature_major(*args)
    calls = []
    real = F._launch_pairs
    monkeypatch.setattr(F, "_launch_pairs", lambda fd, pa, *a, **k: calls.append((fd.shape[1], len(pa), k.get("gather_on_host"))) or real(fd, pa, *a, **k))
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda need, **k: False)
    monkeypatch.setattr(_pair_operand_blocks, "operand_block_limit", lambda *a, **k: 8)
    blocked = F.batch_pair_mi_cuda_feature_major(*args)
    assert len(calls) > 1 and sum(c[1] for c in calls) == len(args[1])
    assert all(c[2] for c in calls)
    assert np.allclose(blocked, whole, rtol=0, atol=1e-15)
