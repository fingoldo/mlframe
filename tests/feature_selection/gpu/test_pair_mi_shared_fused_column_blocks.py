"""The shared-fused pair-MI backend gathers only the columns its pairs reference and streams them in blocks when the card cannot take them all."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from mlframe.feature_selection.filters import _batch_pair_mi_cuda_shared_fused as S

pytestmark = pytest.mark.gpu


def _case(n=3000, ncol=30, npairs=80, nb=5, ky=3, seed=0):
    """Integer codes for ``ncol`` columns, random pairs, labels and their frequencies."""
    rng = np.random.default_rng(seed)
    fd = rng.integers(0, nb, (n, ncol)).astype(np.int32)
    y = rng.integers(0, ky, n).astype(np.int32)
    a = rng.integers(0, ncol, npairs).astype(np.int64)
    b = rng.integers(0, ncol, npairs).astype(np.int64)
    return fd, a, b, np.full(ncol, nb, np.int32), y, np.bincount(y, minlength=ky) / n


def test_blocked_values_equal_the_single_upload(monkeypatch):
    """With a cushion that refuses the whole set and a block limit of 8 columns the MI vector equals the one from a single upload."""
    from mlframe.feature_selection.filters import _fe_gpu_vram, _pair_operand_blocks

    args = _case()
    whole = S.batch_pair_mi_cuda_shared_fused(*args)
    calls = []
    real = S._launch_pairs
    monkeypatch.setattr(S, "_launch_pairs", lambda *a, **k: calls.append(len(a[1])) or real(*a, **k))
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda need, **k: False)
    monkeypatch.setattr(_pair_operand_blocks, "operand_block_limit", lambda *a, **k: 8)
    blocked = S.batch_pair_mi_cuda_shared_fused(*args)
    assert len(calls) > 1 and sum(calls) == len(args[1])
    assert np.allclose(blocked, whole, rtol=0, atol=1e-15)


def test_only_referenced_columns_cross_the_bus_when_blocked(monkeypatch):
    """In the blocked path the host gathers a chunk's columns before the upload: the matrix handed to the bus never has more columns than the block limit."""
    from mlframe.feature_selection.filters import _fe_gpu_vram, _pair_operand_blocks
    from mlframe.feature_selection.filters.info_theory import shared

    widths = []
    real = shared.host_codes_for_upload
    monkeypatch.setattr(shared, "host_codes_for_upload", lambda x, *a, **k: widths.append(x.shape[1]) or real(x, *a, **k))
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda need, **k: False)
    monkeypatch.setattr(_pair_operand_blocks, "operand_block_limit", lambda *a, **k: 6)
    fd, a, b, nbins, y, fy = _case(ncol=40, npairs=30)
    S.batch_pair_mi_cuda_shared_fused(fd, a, b, nbins, y, fy)
    assert widths and max(widths) <= 6 < fd.shape[1]
