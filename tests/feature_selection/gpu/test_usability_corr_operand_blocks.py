"""The usability |corr| warp backend streams an operand matrix that does not fit the card in column blocks."""
import numpy as np
import pytest

from mlframe.feature_selection.filters import batch_pair_usability_corr_gpu as M
from mlframe.feature_selection.filters._pair_operand_blocks import operand_block_limit, pair_chunks_by_operand_block


def test_chunks_cover_every_pair_once_and_respect_the_operand_limit():
    """Each pair lands in exactly one chunk and no chunk touches more than the allowed number of distinct operands."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 30, 400).astype(np.int64)
    b = rng.integers(0, 30, 400).astype(np.int64)
    chunks = pair_chunks_by_operand_block(a, b, 6)
    assert sorted(np.concatenate(chunks).tolist()) == list(range(400))
    for c in chunks:
        assert np.unique(np.concatenate([a[c], b[c]])).size <= 6
    assert len(chunks) > 1


def test_block_limit_halves_until_the_cushion_admits(monkeypatch):
    """The limit is the largest halving of the operand count whose bytes the cushion check accepts, and 0 when two columns do not fit."""
    from mlframe.feature_selection.filters import _fe_gpu_vram

    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda need, **k: need <= 5_000_000)
    assert operand_block_limit(64, 100_000, 0) in (32, 16, 8, 4, 2)
    assert operand_block_limit(64, 100_000, 0) * 100_000 <= 5_000_000
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda need, **k: False)
    assert operand_block_limit(64, 100_000, 0) == 0


@pytest.mark.skipif(not M._CUDA_AVAIL, reason="no CUDA device")
def test_blocked_values_equal_the_unblocked_call_and_the_dispatcher_uses_it(monkeypatch):
    """Pair for pair the blocked table equals the whole-matrix call; with a cushion that refuses the whole matrix the dispatcher returns the blocked backend."""
    from mlframe.feature_selection.filters import _fe_gpu_vram

    rng = np.random.default_rng(1)
    n, k = 1500, 24
    y = rng.standard_normal(n).astype(np.float32)
    ops = rng.standard_normal((k, n)).astype(np.float32)
    a = rng.integers(0, k, 80).astype(np.int64)
    b = rng.integers(0, k, 80).astype(np.int64)
    forms = M.ALL_FORM_IDS
    whole = M.batch_pair_usability_corr_cuda_warp(y, ops, a, b, forms)
    blocked = M.batch_pair_usability_corr_cuda_warp_blocked(y, ops, a, b, forms, 6)
    assert np.array_equal(whole, blocked)
    row = n * 4
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda need, **kw: need <= 8 * (row + 8 * len(forms) * 80) + (1 << 20))
    table, backend = M._dispatch_batch_pair_usability_corr_uncached(y, ops, a, b, forms)
    assert backend == "cuda_warp_blocked"
    assert np.array_equal(table, whole.astype(table.dtype))
