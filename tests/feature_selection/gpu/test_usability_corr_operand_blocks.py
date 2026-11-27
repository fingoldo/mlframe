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


def test_block_limit_is_the_free_budget_over_the_bytes_of_one_operand(monkeypatch):
    """The limit is the number of operand columns (row bytes plus output share, 1 MB reserve) the free budget holds, capped by the columns in use; 0 when two do not fit; every column without cupy."""
    from mlframe.feature_selection.filters import _fe_gpu_vram

    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_bytes_above_cushion", lambda: (1 << 20) + 5_000_000)
    assert operand_block_limit(64, 100_000, 0) == 50
    assert operand_block_limit(40, 100_000, 0) == 40
    assert operand_block_limit(64, 100_000, 25_000) == 40
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_bytes_above_cushion", lambda: (1 << 20) + 150_000)
    assert operand_block_limit(64, 100_000, 0) == 0
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_bytes_above_cushion", lambda: 0)
    assert operand_block_limit(64, 100_000, 0) == 0
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_bytes_above_cushion", lambda: None)
    assert operand_block_limit(64, 100_000, 0) == 64


def test_block_limit_ignores_the_oom_cooldown_that_the_refusal_of_the_whole_matrix_starts(monkeypatch):
    """A refused whole-matrix check starts the OOM cooldown; the block limit that follows must still read the device (a halving search through the cushion check was declined at every size)."""
    from mlframe.feature_selection.filters import _fe_gpu_vram

    cp = pytest.importorskip("cupy")
    free_total = {"v": (3 << 30, 4 << 30)}
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: free_total["v"])
    monkeypatch.setattr(_fe_gpu_vram, "_own_pool_free_bytes", lambda: 0)
    _fe_gpu_vram.note_gpu_oom()
    try:
        assert _fe_gpu_vram._in_oom_cooldown() and not _fe_gpu_vram.fe_gpu_has_vram_cushion(1 << 20)
        limit = operand_block_limit(1141, 2_000_000, 16)
        budget = _fe_gpu_vram.fe_gpu_bytes_above_cushion()
    finally:
        _fe_gpu_vram._reset_gpu_oom_cooldown()
    # The limit is the share of the free memory above the cushion that one operand column takes, read from the device and not from the cooldown.
    assert budget is not None and limit == (budget - (1 << 20)) // (2_000_000 + 16) < 1141


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


def test_an_all_pairs_table_is_cut_into_tiles_not_into_rows():
    """The all-pairs table of 60 columns under a limit of 20 operands is covered by at most ``nb (nb + 1) / 2`` tiles of the ranked blocks (``nb`` = 6), every pair once, no tile over the limit: a cut in ``(a, b)``
    order gave about 20 pairs per chunk (hundreds of launches, each uploading the columns again)."""
    a, b = np.triu_indices(60, 1)
    chunks = pair_chunks_by_operand_block(a.astype(np.int64), b.astype(np.int64), 20)
    assert sorted(np.concatenate(chunks).tolist()) == list(range(a.size))
    assert len(chunks) <= 21
    assert all(np.unique(np.concatenate([a[c], b[c]])).size <= 20 for c in chunks)
    assert max(len(c) for c in chunks) >= 100


def test_the_pair_cap_splits_a_tile_and_an_odd_limit_keeps_the_operand_bound():
    """``max_pairs`` cuts a tile into pieces of at most that many pairs; an odd operand limit still holds (blocks of ``limit // 2``)."""
    rng = np.random.default_rng(1)
    a = rng.integers(0, 40, 700).astype(np.int64)
    b = rng.integers(0, 40, 700).astype(np.int64)
    chunks = pair_chunks_by_operand_block(a, b, 9, max_pairs=30)
    assert sorted(np.concatenate(chunks).tolist()) == list(range(700))
    assert all(len(c) <= 30 and np.unique(np.concatenate([a[c], b[c]])).size <= 9 for c in chunks)
    assert pair_chunks_by_operand_block(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), 8) == []
    whole = pair_chunks_by_operand_block(a, b, 100)
    assert len(whole) == 1 and len(whole[0]) == 700
