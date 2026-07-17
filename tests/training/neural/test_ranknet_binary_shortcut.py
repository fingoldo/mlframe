"""Tests for the binary-relevance pair-index short-circuit.

For binary relevance (rel in {0, 1}), the pair set {(i, j) : rel[i] > rel[j]}
is exactly the Cartesian product of positive-rel indices x negative-rel indices.
The short-circuit replaces the (N, N) bool-mask + torch.where path with an
O(N + n_pairs) build via repeat_interleave + repeat.

These tests pin:
  (1) _binary_pair_indices returns torch.where-equivalent pair tensors for
      binary rel (bit-identical, same row-major ordering).
  (2) Returns None for ordinal / multi-grade rel so callers fall back.
  (3) Empty-pair sentinel (all-pos or all-neg) returns matching empty
      tensors with correct dtype + device.
  (4) ranknet_pairwise_loss large-N path (N > _RANKNET_PAIR_CACHE_MAX_N)
      with binary rel produces the same scalar loss as the torch.where path.
  (5) install_pair_index_cache with a large binary-rel query slice attaches
      identical (i_idx, j_idx) tensors as the torch.where path would.
  (6) Small-N path (N < _BINARY_PAIR_SHORTCUT_MIN_N) still hits the
      original torch.where branch -> no regression for typical fuzz N=10.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _set_of_pairs(i_idx, j_idx):
    return set(zip(i_idx.tolist(), j_idx.tolist()))


def test_binary_pair_indices_matches_torch_where_on_binary_rel():
    from mlframe.training.neural._ranker_losses import _binary_pair_indices

    torch.manual_seed(20260528)
    for n, density in [(10, 0.5), (50, 0.3), (200, 0.1), (1000, 0.05)]:
        rel = (torch.rand(n) < density).to(torch.float32)
        expected = torch.where(rel.unsqueeze(1) > rel.unsqueeze(0))
        actual = _binary_pair_indices(rel)
        assert actual is not None, f"binary helper returned None at N={n}"
        ai, aj = actual
        ei, ej = expected
        assert ai.numel() == ei.numel(), f"pair count mismatch at N={n}"
        # torch.where emits row-major (i outer, j inner) and the helper
        # also produces row-major via pos.repeat_interleave(n_neg) outer +
        # neg.repeat(n_pos) inner -- bit-identical sequence.
        assert torch.equal(ai, ei), f"i_idx order mismatch at N={n}"
        assert torch.equal(aj, ej), f"j_idx order mismatch at N={n}"


def test_binary_pair_indices_returns_none_for_ordinal_rel():
    from mlframe.training.neural._ranker_losses import _binary_pair_indices

    # Multi-grade relevance (0, 1, 2, 3, ...): not binary -> None.
    rel = torch.tensor([0, 1, 2, 1, 3, 0], dtype=torch.float32)
    assert _binary_pair_indices(rel) is None
    # Float-valued (0.5 etc.) likewise rejected.
    rel = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    assert _binary_pair_indices(rel) is None


def test_binary_pair_indices_empty_when_single_class():
    from mlframe.training.neural._ranker_losses import _binary_pair_indices

    for vals in ([0, 0, 0], [1, 1, 1, 1]):
        rel = torch.tensor(vals, dtype=torch.float32)
        result = _binary_pair_indices(rel)
        assert result is not None
        i_idx, j_idx = result
        assert i_idx.numel() == 0 and j_idx.numel() == 0
        assert i_idx.dtype == torch.long and j_idx.dtype == torch.long
        assert i_idx.device == rel.device


def test_binary_pair_indices_preserves_device():
    from mlframe.training.neural._ranker_losses import _binary_pair_indices

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rel = torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32, device="cuda")
    result = _binary_pair_indices(rel)
    assert result is not None
    i_idx, j_idx = result
    assert i_idx.device.type == "cuda"
    assert j_idx.device.type == "cuda"


def test_ranknet_loss_large_n_binary_matches_torch_where_path():
    """Large-N binary rel must produce identical scalar loss vs unmodified torch.where path."""
    from mlframe.training.neural._ranker_losses import (
        _RANKNET_PAIR_CACHE_MAX_N,
        ranknet_pairwise_loss,
    )

    n = _RANKNET_PAIR_CACHE_MAX_N + 200  # forces the large-N fall-through
    torch.manual_seed(42)
    rel = (torch.rand(n) < 0.3).to(torch.float32)
    scores = torch.randn(n, requires_grad=True)

    loss_short = ranknet_pairwise_loss(scores, rel)

    # Reference computation via torch.where path (mimics the original code).
    i_ref, j_ref = torch.where(rel.unsqueeze(1) > rel.unsqueeze(0))
    score_diff = scores[i_ref] - scores[j_ref]
    loss_ref = torch.nn.functional.softplus(-score_diff).mean()

    assert torch.allclose(loss_short, loss_ref, atol=1e-7), f"binary shortcut loss {loss_short.item()} != torch.where loss {loss_ref.item()}"


def test_install_pair_index_cache_large_query_binary_matches_torch_where():
    """Large per-query slice with binary rel: cached (i_idx, j_idx) must match torch.where output."""
    pytest.importorskip("pytorch_lightning")
    from mlframe.training.neural.ranker import _RankerDataset, GroupBatchSampler

    n_per_query = 300  # >= _BINARY_PAIR_SHORTCUT_MIN_N
    n_queries = 2
    n = n_per_query * n_queries

    rng = np.random.default_rng(20260528)
    X = rng.standard_normal((n, 3)).astype(np.float32)
    # Binary relevance for each query: ~30% positives.
    y = (rng.random(n) < 0.3).astype(np.float32)
    group_ids = np.repeat(np.arange(n_queries), n_per_query)

    sampler = GroupBatchSampler(group_ids=group_ids, relevance=y, shuffle=False)
    ds = _RankerDataset(X, y)
    ds.install_pair_index_cache(sampler._query_slices)

    for q_slice in sampler._query_slices:
        rel_q = ds.y[torch.as_tensor(q_slice, dtype=torch.long)]
        i_ref, j_ref = torch.where(rel_q.unsqueeze(1) > rel_q.unsqueeze(0))
        key = tuple(q_slice.tolist())
        i_cached, j_cached = ds._pair_idx_by_query[key]
        assert torch.equal(i_cached, i_ref.to(torch.long))
        assert torch.equal(j_cached, j_ref.to(torch.long))


def test_install_pair_index_cache_small_query_uses_torch_where_path():
    """Small queries (N < _BINARY_PAIR_SHORTCUT_MIN_N) must produce identical
    output to the original torch.where path -- pins no regression for typical
    fuzz N=10-15 queries."""
    pytest.importorskip("pytorch_lightning")
    from mlframe.training.neural.ranker import _RankerDataset, GroupBatchSampler

    n_per_query = 10  # below MIN_N=200
    n_queries = 4
    n = n_per_query * n_queries

    rng = np.random.default_rng(20260528)
    X = rng.standard_normal((n, 3)).astype(np.float32)
    y = (rng.random(n) < 0.4).astype(np.float32)
    group_ids = np.repeat(np.arange(n_queries), n_per_query)

    sampler = GroupBatchSampler(group_ids=group_ids, relevance=y, shuffle=False)
    ds = _RankerDataset(X, y)
    ds.install_pair_index_cache(sampler._query_slices)

    for q_slice in sampler._query_slices:
        rel_q = ds.y[torch.as_tensor(q_slice, dtype=torch.long)]
        i_ref, j_ref = torch.where(rel_q.unsqueeze(1) > rel_q.unsqueeze(0))
        key = tuple(q_slice.tolist())
        cached = ds._pair_idx_by_query[key]
        if i_ref.numel() == 0:
            # No-informative-pair sentinel: (None, None).
            assert cached == (None, None)
        else:
            i_cached, j_cached = cached
            assert torch.equal(i_cached, i_ref.to(torch.long))
            assert torch.equal(j_cached, j_ref.to(torch.long))


def test_ranknet_loss_large_n_ordinal_falls_back():
    """Large-N ordinal (multi-grade) rel must still use torch.where path
    and produce the same loss as before this change."""
    from mlframe.training.neural._ranker_losses import (
        _RANKNET_PAIR_CACHE_MAX_N,
        ranknet_pairwise_loss,
    )

    n = _RANKNET_PAIR_CACHE_MAX_N + 100
    torch.manual_seed(7)
    # Multi-grade rel (0..3) -> binary shortcut returns None -> fallback path.
    rel = torch.randint(0, 4, (n,)).to(torch.float32)
    scores = torch.randn(n)

    loss = ranknet_pairwise_loss(scores, rel)
    i_ref, j_ref = torch.where(rel.unsqueeze(1) > rel.unsqueeze(0))
    score_diff = scores[i_ref] - scores[j_ref]
    loss_ref = torch.nn.functional.softplus(-score_diff).mean()

    assert torch.allclose(loss, loss_ref, atol=1e-7)
