"""Regression tests for the precomputed pair-index fastpath added in iter112.

c0083 profile attributed 138s self-time / 540k calls (= 256us/call) to
``ranknet_pairwise_loss``. Microbench showed ~80us of that was the cache-key
build (tuple(rel.tolist()) forces a GPU->CPU sync each batch on cuda). The
fix builds per-query (i_idx, j_idx) tensors CPU-side in
``_RankerDataset.install_pair_index_cache`` so ``__getitems__`` can attach
them to each batch and the loss skips the cache work entirely.

These tests pin:
  (1) Dataset attaches pair tensors when the cache is installed
  (2) Collate passes the 4-tuple through unchanged
  (3) ``ranknet_pairwise_loss_precomputed`` matches the original on equivalent
      inputs
  (4) MLPRanker.fit uses the fastpath end-to-end without crashing
  (5) Multilabel y short-circuits to the legacy path
  (6) Cached path with no informative pair returns 0 (no crash)
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")


def test_install_pair_index_cache_builds_per_query_entries():
    from mlframe.training.neural.ranker import _RankerDataset, GroupBatchSampler

    rng = np.random.default_rng(20260521)
    X = rng.standard_normal((40, 3)).astype(np.float32)
    y = rng.integers(0, 3, size=40).astype(np.float32)
    group_ids = np.repeat(np.arange(4), 10)
    sampler = GroupBatchSampler(group_ids=group_ids, relevance=y, shuffle=False)
    ds = _RankerDataset(X, y)
    ds.install_pair_index_cache(sampler._query_slices)
    assert ds._pair_idx_by_query is not None
    assert len(ds._pair_idx_by_query) == len(sampler._query_slices)
    # Sanity: a randomly chosen entry's pair tensors should be (n_pairs,) longs.
    # Key scheme: tuple(int(v) for v in indices) (iter115 swap from
    # np.asarray.tobytes for ~4.5x faster per-call key build).
    for indices in sampler._query_slices:
        key = tuple(int(v) for v in indices)
        i_idx, j_idx = ds._pair_idx_by_query[key]
        if i_idx is None:
            # All-equal relevance -> no informative pair
            assert j_idx is None
            continue
        assert i_idx.dtype == torch.long
        assert j_idx.dtype == torch.long
        assert i_idx.shape == j_idx.shape


def test_getitems_returns_four_tuple_when_cache_installed():
    from mlframe.training.neural.ranker import _RankerDataset, GroupBatchSampler

    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4)).astype(np.float32)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0] * 3, dtype=np.float32)
    group_ids = np.repeat(np.arange(3), 10)
    sampler = GroupBatchSampler(group_ids=group_ids, relevance=y, shuffle=False)
    ds = _RankerDataset(X, y)
    ds.install_pair_index_cache(sampler._query_slices)

    indices = list(sampler._query_slices[0])
    batch = ds.__getitems__(indices)
    assert isinstance(batch, list) and len(batch) == 1
    inner = batch[0]
    assert len(inner) == 4
    X_b, y_b, i_idx, j_idx = inner
    assert X_b.shape == (10, 4)
    assert y_b.shape == (10,)
    assert i_idx.numel() > 0
    assert j_idx.numel() == i_idx.numel()


def test_collate_passes_four_tuple_through():
    from mlframe.training.neural.ranker import _ranker_passthrough_collate

    X = torch.randn(8, 3)
    y = torch.randint(0, 3, (8,)).float()
    i_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    j_idx = torch.tensor([3, 4, 5], dtype=torch.long)
    batch = [(X, y, i_idx, j_idx)]
    out = _ranker_passthrough_collate(batch)
    assert isinstance(out, tuple) and len(out) == 4
    assert torch.equal(out[0], X)
    assert torch.equal(out[2], i_idx)


def test_precomputed_loss_matches_original():
    from mlframe.training.neural.ranker import (
        ranknet_pairwise_loss, ranknet_pairwise_loss_precomputed,
    )

    torch.manual_seed(42)
    scores = torch.randn(12, requires_grad=True)
    rel = torch.tensor([3., 0., 2., 1., 3., 0., 2., 1., 3., 0., 2., 1.])
    loss_a = ranknet_pairwise_loss(scores, rel)
    i_idx, j_idx = torch.where(rel.unsqueeze(1) > rel.unsqueeze(0))
    loss_b = ranknet_pairwise_loss_precomputed(scores, i_idx, j_idx)
    assert torch.allclose(loss_a, loss_b, atol=1e-7), (
        f"precomputed {loss_b.item()} != original {loss_a.item()}"
    )


def test_precomputed_loss_zero_on_empty_pairs():
    from mlframe.training.neural.ranker import ranknet_pairwise_loss_precomputed

    scores = torch.randn(5, requires_grad=True)
    # No informative pairs -> sentinel None values
    loss = ranknet_pairwise_loss_precomputed(scores, None, None)
    assert loss.item() == 0.0


def test_mlp_ranker_fit_uses_fastpath_end_to_end():
    from mlframe.training.neural.ranker import MLPRanker

    rng = np.random.default_rng(20260521)
    X = rng.standard_normal((100, 4)).astype(np.float32)
    y = rng.integers(0, 4, size=100).astype(np.float32)
    group_ids = np.repeat(np.arange(10), 10)
    model = MLPRanker(n_estimators=2, early_stopping_patience=None, verbose=0, seed=1)
    model.fit(X, y, group_ids)
    pred = model.predict(X)
    assert pred.shape == (100,)
    assert np.all(np.isfinite(pred))


def test_batch_cache_returns_identical_tuple_across_calls():
    """iter357: install_pair_index_cache must pre-build (X_slice, y_slice,
    i_idx, j_idx) so __getitems__ reduces to a single dict lookup. Calling
    __getitems__ twice with the same indices must return the SAME tuple
    object (identity), not just equal tensors -- this is what proves the
    cache path is hit and the per-batch X[idx]/y[idx] work is skipped."""
    from mlframe.training.neural.ranker import _RankerDataset, GroupBatchSampler

    rng = np.random.default_rng(20260526)
    X = rng.standard_normal((40, 3)).astype(np.float32)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0] * 4, dtype=np.float32)
    group_ids = np.repeat(np.arange(4), 10)
    sampler = GroupBatchSampler(group_ids=group_ids, relevance=y, shuffle=False)
    ds = _RankerDataset(X, y)
    ds.install_pair_index_cache(sampler._query_slices)
    assert ds._batch_by_query is not None
    assert len(ds._batch_by_query) == len(sampler._query_slices)

    indices = list(sampler._query_slices[0])
    batch_1 = ds.__getitems__(indices)
    batch_2 = ds.__getitems__(indices)
    # Identity check: same tuple object returned from the cache.
    assert batch_1[0] is batch_2[0], (
        "batch cache miss: __getitems__ returned a fresh tuple instead of "
        "the cached one"
    )
    # Sanity: contents match what the legacy non-cached path produces.
    idx_tensor = torch.as_tensor(indices, dtype=torch.long)
    assert torch.equal(batch_1[0][0], ds.X[idx_tensor])
    assert torch.equal(batch_1[0][1], ds.y[idx_tensor])


def test_multilabel_y_skips_cache_installation():
    from mlframe.training.neural.ranker import _RankerDataset

    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3)).astype(np.float32)
    y_multilabel = rng.integers(0, 2, size=(20, 3)).astype(np.float32)
    ds = _RankerDataset(X, y_multilabel)
    # Pass a fake query slice list with a slice that would resolve to a 2-D y
    ds.install_pair_index_cache([np.arange(20, dtype=np.int64)])
    # Multilabel branch -> entry skipped, dict stays small
    assert ds._pair_idx_by_query is not None
    # No key present -> getitems returns the 2-tuple path
    batch = ds.__getitems__(list(range(20)))
    assert len(batch[0]) == 2
