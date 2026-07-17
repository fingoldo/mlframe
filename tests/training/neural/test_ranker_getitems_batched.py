"""Regression + biz_value tests for the __getitems__ + passthrough-collate
fast path in mlframe.training.neural.ranker._RankerDataset.

Profile of the 200k-row LTR fuzz combo c0024 attributed 1.69s tottime / 180000
calls to ``_RankerDataset.__getitem__`` (per-row tensor indexing driven by
DataLoader's ``[dataset[i] for i in indices]`` fetch loop). __getitems__
collapses 11 single-row slices per query batch into one batched index;
combined with a passthrough collate_fn the DataLoader skips default_collate's
redundant per-row stack pass.

This test pins:
  (1) __getitem__ and __getitems__ produce equivalent batches
  (2) the passthrough collate correctly unwraps the singleton list
  (3) the passthrough collate falls back to default_collate for the legacy
      per-row __getitem__ output (so older PyTorch keeps working)
  (4) biz_value: batched + passthrough is ≥2x faster than per-row + default
"""

from __future__ import annotations

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from mlframe.training.neural.ranker import _RankerDataset, _ranker_passthrough_collate


def test_getitem_vs_getitems_equivalence():
    """Getitem vs getitems equivalence."""
    rng = np.random.default_rng(20260520)
    X = rng.random((1000, 16), dtype=np.float32)
    y = rng.random(1000, dtype=np.float32)
    ds = _RankerDataset(X, y)

    indices = [3, 17, 42, 99, 256, 511, 777]

    # Per-row __getitem__ path
    rows = [ds[i] for i in indices]
    bx_per_row = torch.stack([r[0] for r in rows])
    by_per_row = torch.stack([r[1] for r in rows])

    # Batched __getitems__ path
    batched = ds.__getitems__(indices)
    assert isinstance(batched, list) and len(batched) == 1, (
        f"__getitems__ must return [(X_batch, y_batch)], got {type(batched).__name__} of len {len(batched) if hasattr(batched, '__len__') else '?'}"
    )
    bx_batched, by_batched = batched[0]

    assert torch.allclose(bx_per_row, bx_batched, atol=0, rtol=0)
    assert torch.allclose(by_per_row, by_batched, atol=0, rtol=0)


def test_passthrough_collate_unwraps_batched_singleton():
    """Passthrough collate unwraps batched singleton."""
    rng = np.random.default_rng(20260520)
    X = rng.random((100, 8), dtype=np.float32)
    y = rng.random(100, dtype=np.float32)
    ds = _RankerDataset(X, y)

    indices = list(range(11))
    batched = ds.__getitems__(indices)
    bx, by = _ranker_passthrough_collate(batched)
    # Shape must be (B, F) / (B,), NOT (1, B, F) / (1, B) -- the default
    # collate's per-row stack would have introduced an unwanted leading 1.
    assert bx.shape == (11, 8), f"unexpected bx shape: {bx.shape}"
    assert by.shape == (11,), f"unexpected by shape: {by.shape}"


def test_passthrough_collate_fallback_for_legacy_per_row():
    """If a caller still feeds a list of (x, y) tuples (legacy per-row path),
    the collate must fall back to default_collate -- preserves backwards
    compatibility with PyTorch < 1.13 or custom callers that bypass
    __getitems__."""
    rng = np.random.default_rng(20260520)
    X = rng.random((100, 8), dtype=np.float32)
    y = rng.random(100, dtype=np.float32)
    ds = _RankerDataset(X, y)

    rows = [ds[i] for i in range(11)]  # list of 11 (x_i, y_i) tuples
    bx, by = _ranker_passthrough_collate(rows)
    assert bx.shape == (11, 8)
    assert by.shape == (11,)


@pytest.mark.biz_transformer
def test_biz_value_batched_path_faster_than_per_row():
    """biz_value: batched + passthrough must be >=2x faster than per-row + default."""
    from torch.utils.data._utils.collate import default_collate

    rng = np.random.default_rng(20260520)
    n_rows, n_features = 200_000, 32
    X = rng.random((n_rows, n_features), dtype=np.float32)
    y = rng.random(n_rows, dtype=np.float32)
    ds = _RankerDataset(X, y)

    batches = [rng.choice(n_rows, 11, replace=False).tolist() for _ in range(50)]

    def per_row(batches):
        """Per row."""
        for indices in batches:
            rows = [ds[i] for i in indices]
            default_collate(rows)

    def batched(batches):
        """Batched."""
        for indices in batches:
            _ranker_passthrough_collate(ds.__getitems__(indices))

    # warmup
    for _ in range(5):
        per_row(batches)
        batched(batches)

    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        per_row(batches)
    t_per = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iters):
        batched(batches)
    t_bat = time.perf_counter() - t0

    speedup = t_per / t_bat
    assert speedup >= 2.0, (
        f"batched __getitems__+passthrough not delivering: speedup={speedup:.2f}x (per_row={t_per * 1000 / iters:.2f}ms, batched={t_bat * 1000 / iters:.2f}ms)"
    )
