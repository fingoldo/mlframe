"""Defensive __getitems__ regression tests for mlframe.training.neural.data.TorchDataset.

The standard mlframe flow doesn't reach __getitems__ — TorchDataModule pairs
TorchDataset(batch_size=B) with DataLoader(batch_size=None), and the DataLoader
iterates integer batch indices (one stacked batch per __getitem__ int call).
This method covers the rare case of an external caller pairing
TorchDataset(batch_size=0) with a custom batch_sampler that yields lists of
indices, akin to the LTR GroupBatchSampler path.

Tests pin:
  (1) sample-mode __getitems__ output equals the per-row __getitem__ + collate path
      (with and without labels, with and without sample_weight)
  (2) batch-mode __getitems__ delegates to __getitem__ unchanged (one batch per int)
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from mlframe.training.neural.data import TorchDataset  # noqa: E402


def _per_row_collated(ds, indices):
    """Mimic torch DataLoader's [ds[i] for i in indices] + default_collate."""
    from torch.utils.data._utils.collate import default_collate

    rows = [ds[i] for i in indices]
    return default_collate(rows)


def test_sample_mode_getitems_matches_per_row_features_only():
    rng = np.random.default_rng(20260520)
    X = rng.random((200, 10), dtype=np.float32)
    ds = TorchDataset(features=X, batch_size=0)

    indices = [3, 7, 17, 42, 99]
    per_row = _per_row_collated(ds, indices)
    via_getitems = ds.__getitems__(indices)

    from torch.utils.data._utils.collate import default_collate

    via_collated = default_collate(via_getitems)
    assert torch.allclose(per_row, via_collated, atol=0, rtol=0)


def test_sample_mode_getitems_matches_per_row_with_labels():
    rng = np.random.default_rng(20260520)
    X = rng.random((200, 10), dtype=np.float32)
    y = rng.integers(0, 3, 200).astype(np.int64)
    ds = TorchDataset(features=X, labels=y, batch_size=0, labels_dtype=torch.int64)

    indices = [3, 7, 17, 42, 99]
    bx_per, by_per = _per_row_collated(ds, indices)
    via = ds.__getitems__(indices)

    from torch.utils.data._utils.collate import default_collate

    bx_via, by_via = default_collate(via)
    assert torch.allclose(bx_per, bx_via, atol=0, rtol=0)
    assert torch.equal(by_per, by_via)


def test_sample_mode_getitems_matches_per_row_with_sample_weight():
    rng = np.random.default_rng(20260520)
    X = rng.random((200, 10), dtype=np.float32)
    y = rng.random(200, dtype=np.float32)
    w = rng.random(200, dtype=np.float32)
    ds = TorchDataset(features=X, labels=y, sample_weight=w, batch_size=0)

    indices = [3, 7, 17, 42, 99]
    bx_per, by_per, bw_per = _per_row_collated(ds, indices)
    via = ds.__getitems__(indices)

    from torch.utils.data._utils.collate import default_collate

    bx_via, by_via, bw_via = default_collate(via)
    assert torch.allclose(bx_per, bx_via, atol=0, rtol=0)
    assert torch.allclose(by_per, by_via, atol=0, rtol=0)
    assert torch.allclose(bw_per, bw_via, atol=0, rtol=0)


def test_batch_mode_getitems_delegates_per_item():
    """In batch_size>0 mode, __getitem__(int) already returns a stacked batch.
    __getitems__ must NOT double-batch — it returns one entry per integer
    index, each being a full stacked batch tuple."""
    rng = np.random.default_rng(20260520)
    X = rng.random((200, 10), dtype=np.float32)
    y = rng.random(200, dtype=np.float32)
    ds = TorchDataset(features=X, labels=y, batch_size=32)

    batch_indices = [0, 1, 2]
    out = ds.__getitems__(batch_indices)
    assert len(out) == 3
    for entry, bi in zip(out, batch_indices):
        bx, by = entry
        # batch_size=32 → bx shape (32, 10), by shape (32,)
        assert bx.shape == (32, 10), f"batch {bi}: bx shape {bx.shape}"
        assert by.shape == (32,)
        # Verify slicing matches the documented semantics
        start, end = bi * 32, (bi + 1) * 32
        assert torch.allclose(bx, ds._extract(ds.features, slice(start, end)))
