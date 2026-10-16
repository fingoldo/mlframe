"""The row-attention call releases the pooled pinned host buffers when it is asked to release memory afterwards."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer import _kernels_cupy as kc
from mlframe.feature_engineering.transformer.row_attention import compute_row_attention


def _sentinel_pinned_buffer() -> None:
    """Plant one entry in the pinned-buffer pool so a release has something to drop."""
    with kc._PINNED_BUFFERS_LOCK:
        kc._PINNED_BUFFERS[("sentinel", 1)] = object()


def test_release_memory_after_drops_pinned_buffers() -> None:
    """With release_memory_after left at its default, the pinned-buffer pool is empty when the call returns."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4)).astype(np.float32)
    y = rng.normal(size=60).astype(np.float32)
    _sentinel_pinned_buffer()
    compute_row_attention(X, y, None, KFold(n_splits=3, shuffle=True, random_state=0), seed=0, n_heads=1, head_dim=2, k=4, gpu_stage4=False)
    assert len(kc._PINNED_BUFFERS) == 0


def test_no_release_keeps_pinned_buffers() -> None:
    """With release_memory_after=False, the pinned-buffer pool is left alone."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4)).astype(np.float32)
    y = rng.normal(size=60).astype(np.float32)
    kc.clear_pinned_buffers()
    _sentinel_pinned_buffer()
    compute_row_attention(X, y, None, KFold(n_splits=3, shuffle=True, random_state=0), seed=0, n_heads=1, head_dim=2, k=4, gpu_stage4=False, release_memory_after=False)
    assert len(kc._PINNED_BUFFERS) == 1
    kc.clear_pinned_buffers()
