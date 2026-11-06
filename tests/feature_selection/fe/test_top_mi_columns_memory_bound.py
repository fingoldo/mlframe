"""``_top_mi_num_cols`` ranks the columns in blocks sized from the free host memory, so a wide frame is never copied whole (the L-27 MemoryError at 10M rows)."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _binned_numeric_agg_batches as batches
from mlframe.feature_selection.filters._extra_fe_families import _top_mi_num_cols


def _frame(n: int, p: int):
    """A float32 frame whose first columns carry the target, and a binary target."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(n, p)).astype(np.float32), columns=[f"x{i}" for i in range(p)])
    y = (X["x3"].to_numpy() + 0.5 * X["x7"].to_numpy() + rng.normal(size=n) > 0).astype(int)
    return X, y


def test_block_width_follows_the_free_memory_and_the_ranking_does_not_change(monkeypatch):
    """With 4 MB of free host memory the blocks are a few columns wide; the top columns equal the unconstrained ranking."""
    X, y = _frame(60_000, 40)
    cols = list(X.columns)
    reference = _top_mi_num_cols(X, cols, y, 6)
    monkeypatch.setattr(batches, "_free_host_bytes", lambda: 4 << 20)
    assert _top_mi_num_cols(X, cols, y, 6) == reference


def test_peak_allocation_stays_below_the_frame_size_when_memory_is_short(monkeypatch):
    """The peak allocation growth of the ranking is under half of the frame's bytes with 8 MB of free host memory (a whole-frame copy is 100 percent, a float64 one 200)."""
    X, y = _frame(200_000, 40)
    frame_bytes = X.shape[0] * X.shape[1] * 4
    monkeypatch.setattr(batches, "_free_host_bytes", lambda: 8 << 20)
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    try:
        _top_mi_num_cols(X, list(X.columns), y, 6)
        peak = tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()
    assert peak < 0.5 * frame_bytes, (peak, frame_bytes)
