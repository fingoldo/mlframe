"""The Fleuret permutation block runs inside the numba parallel guard.

The block is a prange kernel and runs on bootstrap-stability threads when MRMR fits concurrently. The recheck must enter the guard around each
block call, and the test records whether the guard is active at the moment the block kernel is called.
"""

from __future__ import annotations

import contextlib
import importlib

import numpy as np

fp = importlib.import_module("mlframe.feature_selection.filters._fleuret_confidence_prange")


def test_fleuret_block_is_called_inside_the_parallel_guard(monkeypatch):
    """Every block-kernel call happens while a parallel-kernel guard is active on the calling thread."""
    active = {"depth": 0}
    seen = []

    @contextlib.contextmanager
    def _spy_guard():
        """Track nesting depth of the guard while the body runs."""
        active["depth"] += 1
        try:
            yield
        finally:
            active["depth"] -= 1

    def _fake_block(data, factors_nbins, x, y, selected, b0, b1, *rest):
        """Record the guard depth at call time and return all-False flags for the block."""
        seen.append(active["depth"])
        return np.zeros(b1 - b0, dtype=bool)

    monkeypatch.setattr(fp, "parallel_kernel_entry", _spy_guard, raising=False)
    monkeypatch.setattr(fp, "_block_flags", _fake_block)

    nfailed, nchecked = fp.fleuret_confidence_prange(
        None, None, np.arange(12, dtype=np.int64), np.arange(12, dtype=np.int64) % 2, np.array([0], dtype=np.int64),
        npermutations=4, bootstrapped_gain=0.0, max_failed=10, nexisting=1, block=2,
    )

    assert seen, "the permutation block was never called"
    assert all(depth > 0 for depth in seen), f"block called outside the parallel guard (depths {seen})"
    assert (nfailed, nchecked) == (0, 4)
