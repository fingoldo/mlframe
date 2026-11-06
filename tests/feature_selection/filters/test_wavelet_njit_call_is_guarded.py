"""The wavelet leg scan enters its prange kernel through the numba parallel guard.

The scan runs on the enrichment thread pool. Numba's workqueue threading layer (the macOS default) aborts the process when two threads
are inside a parallel region at once, so the call must go through ``parallel_kernel_entry``. Without the guard, macOS shards aborted with
"Concurrent access has been detected" on master.
"""

from __future__ import annotations

import contextlib
import importlib

import numpy as np

mod = importlib.import_module("mlframe.feature_selection.filters._wavelet_basis_fe")


def test_wavelet_leg_scan_enters_the_parallel_guard(monkeypatch):
    """Running the leg scan records at least one entry into the parallel-kernel guard."""
    entries = []

    @contextlib.contextmanager
    def _spy_guard():
        """Count one entry into the guard and yield."""
        entries.append(1)
        yield

    monkeypatch.setattr(mod, "parallel_kernel_entry", _spy_guard, raising=False)
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 1.0, 800)
    y = (x > 0.5).astype(np.float64)

    mod._select_wavelet_legs(x, y, 0.0, 1.0, max_scale=3)

    assert len(entries) >= 1, "the prange leg kernel was called without the parallel guard"
