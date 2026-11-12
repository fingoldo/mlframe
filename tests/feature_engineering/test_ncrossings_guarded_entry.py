"""``compute_ncrossings`` enters the numba mark-parallel kernel only through the process-wide parallel-entry guard.

numaggs fans out over column chunks in a thread pool, and every chunk reaches this prange kernel. Without the guard two
threads can be inside the workqueue threading layer at once, which aborts the process on macOS.
"""

from __future__ import annotations

import contextlib

import numpy as np

from mlframe.feature_engineering import _numerical_counts as counts


def _spy_guard(monkeypatch):
    """Replace the guard with a counting context manager and return the list that records each entry."""
    entries: list[int] = []

    @contextlib.contextmanager
    def _counting_guard():
        """Record one entry and yield."""
        entries.append(1)
        yield

    monkeypatch.setattr(counts, "parallel_kernel_entry", _counting_guard)
    return entries


def test_int32_path_enters_the_parallel_guard_and_matches_the_serial_kernel(monkeypatch):
    """The int32 dispatch goes through the guard once per call and returns the same counts as the serial kernel."""
    entries = _spy_guard(monkeypatch)
    rng = np.random.default_rng(0)
    arr = rng.normal(size=5000)
    marks = np.quantile(arr, [0.1, 0.5, 0.9])

    got = counts.compute_ncrossings(arr, marks, dtype=np.int32)

    assert len(entries) == 1
    np.testing.assert_array_equal(got, counts._compute_ncrossings_serial(arr, marks, np.int32))


def test_serial_dtype_path_does_not_take_the_parallel_guard(monkeypatch):
    """Non-int32 output uses the serial kernel, which has no prange region, so it must not hold the guard."""
    entries = _spy_guard(monkeypatch)
    rng = np.random.default_rng(1)
    arr = rng.normal(size=500)
    marks = np.quantile(arr, [0.25, 0.75])

    counts.compute_ncrossings(arr, marks, dtype=np.float64)

    assert entries == []
