"""The pair family gate's numba prange kernels must run under the process-wide parallel-kernel guard.

``family_pass_batches`` is reached from the FE pair sweep's joblib ``backend="threading"`` workers. Two threads inside one
``parallel=True`` region aborts the process on macOS (numba workqueue layer), so every prange kernel the gate calls has to be
entered while the thread holds ``parallel_kernel_entry()``. Linux does not show the abort, so these tests assert the guard
directly rather than the platform message.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from mlframe._numba_parallel_guard import _entry_lock
from mlframe.feature_selection.filters import _family_maxt_gate as gate

_KERNELS = ("mi_all_columns_njit", "family_null_max_njit", "null_max_bitsets", "quantile_codes_block", "_occupied_bins")


def _family_batches(seed: int, n_batches: int = 3, rows: int = 400):
    """Small random candidate-code batches plus a label vector, shaped like one pair family."""
    rng = np.random.default_rng(seed)
    batches = [rng.integers(0, 4, size=(rows, 5), dtype=np.int64) for _ in range(n_batches)]
    y = rng.integers(0, 3, size=rows, dtype=np.int64)
    return batches, y


def _run(seed: int):
    """One family-gate pass on the seeded fixture; returns the per-batch pass flags."""
    batches, y = _family_batches(seed)
    return gate.family_pass_batches(batches, y, alpha=0.05, n_perm=40, seed=seed)


def test_family_gate_kernels_are_entered_under_parallel_kernel_guard(monkeypatch):
    """Every numba kernel the family gate calls is invoked while the calling thread holds the guard.

    Before the fix the gate called these kernels bare, so the spy saw an unowned lock and the test failed on any platform.
    """
    lock = _entry_lock()
    unguarded: list[str] = []

    def spy_for(name: str):
        """Wrap the named gate kernel so each call records whether the calling thread holds the guard lock."""
        original = getattr(gate, name)

        def wrapper(*args, **kwargs):
            """Record an unguarded call, then forward to the real kernel."""
            if not lock._is_owned():
                unguarded.append(name)
            return original(*args, **kwargs)

        return wrapper

    for name in _KERNELS:
        if hasattr(gate, name):
            monkeypatch.setattr(gate, name, spy_for(name))
    _run(seed=7)
    assert unguarded == [], f"prange kernels entered without parallel_kernel_entry(): {sorted(set(unguarded))}"


def test_family_gate_concurrent_threads_match_serial_results():
    """Scoring several families from concurrent threads gives the same flags as scoring them serially.

    The guard serialises kernel entry, so concurrent callers must neither raise nor diverge from the serial answer.
    """
    seeds = [11, 12, 13, 14]
    serial = [_run(s) for s in seeds]
    with ThreadPoolExecutor(max_workers=len(seeds)) as pool:
        concurrent = list(pool.map(_run, seeds))
    assert serial and concurrent
    for s_flags, c_flags in zip(serial, concurrent):
        assert [bool(f) for f in s_flags] == [bool(f) for f in c_flags]


@pytest.mark.parametrize("seed", [21])
def test_family_gate_reentrant_under_outer_guard(seed):
    """Calling the gate while the same thread already holds the guard must not deadlock (the guard is reentrant)."""
    from mlframe._numba_parallel_guard import parallel_kernel_entry

    with parallel_kernel_entry():
        flags = _run(seed)
    assert len(flags) == 3
