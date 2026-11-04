"""One thread at a time inside a numba ``parallel=True`` region.

numba's default threading layer is not safe to enter concurrently from several Python threads. On Linux
that mostly goes unnoticed; on macOS it aborts the process outright -- mlframe's first three-OS CI run
crashed 92 xdist workers with ``Fatal Python error: Aborted``, every faulthandler dump showing multiple
pool threads stopped at the same ``parallel=True`` call site.

This repository has several places where that can happen: a ``ThreadPoolExecutor`` fans work out over
columns or feature pairs, and each worker's path reaches a prange kernel. A static scan found eleven such
paths across five modules, so the remedy belongs in one place rather than at each call site -- and
``tests/test_meta/test_no_unguarded_nested_parallel.py`` fails on a twelfth appearing unguarded.

Serialising ENTRY costs little: the kernel already parallelises across every core internally, so a second
thread waiting to enter is a thread that had no cores to run on anyway. Everything OUTSIDE the kernel --
sorting, binning, candidate counting, cache lookups -- keeps overlapping, which is where the measured
speedups of these fan-outs actually come from.

Reentrant on purpose: shared dispatchers guard themselves at their own boundary (the module's own
guidance), and callers a few hops up sometimes ALSO guard the same call for defense-in-depth or because
the call graph isn't obvious from the caller's vantage point. A plain ``Lock`` would deadlock the SAME
thread against its own outer acquisition the moment two such guards nest -- confirmed directly (fit_pair_
prewarp_als wrapping build_basis_matrix, and _dispatch_batch_mi_with_noise_gate wrapping the GPU-resident
path's own inner guard, both hung immediately under test). An ``RLock`` costs nothing extra here: the
guarantee that matters is "no two DIFFERENT threads inside at once", and re-entry from the thread already
holding it is by definition still only one thread inside.
"""

from __future__ import annotations

import contextlib
import functools
import threading
import logging

logger = logging.getLogger(__name__)

__all__ = ["parallel_kernel_entry"]

#: Process-wide, deliberately: numba's threading layer is a process-wide resource, so a per-module lock
#: would let two different fan-outs enter it at the same time and reproduce the crash between them.
#: RLock, not Lock: see the reentrancy note above -- nested guards on the SAME thread must not deadlock.
@functools.lru_cache(maxsize=1)
def _entry_lock() -> threading.RLock:
    """The one process-wide RLock; created on first use and cached so every caller receives the same object."""
    return threading.RLock()


def _layer_allows_concurrent_entry() -> bool:
    """Whether numba's ACTIVE threading layer is one that tolerates several Python threads inside ``parallel=True`` regions at once.

    The Intel TBB layer does (its scheduler is a shared, thread-safe pool); ``workqueue``, numba's fallback, does not, and the OpenMP layer depends on the runtime. The layer is only chosen when the first
    parallel kernel is compiled, so a tiny one is compiled here to ask. ``MLFRAME_NUMBA_PARALLEL_GUARD=always`` keeps the lock whatever the layer (the previous behaviour); the answer is computed once.
    """
    import os

    if os.environ.get("MLFRAME_NUMBA_PARALLEL_GUARD", "").strip().lower() == "always":
        return False
    try:
        import numba
        import numpy as np
        from numba import njit, prange

        @njit(parallel=True)
        def _probe(a):  # pragma: no cover - compiled, not traced
            """Smallest ``prange`` kernel: compiling and running it makes numba pick its threading layer."""
            s = 0.0
            for i in prange(a.size):
                s += a[i]
            return s

        _probe(np.ones(8))
        return str(numba.threading_layer()) == "tbb"
    except Exception as _silent_exc:
        logger.debug("_numba_parallel_guard.py: suppressed %r", _silent_exc)
        return False


@functools.lru_cache(maxsize=1)
def _concurrent_entry_ok() -> bool:
    """Cached :func:`_layer_allows_concurrent_entry`."""
    return _layer_allows_concurrent_entry()


def parallel_kernel_entry() -> "contextlib.AbstractContextManager[object]":
    """The context to hold while calling a ``parallel=True`` kernel from a thread that may not be alone.

    Used as ``with parallel_kernel_entry(): out = some_prange_kernel(...)``. With a thread-safe threading layer (TBB) it is a no-op, so concurrent stages do overlap their parallel kernels; otherwise it is the
    process-wide RLock: reentrant (the same thread may hold it across nested guarded calls without deadlocking), and a DIFFERENT thread blocks until every level on the holding thread exits.
    """
    if _concurrent_entry_ok():
        return contextlib.nullcontext()
    return _entry_lock()
