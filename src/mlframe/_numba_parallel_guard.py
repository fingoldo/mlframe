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

import threading

__all__ = ["parallel_kernel_entry"]

#: Process-wide, deliberately: numba's threading layer is a process-wide resource, so a per-module lock
#: would let two different fan-outs enter it at the same time and reproduce the crash between them.
#: RLock, not Lock: see the reentrancy note above -- nested guards on the SAME thread must not deadlock.
_ENTRY = threading.RLock()


def parallel_kernel_entry() -> threading.RLock:
    """The lock to hold while calling a ``parallel=True`` kernel from a thread that may not be alone.

    Used as ``with parallel_kernel_entry(): out = some_prange_kernel(...)``. Reentrant: the same thread
    may hold it across nested guarded calls (a caller guarding a callee that already guards itself)
    without deadlocking; a DIFFERENT thread still blocks until every level on the holding thread exits.
    """
    return _ENTRY
