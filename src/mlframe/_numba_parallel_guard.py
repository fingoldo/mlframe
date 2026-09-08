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
"""

from __future__ import annotations

import threading

__all__ = ["parallel_kernel_entry"]

#: Process-wide, deliberately: numba's threading layer is a process-wide resource, so a per-module lock
#: would let two different fan-outs enter it at the same time and reproduce the crash between them.
_ENTRY = threading.Lock()


def parallel_kernel_entry() -> threading.Lock:
    """The lock to hold while calling a ``parallel=True`` kernel from a thread that may not be alone.

    Used as ``with parallel_kernel_entry(): out = some_prange_kernel(...)``. Safe to hold around a plain
    call: it is a plain lock, so it must not be taken re-entrantly on one thread -- guard the outermost
    kernel call, not every helper along the way.
    """
    return _ENTRY
