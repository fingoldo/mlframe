"""Device sorts that cannot abort the process when the memory pool is short.

cupy sorts through thrust. A thrust temporary allocation that does not fit a memory pool with a hard limit (the default pool under
``ensure_fe_gpu_pool_limit``, a stage slot pool under FE stage concurrency) raises a C++ ``bad_alloc`` inside the allocator callback, and the
process terminates (``Fatal Python error: Aborted``) instead of the ``OutOfMemoryError`` every GPU path catches to take its host twin. Here the
pool behind the current allocator is checked first (its limit and the device's free memory), then the sort's room is reserved through the pool
under the pool's lock (``sort_room``): a short pool or device raises a cupy ``OutOfMemoryError`` in Python, before thrust runs.

KNOWN GAP (GPU11-15, audits/2026-11-09/_TRACKER.md, found 2026-11-11): ``sort_room``'s lock (``_pool_lock``) is a lock THIS MODULE owns, not
cupy's own pool-internal lock. It serializes `sort_room`/`device_argsort`/`device_sort` CALLS against EACH OTHER on the same pool, but any OTHER
code that allocates directly against the SAME pool object (``cp.empty(...)`` under the same ``using_allocator``, bypassing this module entirely)
is invisible to it and can still race the reserve-then-sort window, reproducing the exact ``Fatal Python error: Aborted`` this module exists to
prevent (reproduced in a single-threaded-equivalent test harness; see the tracker entry for the measured mechanism). Fixing this for real means
either threading every cupy allocation site in the project through the same lock (a large, cross-cutting change) or accepting and documenting
that concurrent allocation on a SHARED pool outside this module is unsupported - a design decision for whoever picks this up, not attempted here.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from typing import Any, Iterator, Optional

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

#: Bytes a thrust sort of ``a`` along an axis needs beyond ``a``: a copy of the values, int64 indices, int64 segment keys, the int64 result and cub's scratch.
#: Measured peaks (pool high-water, result included): 40 B per element for an axis-0 argsort of (100000, 40) in float64 and in float32 alike, 24 B for a 1-D
#: float64 argsort; the estimate covers the largest (a float32 matrix under the old 2 x nbytes + 24 B estimate got 32 B and thrust aborted the fit).
_INDEX_BYTES = 8
_SCRATCH_MARGIN = 1 << 20


def _sort_need_bytes(a: Any) -> int:
    """Upper estimate of the device memory a sort or argsort of ``a`` allocates."""
    return 2 * int(a.nbytes) + 4 * int(a.size) * _INDEX_BYTES + _SCRATCH_MARGIN


def _current_pool(cp: Any) -> Optional[Any]:
    """The memory pool behind the current thread's allocator (a stage slot pool or the default pool), or ``None`` when it is not a cupy pool."""
    alloc = cp.cuda.get_allocator()
    pool = getattr(alloc, "__self__", None)
    return pool if isinstance(pool, cp.cuda.MemoryPool) else None


def _room(cp: Any, pool: Any) -> int:
    """Bytes a new allocation can still get: the room under the pool's limit (if it has one), and at most the device's free memory plus the pool's
    cached free blocks (other processes or pools can fill the device while the pool's own limit still has room)."""
    free_dev, _total = cp.cuda.runtime.memGetInfo()
    room = int(free_dev)
    if pool is not None:
        room += int(pool.free_bytes())
        limit = int(pool.get_limit())
        if limit > 0:
            room = min(room, limit - int(pool.used_bytes()))
    return room


def device_room(cp: Any) -> int:
    """Bytes the calling thread's allocations can still get: the room under the limit of the pool behind its allocator (a stage slot's pool or the default one) and the device's free memory plus
    that pool's cached free blocks. Block sizes for streamed device work should be taken from this, not from the device-wide free memory: a concurrent stage runs under a fraction of the card."""
    return _room(cp, _current_pool(cp))


_POOL_LOCKS: "dict[int, tuple[Any, threading.Lock]]" = {}
_POOL_LOCKS_GUARD = threading.Lock()


def _pool_lock(pool: Any) -> "threading.Lock":
    """One lock per memory pool: threads that allocate from the same pool take turns between reserving a sort's room and the sort itself.

    The entry holds the pool itself, so its ``id`` cannot be reused by another pool while the entry lives, and a lookup checks identity."""
    with _POOL_LOCKS_GUARD:
        entry = _POOL_LOCKS.get(id(pool))
        if entry is None or entry[0] is not pool:
            entry = _POOL_LOCKS[id(pool)] = (pool, threading.Lock())
        return entry[1]


def _trace_sort(cp: Any, a: Any, pool: Any) -> None:
    """Append one line per device sort to the file named by ``MLFRAME_SORT_ROOM_TRACE`` (shape, dtype, estimated need, device and pool room, allocator type, thread): a process that
    aborts inside thrust leaves the sort that killed it as the last line. Off (one dict lookup) when the variable is unset."""
    path = os.environ.get("MLFRAME_SORT_ROOM_TRACE", "").strip()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(
                f"{time.time():.3f} thread={threading.get_ident()} shape={tuple(a.shape)} dtype={a.dtype} need={_sort_need_bytes(a)} room={_room(cp, pool)} "
                f"pool={'none' if pool is None else (int(pool.used_bytes()), int(pool.total_bytes()), int(pool.get_limit()))} allocator={type(cp.cuda.get_allocator()).__name__}\n"
            )
    except OSError as exc:
        log_throttle(logger, "_device_sort_safe.trace", logging.WARNING, "sort trace file %s not writable: %s", path, exc)


@contextlib.contextmanager
def sort_room(a: Any) -> "Iterator[None]":
    """Hold room for a sort of ``a`` in the current pool for the duration of the block, or raise cupy's ``OutOfMemoryError``.

    A free-memory check alone races: under FE stage concurrency another thread allocated between the check and thrust's own allocation, and
    thrust aborted the process (warm 100k fits, three cascades). Here the need is allocated through the pool (a short pool raises the ordinary
    ``OutOfMemoryError`` there) and released at once, so the block stays cached in the pool for thrust's temporaries; the pool's lock keeps the
    other threads of the same pool from taking it before the sort ran."""
    import cupy as cp

    ensure_sort_headroom(a)
    pool = _current_pool(cp)
    _trace_sort(cp, a, pool)
    if pool is None:
        yield
        return
    need = _sort_need_bytes(a)
    with _pool_lock(pool):
        hold = None
        for attempt in range(2):  # the first OOM frees the pool's cached blocks and retries once; a second OOM is real and propagates
            try:
                hold = cp.empty(need, dtype=cp.uint8)
                break
            except cp.cuda.memory.OutOfMemoryError:
                if attempt:
                    raise
                pool.free_all_blocks()
        del hold
        yield


def ensure_sort_headroom(a: Any) -> None:
    """Raise cupy's ``OutOfMemoryError`` when a sort of ``a`` would not fit: the current pool's limit or the device itself (a sort that runs out of
    device memory inside thrust aborts the process just as one over the pool limit does), after releasing the pool's cached free blocks."""
    import cupy as cp

    pool = _current_pool(cp)
    need = _sort_need_bytes(a)
    if _room(cp, pool) >= need:
        return
    if pool is not None:
        pool.free_all_blocks()
    room = _room(cp, pool)
    if room < need:
        raise cp.cuda.memory.OutOfMemoryError(need, room, int(pool.get_limit()) if pool is not None else 0)


#: Vector length from which a 2-D sort runs as one 1-D sort per column or row. cupy sorts a 2-D array in one batched call that is 3.5 to 5 times slower than the 1-D sorts of the same vectors
#: at 300000 rows (32 columns: 103 ms against 19 ms, 1.6 times at 100000 rows) and faster than them below ~50000 rows, where a launch per column costs more than it saves; results are identical,
#: ties included.
_PER_VECTOR_SORT_MIN_LEN = 80_000


def _vectors_of(a: Any, axis: int) -> "tuple[int, int] | None":
    """``(vector length, vector count)`` of a 2-D ``a`` sorted along ``axis``, or ``None`` when the sort is not a long-vector 2-D one."""
    if getattr(a, "ndim", 0) != 2:
        return None
    ax = axis % 2
    length, count = int(a.shape[ax]), int(a.shape[1 - ax])
    return (length, count) if length >= _PER_VECTOR_SORT_MIN_LEN and count > 1 else None


def _sorted_per_vector(cp: Any, a: Any, axis: int, argsort: bool) -> Any:
    """``cupy.argsort`` / ``cupy.sort`` of 2-D ``a`` along ``axis`` as one 1-D sort per vector, written into one output."""
    ax = axis % 2
    out = cp.empty(a.shape, dtype=cp.int64 if argsort else a.dtype)
    fn = cp.argsort if argsort else cp.sort
    for j in range(int(a.shape[1 - ax])):
        if ax == 0:
            out[:, j] = fn(a[:, j])
        else:
            out[j] = fn(a[j])
    return out


def device_argsort(a: Any, axis: int = -1, **kwargs: Any) -> Any:
    """``cupy.argsort(a, axis, **kwargs)`` that raises ``OutOfMemoryError`` instead of aborting when the pool is short."""
    import cupy as cp

    with sort_room(a):
        if not kwargs and _vectors_of(a, axis) is not None:
            return _sorted_per_vector(cp, a, axis, True)
        return cp.argsort(a, axis=axis, **kwargs)


def device_sort(a: Any, axis: int = -1, **kwargs: Any) -> Any:
    """``cupy.sort(a, axis, **kwargs)`` that raises ``OutOfMemoryError`` instead of aborting when the pool is short."""
    import cupy as cp

    with sort_room(a):
        if not kwargs and _vectors_of(a, axis) is not None:
            return _sorted_per_vector(cp, a, axis, False)
        return cp.sort(a, axis=axis, **kwargs)


def _guarded(name: str) -> Any:
    """``cupy.<name>`` behind :func:`ensure_sort_headroom` on its first argument (the routine sorts it through thrust)."""

    def call(a: Any, *args: Any, **kwargs: Any) -> Any:
        """``cupy.<name>(a, *args, **kwargs)`` that raises ``OutOfMemoryError`` instead of aborting when the pool is short."""
        import cupy as cp

        if not hasattr(a, "nbytes"):
            return getattr(cp, name)(a, *args, **kwargs)
        with sort_room(a):
            return getattr(cp, name)(a, *args, **kwargs)

    call.__name__ = f"device_{name}"
    return call


#: Sorting routines of cupy beyond sort / argsort (unique, median, percentile and quantile sort their input through thrust).
device_unique = _guarded("unique")
device_median = _guarded("median")
device_percentile = _guarded("percentile")
device_quantile = _guarded("quantile")

__all__ = ["device_argsort", "device_median", "device_percentile", "device_quantile", "device_sort", "device_unique", "ensure_sort_headroom", "sort_room"]
