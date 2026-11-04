"""Device sorts that cannot abort the process when the memory pool is short.

cupy sorts through thrust. A thrust temporary allocation that does not fit a memory pool with a hard limit (the default pool under
``ensure_fe_gpu_pool_limit``, a stage slot pool under FE stage concurrency) raises a C++ ``bad_alloc`` inside the allocator callback, and the
process terminates (``Fatal Python error: Aborted``) instead of the ``OutOfMemoryError`` every GPU path catches to take its host twin. Here the
pool behind the current allocator is checked first: when the sort's temporaries do not fit its remaining room, the cached free blocks are
released and the room checked again, and if it is still short a cupy ``OutOfMemoryError`` is raised in Python, before thrust runs.
"""

from __future__ import annotations

from typing import Any, Optional

#: Bytes a thrust sort of ``a`` along an axis needs beyond ``a``: a copy of the values, int64 indices, int64 segment keys and cub's scratch.
_INDEX_BYTES = 8


def _sort_need_bytes(a: Any) -> int:
    """Upper estimate of the device memory a sort or argsort of ``a`` allocates."""
    return 2 * int(a.nbytes) + 3 * int(a.size) * _INDEX_BYTES


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


def device_argsort(a: Any, axis: int = -1, **kwargs: Any) -> Any:
    """``cupy.argsort(a, axis, **kwargs)`` that raises ``OutOfMemoryError`` instead of aborting when the pool is short."""
    import cupy as cp

    ensure_sort_headroom(a)
    return cp.argsort(a, axis=axis, **kwargs)


def device_sort(a: Any, axis: int = -1, **kwargs: Any) -> Any:
    """``cupy.sort(a, axis, **kwargs)`` that raises ``OutOfMemoryError`` instead of aborting when the pool is short."""
    import cupy as cp

    ensure_sort_headroom(a)
    return cp.sort(a, axis=axis, **kwargs)


def _guarded(name: str) -> Any:
    """``cupy.<name>`` behind :func:`ensure_sort_headroom` on its first argument (the routine sorts it through thrust)."""

    def call(a: Any, *args: Any, **kwargs: Any) -> Any:
        """``cupy.<name>(a, *args, **kwargs)`` that raises ``OutOfMemoryError`` instead of aborting when the pool is short."""
        import cupy as cp

        if hasattr(a, "nbytes"):
            ensure_sort_headroom(a)
        return getattr(cp, name)(a, *args, **kwargs)

    call.__name__ = f"device_{name}"
    return call


#: Sorting routines of cupy beyond sort / argsort (unique, median, percentile and quantile sort their input through thrust).
device_unique = _guarded("unique")
device_median = _guarded("median")
device_percentile = _guarded("percentile")
device_quantile = _guarded("quantile")

__all__ = ["device_argsort", "device_median", "device_percentile", "device_quantile", "device_sort", "device_unique", "ensure_sort_headroom"]
