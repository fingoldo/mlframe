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


def ensure_sort_headroom(a: Any) -> None:
    """Raise cupy's ``OutOfMemoryError`` when a sort of ``a`` would not fit the current pool's limit, after releasing its cached free blocks."""
    import cupy as cp

    pool = _current_pool(cp)
    if pool is None:
        return
    limit = int(pool.get_limit())
    if limit <= 0:
        return  # no hard limit: thrust allocates from the device as any other array
    need = _sort_need_bytes(a)
    if limit - int(pool.used_bytes()) >= need:
        return
    pool.free_all_blocks()
    room = limit - int(pool.used_bytes())
    if room < need:
        raise cp.cuda.memory.OutOfMemoryError(need, room, limit)


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


__all__ = ["device_argsort", "device_sort", "ensure_sort_headroom"]
