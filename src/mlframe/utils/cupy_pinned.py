"""Pinned-memory policy for cupy host-to-device uploads.

``cp.asarray(numpy_array)`` stages the upload through a page-locked (pinned) host buffer of the array's size, taken from cupy's pinned pool.
Two problems with that default on a RAM-tight host:

* When ``cudaHostAlloc`` fails, the Windows driver can report ``cudaErrorAlreadyMapped`` instead of ``cudaErrorMemoryAllocation``. cupy
  falls back to a synchronous pageable copy only on the latter, so the upload raised and every GPU MI batch of a wide fit failed.
* Staging is an extra host copy into a freshly page-locked buffer; measured on a pageable numpy source, it is slower than the pageable copy
  at every size (1 MB 0.31 vs 0.24 ms, 16 MB 4.84 vs 3.31 ms, 128 MB 32.8 vs 23.1 ms), and the pool keeps the locked, unswappable blocks.

The policy wraps cupy's pinned allocator: requests above ``MLFRAME_PINNED_STAGING_MAX_MB`` and any failed allocation are
reported as ``cudaErrorMemoryAllocation``, which makes cupy take its own synchronous-copy path. The default cap is 0: staging was slower at
every measured size, so pinned memory is kept for sources that are page-locked from the start (no staging copy), where it can be faster.
"""

from __future__ import annotations

import logging
import warnings

from mlframe.utils.env_flags import env_int

logger = logging.getLogger(__name__)

_CUDA_ERROR_MEMORY_ALLOCATION = 2
_installed = False


def _make_pool(pinned_memory):
    """The pinned pool the wrapper draws from (a function so tests can substitute a failing pool)."""
    return pinned_memory.PinnedMemoryPool()


def install_pinned_policy() -> None:
    """Install the pinned-allocator wrapper once per process (no-op without cupy or a device)."""
    global _installed
    if _installed:
        return
    _installed = True
    try:
        import cupy as cp
        from cupy.cuda import pinned_memory
        from cupy_backends.cuda.api import runtime
    except Exception as e:  # nosec B110 - no cupy means no pinned uploads to manage
        logger.debug("pinned policy not installed: %s", e)
        return
    cap = env_int("MLFRAME_PINNED_STAGING_MAX_MB", 0, minimum=0) << 20
    pool = _make_pool(pinned_memory)

    def _alloc(nbytes: int):
        """Pinned block from the pool, or the out-of-memory status that sends cupy to its synchronous pageable copy."""
        if nbytes > cap:
            raise runtime.CUDARuntimeError(_CUDA_ERROR_MEMORY_ALLOCATION)
        try:
            return pool.malloc(nbytes)
        except runtime.CUDARuntimeError as e:
            logger.debug("pinned staging of %d bytes failed (%s); synchronous copy instead", nbytes, e)
            raise runtime.CUDARuntimeError(_CUDA_ERROR_MEMORY_ALLOCATION) from e

    cp.cuda.set_pinned_memory_allocator(_alloc)
    # cupy warns on every synchronous fallback; here it is the chosen path, not a degradation.
    warnings.filterwarnings("ignore", message="Using synchronous transfer as pinned memory")


__all__ = ["install_pinned_policy"]
