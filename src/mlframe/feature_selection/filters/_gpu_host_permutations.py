"""Host-generated y-permutations shared by the CPU permutation kernels and the GPU batched permutation tests.

Permutation ``k`` of ``base_seed`` is the Fisher-Yates shuffle driven by the same per-permutation LCG that
``parallel_mi_prange`` / ``parallel_mi`` use on the CPU, so the permutation null does not depend on which backend
evaluates it. The GPU paths upload these rows instead of drawing a separate cuRAND stream.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

import numpy as np
from numba import njit, prange

HOST_PERM_BUDGET_BYTES = 256 * 1024 * 1024
# The permutation stream is a pure function of (target codes, seed, first permutation, count), and a fit asks for the same batches of the same target
# tens of thousands of times (one permutation test per candidate pair and transform). The batches are kept for reuse, host and device separately, each
# bounded; a cached batch is the very array the generator returned, so cached and uncached results are bit-identical.
HOST_PERM_CACHE_BYTES = 128 * 1024 * 1024
DEVICE_PERM_CACHE_BYTES = 128 * 1024 * 1024
_HOST_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_HOST_CACHE_BYTES = 0
_DEVICE_CACHE: "OrderedDict[int, tuple]" = OrderedDict()
_DEVICE_CACHE_BYTES = 0
_CACHE_LOCK = threading.Lock()


@njit(parallel=True, nogil=True, cache=True)
def _lcg_permuted_y_batch_kernel(classes_y: np.ndarray, base_seed: np.uint64, first_perm: np.int64, count: np.int64) -> np.ndarray:
    """Rows ``first_perm .. first_perm+count-1`` of the shared LCG permutation stream applied to ``classes_y``."""
    n = classes_y.shape[0]
    out = np.empty((count, n), dtype=np.int32)
    for b in prange(count):
        state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(first_perm + b + 1)
        for i in range(n):
            out[b, i] = classes_y[i]
        for j in range(n - 1, 0, -1):
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            k = int(state >> np.uint64(33)) % (j + 1)
            tmp = out[b, j]
            out[b, j] = out[b, k]
            out[b, k] = tmp
    return out


def host_permuted_y_batch(classes_y: np.ndarray, base_seed: object, first_perm: int, count: int) -> np.ndarray:
    """``(count, n)`` int32 matrix whose row ``b`` is ``classes_y`` shuffled by permutation ``first_perm + b`` of ``base_seed``.

    ``base_seed=None`` maps to 0, the ``mi_direct`` default, so an unseeded GPU call draws the same stream as an unseeded CPU call.
    The matrix is read-only and repeated requests for the same target content, seed and slice return the same array (bounded by ``HOST_PERM_CACHE_BYTES``).
    """
    global _HOST_CACHE_BYTES
    from mlframe._numba_parallel_guard import parallel_kernel_entry

    seed = np.uint64(0 if base_seed is None else int(base_seed))  # type: ignore[call-overload]
    y = np.ascontiguousarray(classes_y, dtype=np.int32)
    key = (hashlib.blake2b(y.data, digest_size=16).digest(), y.shape[0], int(seed), int(first_perm), int(count))
    with _CACHE_LOCK:
        hit = _HOST_CACHE.get(key)
        if hit is not None:
            _HOST_CACHE.move_to_end(key)
            return hit
        with parallel_kernel_entry():
            out = np.asarray(_lcg_permuted_y_batch_kernel(y, seed, np.int64(first_perm), np.int64(count)))
        out.flags.writeable = False
        if out.nbytes <= HOST_PERM_CACHE_BYTES and key not in _HOST_CACHE:
            _HOST_CACHE[key] = out
            _HOST_CACHE_BYTES += out.nbytes
            while _HOST_CACHE_BYTES > HOST_PERM_CACHE_BYTES and len(_HOST_CACHE) > 1:
                _, dropped = _HOST_CACHE.popitem(last=False)
                _HOST_CACHE_BYTES -= dropped.nbytes
    return out


def device_array_cached(cp, host: np.ndarray):
    """``cp.asarray(host)``, reusing the device copy of a read-only host batch that was uploaded before (bounded by ``DEVICE_PERM_CACHE_BYTES``).

    Only read-only arrays are cached, since a writable one could change after the upload; the copy is completed before it is shared so any stream may read it.
    """
    global _DEVICE_CACHE_BYTES
    if host.flags.writeable or host.nbytes > DEVICE_PERM_CACHE_BYTES:
        return cp.asarray(host)
    key = id(host)
    with _CACHE_LOCK:
        hit = _DEVICE_CACHE.get(key)
        if hit is not None and hit[0] is host:
            _DEVICE_CACHE.move_to_end(key)
            return hit[1]
        dev = cp.asarray(host)
        cp.cuda.get_current_stream().synchronize()
        if key not in _DEVICE_CACHE:
            _DEVICE_CACHE[key] = (host, dev)  # holding the host array keeps its id from being reused
            _DEVICE_CACHE_BYTES += host.nbytes
            while _DEVICE_CACHE_BYTES > DEVICE_PERM_CACHE_BYTES and len(_DEVICE_CACHE) > 1:
                _, (dropped_host, _dropped_dev) = _DEVICE_CACHE.popitem(last=False)
                _DEVICE_CACHE_BYTES -= dropped_host.nbytes
    return dev


def clear_permutation_caches() -> None:
    """Drop every cached permutation batch (host and device)."""
    global _HOST_CACHE_BYTES, _DEVICE_CACHE_BYTES
    with _CACHE_LOCK:
        _HOST_CACHE.clear()
        _DEVICE_CACHE.clear()
        _HOST_CACHE_BYTES = 0
        _DEVICE_CACHE_BYTES = 0


def host_perm_batch_cap(n: int, budget_bytes: int = HOST_PERM_BUDGET_BYTES) -> int:
    """Most permutations of length ``n`` whose int32 host matrix fits ``budget_bytes`` (at least 1)."""
    return max(1, int(budget_bytes) // (4 * max(1, int(n))))
