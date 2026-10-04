"""Host-generated y-permutations shared by the CPU permutation kernels and the GPU batched permutation tests.

Permutation ``k`` of ``base_seed`` is the Fisher-Yates shuffle driven by the same per-permutation LCG that
``parallel_mi_prange`` / ``parallel_mi`` use on the CPU, so the permutation null does not depend on which backend
evaluates it. The GPU paths upload these rows instead of drawing a separate cuRAND stream.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

HOST_PERM_BUDGET_BYTES = 256 * 1024 * 1024


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
    """
    from mlframe._numba_parallel_guard import parallel_kernel_entry

    seed = np.uint64(0 if base_seed is None else int(base_seed))  # type: ignore[call-overload]
    y = np.ascontiguousarray(classes_y, dtype=np.int32)
    with parallel_kernel_entry():
        return np.asarray(_lcg_permuted_y_batch_kernel(y, seed, np.int64(first_perm), np.int64(count)))


def host_perm_batch_cap(n: int, budget_bytes: int = HOST_PERM_BUDGET_BYTES) -> int:
    """Most permutations of length ``n`` whose int32 host matrix fits ``budget_bytes`` (at least 1)."""
    return max(1, int(budget_bytes) // (4 * max(1, int(n))))
