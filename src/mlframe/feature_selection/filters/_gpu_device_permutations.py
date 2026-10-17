"""Device-generated y-permutations for the GPU permutation tests (CUDA kernel, bit-identical to the host generator).

:func:`_gpu_host_permutations.host_permuted_y_batch` draws permutation ``k`` of ``base_seed`` with the LCG Fisher-Yates that the CPU
kernels use, on the host, and the GPU paths then copy the rows up. Here the same recurrence runs in one CUDA thread per permutation and
the rows are written straight into device memory, so the host does no per-permutation work and no matrix crosses the bus. The
arithmetic is the same uint64 recurrence (wrap-around multiply, ``(state >> 33) % (j + 1)``), so the rows equal the host rows exactly.

bench-attempt-rejected (2026-10-15): device generator vs host numba generator, 99 x 2000 int32 rows, GTX 1050 Ti: device 2.76 ms, host
prange 0.55 ms. One block of 99 sequential Fisher-Yates threads does not beat the host kernel at this size, so the GPU paths keep
``host_permuted_y_batch``. Rows are bit-identical (0 mismatches over 72 cases); the kernel is kept as the parity reference.
Breakdown (CUDA events, GTX 1050 Ti, 99 permutations): the kernel dominates. At n = 2000 it takes 2.2 ms against 0.11 ms for the input upload and
0.27 ms for the 99-row readback; at n = 100000 it takes 81 ms against 12 ms for the readback. The bottleneck is the design, one block of 99
threads each running a sequential n-step Fisher-Yates, not the transfers. A size gate is therefore not justified by this kernel; it would
need a parallel shuffle, which changes the stream and so loses bit-identity with the host generator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import cuda

if TYPE_CHECKING:
    import cupy as cp


@cuda.jit
def _lcg_permuted_y_cuda_kernel(classes_y, base_seed, first_perm, count, out):
    """One thread per permutation row ``b``: copy ``classes_y`` and apply the shared LCG Fisher-Yates for ``first_perm + b``."""
    b = cuda.grid(1)
    if b >= count:
        return
    n = classes_y.shape[0]
    for i in range(n):
        out[b, i] = classes_y[i]
    state = base_seed * np.uint64(2654435761) + np.uint64(first_perm + b + 1)
    for j in range(n - 1, 0, -1):
        state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
        k = (state >> np.uint64(33)) % np.uint64(j + 1)
        tmp = out[b, j]
        out[b, j] = out[b, k]
        out[b, k] = tmp


def device_permuted_y_batch(classes_y: np.ndarray, base_seed: object, first_perm: int, count: int) -> "cp.ndarray":
    """``(count, n)`` int32 cupy matrix whose row ``b`` is ``classes_y`` shuffled by permutation ``first_perm + b`` of ``base_seed``.

    Same contract as :func:`_gpu_host_permutations.host_permuted_y_batch` (``base_seed=None`` maps to 0), but the result stays on the device.
    """
    import cupy as cp

    seed = np.uint64(0 if base_seed is None else int(base_seed))  # type: ignore[call-overload]
    y_dev = cp.asarray(np.ascontiguousarray(classes_y, dtype=np.int32))
    n = int(y_dev.shape[0])
    out = cp.empty((int(count), n), dtype=cp.int32)
    threads = 128
    blocks = (int(count) + threads - 1) // threads
    _lcg_permuted_y_cuda_kernel[blocks, threads](y_dev, seed, np.int64(first_perm), np.int64(count), out)
    return out


__all__ = ["device_permuted_y_batch"]
