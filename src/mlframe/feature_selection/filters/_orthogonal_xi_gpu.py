"""Device twin of ``_orthogonal_xi_fe._xi_columns_njit``: Chatterjee's Xi of many columns against one target.

The host kernel orders each column's rows by ``np.lexsort((perm, col))`` (ties in the column broken by the column's own random permutation) and walks the target's tie-corrected right ranks
``r`` and left ranks ``l`` in that order: ``xi = 1 - n * sum |r[i+1] - r[i]| / (2 * sum l (n - l))``. Here one block of columns is done in four steps without leaving the device:

1. ``xi_inverse_perm``: ``ra[j, perm[j, i]] = i``, the row that has rank ``t`` in the permutation's order;
2. ``xi_gather``: the column's values in that row order, ``keys[j, t] = col[j, ra[j, t]]``;
3. a stable argsort of ``keys`` along each row (ties keep the permutation order, which is what ``lexsort`` does with ``perm`` as its secondary key);
4. ``xi_walk``: one thread per sorted position reads its row and its predecessor's row and adds ``|r - r_prev|`` and ``l (n - l)`` into two per-column sums.

The ranks are integer counts, so every term is an integer held exactly in a double and the block sums are exact in any order: the result does not depend on how the threads are scheduled and it
equals the host's value (the host sums the same integers in the same double precision). The permutations are drawn on the host from the same generator as the host path, so the tie-breaking
stream is the same. Columns go in blocks that fit ``_BUDGET_FRACTION`` of the free device memory; any device fault returns ``None`` and the caller runs the host kernel.

``MLFRAME_XI_DEVICE=0|1`` forces the path; otherwise the kernel-tuning cache decides by row count (fallback ``DEVICE_FALLBACK_ROWS``, the host until a sweep says otherwise).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from ._device_sort_safe import device_argsort
from ._ktc_fallback import device_from_rows
from ._raw_module_cache import RawModuleCache
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

_BUDGET_FRACTION = 0.25
DEVICE_FALLBACK_ROWS = 10**9  # the host radix kernel is as fast or faster at every measured size on the development card (audits profile note): the device runs only when the tuner or the env switch says so
_THREADS = 256

_SRC = r"""
// dtype-ok: Intentional float64: the rank sums are integers held exactly in doubles; the column type T is the host column's own dtype.
extern "C" __global__ void xi_inverse_perm(const long long* __restrict__ perm, int* __restrict__ ra, const long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long base = (long long)blockIdx.y * n;
    ra[base + perm[base + i]] = (int)i;
}

extern "C" __global__ void xi_gather(const T* __restrict__ cols, const int* __restrict__ ra, T* __restrict__ keys, const long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long base = (long long)blockIdx.y * n;
    T v = cols[base + ra[base + i]];
    if (v == (T)0) v = (T)0;  // -0.0 orders as +0.0, as the host comparison treats it
    keys[base + i] = v;
}

extern "C" __global__ void xi_walk(const long long* __restrict__ order, const int* __restrict__ ra, const double* __restrict__ r_row, const double* __restrict__ l_row,
                                   double* __restrict__ sums, const long long n) {
    __shared__ double red_s[256];
    __shared__ double red_d[256];
    const long long base = (long long)blockIdx.y * n;
    double s = 0.0, d = 0.0;
    for (long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x; q < n; q += (long long)gridDim.x * blockDim.x) {
        const int row = ra[base + order[base + q]];
        const double lq = l_row[row];
        d += lq * ((double)n - lq);
        if (q > 0) {
            const int prev = ra[base + order[base + q - 1]];
            const double diff = r_row[row] - r_row[prev];
            s += diff < 0.0 ? -diff : diff;
        }
    }
    red_s[threadIdx.x] = s;
    red_d[threadIdx.x] = d;
    __syncthreads();
    for (int h = blockDim.x >> 1; h > 0; h >>= 1) {
        if (threadIdx.x < h) {
            red_s[threadIdx.x] += red_s[threadIdx.x + h];
            red_d[threadIdx.x] += red_d[threadIdx.x + h];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&sums[2 * blockIdx.y], red_s[0]);
        atomicAdd(&sums[2 * blockIdx.y + 1], red_d[0]);
    }
}
"""

_MODULES = {np.dtype(np.float32): RawModuleCache("#define T float\n" + _SRC), np.dtype(np.float64): RawModuleCache("#define T double\n" + _SRC)}


def xi_columns_device(cols: np.ndarray, perms: np.ndarray, r_row: np.ndarray, l_row: np.ndarray) -> Optional[np.ndarray]:
    """Xi of each row of ``cols`` (columns, n) with the x-ties of row ``j`` broken by ``perms[j]``; ``None`` when the device cannot take it (the caller runs the host kernel).

    ``r_row`` / ``l_row`` are the target's tie-corrected right and left ranks per row (double precision, length n)."""
    kk, n = cols.shape
    if kk == 0 or n < 2 or cols.dtype not in _MODULES:
        return None
    try:
        import cupy as cp

        module = _MODULES[cols.dtype].get(cp)
        inv_k, gather_k, walk_k = module.get_function("xi_inverse_perm"), module.get_function("xi_gather"), module.get_function("xi_walk")
        r_dev = cp.asarray(np.ascontiguousarray(r_row, dtype=np.float64))  # Intentional float64: integer rank counts held exactly
        l_dev = cp.asarray(np.ascontiguousarray(l_row, dtype=np.float64))  # Intentional float64: integer rank counts held exactly
        free = int(cp.cuda.runtime.memGetInfo()[0]) + int(cp.get_default_memory_pool().free_bytes())
        per_col = n * (8 + 4 + int(cols.dtype.itemsize) * 2 + 8 + 8 + 8 + 4 + 8)  # perm, ra, cols + keys, order, sort temporaries
        block = int(max(1, min(kk, (_BUDGET_FRACTION * free) // per_col)))
        out = np.empty(kk, dtype=np.float64)  # Intentional float64: accumulator output
        grid_x = (n + _THREADS - 1) // _THREADS
        walk_blocks = min(grid_x, 64)
        for c0 in range(0, kk, block):
            w = min(block, kk - c0)
            perm_d = cp.asarray(np.ascontiguousarray(perms[c0 : c0 + w], dtype=np.int64))
            col_d = cp.asarray(np.ascontiguousarray(cols[c0 : c0 + w]))
            ra = cp.empty((w, n), dtype=cp.int32)
            inv_k((grid_x, w), (_THREADS,), (perm_d, ra, np.int64(n)))
            del perm_d
            keys = cp.empty((w, n), dtype=col_d.dtype)
            gather_k((grid_x, w), (_THREADS,), (col_d, ra, keys, np.int64(n)))
            del col_d
            order = device_argsort(keys, axis=1)
            del keys
            sums = cp.zeros(2 * w, dtype=cp.float64)
            walk_k((walk_blocks, w), (_THREADS,), (order, ra, r_dev, l_dev, sums, np.int64(n)))
            s = cp.asnumpy(sums).reshape(w, 2)
            den = 2.0 * s[:, 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                out[c0 : c0 + w] = np.where(den > 0.0, 1.0 - float(n) * s[:, 0] / den, 0.0)
            del order, ra, sums
        return out
    except Exception as exc:
        log_throttle(logger, "xi_gpu.failed", logging.WARNING, "device Xi failed (%s: %s); the host kernel scores the columns", type(exc).__name__, exc)
        return None


def _sweep_inputs(dims: dict) -> tuple:
    """A 40-column float32 block of ``dims["n_samples"]`` rows, its permutations and the ranks of a 3-class target."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    cols = rng.standard_normal((40, n)).astype(np.float32)
    perms = np.stack([rng.permutation(n) for _ in range(40)])
    y = rng.integers(0, 3, n).astype(np.float64)  # Intentional float64: target labels
    _u, inv, counts = np.unique(y, return_counts=True, return_inverse=True)
    r_row = np.cumsum(counts)[inv].astype(np.float64)  # Intentional float64: integer rank counts
    l_row = np.cumsum(counts[::-1])[::-1][inv].astype(np.float64)  # Intentional float64: integer rank counts
    return (cols, perms, r_row, l_row)


def _host(cols: np.ndarray, perms: np.ndarray, r_row: np.ndarray, l_row: np.ndarray) -> np.ndarray:
    """The host kernel's Xi per column."""
    from ._orthogonal_xi_fe import _xi_columns_radix_njit

    out = np.empty(cols.shape[0], dtype=np.float64)  # Intentional float64: accumulator output
    _xi_columns_radix_njit(np.ascontiguousarray(cols), perms, r_row, l_row, out)
    return out


def _device(cols: np.ndarray, perms: np.ndarray, r_row: np.ndarray, l_row: np.ndarray) -> np.ndarray:
    """The device kernel's Xi per column; a failed launch raises so the sweep does not record a host fallback as a device time."""
    out = xi_columns_device(cols, perms, r_row, l_row)
    if out is None:
        raise RuntimeError("device Xi unavailable")
    return out


def _run_sweep() -> list:
    """Time the host and the device kernel per row count; the device must agree with the host."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return sweep_backend_grid({"host": _host, "device": _device}, {"n_samples": _SWEEP_ROWS}, _sweep_inputs, reference="host", repeats=3, equiv_rtol=1e-12, equiv_atol=1e-12)


_SWEEP_ROWS = [5_000, 30_000, 100_000]
_SALT = 1


def _fallback_choice(n_samples: int) -> str:
    """Pre-sweep fallback: the device from ``DEVICE_FALLBACK_ROWS`` rows."""
    return device_from_rows(n_samples, DEVICE_FALLBACK_ROWS)


try:
    from pyutilz.performance.kernel_tuning.registry import TunerSpec, kernel_tuner

    _SPEC: "TunerSpec | None" = kernel_tuner(
        kernel_name="fe_xi_columns_device",
        variant_fns=(),  # GPU-only cupy path; covered by salt
        tuner=_run_sweep,
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fe_xi_columns_device",
    )
except Exception as e:
    logger.warning("kernel_tuner spec build for fe_xi_columns_device failed: %s", e)
    _SPEC = None


def use_device(n_rows: int) -> bool:
    """Whether ``n_rows`` rows take the device kernel: ``MLFRAME_XI_DEVICE=0|1`` forces a path, else the per-host tuner (fallback ``DEVICE_FALLBACK_ROWS``, the host until a sweep says otherwise)."""
    from ._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled():
        return False
    forced = os.environ.get("MLFRAME_XI_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("fe_xi_columns_device: KTC choose() failed, using the fallback: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS


__all__ = ["xi_columns_device", "use_device"]
