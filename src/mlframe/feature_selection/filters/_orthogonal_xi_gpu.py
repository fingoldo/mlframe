"""Device twin of ``_orthogonal_xi_fe._xi_columns_radix_njit``: Chatterjee's Xi of many float32 columns against one target.

Both paths order the rows of a column by ``(value, tie key, row)`` where the tie key is a 32-bit counter hash of ``(column seed, row)`` (``_orthogonal_xi_fe.column_seeds`` / ``_tie_key``), so the
tie-break stream is arithmetic, not a generator: nothing is drawn on the host or sent over the bus. A block of columns is processed in three launches with no table written between them that the
next one does not need:

1. ``xi_sort_keys``: one 64-bit key per element, the order-preserving 32-bit code of the value (``-0.0`` coded as ``+0.0``) in the high word and the tie key in the low word;
2. a batched stable argsort of the keys (``device_argsort``, cub radix sort): the result is the row order itself, so the walk needs no inverse permutation and no second gather;
3. ``xi_walk``: one thread per sorted position reads its row and its predecessor's row, looks the target class of each up in a small table (``codes`` int32 per row, the right and left ranks per
   class in ``r_tab`` / ``l_tab``, read through the read-only cache) and adds ``|r - r_prev|`` and ``l (n - l)`` into two per-column sums.

The rank sums are integer counts, so every term is an integer held exactly in a double and the block sums are exact in any order (``{ACC}`` is double and is never lowered): the result equals the
host's value to the last bit. Values are ``{REAL}`` (float32 here; a float64 column goes to the host kernel). Operands may already be resident: ``cols`` is a numpy array that is uploaded, or a cupy array
that is used as is. Columns go in blocks that fit ``_BUDGET_FRACTION`` of the free device memory; any device fault returns ``None`` and the caller runs the host kernel. The host/device choice is the
kernel-tuning cache entry ``fe_xi_columns_device`` (fallback: the host).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from ._device_sort_safe import device_argsort, device_room
from ._ktc_fallback import device_from_rows
from ._raw_module_cache import RawModuleCache
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

_BUDGET_FRACTION = 0.25
DEVICE_FALLBACK_ROWS = 10**9  # the tuner decides; until a sweep says the device wins the host kernel runs
_THREADS = 256

_SRC = r"""
// dtype-ok: Intentional float64: {ACC} holds integer rank sums exactly; {REAL} is the column type.
typedef unsigned long long u64;
typedef unsigned int u32;

__device__ __forceinline__ u64 mix64(u64 x) {
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

extern "C" __global__ void xi_sort_keys(const {REAL}* __restrict__ cols, const u64* __restrict__ seeds, u64* __restrict__ keys, const long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long at = (long long)blockIdx.y * n + i;
    u32 u = __float_as_uint(cols[at]);
    if (u == 0x80000000u) u = 0u;  // -0.0 orders as +0.0, as the host comparison treats it
    const u32 ord = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    const u32 tie = (u32)(mix64(seeds[blockIdx.y] + ((u64)i + 1ULL) * 0x9E3779B97F4A7C15ULL) >> 32);
    keys[at] = ((u64)ord << 32) | (u64)tie;
}

extern "C" __global__ void xi_walk(const long long* __restrict__ order, const int* __restrict__ codes, const {ACC}* __restrict__ r_tab, const {ACC}* __restrict__ l_tab,
                                   {ACC}* __restrict__ sums, const long long n) {
    __shared__ {ACC} red_s[256];
    __shared__ {ACC} red_d[256];
    const long long base = (long long)blockIdx.y * n;
    {ACC} s = 0.0, d = 0.0;
    for (long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x; q < n; q += (long long)gridDim.x * blockDim.x) {
        const int c = codes[order[base + q]];
        const {ACC} lq = __ldg(&l_tab[c]);
        d += lq * (({ACC})n - lq);
        if (q > 0) {
            const int p = codes[order[base + q - 1]];
            const {ACC} diff = __ldg(&r_tab[c]) - __ldg(&r_tab[p]);
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

_MODULE = RawModuleCache(_SRC.replace("{REAL}", "float").replace("{ACC}", "double"))


class YSide:
    """The target-side tables of the walk: the class code of each row and the right / left rank of each class, built once per batch and uploaded once."""

    def __init__(self, r_row: np.ndarray, l_row: np.ndarray) -> None:
        uniq, inv = np.unique(r_row, return_inverse=True)
        first = np.zeros(uniq.size, dtype=np.int64)
        first[inv[::-1]] = np.arange(inv.size - 1, -1, -1)  # the first row of each class
        self.codes = np.ascontiguousarray(inv.reshape(-1), dtype=np.int32)
        self.r_tab = np.ascontiguousarray(r_row[first], dtype=np.float64)  # Intentional float64: integer rank counts held exactly
        self.l_tab = np.ascontiguousarray(l_row[first], dtype=np.float64)  # Intentional float64: integer rank counts held exactly
        self._device: Optional[tuple] = None

    def device(self, cp: Any) -> tuple:
        """``(codes, r_tab, l_tab)`` as device arrays, uploaded on first use."""
        if self._device is None:
            self._device = (cp.asarray(self.codes), cp.asarray(self.r_tab), cp.asarray(self.l_tab))
        return self._device


def xi_columns_device(cols: Any, seeds: np.ndarray, r_row: np.ndarray, l_row: np.ndarray, y_side: "YSide | None" = None) -> Optional[np.ndarray]:
    """Xi of each row of ``cols`` (columns, n; float32, numpy or a resident cupy array) with the column tie seeds ``seeds``; ``None`` when the device cannot take it (the caller runs the host kernel)."""
    kk, n = cols.shape
    if kk == 0 or n < 2 or str(cols.dtype) != "float32":
        return None
    try:
        import cupy as cp

        module = _MODULE.get(cp)
        keys_k, walk_k = module.get_function("xi_sort_keys"), module.get_function("xi_walk")
        ys = y_side if y_side is not None else YSide(r_row, l_row)
        codes_d, r_tab_d, l_tab_d = ys.device(cp)
        seeds_d = cp.asarray(np.ascontiguousarray(seeds, dtype=np.uint64))
        resident = isinstance(cols, cp.ndarray)
        free = device_room(cp)
        per_col = n * (4 + 8 + 8 + 8 + 8 + 16)  # values, keys, order, sort temporaries
        block = int(max(1, min(kk, (_BUDGET_FRACTION * free) // per_col)))
        out = np.empty(kk, dtype=np.float64)  # Intentional float64: accumulator output
        grid_x = (n + _THREADS - 1) // _THREADS
        walk_blocks = min(grid_x, 64)
        for c0 in range(0, kk, block):
            w = min(block, kk - c0)
            col_d = cols[c0 : c0 + w] if resident else cp.asarray(np.ascontiguousarray(cols[c0 : c0 + w]))
            keys = cp.empty((w, n), dtype=cp.uint64)
            keys_k((grid_x, w), (_THREADS,), (col_d, seeds_d[c0 : c0 + w], keys, np.int64(n)))
            if not resident:
                del col_d
            order = device_argsort(keys, axis=1)
            del keys
            sums = cp.zeros(2 * w, dtype=cp.float64)
            walk_k((walk_blocks, w), (_THREADS,), (order, codes_d, r_tab_d, l_tab_d, sums, np.int64(n)))
            s = cp.asnumpy(sums).reshape(w, 2)
            den = 2.0 * s[:, 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                out[c0 : c0 + w] = np.where(den > 0.0, 1.0 - float(n) * s[:, 0] / den, 0.0)
            del order, sums
        return out
    except Exception as exc:
        log_throttle(logger, "xi_gpu.failed", logging.WARNING, "device Xi failed (%s: %s); the host kernel scores the columns", type(exc).__name__, exc)
        return None


def _sweep_inputs(dims: dict) -> tuple:
    """A 40-column float32 block of ``dims["n_samples"]`` rows (a quarter of them rounded, so with ties), the column seeds and the ranks of a 3-class target."""
    from ._orthogonal_xi_fe import _target_ranks, column_seeds

    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    cols = rng.standard_normal((40, n)).astype(np.float32)
    cols[:10] = np.round(cols[:10] * 3)
    y = rng.integers(0, 3, n).astype(np.float64)  # Intentional float64: target labels
    r_row, l_row = _target_ranks(y)
    return (cols, column_seeds(0, 40), r_row, l_row)


def _host(cols: np.ndarray, seeds: np.ndarray, r_row: np.ndarray, l_row: np.ndarray) -> np.ndarray:
    """The host kernel's Xi per column."""
    from ._orthogonal_xi_fe import _xi_columns_radix_njit

    out = np.empty(cols.shape[0], dtype=np.float64)  # Intentional float64: accumulator output
    _xi_columns_radix_njit(np.ascontiguousarray(cols), seeds, r_row, l_row, out)
    return out


def _device(cols: np.ndarray, seeds: np.ndarray, r_row: np.ndarray, l_row: np.ndarray) -> np.ndarray:
    """The device kernel's Xi per column; a failed launch raises so the sweep does not record a host fallback as a device time."""
    out = xi_columns_device(cols, seeds, r_row, l_row)
    if out is None:
        raise RuntimeError("device Xi unavailable")
    return out


def _run_sweep() -> list:
    """Time the host and the device kernel per row count; the device must agree with the host."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return sweep_backend_grid({"host": _host, "device": _device}, {"n_samples": _SWEEP_ROWS}, _sweep_inputs, reference="host", repeats=3, equiv_rtol=1e-12, equiv_atol=1e-12)


_SWEEP_ROWS = [5_000, 30_000, 100_000]
_SALT = 2


def _fallback_choice(n_samples: int) -> str:
    """Pre-sweep fallback: the host unless ``DEVICE_FALLBACK_ROWS`` is lowered."""
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
    """Whether ``n_rows`` rows take the device kernel: the per-host kernel-tuning cache decides (fallback ``DEVICE_FALLBACK_ROWS``, the host until a sweep says otherwise)."""
    from ._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled():
        return False
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("fe_xi_columns_device: KTC choose() failed, using the fallback: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS


__all__ = ["YSide", "xi_columns_device", "use_device"]
