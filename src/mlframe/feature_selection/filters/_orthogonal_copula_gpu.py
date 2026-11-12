"""Device twin of ``_orthogonal_copula_mi_fe._copula_columns_njit``: the Miller-Madow copula MI of many all-finite columns against one target.

The host kernel sorts each column, gives every tie run the mean of its 1-based ranks, scales by ``n + 1``, bins the rank into ``nb`` equal-width bins, counts the joint table against the
target's bin codes ``vi`` and forms the plug-in MI with the Miller-Madow correction. Here a block of columns is sorted in one batched device argsort, ``copula_joint`` gives one thread to
each sorted position (binary searches for the bounds of its tie run, the run's mean rank, the bin, one shared-memory histogram per block flushed into the column's ``nb x nb`` table with
integer atomics), and the MI of the integer tables is formed on the host in the host kernel's loop order. The counts are integers, so the tables equal the host's exactly; the MI then
differs from the host's value only by the order of the final floating-point sum, which is the same here (same loops over the same integers).

Columns go in blocks that fit ``_BUDGET_FRACTION`` of the free device memory; any device fault returns ``None`` and the caller runs the host kernel. ``MLFRAME_COPULA_DEVICE=0|1`` forces the
The kernel-tuning cache decides by row count (fallback: the host).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for

from ._device_sort_safe import device_argsort, device_room
from ._ktc_fallback import host_device_sweep, rows_fallback
from ._raw_module_cache import RawModuleCache
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

_BUDGET_FRACTION = 0.25
DEVICE_FALLBACK_ROWS = 10**9  # the tuner decides; until a sweep says the device wins the host kernel runs
_MAX_BINS = 32  # the shared histograms hold _REPL copies of nb * nb counts
_REPL = 8  # must equal REPL in the kernel source

_SRC = r"""
// dtype-ok: Intentional float64: {ACC} evaluates the rank-to-bin map exactly as the host kernel does; {REAL} is the column type.
#define REPL 8  // histogram copies per block: the lanes of a warp spread over them, so equal (rank bin, target bin) cells of neighbouring lanes do not serialise on one address
extern "C" __global__ void copula_joint(const {REAL}* __restrict__ xs, const long long* __restrict__ order, const int* __restrict__ vi, int* __restrict__ joint,
                                        const long long n, const int nb) {
    extern __shared__ int hist[];
    const int cells = nb * nb;
    for (int c = threadIdx.x; c < REPL * cells; c += blockDim.x) hist[c] = 0;
    __syncthreads();
    const long long base = (long long)blockIdx.y * n;
    const {REAL}* x = xs + base;
    int* mine = hist + (threadIdx.x & (REPL - 1)) * cells;
    for (long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x; q < n; q += (long long)gridDim.x * blockDim.x) {
        const {REAL} v = x[q];
        long long start = q, end = q + 1;
        const bool lone = (q == 0 || x[q - 1] < v) && (q + 1 == n || x[q + 1] > v);  // a value with no equal neighbour is its own tie run: the common case on a continuous column
        if (!lone) {
            long long lo = 0, hi = q;  // first index with value v
            while (lo < hi) { const long long mid = (lo + hi) >> 1; if (x[mid] < v) lo = mid + 1; else hi = mid; }
            start = lo;
            lo = q + 1; hi = n;  // one past the last index with value v
            while (lo < hi) { const long long mid = (lo + hi) >> 1; if (x[mid] <= v) lo = mid + 1; else hi = mid; }
            end = lo;
        }
        const {ACC} u = 0.5 * ({ACC})(start + 1 + end) / (({ACC})n + 1.0);
        int ub = (int)floor(u * ({ACC})nb);
        if (ub > nb - 1) ub = nb - 1;
        atomicAdd(&mine[ub * nb + __ldg(&vi[order[base + q]])], 1);
    }
    __syncthreads();
    int* out = joint + (long long)blockIdx.y * cells;
    for (int c = threadIdx.x; c < cells; c += blockDim.x) {
        int t = 0;
        for (int r = 0; r < REPL; ++r) t += hist[r * cells + c];
        if (t) atomicAdd(&out[c], t);
    }
}
"""

_MODULES = {np.dtype(np.float32): RawModuleCache(_SRC.replace("{REAL}", "float").replace("{ACC}", "double")), np.dtype(np.float64): RawModuleCache(_SRC.replace("{REAL}", "double").replace("{ACC}", "double"))}


def _mi_from_joint(joint: np.ndarray, total: float) -> float:
    """The host kernel's Miller-Madow MI of one ``nb x nb`` integer table (same loops, same order)."""
    nb = joint.shape[0]
    px = joint.sum(axis=1).astype(np.float64)  # Intentional float64: count sums
    py = joint.sum(axis=0).astype(np.float64)  # Intentional float64: count sums
    mi = 0.0
    kxy = 0
    for a in range(nb):
        for b in range(nb):
            c = int(joint[a, b])
            if c > 0:
                pxy = c / total
                mi += pxy * (math.log(pxy) - math.log(px[a] / total * (py[b] / total)))
                kxy += 1
    kx = int((px > 0).sum())
    ky = int((py > 0).sum())
    mi -= (kxy - kx - ky + 1) / (2.0 * total)
    return mi if mi > 0.0 else 0.0


def _mi_from_tables(tables: np.ndarray, total: float) -> np.ndarray:
    """Miller-Madow MI of every ``nb x nb`` integer table of ``tables`` (columns, nb, nb), vectorised; equal to :func:`_mi_from_joint` per table up to the order of the floating-point sum."""
    t = tables.astype(np.float64)  # Intentional float64: counts and their logarithms
    px = t.sum(axis=2)
    py = t.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pxy = t / total
        term = np.where(t > 0, pxy * (np.log(pxy) - np.log(px[:, :, None] / total * (py[:, None, :] / total))), 0.0)
    mi = term.sum(axis=(1, 2))
    kxy = (t > 0).sum(axis=(1, 2))
    kx = (px > 0).sum(axis=1)
    ky = (py > 0).sum(axis=1)
    mi = mi - (kxy - kx - ky + 1) / (2.0 * total)
    return np.where(mi > 0.0, mi, 0.0)


def copula_columns_device(cols_rows: np.ndarray, vi: np.ndarray, nb: int) -> Optional[np.ndarray]:
    """Copula MI of each row of the all-finite ``cols_rows`` (columns, n; numpy, or a resident cupy array that is used as is) against the target bin codes ``vi``; ``None`` when the device cannot take it (the caller runs the host kernel)."""
    kk, n = cols_rows.shape
    if kk == 0 or n < 2 or np.dtype(str(cols_rows.dtype)) not in _MODULES or nb > _MAX_BINS:
        return None
    try:
        import cupy as cp

        resident = isinstance(cols_rows, cp.ndarray)

        kernel = _MODULES[np.dtype(str(cols_rows.dtype))].get(cp).get_function("copula_joint")
        vi_d = cp.asarray(np.ascontiguousarray(vi, dtype=np.int32))
        free = device_room(cp)
        per_col = n * (int(cols_rows.dtype.itemsize) * 2 + 8 * 4)  # values, sorted values, order and sort temporaries
        block = int(max(1, min(kk, (_BUDGET_FRACTION * free) // per_col)))
        dyn_smem = _REPL * nb * nb * 4
        threads = block_size_for(kernel, dyn_smem=dyn_smem, work_items=n, pow2=True)
        grid_x = min((n + threads - 1) // threads, 64)
        tables_dev = cp.zeros((kk, nb, nb), dtype=cp.int32)  # one resident joint-histogram table; one D2H after the loop instead of one per chunk
        for c0 in range(0, kk, block):
            w = min(block, kk - c0)
            xd = cols_rows[c0 : c0 + w] if resident else cp.asarray(np.ascontiguousarray(cols_rows[c0 : c0 + w]))
            order = device_argsort(xd, axis=1)
            xs = cp.take_along_axis(xd, order, axis=1)
            if not resident:
                del xd
            joint = tables_dev[c0 : c0 + w]
            kernel((grid_x, w), (threads,), (xs, order, vi_d, joint, np.int64(n), np.int32(nb)), shared_mem=dyn_smem)
            del xs, order, joint
        return _mi_from_tables(cp.asnumpy(tables_dev), float(n))
    except Exception as exc:
        log_throttle(logger, "copula_gpu.failed", logging.WARNING, "device copula MI failed (%s: %s); the host kernel scores the columns", type(exc).__name__, exc)
        return None


def _sweep_inputs(dims: dict) -> tuple:
    """A 40-column float32 block of ``dims["n_samples"]`` rows and the 20 equal-width bin codes of a normal target's ranks."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    cols = rng.standard_normal((40, n)).astype(np.float32)
    cols[:5] += rng.standard_normal(n).astype(np.float32)
    y = cols[0] + rng.standard_normal(n).astype(np.float32)
    ranks = np.argsort(np.argsort(y, kind="stable"), kind="stable") + 1.0
    vi = np.clip(np.floor(ranks / (n + 1.0) * 20).astype(np.int64), 0, 19)
    return (cols, vi, 20)


def _host(cols: np.ndarray, vi: np.ndarray, nb: int) -> np.ndarray:
    """The host kernel's copula MI per column."""
    from ._orthogonal_copula_mi_fe import _copula_columns_njit, _copula_columns_radix_njit

    out = np.empty(cols.shape[0], dtype=np.float64)  # Intentional float64: statistic output
    (_copula_columns_radix_njit if cols.dtype == np.float32 else _copula_columns_njit)(np.ascontiguousarray(cols), vi, nb, out)
    return out


def _device(cols: np.ndarray, vi: np.ndarray, nb: int) -> np.ndarray:
    """The device kernel's copula MI per column; a failed launch raises so the sweep does not record a host fallback as a device time."""
    out = copula_columns_device(cols, vi, nb)
    if out is None:
        raise RuntimeError("device copula MI unavailable")
    return out


_run_sweep = host_device_sweep(__name__, _host, _device, repeats=3, rtol=1e-12)


_SWEEP_ROWS = [5_000, 30_000, 100_000]
_SALT = 1


_fallback_choice = rows_fallback(__name__)


try:
    from pyutilz.performance.kernel_tuning.registry import TunerSpec, kernel_tuner

    _SPEC: "TunerSpec | None" = kernel_tuner(
        kernel_name="fe_copula_columns_device",
        variant_fns=(),  # GPU-only cupy path; covered by salt
        tuner=_run_sweep,
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fe_copula_columns_device",
    )
except Exception as e:
    logger.warning("kernel_tuner spec build for fe_copula_columns_device failed: %s", e)
    _SPEC = None


def use_device(n_rows: int) -> bool:
    """Whether ``n_rows`` rows take the device kernel: the per-host kernel-tuning cache decides (fallback: the host)."""
    from ._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled():
        return False
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("fe_copula_columns_device: KTC choose() failed, using the fallback: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS


__all__ = ["copula_columns_device", "use_device"]
