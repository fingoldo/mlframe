"""Device twin of ``_ksg_cd_fast._mi_cd_columns``: the Ross continuous-versus-discrete k-NN MI of many one-dimensional columns.

The host kernel walks, for every point of a column sorted by value, outward through the points of its own class to find the k-th neighbour distance ``r`` and then counts the points of
ALL classes strictly closer than ``r``. Here the columns are sorted on the device (one batched argsort), and one thread takes one point of one column: it walks the sorted column
skipping the points of other classes (the neighbours of a point sit next to it, so adjacent threads read adjacent memory), then runs the host's exact widen-then-trim count with the same
comparisons. Only ``sum psi(m)`` depends on the data; ``sum psi(k)`` and ``sum psi(class count)`` are class sums formed on the host. Each block writes its partial sum of ``psi(m)``
and the partials are added in a fixed order, so a run is reproducible; they differ from the host's sequential sum only by floating-point rounding (measured: 7e-12 absolute at 100000 rows).

Rows of classes with a single member are dropped before the sort (the host drops them from the sorted list, which leaves the order of the others unchanged). Columns go to the device in blocks
that fit ``_BUDGET_FRACTION`` of the free memory; on any device fault the caller runs the host kernel.

The kernel-tuning cache decides by row count (fallback ``DEVICE_FALLBACK_ROWS``).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np


from ._device_sort_safe import device_argsort, device_room
from ._ktc_fallback import device_from_rows
from ._raw_module_cache import RawModuleCache
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

_BUDGET_FRACTION = 0.25
DEVICE_FALLBACK_ROWS = 20_000
_BLOCKS_PER_COLUMN = 64

_SRC = r"""
// dtype-ok: Intentional float64: the host kernel works on noised double columns; the distance comparisons must match it exactly.
#define POS_INF __longlong_as_double(0x7ff0000000000000LL)
extern "C" __global__ void ksg_psi_m_sums(const double* __restrict__ xs, const int* __restrict__ ds, const long long nk, const int* __restrict__ counts,
                                          const int nn, const double* __restrict__ psi, double* __restrict__ partial) {
    __shared__ double red[256];
    const int col = blockIdx.y;
    const double* x = xs + (long long)col * nk;
    const int* d = ds + (long long)col * nk;
    double acc = 0.0;
    for (long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x; j < nk; j += (long long)gridDim.x * blockDim.x) {
        const int c = d[j];
        const int cnt = counts[c];
        const int k = min(nn, cnt - 1);
        const double v = x[j];
        long long lo = j - 1, hi = j + 1;
        double r = 0.0;
        for (int t = 0; t < k; ++t) {
            while (lo >= 0 && d[lo] != c) --lo;
            while (hi < nk && d[hi] != c) ++hi;
            const double dl = lo >= 0 ? v - x[lo] : POS_INF;
            const double dh = hi < nk ? x[hi] - v : POS_INF;
            if (dl <= dh) { r = dl; --lo; } else { r = dh; ++hi; }
        }
        const double lo_key = v - r, hi_key = v + r;
        long long lo_m = j;
        while (lo_m > 0 && x[lo_m - 1] >= lo_key) --lo_m;
        long long hi_m = j + 1;
        while (hi_m < nk && x[hi_m] <= hi_key) ++hi_m;
        while (lo_m < hi_m && v - x[lo_m] >= r) ++lo_m;
        while (hi_m > lo_m && x[hi_m - 1] - v >= r) --hi_m;
        acc += psi[hi_m - lo_m];
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[(long long)col * gridDim.x + blockIdx.x] = red[0];
}
"""

_MODULE = RawModuleCache(_SRC)
_THREADS = 256


def mi_cd_columns_device(X: np.ndarray, d: np.ndarray, n_classes: int, n_neighbors: int) -> Optional[np.ndarray]:
    """Ross MI (nats) per column of the noised, scaled ``X`` (rows, columns) with integer labels ``d``; ``None`` when the device cannot take it (the caller runs the host kernel)."""
    n, n_cols = X.shape
    if n == 0 or n_cols == 0:
        return None
    try:
        import cupy as cp

        counts = np.bincount(d, minlength=n_classes).astype(np.int64)
        keep = counts[d] > 1
        n_kept = int(keep.sum())
        if n_kept == 0:
            return np.zeros(n_cols)
        # psi at integers 1..n via psi(m+1) = psi(m) + 1/m, as the host table
        psi = np.zeros(n + 2)
        psi[1] = -0.5772156649015329
        psi[2:] = psi[1] + np.cumsum(1.0 / np.arange(1, n + 1))
        multi = counts > 1
        k_c = np.minimum(n_neighbors, counts - 1)
        sum_k = float((counts[multi] * psi[k_c[multi]]).sum())
        sum_cnt = float((counts[multi] * psi[counts[multi]]).sum())
        d_dev = cp.asarray(np.ascontiguousarray(d[keep], dtype=np.int32))
        counts_dev = cp.asarray(counts.astype(np.int32))
        psi_dev = cp.asarray(psi)
        kernel = _MODULE.get(cp).get_function("ksg_psi_m_sums")
        free = device_room(cp)
        per_col = n_kept * (8 + 8 + 8 + 4)  # values, sort order, sorted values, sorted labels
        block = int(max(1, min(n_cols, (_BUDGET_FRACTION * free) // per_col)))
        X_kept = X if n_kept == n else X[keep]
        out = np.empty(n_cols)
        for c0 in range(0, n_cols, block):
            xd = cp.ascontiguousarray(cp.asarray(np.ascontiguousarray(X_kept[:, c0 : c0 + block])).T)  # uploaded as laid out on the host, transposed to (columns, rows) on the device
            order = device_argsort(xd, axis=1)
            xs = cp.ascontiguousarray(cp.take_along_axis(xd, order, axis=1))
            ds = d_dev[order]
            del xd
            width = xs.shape[0]
            partial = cp.empty((width, _BLOCKS_PER_COLUMN), dtype=cp.float64)
            kernel((_BLOCKS_PER_COLUMN, width), (_THREADS,), (xs, ds, np.int64(n_kept), counts_dev, np.int32(n_neighbors), psi_dev, partial))
            sum_m = cp.asnumpy(partial.sum(axis=1))
            out[c0 : c0 + width] = np.maximum(psi[n_kept] + (sum_k - sum_cnt - sum_m) / n_kept, 0.0)
            del xs, ds, order, partial
        return out
    except Exception as exc:
        log_throttle(logger, "ksg_cd_gpu.failed", logging.WARNING, "device KSG MI failed (%s: %s); the host kernel scores the columns", type(exc).__name__, exc)
        return None


def _sweep_inputs(dims: dict) -> tuple:
    """A noised 60-column block of ``dims["n_samples"]`` rows with a 3-class label."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    X = rng.standard_normal((n, 60))
    d = rng.integers(0, 3, n).astype(np.int64)
    X[:, :10] += d[:, None] * 0.5
    return (X, d)


def _host(X: np.ndarray, d: np.ndarray) -> np.ndarray:
    """The host kernel's MI per column."""
    from ._ksg_cd_fast import _mi_cd_columns

    return np.asarray(_mi_cd_columns(np.ascontiguousarray(X), d, int(d.max()) + 1, 3))


def _device(X: np.ndarray, d: np.ndarray) -> np.ndarray:
    """The device kernel's MI per column; a failed launch raises so the sweep does not record a host fallback as a device time."""
    out = mi_cd_columns_device(X, d, int(d.max()) + 1, 3)
    if out is None:
        raise RuntimeError("device KSG unavailable")
    return out


def _run_sweep() -> list:
    """Time the host and the device kernel per row count; the device must agree with the host."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return sweep_backend_grid({"host": _host, "device": _device}, {"n_samples": _SWEEP_ROWS}, _sweep_inputs, reference="host", repeats=3, equiv_rtol=1e-9, equiv_atol=1e-9)


_SWEEP_ROWS = [5_000, 30_000, 100_000]
_SALT = 1


def _fallback_choice(n_samples: int) -> str:
    """Pre-sweep fallback: the device from ``DEVICE_FALLBACK_ROWS`` rows."""
    return device_from_rows(n_samples, DEVICE_FALLBACK_ROWS)


try:
    from pyutilz.performance.kernel_tuning.registry import TunerSpec, kernel_tuner

    _SPEC: "TunerSpec | None" = kernel_tuner(
        kernel_name="fe_ksg_cd_device",
        variant_fns=(),  # GPU-only cupy path; covered by salt
        tuner=_run_sweep,
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fe_ksg_cd_device",
    )
except Exception as e:
    logger.warning("kernel_tuner spec build for fe_ksg_cd_device failed: %s", e)
    _SPEC = None


def use_device(n_rows: int) -> bool:
    """Whether ``n_rows`` rows take the device kernel: the per-host kernel-tuning cache decides (fallback ``DEVICE_FALLBACK_ROWS``)."""
    from ._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled():
        return False
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("fe_ksg_cd_device: KTC choose() failed, using the fallback: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS


__all__ = ["mi_cd_columns_device", "use_device"]
