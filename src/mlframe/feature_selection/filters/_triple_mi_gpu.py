"""Device twin of ``_orthogonal_triplet_fe._triple_mi_kernel``: the plug-in MI of every triple of joint quantile codes with the target.

Win axis: the host kernel makes one counting pass over the rows per triple (C(p, 3) passes, each into a ``base**3 * n_y`` table). Here one block owns one triple: its threads stride over the rows ONCE, reading the three
codes (one byte each, coalesced along the column) and the class, and count into a shared-memory table of ``base**3 * n_y`` integers; the MI of the table is finished in the same kernel with the host's formula and cell
order (cell-major, class-minor), so the value equals the host's up to the device ``log``. Triples are enumerated in lexicographic order, as the host writes them.

Handled when the table fits the shared budget (``base**3 * n_y * 4`` bytes) and the codes fit a byte; otherwise the host kernel runs.
"""

from __future__ import annotations

import logging
from typing import Any
import os

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for

from mlframe.utils.log_throttle import log_throttle

from ._raw_module_cache import RawModuleCache
from ._kernel_types import render

logger = logging.getLogger(__name__)

SHARED_BUDGET_BYTES = 40 * 1024
#: Rows x triples from which the device is the default choice (the host kernel is memory-latency bound well below this).
DEVICE_MIN_WORK = 5_000_000
TRIPLES_PER_LAUNCH = 1 << 20

_SRC = render(r"""
extern "C" __global__ void triple_mi(const signed char* __restrict__ codes, const unsigned char* __restrict__ Y, const int* __restrict__ TI, const int* __restrict__ TJ,
                                     const int* __restrict__ TK, const long long n, const long long stride, const int base, const int n_y, const int n_cells, {ACC}* __restrict__ out) {
    extern __shared__ int sh[];                       // (n_cells, n_y) joint table, cell-major
    const int t = blockIdx.x;
    const signed char* ci = codes + (long long)TI[t] * stride;   // rows are padded to a multiple of 4 bytes (``stride``), so every row starts on a 4-byte boundary
    const signed char* cj = codes + (long long)TJ[t] * stride;
    const signed char* ck = codes + (long long)TK[t] * stride;
    const int size = n_cells * n_y;
    for (int s = threadIdx.x; s < size; s += blockDim.x) sh[s] = 0;
    __syncthreads();
    // four rows per thread: one 4-byte load per array instead of four 1-byte loads, then the (up to three) rows left over
    const unsigned int* ci4 = (const unsigned int*)ci;
    const unsigned int* cj4 = (const unsigned int*)cj;
    const unsigned int* ck4 = (const unsigned int*)ck;
    const unsigned int* y4 = (const unsigned int*)Y;
    for (long long q = threadIdx.x; q < (n >> 2); q += blockDim.x) {
        const unsigned int a = ci4[q], b = cj4[q], c = ck4[q], yy = y4[q];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int sh8 = 8 * u;
            const int cell = (((int)(signed char)((a >> sh8) & 0xff) * base + (int)(signed char)((b >> sh8) & 0xff)) * base + (int)(signed char)((c >> sh8) & 0xff)) * n_y + (int)((yy >> sh8) & 0xff);
            atomicAdd(&sh[cell], 1);
        }
    }
    for (long long r = (n & ~3LL) + threadIdx.x; r < n; r += blockDim.x) {
        const int cell = (((int)ci[r] * base + (int)cj[r]) * base + (int)ck[r]) * n_y + (int)Y[r];
        atomicAdd(&sh[cell], 1);
    }
    __syncthreads();
    // The MI tail is spread over the block: class totals, then one term per cell, then a short ordered sum.
    __shared__ long long pys[16];
    __shared__ {ACC} term[1024];
    if (threadIdx.x < n_y) {
        long long tot = 0;
        for (int c = 0; c < n_cells; ++c) tot += sh[c * n_y + threadIdx.x];
        pys[threadIdx.x] = tot;
    }
    __syncthreads();
    const {ACC} nn = ({ACC})n;
    for (int c = threadIdx.x; c < n_cells; c += blockDim.x) {
        long long px = 0;
        for (int v = 0; v < n_y; ++v) px += sh[c * n_y + v];
        {ACC} tc = 0.0;
        if (px != 0) {
            for (int v = 0; v < n_y; ++v) {
                const long long cnt = sh[c * n_y + v];
                if (cnt == 0) continue;
                tc += (({ACC})cnt / nn) * log((({ACC})cnt * nn) / (({ACC})px * ({ACC})pys[v]));
            }
        }
        term[c] = tc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        {ACC} mi = 0.0;
        for (int c = 0; c < n_cells; ++c) mi += term[c];
        out[t] = mi;
    }
}
""")

_MODULE = RawModuleCache(_SRC)


def _triples(p: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Every ``i < j < k`` of ``range(p)`` in lexicographic order (the order the host kernel writes them)."""
    i, j, k = np.nonzero(_strict_order_mask(p))
    return i.astype(np.int32), j.astype(np.int32), k.astype(np.int32)


def _strict_order_mask(p: int) -> np.ndarray:
    """Boolean ``(p, p, p)`` mask of the index triples with ``i < j < k``."""
    idx = np.arange(p)
    return (idx[:, None, None] < idx[None, :, None]) & (idx[None, :, None] < idx[None, None, :])


def triple_mi_gpu(codes: np.ndarray, y: np.ndarray, base: int, n_y: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None":
    """``(out_i, out_j, out_k, out_mi)`` as the host kernel returns them, or ``None`` when the device path does not apply (table too big for shared memory, codes wider than a byte, more than 16 classes, no cupy).

    ``codes`` is ``(p, n)`` with values in ``[0, base)``; ``y`` the dense class code of each row.
    """
    p, n = int(codes.shape[0]), int(codes.shape[1])
    n_cells = int(base) ** 3
    if base > 127 or n_y > 16 or n_cells > 1024 or n_cells * n_y * 4 > SHARED_BUDGET_BYTES or p < 3:
        return None
    try:
        import cupy as cp
    except Exception as exc:
        log_throttle(logger, "triple_mi_gpu.cupy_unavailable", logging.WARNING, "device triple MI unavailable (%s); the host triple kernel runs", exc)
        return None
    try:
        return _triple_mi_device(cp, codes, y, base, n_y, p, n, n_cells)
    except Exception as exc:
        log_throttle(logger, "triple_mi_gpu.launch_failed", logging.WARNING, "device triple MI failed (%s: %s); the host triple kernel runs", type(exc).__name__, exc)
        return None


def _triple_mi_device(cp: Any, codes: np.ndarray, y: np.ndarray, base: int, n_y: int, p: int, n: int, n_cells: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """The launch loop of ``triple_mi_gpu``: ``(out_i, out_j, out_k, out_mi)`` for every triple, one block per triple."""
    ti, tj, tk = _triples(p)
    m = int(ti.size)
    kernel = _MODULE.get(cp).get_function("triple_mi")
    stride = (n + 3) & ~3  # rows padded to 4 bytes for the 4-byte loads of the kernel; the pad is never read as a row
    y_dev = cp.zeros(stride, dtype=cp.uint8)
    y_dev[:n] = cp.asarray(np.ascontiguousarray(y, dtype=np.uint8))
    out = np.empty(m, dtype=np.float64)  # Intentional float64: MI statistic vector compared with the host's
    threads = block_size_for(kernel, dyn_smem=n_cells * n_y * 4, work_items=n)

    def upload(cols: np.ndarray) -> Any:
        """The code rows ``cols`` as a padded byte matrix on the device (a column block of the code matrix)."""
        dev = cp.zeros((cols.shape[0], stride), dtype=cp.int8)
        dev[:, :n] = cp.asarray(np.ascontiguousarray(cols, dtype=np.int8))
        return dev

    whole = None  # the full code matrix, uploaded once when a block of triples touches every column (the usual case)
    for s in range(0, m, TRIPLES_PER_LAUNCH):
        e = min(m, s + TRIPLES_PER_LAUNCH)
        used = np.unique(np.concatenate([ti[s:e], tj[s:e], tk[s:e]]))
        if used.shape[0] == p:
            if whole is None:
                whole = upload(codes)
            codes_dev, bi, bj, bk = whole, ti[s:e], tj[s:e], tk[s:e]
        else:  # a block of triples that touches few columns uploads only those columns, so the device holds the block's columns, not the whole matrix
            remap = np.zeros(p, dtype=np.int32)
            remap[used] = np.arange(used.shape[0], dtype=np.int32)
            codes_dev, bi, bj, bk = upload(codes[used]), remap[ti[s:e]], remap[tj[s:e]], remap[tk[s:e]]
        # Intentional float64: MI statistic vector compared with the host's
        d_out = cp.empty(e - s, dtype=cp.float64)
        kernel((e - s,), (threads,), (codes_dev, y_dev, cp.asarray(bi), cp.asarray(bj), cp.asarray(bk), np.int64(n), np.int64(stride), np.int32(base), np.int32(n_y), np.int32(n_cells), d_out), shared_mem=n_cells * n_y * 4)  # launch-ok: one block per triple is the design
        out[s:e] = cp.asnumpy(d_out)
    return ti.astype(np.int64), tj.astype(np.int64), tk.astype(np.int64), out


def use_device_triples(n_rows: int, n_triples: int) -> bool:
    """Whether ``n_triples`` triples over ``n_rows`` rows are scored on the device: ``MLFRAME_TRIPLE_MI_DEVICE=0|1`` forces a path, else the per-host kernel-tuning entry ``triple_mi_backend`` (swept over the
    row-triple work), else ``DEVICE_MIN_WORK`` row-triples."""
    forced = os.environ.get("MLFRAME_TRIPLE_MI_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    work = int(n_rows) * int(n_triples)
    if _SPEC is not None:
        try:
            return str(_SPEC.choose(n_samples=work)) == "device"
        except Exception as exc:
            log_throttle(logger, "triple_mi_gpu.choose_failed", logging.WARNING, "triple_mi_backend: KTC choose() failed, using the floor: %s", exc)
    return work >= DEVICE_MIN_WORK


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Kernel-tuning entry ``triple_mi_backend``: the numba triple scan against this device scan, swept over rows x triples
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
_SWEEP_N = 20_000
_SWEEP_ROWS = [_SWEEP_N * k for k in (50, 500, 4000, 34000)]
_SWEEP_BASE = 6
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """Codes of ``p`` columns over 20000 rows with ``p`` chosen so that C(p, 3) is about ``dims["n_samples"] / 20000``, base 6, a binary target tied to the first two columns."""
    from math import comb

    rng = np.random.default_rng(0)
    want = max(1, int(dims["n_samples"]) // _SWEEP_N)
    p = 3
    while comb(p + 1, 3) <= want:
        p += 1
    codes = rng.integers(0, _SWEEP_BASE, (p, _SWEEP_N)).astype(np.int8)
    y = ((codes[0].astype(np.int64) + codes[1]) % 2).astype(np.int64)
    return (codes, y, _SWEEP_BASE, 2)


def _sweep_host(codes: np.ndarray, y: np.ndarray, base: int, n_y: int) -> np.ndarray:
    """The numba triple scan; its MI vector."""
    from ._orthogonal_triplet_fe import _triple_mi_kernel

    return np.asarray(_triple_mi_kernel(codes.astype(np.int64), y, base, n_y)[3])


def _sweep_device(codes: np.ndarray, y: np.ndarray, base: int, n_y: int) -> np.ndarray:
    """The device scan; a failed launch raises, so the sweep never records a host fallback as a device time."""
    res = triple_mi_gpu(codes, y, base, n_y)
    if res is None:
        raise RuntimeError("device triple MI does not apply")
    return res[3]


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    from ._ktc_fallback import host_device_sweep, rows_fallback

    _SPEC: Any = kernel_tuner(
        kernel_name="triple_mi_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-9, atol=1e-12),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=rows_fallback(__name__, "DEVICE_MIN_WORK"),
        gpu_capable=True,
        salt=_SALT,
        cli_label="triple_mi_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the floor decides
    logger.warning("kernel_tuner spec build for triple_mi_backend failed: %s", e)
    _SPEC = None


__all__ = ["triple_mi_gpu", "use_device_triples"]
