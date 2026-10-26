"""Pair-level residual test for the sufficient-summary early stop.

The early stop in ``_fe_sufficient_summary`` argues that no engineered feature can help when no raw column carries information about the
residual. That holds for a feature built from ONE column. A feature built from TWO columns can carry information about the residual
while each column alone carries none (a product, an XOR), because the data-processing inequality bounds the joint ``I(r; raws)``, not the
per-column marginals. This module closes the gap: it scores the joint MI of the residual with every pair of raw columns and compares the
best pair with a Bonferroni-corrected chi-square ceiling over the number of pairs.

The pair grid is coarse (``PAIR_BINS`` levels per column) so the table stays dense at moderate row counts, and the chi-square null
``2 n MI ~ chi2(df)`` holds. A false alarm only keeps the FE search running, which is the safe direction; a too-high ceiling can only fall back
to the old behaviour.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

import numpy as np
from numba import njit, prange

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

PAIR_BINS = 5  # levels per raw column in the pair grid: 5 x 5 x 10 cells stay dense at a few thousand rows
PAIR_ALPHA = 0.001  # family-wise level of the pair ceiling, split over the pairs (Bonferroni); a real interaction clears it by a wide margin
PAIR_WORK_BUDGET = 4_000_000_000  # pair-rows scored per check on the CPU: a few seconds on a multi-core host
PAIR_WORK_BUDGET_GPU = 100_000_000_000  # pair-rows scored per check when the GPU backend is chosen
PAIR_MIN_ROWS = 1500  # fewer rows than this cannot certify a pair: the caller keeps the FE search running
BLOCK_COLS = 256  # raw columns per streamed block: two blocks of int8 codes are all that is held at once
BLOCK_RAM_FRACTION = 0.2  # share of the free RAM the two code blocks may take


@njit(cache=True, parallel=True)
def _pair_mi_best_per_first_v1(codes_t: np.ndarray, r: np.ndarray, k: int, kr: int) -> tuple:
    """First version of the kernel, kept as the reference for the equivalence test: for each column ``i``, the best joint MI of ``(x_i, x_j; r)`` over ``j > i`` and the arg ``j`` (-1 when there is no ``j``)."""
    p, n = codes_t.shape
    # Intentional float64: count / best accumulators of the host reference
    best = np.zeros(p, dtype=np.float64)
    best_j = np.full(p, -1, dtype=np.int64)
    # Intentional float64: count accumulators
    cr = np.zeros(kr, dtype=np.float64)
    for t in range(n):
        cr[r[t]] += 1.0
    for i in prange(p):
        # Intentional float64: count accumulators
        cnt = np.zeros(k * k * kr, dtype=np.float64)
        # Intentional float64: count accumulators
        ca = np.zeros(k * k, dtype=np.float64)
        bi = 0.0
        bj = -1
        for j in range(i + 1, p):
            cnt[:] = 0.0
            ca[:] = 0.0
            for t in range(n):
                cell = np.int64(codes_t[i, t]) * k + np.int64(codes_t[j, t])
                cnt[cell * kr + r[t]] += 1.0
                ca[cell] += 1.0
            mi = 0.0
            for a in range(k * k):
                if ca[a] > 0.0:
                    for q in range(kr):
                        c = cnt[a * kr + q]
                        if c > 0.0:
                            mi += (c / n) * np.log(c * n / (ca[a] * cr[q]))
            if mi > bi:
                bi = mi
                bj = j
        best[i] = bi
        best_j[i] = bj
    return best, best_j


@njit(cache=True)
def _row_best(codes_t: np.ndarray, r: np.ndarray, cr: np.ndarray, i: int, k: int, kr: int) -> tuple:
    """Best joint MI of ``(x_i, x_j; r)`` over ``j > i`` and its ``j``: one pass per ``j`` fills a dense count table, the marginals come from the table."""
    p, n = codes_t.shape
    base = np.empty(n, dtype=np.int32)
    for t in range(n):
        base[t] = np.int32(codes_t[i, t]) * k * kr + np.int32(r[t])
    cnt = np.zeros(k * k * kr, dtype=np.int32)
    best = 0.0
    best_j = -1
    nf = float(n)
    for j in range(i + 1, p):
        cnt[:] = 0
        for t in range(n):
            cnt[base[t] + np.int32(codes_t[j, t]) * kr] += 1
        mi = 0.0
        for a in range(k * k):
            s = 0
            for q in range(kr):
                s += cnt[a * kr + q]
            if s > 0:
                sf = float(s)
                for q in range(kr):
                    c = cnt[a * kr + q]
                    if c > 0:
                        cf = float(c)
                        mi += (cf / nf) * np.log(cf * nf / (sf * cr[q]))
        if mi > best:
            best = mi
            best_j = j
    return best, best_j


@njit(cache=True, parallel=True)
def _pair_mi_best_per_first(codes_t: np.ndarray, r: np.ndarray, k: int, kr: int) -> tuple:
    """For each column ``i``, the best joint MI of ``(x_i, x_j; r)`` over ``j > i`` and the arg ``j`` (-1 when there is no ``j``).

    Row ``i`` costs ``p - i`` pair passes, so a plain split of the rows over the threads leaves the first threads with most of the work. Rows are
    handed out in mirrored pairs ``(m, p - 1 - m)``, which makes every unit the same size.
    """
    p, n = codes_t.shape
    # Intentional float64: count / best accumulators
    best = np.zeros(p, dtype=np.float64)
    best_j = np.full(p, -1, dtype=np.int64)
    # Intentional float64: count accumulators
    cr = np.zeros(kr, dtype=np.float64)
    for t in range(n):
        cr[r[t]] += 1.0
    half = (p + 1) // 2
    for m in prange(half):
        b, bj = _row_best(codes_t, r, cr, m, k, kr)
        best[m] = b
        best_j[m] = bj
        mirror = p - 1 - m
        if mirror != m:
            b, bj = _row_best(codes_t, r, cr, mirror, k, kr)
            best[mirror] = b
            best_j[mirror] = bj
    return best, best_j


@njit(cache=True, parallel=True)
def _pair_mi_block_cpu(a_t: np.ndarray, b_t: np.ndarray, r: np.ndarray, cr: np.ndarray, k: int, kr: int, same: bool) -> tuple:
    """Best joint MI of ``(a_i, b_j; r)`` for one block pair: per row ``i`` of ``a_t`` the best over the columns ``j`` of ``b_t`` (only ``j > i`` when the two blocks are the same block)."""
    pa, n = a_t.shape
    pb = b_t.shape[0]
    # Intentional float64: best accumulator
    best = np.zeros(pa, dtype=np.float64)
    best_j = np.full(pa, -1, dtype=np.int64)
    nf = float(n)
    for i in prange(pa):
        base = np.empty(n, dtype=np.int32)
        for t in range(n):
            base[t] = np.int32(a_t[i, t]) * k * kr + np.int32(r[t])
        cnt = np.zeros(k * k * kr, dtype=np.int32)
        bi = 0.0
        bj = -1
        start = i + 1 if same else 0
        for j in range(start, pb):
            cnt[:] = 0
            for t in range(n):
                cnt[base[t] + np.int32(b_t[j, t]) * kr] += 1
            mi = 0.0
            for a in range(k * k):
                s = 0
                for q in range(kr):
                    s += cnt[a * kr + q]
                if s > 0:
                    sf = float(s)
                    for q in range(kr):
                        c = cnt[a * kr + q]
                        if c > 0:
                            cf = float(c)
                            mi += (cf / nf) * np.log(cf * nf / (sf * cr[q]))
            if mi > bi:
                bi = mi
                bj = j
        best[i] = bi
        best_j[i] = bj
    return best, best_j


_FUSED_SRC = r"""
extern "C" __global__
void pair_mi_block(const signed char* A, const signed char* B, const int* R, {ACC}* out, const {ACC}* cr,
                   int pa, int pb, int n, int k, int kr, int same)
{
    int b = blockIdx.x;
    int a = blockIdx.y;
    if (same && b <= a) {
        if (threadIdx.x == 0) out[(size_t)a * pb + b] = -1;
        return;
    }
    extern __shared__ int hist[];
    __shared__ {ACC} red[128];
    int groups = k * k;
    int cells = groups * kr;
    for (int c = threadIdx.x; c < cells; c += blockDim.x) hist[c] = 0;
    __syncthreads();
    const signed char* ap = A + (size_t)a * n;
    const signed char* bp = B + (size_t)b * n;
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        int idx = ((int)ap[t] * k + (int)bp[t]) * kr + R[t];  // width-ok: a histogram cell below k*k*kr (a few hundred cells in shared memory)
        atomicAdd(&hist[idx], 1);
    }
    __syncthreads();
    {ACC} part = 0;
    {REAL} nf = ({REAL})n;
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        int s = 0;
        for (int q = 0; q < kr; ++q) s += hist[g * kr + q];
        if (s > 0) {
            {REAL} sf = ({REAL})s;
            for (int q = 0; q < kr; ++q) {
                int c = hist[g * kr + q];
                if (c > 0) {
                    {REAL} cf = ({REAL})c;
                    {REAL} term = (cf / nf) * log(cf * nf / (sf * ({REAL})cr[q]));
                    part += ({ACC})term;
                }
            }
        }
    }
    red[threadIdx.x] = part;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) red[threadIdx.x] += red[threadIdx.x + s2];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[(size_t)a * pb + b] = red[0];
}
"""
_FUSED_SRC_V2 = r"""
extern "C" __global__
void pair_mi_block_v2(const signed char* A, const signed char* B, const int* R, {ACC}* out, const {ACC}* cr,
                      int pa, int pb, int n, int k, int kr, int same)
{
    int b = blockIdx.x;
    int a = blockIdx.y;
    if (same && b <= a) {
        if (threadIdx.x == 0) out[(size_t)a * pb + b] = -1;
        return;
    }
    extern __shared__ int hist[];
    __shared__ {ACC} warp_part[32];
    int cells = k * k * kr;
    for (int c = threadIdx.x; c < cells; c += blockDim.x) hist[c] = 0;
    __syncthreads();
    const signed char* ap = A + (size_t)a * n;
    const signed char* bp = B + (size_t)b * n;
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        int idx = ((int)ap[t] * k + (int)bp[t]) * kr + R[t];  // width-ok: a histogram cell below k*k*kr (a few hundred cells in shared memory)
        atomicAdd(&hist[idx], 1);
    }
    __syncthreads();
    {ACC} part = 0;
    {REAL} nf = ({REAL})n;
    for (int c = threadIdx.x; c < cells; c += blockDim.x) {
        int cnt = hist[c];
        if (cnt > 0) {
            int g0 = (c / kr) * kr;
            int s = 0;
            for (int q = 0; q < kr; ++q) s += hist[g0 + q];
            {REAL} cf = ({REAL})cnt;
            {REAL} term = (cf / nf) * log(cf * nf / (({REAL})s * ({REAL})cr[c - g0]));
            part += ({ACC})term;
        }
    }
    for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xffffffffu, part, off);
    if ((threadIdx.x & 31) == 0) warp_part[threadIdx.x >> 5] = part;
    __syncthreads();
    if (threadIdx.x == 0) {
        {ACC} total = 0;
        for (int w = 0; w < (int)(blockDim.x >> 5); ++w) total += warp_part[w];
        out[(size_t)a * pb + b] = total;
    }
}
"""
_FUSED_NAMES = {"v1": (_FUSED_SRC, "pair_mi_block"), "v2": (_FUSED_SRC_V2, "pair_mi_block_v2")}
FUSED_VERSION = "v2"  # v1: a thread per cell group, tree reduction with 7 barriers; v2: a thread per cell, warp-shuffle reduction. v2 is the default (see the profile record)
_FUSED_THREADS = 128  # one block per pair; a multiple of 32 because of the warp shuffle
_FUSED_KERNELS: dict = {}
_FUSED_LOCK = threading.Lock()  # compile-once get-or-create of the kernels is one critical section
_ACC_TYPE = "double"  # Intentional float64: the MI sum and the counts it divides by are statistics compared with a threshold; only the per-cell terms use the crit type


def gpu_real_type() -> str:
    """The ``{REAL}`` type of the fused kernel: ``"float"`` under the crit-dtype relaxation (the default), ``"double"`` when it is switched off."""
    from ._fe_gpu_batch._devices import crit_float_dtype

    return "float" if crit_float_dtype().__name__ == "float32" else "double"


def pair_equiv_tolerance(real: Optional[str] = None) -> tuple:
    """``(rtol, atol)`` within which the GPU variant must agree with the double CPU reference, taken from the machine epsilon of the ``{REAL}`` type as in the
    other crit-dtype tolerances (``max(1e-9, 8 * eps)``): about 1e-6 for the float32 terms, the 1e-9 floor for double.

    The tuning sweep drops a variant that diverges from its reference by more than this. A tolerance tighter than the real rounding of the type drops the GPU
    from every cell, and the sweep then silently picks the CPU everywhere.
    """
    real = gpu_real_type() if real is None else real
    # Intentional float64: tolerance of the double CPU reference
    eps = float(np.finfo(np.float32 if real == "float" else np.float64).eps)
    rtol = max(1e-9, 8 * eps)
    return rtol, rtol * 1e-3


def _fused_kernel(real: str, version: Optional[str] = None):
    """The compiled fused kernel for a ``{REAL}`` type and a kernel version (one module per pair, compiled once per process; cupy also caches the binary on disk)."""
    version = FUSED_VERSION if version is None else version
    with _FUSED_LOCK:
        kernel = _FUSED_KERNELS.get((real, version))
        if kernel is None:
            import cupy as cp

            source, name = _FUSED_NAMES[version]
            kernel = _FUSED_KERNELS[(real, version)] = cp.RawKernel(source.replace("{REAL}", real).replace("{ACC}", _ACC_TYPE), name)
        return kernel


def _block_best_gpu_fused(a_dev, b_dev, r_dev, cr_dev, k: int, kr: int, same: bool, real: str, version: Optional[str] = None) -> tuple:
    """One block pair scored by ONE fused kernel: a CUDA block per pair fills the joint histogram in shared memory and reduces the MI in place.

    ``a_dev``/``b_dev`` are ``(columns, n)`` int8 device arrays, ``r_dev`` the int32 residual codes and ``cr_dev`` its double counts, all resident, so a
    block pair costs one launch and a device argmax: nothing but the winner leaves the device. ``real`` is the ``{REAL}`` type of the per-cell terms.
    """
    import cupy as cp

    pa, n = a_dev.shape
    pb = b_dev.shape[0]
    # Intentional float64: score accumulator of the device reduction
    out = cp.empty(pa * pb, dtype=cp.float64)
    shared = k * k * kr * 4
    _fused_kernel(real, version)((pb, pa), (_FUSED_THREADS,), (a_dev, b_dev, r_dev, out, cr_dev, pa, pb, n, k, kr, int(same)), shared_mem=shared)
    m = int(cp.argmax(out))
    return float(out[m]), (m // pb, m % pb)


def _gpu_block_cols(n_used: int, p: int) -> int:
    """Columns per block on the device: two int8 blocks plus the pair table stay inside a third of the free VRAM."""
    import cupy as cp

    free, _total = cp.cuda.runtime.memGetInfo()
    fit = int((free // 3) // (2 * max(1, n_used) + 8 * BLOCK_COLS))
    return int(max(16, min(BLOCK_COLS, fit, p)))


def _block_best_gpu(a_t: np.ndarray, b_t: np.ndarray, r: np.ndarray, k: int, kr: int, same: bool) -> tuple:
    """The same block pair scored by the project pair-MI dispatcher (numba.cuda / cupy, with its own VRAM fit check and CPU fallback).

    Kept as the measured baseline of the generic path: on 80 thousand pairs at 5000 rows it is about twelve times slower than the CPU kernel, because it is built
    for few pairs over very many rows. The fused kernel above is the GPU path the dispatch chooses."""
    from .batch_pair_mi_gpu import dispatch_batch_pair_mi
    from ._fe_usability_signal import _crit_np_dtype

    pa, pb = a_t.shape[0], b_t.shape[0]
    n = a_t.shape[1]
    if same:
        ia, ib = np.triu_indices(pa, k=1)
        pair_a, pair_b = ia.astype(np.int64), ib.astype(np.int64) + pa
    else:
        pair_a = np.repeat(np.arange(pa, dtype=np.int64), pb)
        pair_b = np.tile(np.arange(pb, dtype=np.int64), pa) + pa
    if pair_a.size == 0:
        return 0.0, (-1, -1)
    factors = np.empty((n, pa + pb), dtype=np.int32)
    factors[:, :pa] = a_t.T
    factors[:, pa:] = b_t.T
    nbins = np.full(pa + pb, k, dtype=np.int32)
    classes_y = r.astype(np.int32)
    freqs_y = np.bincount(r, minlength=kr).astype(_crit_np_dtype()) / max(1, n)
    mi, _backend = dispatch_batch_pair_mi(factors, pair_a, pair_b, nbins, classes_y, freqs_y)
    m = int(np.argmax(mi))
    return float(mi[m]), (int(pair_a[m]), int(pair_b[m] - pa))


def _block_cols_for(n_used: int, p: int) -> int:
    """Columns per streamed block: two int8 blocks of ``n_used`` rows stay inside ``BLOCK_RAM_FRACTION`` of the free RAM."""
    try:
        import psutil

        free = int(psutil.virtual_memory().available)
    except Exception as exc:
        logger.warning("_block_cols_for: psutil free-RAM probe failed (%s); using the fixed block width %d", exc, BLOCK_COLS)
        return int(min(BLOCK_COLS, p))
    fit = int(free * BLOCK_RAM_FRACTION // (2 * max(1, n_used)))
    return int(max(16, min(BLOCK_COLS, fit, p)))


def best_pair_streamed(get_block: Callable[[int, int], np.ndarray], p: int, r: np.ndarray, k: int, kr: int, backend: str, block_cols: int, real: Optional[str] = None) -> tuple:
    """Best ``(mi, (i, j))`` over all pairs of ``p`` columns, streamed in column blocks so only two blocks of codes are alive at once.

    ``get_block(c0, c1)`` returns the ``(c1 - c0, n)`` int8 codes of those columns. ``backend`` is ``"njit"`` (the CPU kernel), ``"gpu_fused"`` (the fused
    cupy kernel, data resident on the device) or ``"gpu_dispatch"`` (the generic project dispatcher, kept as the measured baseline). The diagonal blocks score
    the pairs inside a block, the off-diagonal blocks score the rectangle between block ``I`` and a later block ``J``; together they cover every pair once.
    A device that runs out of memory hands the remaining work to the CPU kernel instead of failing the check.
    """
    # Intentional float64: residual counts accumulated in double
    cr = np.bincount(r, minlength=kr).astype(np.float64)
    best_mi, best_pair = 0.0, (-1, -1)
    starts = list(range(0, p, block_cols))
    xp: Any = None
    r_dev = cr_dev = None
    fused_real = ""
    if backend == "gpu_fused":
        import cupy as cp

        xp = cp
        fused_real = gpu_real_type() if real is None else real
        r_dev, cr_dev = cp.asarray(r.astype(np.int32)), cp.asarray(cr)
    for ia, c0 in enumerate(starts):
        a_t = get_block(c0, min(c0 + block_cols, p))
        a_dev = xp.asarray(a_t) if xp is not None else None
        for jb in range(ia, len(starts)):
            d0 = starts[jb]
            same = jb == ia
            b_t = a_t if same else get_block(d0, min(d0 + block_cols, p))
            try:
                if backend == "gpu_fused":
                    mi, (i, j) = _block_best_gpu_fused(a_dev, a_dev if same else xp.asarray(b_t), r_dev, cr_dev, k, kr, same, fused_real)
                elif backend == "gpu_dispatch":
                    mi, (i, j) = _block_best_gpu(a_t, b_t, r, k, kr, same)
                else:
                    mi, i, j = _block_best_cpu(a_t, b_t, r, cr, k, kr, same)
            except Exception as exc:  # a device out of memory or a driver fault: finish this block pair on the CPU
                if backend == "njit":
                    raise
                log_throttle(logger, "pair_residual_backend_failed", logging.WARNING, "pair residual test: the %s backend failed (%s: %s); this block pair runs on the CPU.", backend, type(exc).__name__, exc)
                mi, i, j = _block_best_cpu(a_t, b_t, r, cr, k, kr, same)
            if mi > best_mi and j >= 0:
                best_mi, best_pair = mi, (c0 + i, d0 + j)
    return best_mi, best_pair


def _block_best_cpu(a_t: np.ndarray, b_t: np.ndarray, r: np.ndarray, cr: np.ndarray, k: int, kr: int, same: bool) -> tuple:
    """One block pair on the CPU kernel: ``(best MI, i, j)`` with positions inside the two blocks."""
    bests, bj = _pair_mi_block_cpu(a_t, b_t, r, cr, k, kr, same)
    i = int(np.argmax(bests))
    return float(bests[i]), i, int(bj[i])


def _coarse_block_getter(raw_codes: np.ndarray, levels: np.ndarray, k: int, rows: slice):
    """``get_block`` that coarsens raw columns to ``k`` levels one column at a time: no ``n x p`` int64 temporary is ever built."""
    n_used = len(range(*rows.indices(raw_codes.shape[0])))

    def get_block(c0: int, c1: int) -> np.ndarray:
        """Coarse int8 codes of columns ``c0:c1``."""
        out = np.empty((c1 - c0, n_used), dtype=np.int8)
        for c in range(c0, c1):
            col = raw_codes[rows, c].astype(np.int64)
            out[c - c0] = np.minimum((col * k) // levels[c], k - 1)
        return out

    return get_block


def _pair_residual_fallback_choice(n_samples: int, n_pairs: int) -> str:
    """Pre-sweep heuristic for the backend: the GPU only when it is present and the work is large enough to repay the launch and the upload."""
    return "gpu_fused" if _fused_gpu_available() and int(n_samples) * int(n_pairs) >= PRT_GPU_MIN_PAIR_ROWS else "njit"


def _fused_gpu_available() -> bool:
    """Whether the fused kernel can run here: cupy imports and sees a device, and the project GPU policy does not switch the GPU off."""
    from ._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled():
        return False
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception as exc:
        logger.warning("_fused_gpu_available: no usable CUDA device (%s); the CPU kernel serves", exc)
        return False


# Measured on the GTX 1050 Ti host (warm, synchronized): the fused GPU kernel has a fixed cost of about 0.5 ms (launch, upload, argmax), so the CPU kernel wins
# while pairs x rows stays under about 1e6 (100 pairs at 1500 and 5000 rows are CPU; 100 pairs at 20000 rows, 1000 pairs at 1500 rows and more are GPU). The
# fallback heuristic uses that product; the sweep grid below starts under the crossover so the cache measures both sides of it.
PRT_GPU_MIN_PAIR_ROWS = 1_000_000
_PRT_SWEEP_N_SAMPLES = [1_500, 5_000, 20_000, 100_000]
_PRT_SWEEP_N_PAIRS_GRID = [100, 1_000, 5_000, 80_000, 500_000]
_PRT_SALT = 2


def _make_pair_residual_inputs(dims: dict) -> tuple:
    """Synthetic coarse codes for ``dims['n_pairs']`` pairs at ``dims['n_samples']`` rows, plus a residual that depends on one planted pair."""
    n = int(dims["n_samples"])
    p = max(4, int(np.ceil((1 + np.sqrt(1 + 8 * int(dims["n_pairs"]))) / 2)))
    rng = np.random.default_rng(0)
    k, kr = PAIR_BINS, 10
    codes = rng.integers(0, k, size=(p, n)).astype(np.int8)
    r = rng.integers(0, kr, size=n).astype(np.int64)
    r[: n // 2] = (codes[1, : n // 2].astype(np.int64) * 2 + codes[2, : n // 2]) % kr
    return codes, r, k, kr


def _run_pair_residual_sweep() -> list:
    """Grid sweep over ``(n_samples, n_pairs)`` -> the fastest equivalent backend per cell (njit or the GPU dispatcher)."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    def run(backend: str):
        """Sweep variant for one backend."""

        def fn(codes, r, k, kr):
            """Best pair MI of the streamed test as a one-element array."""
            mi, _pair = best_pair_streamed(lambda c0, c1: codes[c0:c1], codes.shape[0], r, k, kr, backend, BLOCK_COLS)
            return np.array([mi])

        return fn

    variants = {"njit": run("njit")}
    if _fused_gpu_available():
        variants["gpu_fused"] = run("gpu_fused")  # the generic dispatcher path is not swept: it is orders of magnitude slower on this workload
    grid = {"n_samples": _PRT_SWEEP_N_SAMPLES, "n_pairs": _PRT_SWEEP_N_PAIRS_GRID}
    rtol, atol = pair_equiv_tolerance()
    return list(sweep_backend_grid(variants, grid, _make_pair_residual_inputs, reference="njit", repeats=3, equiv_rtol=rtol, equiv_atol=atol))


from pyutilz.performance.kernel_tuning.registry import kernel_tuner

_PRT_SPEC = kernel_tuner(
    kernel_name="pair_residual_test",
    variant_fns=(_pair_mi_block_cpu,),  # CPU body; the GPU variant is covered by the salt
    tuner=_run_pair_residual_sweep,
    axes={"n_samples": list(_PRT_SWEEP_N_SAMPLES), "n_pairs": list(_PRT_SWEEP_N_PAIRS_GRID)},
    fallback=_pair_residual_fallback_choice,
    gpu_capable=True,
    salt=_PRT_SALT,
    cli_label="pair_residual_test",
)


def pair_residual_backend_choice(n_samples: int, n_pairs: int) -> str:
    """Per-host backend (``"njit"`` or ``"gpu_fused"``) for this shape through the kernel-tuning cache, with the heuristic as the fallback."""
    if not _fused_gpu_available():
        return "njit"
    try:
        choice = _PRT_SPEC.choose(n_samples=int(n_samples), n_pairs=int(n_pairs))
    except Exception as e:  # nosec B110 - a broken cache must not break the check; the heuristic decides
        logger.debug("pair residual kernel-tuning lookup failed (%r), the heuristic decides", e)
        choice = _pair_residual_fallback_choice(n_samples, n_pairs)
    return "gpu_fused" if choice in ("gpu_fused", "gpu", "cuda", "cupy") else "njit"


def pair_residual_test(raw_codes: np.ndarray, raw_nbins: np.ndarray, r_codes: np.ndarray, backend: Optional[str] = None) -> Optional[tuple]:
    """Best pair of raw columns against the residual codes.

    Returns ``(max_pair_mi, ceiling, (i, j))`` with ``i, j`` positions in the raw pool, or ``None`` when the check cannot be made
    (fewer than two columns, too many pairs for the row budget, or too few rows), in which case the caller must not stop the FE search.

    Memory is bounded: rows are strided to the work budget before anything is built, the columns are coarsened one at a time into int8 blocks, and the pairs
    are scored block pair by block pair, so the peak is two blocks of codes plus the dispatcher's own chunk, however wide the pool is. ``backend`` forces
    ``"njit"``, ``"gpu_fused"`` or ``"gpu_dispatch"``; ``None`` takes the kernel-tuning-cache choice.
    """
    from scipy.stats import chi2

    n, p = raw_codes.shape
    if p < 2:
        return None
    n_pairs = p * (p - 1) // 2
    chosen = pair_residual_backend_choice(n, n_pairs) if backend is None else backend
    chosen = "gpu_fused" if chosen == "gpu" else chosen
    budget = PAIR_WORK_BUDGET_GPU if chosen.startswith("gpu") else PAIR_WORK_BUDGET
    stride = 1
    if n_pairs * n > budget:
        stride = int(np.ceil(n_pairs * n / budget))
    n_used = (n + stride - 1) // stride
    if n_used < PAIR_MIN_ROWS:
        return None
    k = int(min(PAIR_BINS, max(2, int(raw_nbins.max()))))
    rows = slice(None, None, stride)
    levels = np.maximum(raw_nbins, 1).astype(np.int64)
    r = np.ascontiguousarray(r_codes[rows].astype(np.int64))
    kr = int(r.max()) + 1
    if kr < 2:
        return None
    block_cols = _gpu_block_cols(n_used, p) if chosen == "gpu_fused" else _block_cols_for(n_used, p)
    best_mi, (i, j) = best_pair_streamed(_coarse_block_getter(raw_codes, levels, k, rows), p, r, k, kr, chosen, block_cols)
    if i < 0:
        return None
    df = (k * k - 1) * (kr - 1)
    ceiling = float(chi2.isf(PAIR_ALPHA / n_pairs, df) / (2.0 * n_used))
    return float(best_mi), ceiling, (int(i), int(j))
