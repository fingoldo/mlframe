"""Multi-frequency Fourier detector for a BATCH of columns, resident on the device.

``_fourier_detect_gpu_resident.detect_fourier_freqs_for_col_gpu`` runs a short sequential loop (coarse periodogram scan, two refinement scans, a held-out confirmation, a least-squares
deflation) for ONE column; it is launch- and sync-bound (about 100 launches and ten blocking scalar reads per column on a few tens of thousands of rows). The columns of one fit are
independent and share their row split, so here the whole loop runs for ``C`` columns at once: every kernel takes a ``(C, n)`` block and a per-column frequency list, the host reads one
small scalar vector per phase of an iteration instead of one per column, and the per-column decisions (stop, duplicate, tail alias, accept) are made on the host from those vectors.

The sample and target blocks keep the dtype they arrive in (the critical dtype, float32 by default) and the kernels are compiled for it; every sum, the frequencies and the power statistic are double precision. The detected frequencies are grid points of a 0.0125 refinement, far coarser than float32 rounding.
A column whose cubic detrend or deflation system is singular is handled by the single-column path.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

from pyutilz.performance.gpu_hw_launch import block_size_for

import numpy as np

from mlframe._numerical_floors import VARIANCE_FLOOR
from mlframe.utils.env_flags import env_int
from mlframe.feature_selection.filters._kernel_types import render
from .._gpu_raw_module import LazyRawModule
from ._fourier_core_cycles import core_span, freq_is_tail_aliased

logger = logging.getLogger(__name__)

__all__ = ["detect_fourier_freqs_batch_gpu"]


def _block_for(kernel: Any, n: int) -> int:
    """Threads per block: the kernels keep one partial per warp in shared arrays of 32 rows, so any warp multiple up to 1024 fits; derived from the device."""
    return block_size_for(kernel, work_items=n, max_width=1024)


_BLOCK_COLS: Optional[int] = env_int("MLFRAME_FOURIER_BLOCK_COLS", 0, minimum=0) or None  # 0/unset: derive the width from the free VRAM
_MIN_STD = 1e-9
_MIN_Y_SS = VARIANCE_FLOOR
_DUPLICATE_WINDOW = 0.25
_MIN_VAL_CORR_FLOOR = 0.30
_REFINE_COARSE = (0.25, 0.05)  # (half width, step) of the first local scan around the coarse peak
_REFINE_FINE = (0.05, 0.0125)  # (half width, step) of the second scan around the first refinement
_MIN_REFINE_FREQ = 0.05

_SRC_TEMPLATE = r"""
// sin / cos of w * z for a sample type T. A 4-byte T: the argument is reduced to [-pi, pi] in {ACC} (a few {ACC} operations) and the sine and cosine are taken in T, the full {ACC} sincos
// being ~30 times slower on the cards this runs on; an 8-byte T keeps the {ACC} sincos. One template: the branch is on sizeof(T), a compile-time constant. Outside the extern "C" block: C linkage cannot template.
template <typename T>
__device__ __forceinline__ void trig(const {ACC} w, const T z, {ACC}* s, {ACC}* c) {
    if (sizeof(T) == 4) {
        const {ACC} a = w * ({ACC})z;
        const {ACC} r = a - 6.283185307179586 * rint(a * 0.15915494309189535);
        T sf, cf;
        sincos((T)r, &sf, &cf);
        *s = ({ACC})sf;
        *c = ({ACC})cf;
    } else {
        sincos(w * ({ACC})z, s, c);
    }
}
// Kahan-compensated add of v into acc with the running compensation comp (both in the sample type)
#define KAHAN_ADD(acc, comp, v) { const {REAL} yk_ = (v) - comp; const {REAL} tk_ = acc + yk_; comp = (tk_ - acc) - yk_; acc = tk_; }
extern "C" {
__device__ __forceinline__ {ACC} power_term({ACC} sv, {ACC} vv, {ACC} vy, {ACC} n, {ACC} y_ss) {
    const {ACC} v_ss = vv - sv * sv / n;
    const bool ok = (v_ss > 1e-12 * vv) && (v_ss >= @VARIANCE_FLOOR@) && (y_ss >= @VARIANCE_FLOOR@);
    const {ACC} denom = v_ss * y_ss;
    return (ok && denom > 0.0) ? (vy * vy) / denom : 0.0;
}
// grid (F, C): power of column c's centred target at its f-th frequency; padded frequencies (f >= nfreq[c]) get -1
__global__ void periodogram_power_b(const {REAL}* Z, const {REAL}* YC, const {ACC}* freqs, const int* nfreq, const {REAL}* y_ss, {ACC}* out, const long long n, const int F) {
    const int c = blockIdx.y, f = blockIdx.x;
    if (f >= nfreq[c]) { if (threadIdx.x == 0) out[(long long)c * F + f] = -1.0; return; }
    const {REAL}* z = Z + (long long)c * n;
    const {REAL}* yc = YC + (long long)c * n;
    const {ACC} w = 6.283185307179586 * freqs[(long long)c * F + f];
    // Per-thread partials in the SAMPLE type with Kahan compensation, combined in {ACC} below: the sine and cosine are taken in the sample type anyway (trig above), so {ACC} partials
    // would only pay the 1:32 FP64 rate of the card for no accuracy the inputs carry (30000 x 64 columns x 96 frequencies: 77.6 -> 35.6 ms, power within 2.4e-5 relative, p99 4e-7, the argmax
    // frequency of every column unchanged); the compensation keeps the long per-thread sums at the precision of their terms.
    {REAL} ss = 0, sc = 0, vvs = 0, vvc = 0, vys = 0, vyc = 0;
    {REAL} c_ss = 0, c_sc = 0, c_vvs = 0, c_vvc = 0, c_vys = 0, c_vyc = 0;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        {ACC} s_, co_;
        trig(w, z[i], &s_, &co_);
        const {REAL} s = ({REAL})s_, co = ({REAL})co_, y = yc[i];
        KAHAN_ADD(ss, c_ss, s) KAHAN_ADD(sc, c_sc, co) KAHAN_ADD(vvs, c_vvs, s * s) KAHAN_ADD(vvc, c_vvc, co * co) KAHAN_ADD(vys, c_vys, s * y) KAHAN_ADD(vyc, c_vyc, co * y)
    }
    __shared__ {ACC} sh[6][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    {ACC} vals[6] = {({ACC})ss, ({ACC})sc, ({ACC})vvs, ({ACC})vvc, ({ACC})vys, ({ACC})vyc};
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        {ACC} v = vals[k];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) sh[k][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        {ACC} tot[6];
        const int nw = (blockDim.x + 31) >> 5;
        for (int k = 0; k < 6; ++k) { {ACC} t = 0.0; for (int w2 = 0; w2 < nw; ++w2) t += sh[k][w2]; tot[k] = t; }
        out[(long long)c * F + f] = power_term(tot[0], tot[2], tot[4], ({ACC})n, y_ss[c]) + power_term(tot[1], tot[3], tot[5], ({ACC})n, y_ss[c]);
    }
}
// grid (blocks, C): packed A'A (upper, 6) + A'y (3) of A = [1, sin, cos] per column and block
__global__ void sincos_gram_b(const {REAL}* Z, const {REAL}* Y, const {ACC}* fvec, {ACC}* part, const long long n, const int blocks) {
    const int c = blockIdx.y;
    const {REAL}* z = Z + (long long)c * n;
    const {REAL}* y = Y + (long long)c * n;
    {ACC} acc[9];
    #pragma unroll
    for (int t = 0; t < 9; ++t) acc[t] = 0.0;
    const {ACC} w = 6.283185307179586 * fvec[c];
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        {ACC} s, co;
        trig(w, z[i], &s, &co);
        const {ACC} yi = y[i];
        acc[0] += 1.0; acc[1] += s; acc[2] += co; acc[3] += s * s; acc[4] += s * co; acc[5] += co * co;
        acc[6] += yi; acc[7] += s * yi; acc[8] += co * yi;
    }
    __shared__ {ACC} sh[9][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        {ACC} v = acc[k];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) sh[k][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < 9) {
        {ACC} t = 0.0;
        const int nw = (blockDim.x + 31) >> 5;
        for (int w2 = 0; w2 < nw; ++w2) t += sh[threadIdx.x][w2];
        part[((long long)c * blocks + blockIdx.x) * 9 + threadIdx.x] = t;
    }
}
// grid (C): compensated merge of the block partials and the 3x3 solve; coef = 0 for an inactive column, NaN for a singular system
__global__ void solve3_b(const {ACC}* part, const int blocks, const int* active, {ACC}* coef) {
    const int c = blockIdx.x;
    if (!active[c]) { if (threadIdx.x < 3) coef[c * 3 + threadIdx.x] = 0.0; return; }
    __shared__ {ACC} tot[9];
    if (threadIdx.x < 9) {
        {ACC} s = 0.0, comp = 0.0;
        for (int b = 0; b < blocks; ++b) {
            const {ACC} x = part[((long long)c * blocks + b) * 9 + threadIdx.x];
            const {ACC} t = s + x;
            if (fabs(s) >= fabs(x)) comp += (s - t) + x; else comp += (x - t) + s;
            s = t;
        }
        tot[threadIdx.x] = s + comp;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    {ACC} M[3][4] = {{tot[0], tot[1], tot[2], tot[6]}, {tot[1], tot[3], tot[4], tot[7]}, {tot[2], tot[4], tot[5], tot[8]}};
    bool bad = false;
    for (int col = 0; col < 3 && !bad; ++col) {
        int piv = col; {ACC} best = fabs(M[col][col]);
        for (int r = col + 1; r < 3; ++r) if (fabs(M[r][col]) > best) { best = fabs(M[r][col]); piv = r; }
        if (!(best > 0.0) || !isfinite(best)) { bad = true; break; }
        if (piv != col) for (int k = 0; k < 4; ++k) { {ACC} tmp = M[col][k]; M[col][k] = M[piv][k]; M[piv][k] = tmp; }
        for (int r = col + 1; r < 3; ++r) { const {ACC} fct = M[r][col] / M[col][col]; for (int k = col; k < 4; ++k) M[r][k] -= fct * M[col][k]; }
    }
    if (bad) { coef[c * 3] = nan(""); coef[c * 3 + 1] = nan(""); coef[c * 3 + 2] = nan(""); return; }
    {ACC} x[3];
    for (int r = 2; r >= 0; --r) { {ACC} s = M[r][3]; for (int k = r + 1; k < 3; ++k) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
    coef[c * 3] = x[0]; coef[c * 3 + 1] = x[1]; coef[c * 3 + 2] = x[2];
}
// grid (blocks, C): y minus its projection (a zero coefficient vector leaves the column unchanged)
__global__ void apply_deflate_b(const {REAL}* Z, const {REAL}* Y, const {ACC}* coef, const {ACC}* fvec, {REAL}* out, const long long n) {
    const int c = blockIdx.y;
    const {ACC} w = 6.283185307179586 * fvec[c];
    const {ACC} c0 = coef[c * 3], c1 = coef[c * 3 + 1], c2 = coef[c * 3 + 2];
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        {ACC} s, co;
        trig(w, Z[(long long)c * n + i], &s, &co);
        out[(long long)c * n + i] = Y[(long long)c * n + i] - (c0 + c1 * s + c2 * co);
    }
}
}
"""


_SRC = render(_SRC_TEMPLATE.replace("@VARIANCE_FLOOR@", repr(VARIANCE_FLOOR)))  # Intentional float64: {ACC} is float64; {REAL} stays for the per-type modules
_MODULES_BY_REAL = {
    "double": LazyRawModule(_SRC.replace("{REAL}", "double"), ("-std=c++14",)),
    "float": LazyRawModule(_SRC.replace("{REAL}", "float"), ("-std=c++14",)),
}


def _module_for(Z: Any) -> Any:
    """The module compiled for the element type of the sample block ``Z``; the target blocks are cast to that type, the sums accumulate in double precision."""
    return _MODULES_BY_REAL["float" if Z.dtype.itemsize == 4 else "double"]


def _power(cp: Any, Z: Any, YC: Any, freqs: Any, nfreq: Any, y_ss: Any) -> Any:
    """``(C, F)`` periodogram power of each column's centred target at its own frequencies (padded entries are -1)."""
    C, n = int(Z.shape[0]), int(Z.shape[1])
    F = int(freqs.shape[1])
    YC = cp.ascontiguousarray(YC, dtype=Z.dtype)
    y_ss = cp.ascontiguousarray(y_ss, dtype=Z.dtype)
    # Intentional float64: the power statistic (a ratio of sums), accumulated and stored in double precision.
    out = cp.empty((C, F), dtype=cp.float64)
    power = _module_for(Z).get(cp).get_function("periodogram_power_b")
    power((F, C), (_block_for(power, n),), (Z, YC, freqs, nfreq, y_ss, out, cp.int64(n), cp.int32(F)))
    return out


def _deflate(cp: Any, Z: Any, Y: Any, fvec: Any, active: Any) -> "tuple[Any, np.ndarray]":
    """Deflate ``[1, sin, cos]`` at ``fvec[c]`` out of every ACTIVE column; returns the new block and the host mask of columns whose 3x3 system was singular (their rows hold NaN)."""
    C, n = int(Z.shape[0]), int(Z.shape[1])
    mod = _module_for(Z).get(cp)
    gram = mod.get_function("sincos_gram_b")
    deflate = mod.get_function("apply_deflate_b")
    gram_threads = _block_for(gram, n)
    blocks = max(1, min(64, (n + gram_threads - 1) // gram_threads))
    Y = cp.ascontiguousarray(Y, dtype=Z.dtype)
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    part = cp.empty((C * blocks, 9), dtype=cp.float64)
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    coef = cp.empty((C, 3), dtype=cp.float64)
    out = cp.empty((C, n), dtype=Z.dtype)  # the deflated target keeps the sample block's dtype; the projection is accumulated in double precision
    gram((blocks, C), (gram_threads,), (Z, Y, fvec, part, cp.int64(n), cp.int32(blocks)))
    mod.get_function("solve3_b")(
        (C,), (32,), (part, cp.int32(blocks), active, coef)
    )  # launch-ok: one warp: threads 0-8 fold the nine partial sums, thread 0 solves the 3x3 system; a wider block adds nothing
    deflate((blocks, C), (_block_for(deflate, n),), (Z, Y, coef, fvec, out, cp.int64(n)))
    singular = ~np.isfinite(cp.asnumpy(coef)).all(axis=1)
    return out, singular


def _scan_freqs(centers: np.ndarray, half: float, step: float) -> "tuple[np.ndarray, np.ndarray]":
    """Per-column candidate frequencies of one local refinement scan: ``[center, lo, lo + step, ...]`` with ``lo = max(0.05, center - half)``, padded to a common width; plus the counts."""
    lists = []
    for ctr in centers:
        lo = max(_MIN_REFINE_FREQ, float(ctr) - half)
        hi = float(ctr) + half
        n_steps = round((hi - lo) / step) + 1
        # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
        lists.append(np.concatenate([[float(ctr)], lo + step * np.arange(n_steps, dtype=np.float64)]))
    width = max(len(a) for a in lists)
    mat = np.zeros((len(lists), width))
    for k, a in enumerate(lists):
        mat[k, : len(a)] = a
    return mat, np.array([len(a) for a in lists], dtype=np.int32)


def _argmax_freq(cp: Any, power: Any, freqs_host: np.ndarray) -> np.ndarray:
    """Frequency at the first maximum of each row of ``power`` (one small transfer)."""
    best = cp.asnumpy(cp.argmax(power, axis=1))
    return np.asarray(freqs_host[np.arange(len(best)), best])


def _run_group(
    cp: Any,
    Z_tr: np.ndarray,
    Z_va: np.ndarray,
    Y_tr: np.ndarray,
    Y_va: np.ndarray,
    grids: "list[list[float]]",
    spans: "list[Any]",
    min_val_corr: float,
    max_freqs: int,
    single: Callable[[int], list],
) -> "list[list[float]]":
    """The detector loop for one group of columns with the same train / validation sizes. ``single(k)`` runs the single-column path for column ``k`` (singular systems)."""
    C = Z_tr.shape[0]
    results: "list[list[float]]" = [[] for _ in range(C)]
    zt, zv = cp.asarray(Z_tr), cp.asarray(Z_va)
    yt, yv = cp.asarray(Y_tr), cp.asarray(Y_va)

    # cubic detrend of the target in z, fitted on the train rows and applied to both splits (batched normal equations)
    def vander(z):
        """``(C, n, 4)`` design ``[z^3, z^2, z, 1]``."""
        z2 = z * z
        return cp.stack([z2 * z, z2, z, cp.ones_like(z)], axis=2)

    Vt, Vv = vander(zt), vander(zv)
    try:
        VtT = Vt.transpose(0, 2, 1)
        coef = cp.linalg.solve(VtT @ Vt, VtT @ yt[:, :, None])
    except np.linalg.LinAlgError as _exc:
        logger.warning("_run_group fell back to its default after %s", type(_exc).__name__)
        return [single(k) for k in range(C)]
    yt = yt - (Vt @ coef)[:, :, 0]
    yv = yv - (Vv @ coef)[:, :, 0]
    stds = cp.asnumpy(cp.stack([cp.std(yt, axis=1), cp.std(yv, axis=1)]))
    active = [k for k in range(C) if stds[0, k] >= _MIN_STD and stds[1, k] >= _MIN_STD]
    eff_min = max(float(min_val_corr), _MIN_VAL_CORR_FLOOR)

    g_width = max(len(g) for g in grids)
    g_host = np.zeros((C, g_width))
    g_count = np.array([len(g) for g in grids], dtype=np.int32)
    for k, g in enumerate(grids):
        g_host[k, : len(g)] = g
    g_dev, g_count_dev = cp.asarray(g_host), cp.asarray(g_count)

    for _ in range(max(1, int(max_freqs))):
        if not active:
            break
        yc = yt - yt.mean(axis=1, keepdims=True)
        sc = cp.asnumpy(cp.stack([cp.std(yt, axis=1), cp.std(yv, axis=1), (yc * yc).sum(axis=1)]))
        live = [k for k in active if sc[0, k] >= _MIN_STD and sc[1, k] >= _MIN_STD and sc[2, k] >= _MIN_Y_SS]
        active = live
        if not active:
            break
        y_ss = cp.asarray(np.where(sc[2] >= _MIN_Y_SS, sc[2], 1.0))
        coarse = _argmax_freq(cp, _power(cp, zt, yc, g_dev, g_count_dev, y_ss), g_host)
        for half, step in (_REFINE_COARSE, _REFINE_FINE):
            fm, fc = _scan_freqs(coarse, half, step)
            coarse = _argmax_freq(cp, _power(cp, zt, yc, cp.asarray(fm), cp.asarray(fc), y_ss), fm)
        refined = coarse
        deflate_only, confirm = [], []
        for k in active:
            if any(abs(refined[k] - g) < _DUPLICATE_WINDOW for g in results[k]) or freq_is_tail_aliased(float(refined[k]), spans[k]):
                deflate_only.append(k)
            else:
                confirm.append(k)
        accepted, stopped = [], []
        if confirm:
            yvc = yv - yv.mean(axis=1, keepdims=True)
            yv_ss = (yvc * yvc).sum(axis=1)
            rf = cp.asarray(refined.reshape(C, 1))
            vp = _power(cp, zv, yvc, rf, cp.asarray(np.ones(C, dtype=np.int32)), cp.where(yv_ss >= _MIN_Y_SS, yv_ss, 1.0))
            vp_h, yvss_h = cp.asnumpy(cp.stack([vp[:, 0], yv_ss]))
            for k in confirm:
                if yvss_h[k] < _MIN_Y_SS or vp_h[k] <= 0.0 or np.sqrt(vp_h[k]) < eff_min:
                    stopped.append(k)
                else:
                    accepted.append(k)
                    results[k].append(float(refined[k]))
        to_deflate = sorted(deflate_only + accepted)
        active = to_deflate
        if not to_deflate:
            break
        mask = np.zeros(C, dtype=np.int32)
        mask[to_deflate] = 1
        mask_dev, fvec = cp.asarray(mask), cp.asarray(refined)
        yt_new, sing_t = _deflate(cp, zt, yt, fvec, mask_dev)
        yv_new, sing_v = _deflate(cp, zv, yv, fvec, mask_dev)
        yt, yv = yt_new, yv_new
        for k_sing in np.flatnonzero((sing_t | sing_v) & (mask == 1)):
            k = int(k_sing)
            # singular [1, sin, cos] system: the single-column path decides (its lstsq / graceful fallback); the rest of this column's loop runs there too
            results[k] = single(int(k))
            active = [a for a in active if a != int(k)]
    return results


def detect_fourier_freqs_batch_gpu(
    jobs: "Sequence[tuple[np.ndarray, np.ndarray, Sequence[float]]]",
    *,
    min_val_corr: float,
    min_rows: int,
    max_freqs: int,
    fourier_detect_max_n: int,
    single: Callable[[np.ndarray, np.ndarray, Sequence[float]], list],
) -> "list[list[float]]":
    """Detected z-space frequencies for each ``(z01, y, f_grid)`` job, the same lists as the single-column detector.

    Columns the single-column preamble rejects (guards, tiny splits) give ``[]``; the others are grouped by their split sizes and run together. ``single(z01, y, f_grid)`` is the single-column
    detector, used for a column whose cubic detrend or deflation system is singular."""
    import cupy as cp

    from ._fourier_detect_gpu_resident import _host_split_for_detect

    results: "list[list[float]]" = [[] for _ in jobs]
    groups: "dict[tuple, list]" = {}
    for j, (z01, y, f_grid) in enumerate(jobs):
        # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
        z01 = np.asarray(z01, dtype=np.float64).ravel()
        prep = _host_split_for_detect(z01, y, f_grid, min_rows=min_rows, fourier_detect_max_n=fourier_detect_max_n)
        if prep is None:
            continue
        grid, z_tr, z_va, y_tr, y_va = prep
        groups.setdefault((z_tr.size, z_va.size), []).append((j, grid, z_tr, z_va, y_tr, y_va, core_span(z01)))
    for all_members in groups.values():
        width = _column_block_width(all_members[0][2].size + all_members[0][3].size, len(all_members))
        for start in range(0, len(all_members), width):
            members = all_members[start : start + width]
            _run_member_block(cp, members, jobs, results, single, min_val_corr, max_freqs)
    return results


def _column_block_width(n_rows: int, n_cols: int) -> int:
    """Columns per column block: the widest block whose working set fits the VRAM cushion, found by halving from the whole group (set ``_BLOCK_COLS`` to force a width).

    One column holds both splits of the column and of the target, the cubic design matrices (four columns per split) and about eight more row vectors of the periodogram and deflation steps:
    ``n_rows * 8 * 22`` bytes in double.
    """
    if _BLOCK_COLS is not None:
        return max(1, int(_BLOCK_COLS))
    from .._fe_gpu_vram import fe_gpu_has_vram_cushion

    width = max(1, int(n_cols))
    while width > 1 and not fe_gpu_has_vram_cushion(width * int(n_rows) * 8 * 22):
        width //= 2
    return width


def _run_member_block(cp: Any, members: list, jobs: Any, results: list, single: Any, min_val_corr: float, max_freqs: int) -> None:
    """One column block of a group: stack its columns, run the detector loop and write the lists into ``results``."""
    Z_tr = np.ascontiguousarray(np.stack([m[2] for m in members]))
    Z_va = np.ascontiguousarray(np.stack([m[3] for m in members]))
    Y_tr = np.ascontiguousarray(np.stack([m[4] for m in members]))
    Y_va = np.ascontiguousarray(np.stack([m[5] for m in members]))
    job_ids = [m[0] for m in members]

    def single_k(k: int, _ids=job_ids) -> list:
        """Single-column detector for the group's column ``k``."""
        z01, y, f_grid = jobs[_ids[k]]
        found: list = single(z01, y, f_grid)
        return found

    out = _run_group(cp, Z_tr, Z_va, Y_tr, Y_va, [m[1] for m in members], [m[6] for m in members], min_val_corr, max_freqs, single_k)
    for k, j in enumerate(job_ids):
        results[j] = out[k]
