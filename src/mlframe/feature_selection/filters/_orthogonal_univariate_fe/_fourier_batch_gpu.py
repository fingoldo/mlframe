"""Multi-frequency Fourier detector for a BATCH of columns, resident on the device.

``_fourier_detect_gpu_resident.detect_fourier_freqs_for_col_gpu`` runs a short sequential loop (coarse periodogram scan, two refinement scans, a held-out confirmation, a least-squares
deflation) for ONE column; it is launch- and sync-bound (about 100 launches and ten blocking scalar reads per column on a few tens of thousands of rows). The columns of one fit are
independent and share their row split, so here the whole loop runs for ``C`` columns at once: every kernel takes a ``(C, n)`` block and a per-column frequency list, the host reads one
small scalar vector per phase of an iteration instead of one per column, and the per-column decisions (stop, duplicate, tail alias, accept) are made on the host from those vectors.

Everything is float64 (the single-column path rounds the inputs to float32 first; the detected frequencies are grid points of a 0.0125 refinement, far coarser than that rounding).
A column whose cubic detrend or deflation system is singular is handled by the single-column path.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ._fourier_core_cycles import core_span, freq_is_tail_aliased

logger = logging.getLogger(__name__)

__all__ = ["detect_fourier_freqs_batch_gpu"]

_LOCK = threading.Lock()
_MODULE: Optional[Any] = None
_THREADS = 256
_MIN_STD = 1e-9
_MIN_Y_SS = 1e-24
_DUPLICATE_WINDOW = 0.25
_MIN_VAL_CORR_FLOOR = 0.30
_REFINE_COARSE = (0.25, 0.05)  # (half width, step) of the first local scan around the coarse peak
_REFINE_FINE = (0.05, 0.0125)  # (half width, step) of the second scan around the first refinement
_MIN_REFINE_FREQ = 0.05

_SRC = r"""
extern "C" {
__device__ __forceinline__ double power_term(double sv, double vv, double vy, double n, double y_ss) {
    const double v_ss = vv - sv * sv / n;
    const bool ok = (v_ss > 1e-12 * vv) && (v_ss >= 1e-24) && (y_ss >= 1e-24);
    const double denom = v_ss * y_ss;
    return (ok && denom > 0.0) ? (vy * vy) / denom : 0.0;
}
// grid (F, C): power of column c's centred target at its f-th frequency; padded frequencies (f >= nfreq[c]) get -1
__global__ void periodogram_power_b(const double* Z, const double* YC, const double* freqs, const int* nfreq, const double* y_ss, double* out, const long long n, const int F) {
    const int c = blockIdx.y, f = blockIdx.x;
    if (f >= nfreq[c]) { if (threadIdx.x == 0) out[(long long)c * F + f] = -1.0; return; }
    const double* z = Z + (long long)c * n;
    const double* yc = YC + (long long)c * n;
    const double w = 6.283185307179586 * freqs[(long long)c * F + f];
    double ss = 0.0, sc = 0.0, vvs = 0.0, vvc = 0.0, vys = 0.0, vyc = 0.0;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        double s, co;
        sincos(w * z[i], &s, &co);
        const double y = yc[i];
        ss += s; sc += co; vvs += s * s; vvc += co * co; vys += s * y; vyc += co * y;
    }
    __shared__ double sh[6][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    double vals[6] = {ss, sc, vvs, vvc, vys, vyc};
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        double v = vals[k];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) sh[k][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double tot[6];
        const int nw = (blockDim.x + 31) >> 5;
        for (int k = 0; k < 6; ++k) { double t = 0.0; for (int w2 = 0; w2 < nw; ++w2) t += sh[k][w2]; tot[k] = t; }
        out[(long long)c * F + f] = power_term(tot[0], tot[2], tot[4], (double)n, y_ss[c]) + power_term(tot[1], tot[3], tot[5], (double)n, y_ss[c]);
    }
}
// grid (blocks, C): packed A'A (upper, 6) + A'y (3) of A = [1, sin, cos] per column and block
__global__ void sincos_gram_b(const double* Z, const double* Y, const double* fvec, double* part, const long long n, const int blocks) {
    const int c = blockIdx.y;
    const double* z = Z + (long long)c * n;
    const double* y = Y + (long long)c * n;
    double acc[9];
    #pragma unroll
    for (int t = 0; t < 9; ++t) acc[t] = 0.0;
    const double w = 6.283185307179586 * fvec[c];
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        double s, co;
        sincos(w * z[i], &s, &co);
        const double yi = y[i];
        acc[0] += 1.0; acc[1] += s; acc[2] += co; acc[3] += s * s; acc[4] += s * co; acc[5] += co * co;
        acc[6] += yi; acc[7] += s * yi; acc[8] += co * yi;
    }
    __shared__ double sh[9][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        double v = acc[k];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) sh[k][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < 9) {
        double t = 0.0;
        const int nw = (blockDim.x + 31) >> 5;
        for (int w2 = 0; w2 < nw; ++w2) t += sh[threadIdx.x][w2];
        part[((long long)c * blocks + blockIdx.x) * 9 + threadIdx.x] = t;
    }
}
// grid (C): compensated merge of the block partials and the 3x3 solve; coef = 0 for an inactive column, NaN for a singular system
__global__ void solve3_b(const double* part, const int blocks, const int* active, double* coef) {
    const int c = blockIdx.x;
    if (!active[c]) { if (threadIdx.x < 3) coef[c * 3 + threadIdx.x] = 0.0; return; }
    __shared__ double tot[9];
    if (threadIdx.x < 9) {
        double s = 0.0, comp = 0.0;
        for (int b = 0; b < blocks; ++b) {
            const double x = part[((long long)c * blocks + b) * 9 + threadIdx.x];
            const double t = s + x;
            if (fabs(s) >= fabs(x)) comp += (s - t) + x; else comp += (x - t) + s;
            s = t;
        }
        tot[threadIdx.x] = s + comp;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    double M[3][4] = {{tot[0], tot[1], tot[2], tot[6]}, {tot[1], tot[3], tot[4], tot[7]}, {tot[2], tot[4], tot[5], tot[8]}};
    bool bad = false;
    for (int col = 0; col < 3 && !bad; ++col) {
        int piv = col; double best = fabs(M[col][col]);
        for (int r = col + 1; r < 3; ++r) if (fabs(M[r][col]) > best) { best = fabs(M[r][col]); piv = r; }
        if (!(best > 0.0) || !isfinite(best)) { bad = true; break; }
        if (piv != col) for (int k = 0; k < 4; ++k) { double tmp = M[col][k]; M[col][k] = M[piv][k]; M[piv][k] = tmp; }
        for (int r = col + 1; r < 3; ++r) { const double fct = M[r][col] / M[col][col]; for (int k = col; k < 4; ++k) M[r][k] -= fct * M[col][k]; }
    }
    if (bad) { coef[c * 3] = nan(""); coef[c * 3 + 1] = nan(""); coef[c * 3 + 2] = nan(""); return; }
    double x[3];
    for (int r = 2; r >= 0; --r) { double s = M[r][3]; for (int k = r + 1; k < 3; ++k) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
    coef[c * 3] = x[0]; coef[c * 3 + 1] = x[1]; coef[c * 3 + 2] = x[2];
}
// grid (blocks, C): y minus its projection (a zero coefficient vector leaves the column unchanged)
__global__ void apply_deflate_b(const double* Z, const double* Y, const double* coef, const double* fvec, double* out, const long long n) {
    const int c = blockIdx.y;
    const double w = 6.283185307179586 * fvec[c];
    const double c0 = coef[c * 3], c1 = coef[c * 3 + 1], c2 = coef[c * 3 + 2];
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        double s, co;
        sincos(w * Z[(long long)c * n + i], &s, &co);
        out[(long long)c * n + i] = Y[(long long)c * n + i] - (c0 + c1 * s + c2 * co);
    }
}
}
"""


def _module(cp: Any) -> Any:
    """Compile the kernels once."""
    global _MODULE
    with _LOCK:
        if _MODULE is None:
            _MODULE = cp.RawModule(code=_SRC, options=("-std=c++14",))
        return _MODULE


def _power(cp: Any, Z: Any, YC: Any, freqs: Any, nfreq: Any, y_ss: Any) -> Any:
    """``(C, F)`` periodogram power of each column's centred target at its own frequencies (padded entries are -1)."""
    C, n = int(Z.shape[0]), int(Z.shape[1])
    F = int(freqs.shape[1])
    out = cp.empty((C, F), dtype=cp.float64)
    _module(cp).get_function("periodogram_power_b")((F, C), (_THREADS,), (Z, YC, freqs, nfreq, y_ss, out, cp.int64(n), cp.int32(F)))
    return out


def _deflate(cp: Any, Z: Any, Y: Any, fvec: Any, active: Any) -> "tuple[Any, np.ndarray]":
    """Deflate ``[1, sin, cos]`` at ``fvec[c]`` out of every ACTIVE column; returns the new block and the host mask of columns whose 3x3 system was singular (their rows hold NaN)."""
    C, n = int(Z.shape[0]), int(Z.shape[1])
    blocks = max(1, min(64, (n + _THREADS - 1) // _THREADS))
    mod = _module(cp)
    part = cp.empty((C * blocks, 9), dtype=cp.float64)
    coef = cp.empty((C, 3), dtype=cp.float64)
    out = cp.empty((C, n), dtype=cp.float64)
    mod.get_function("sincos_gram_b")((blocks, C), (_THREADS,), (Z, Y, fvec, part, cp.int64(n), cp.int32(blocks)))
    mod.get_function("solve3_b")((C,), (32,), (part, cp.int32(blocks), active, coef))
    mod.get_function("apply_deflate_b")((blocks, C), (_THREADS,), (Z, Y, coef, fvec, out, cp.int64(n)))
    singular = ~np.isfinite(cp.asnumpy(coef)).all(axis=1)
    return out, singular


def _scan_freqs(centers: np.ndarray, half: float, step: float) -> "tuple[np.ndarray, np.ndarray]":
    """Per-column candidate frequencies of one local refinement scan: ``[center, lo, lo + step, ...]`` with ``lo = max(0.05, center - half)``, padded to a common width; plus the counts."""
    lists = []
    for ctr in centers:
        lo = max(_MIN_REFINE_FREQ, float(ctr) - half)
        hi = float(ctr) + half
        n_steps = round((hi - lo) / step) + 1
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
    cp: Any, Z_tr: np.ndarray, Z_va: np.ndarray, Y_tr: np.ndarray, Y_va: np.ndarray, grids: "list[list[float]]", spans: "list[Any]", min_val_corr: float, max_freqs: int,
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
    except np.linalg.LinAlgError:
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
        for k in np.flatnonzero((sing_t | sing_v) & (mask == 1)):
            # singular [1, sin, cos] system: the single-column path decides (its lstsq / graceful fallback); the rest of this column's loop runs there too
            results[k] = single(int(k))
            active = [a for a in active if a != int(k)]
    return results


def detect_fourier_freqs_batch_gpu(
    jobs: "Sequence[tuple[np.ndarray, np.ndarray, Sequence[float]]]", *, min_val_corr: float, min_rows: int, max_freqs: int, fourier_detect_max_n: int,
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
        z01 = np.asarray(z01, dtype=np.float64).ravel()
        prep = _host_split_for_detect(z01, y, f_grid, min_rows=min_rows, fourier_detect_max_n=fourier_detect_max_n)
        if prep is None:
            continue
        grid, z_tr, z_va, y_tr, y_va = prep
        groups.setdefault((z_tr.size, z_va.size), []).append((j, grid, z_tr, z_va, y_tr, y_va, core_span(z01)))
    for members in groups.values():
        Z_tr = np.ascontiguousarray(np.stack([m[2] for m in members]))
        Z_va = np.ascontiguousarray(np.stack([m[3] for m in members]))
        Y_tr = np.ascontiguousarray(np.stack([m[4] for m in members]))
        Y_va = np.ascontiguousarray(np.stack([m[5] for m in members]))
        job_ids = [m[0] for m in members]

        def single_k(k: int, _ids=job_ids) -> list:
            """Single-column detector for the group's column ``k``."""
            z01, y, f_grid = jobs[_ids[k]]
            return single(z01, y, f_grid)

        out = _run_group(cp, Z_tr, Z_va, Y_tr, Y_va, [m[1] for m in members], [m[6] for m in members], min_val_corr, max_freqs, single_k)
        for k, j in enumerate(job_ids):
            results[j] = out[k]
    return results
