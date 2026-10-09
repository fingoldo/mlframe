"""Fused kernels for the resident multi-frequency Fourier detector.

The detector (:func:`_fourier_detect_gpu_resident.detect_fourier_freqs_for_col_gpu`) is a short sequential loop on a few tens of thousands of rows, so its cost is launches, not
arithmetic: ~185 launches and ~2 ms of GPU work per column. Two pieces account for nearly all of it, and each becomes a handful of kernels here:

* ``periodogram_power``: for every frequency in a list, the squared correlation of ``sin(2 pi f z)`` and of ``cos(2 pi f z)`` with the centred target, summed - the same raw-moment
  form (``v_ss = v.v - (sum v)^2 / n``, the same relative degeneracy guard) as the cupy batch, with the sin/cos computed in the kernel instead of materialising (F, n) planes. One block per
  frequency, so the coarse grid, both refinement scans and the held-out confirmation are one launch each.
* ``deflate``: least-squares removal of ``[1, sin, cos]`` at a frequency, as a block-partial Gram kernel, a one-block 3x3 solve and an elementwise apply (3 launches instead of a stack,
  two products, a cusolver solve and a matvec). A singular system leaves NaN coefficients and the caller takes the lstsq path.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

_LOCK = threading.Lock()
_MODULE: Optional[Any] = None

_SRC = r"""
extern "C" {
__device__ __forceinline__ double power_term(double sv, double vv, double vy, double n, double y_ss) {
    const double v_ss = vv - sv * sv / n;
    const bool ok = (v_ss > 1e-12 * vv) && (v_ss >= 1e-24) && (y_ss >= 1e-24);
    const double denom = v_ss * y_ss;
    return (ok && denom > 0.0) ? (vy * vy) / denom : 0.0;
}
__global__ void periodogram_power(const double* z, const double* yc, const double* freqs, double* out, const long long n, const double y_ss) {
    const double w = 6.283185307179586 * freqs[blockIdx.x];
    double ss = 0.0, sc = 0.0, vvs = 0.0, vvc = 0.0, vys = 0.0, vyc = 0.0;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        double s, c;
        sincos(w * z[i], &s, &c);
        const double y = yc[i];
        ss += s; sc += c; vvs += s * s; vvc += c * c; vys += s * y; vyc += c * y;
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
        out[blockIdx.x] = power_term(tot[0], tot[2], tot[4], (double)n, y_ss) + power_term(tot[1], tot[3], tot[5], (double)n, y_ss);
    }
}
__global__ void sincos_gram(const double* z, const double* y, const double f, double* part, const long long n) {
    // A = [1, sin, cos]: packed A'A (upper, 6) + A'y (3)
    double acc[9];
    #pragma unroll
    for (int t = 0; t < 9; ++t) acc[t] = 0.0;
    const double w = 6.283185307179586 * f;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        double s, c;
        sincos(w * z[i], &s, &c);
        const double yi = y[i];
        acc[0] += 1.0; acc[1] += s; acc[2] += c; acc[3] += s * s; acc[4] += s * c; acc[5] += c * c;
        acc[6] += yi; acc[7] += s * yi; acc[8] += c * yi;
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
        part[(long long)blockIdx.x * 9 + threadIdx.x] = t;
    }
}
__global__ void solve3(const double* part, const int nblocks, double* coef) {
    __shared__ double tot[9];
    if (threadIdx.x < 9) {
        double s = 0.0, comp = 0.0;
        for (int b = 0; b < nblocks; ++b) {
            const double x = part[(long long)b * 9 + threadIdx.x];
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
    if (bad) { coef[0] = nan(""); coef[1] = nan(""); coef[2] = nan(""); return; }
    for (int r = 2; r >= 0; --r) { double s = M[r][3]; for (int k = r + 1; k < 3; ++k) s -= M[r][k] * coef[k]; coef[r] = s / M[r][r]; }
}
__global__ void apply_deflate(const double* z, const double* y, const double* coef, const double f, double* out, const long long n) {
    const double w = 6.283185307179586 * f;
    const double c0 = coef[0], c1 = coef[1], c2 = coef[2];
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        double s, c;
        sincos(w * z[i], &s, &c);
        out[i] = y[i] - (c0 + c1 * s + c2 * c);
    }
}
}
"""


def _module(cp):
    """Compile the kernel module once."""
    global _MODULE
    with _LOCK:
        if _MODULE is None:
            _MODULE = cp.RawModule(code=_SRC, options=("-std=c++14",))
        return _MODULE


def power_sincos(cp: Any, z: Any, yc: Any, y_ss: float, freqs: Any) -> Any:
    """Resident ``(F,)`` periodogram power of ``yc`` at each frequency in ``freqs`` (one block per frequency). ``z`` and ``yc`` are float64 ``(n,)``."""
    f = cp.ascontiguousarray(freqs, dtype=cp.float64)
    out = cp.empty(int(f.shape[0]), dtype=cp.float64)
    _module(cp).get_function("periodogram_power")((int(f.shape[0]),), (256,), (z, yc, f, out, cp.int64(z.shape[0]), cp.float64(y_ss)))
    return out


def deflate_sincos(cp: Any, z: Any, y: Any, freq: float) -> Optional[Any]:
    """``y`` minus its least-squares projection onto ``[1, sin(2 pi f z), cos(2 pi f z)]`` as a new float64 array, or ``None`` when the 3x3 system is singular (caller falls back).

    The three coefficients come back in one small transfer: whether the system was solvable has to be known before the result is used."""
    import numpy as np

    n = int(z.shape[0])
    mod = _module(cp)
    blocks = max(1, min(64, (n + 255) // 256))
    part = cp.empty((blocks, 9), dtype=cp.float64)
    coef = cp.empty(3, dtype=cp.float64)
    out = cp.empty(n, dtype=cp.float64)
    mod.get_function("sincos_gram")((blocks,), (256,), (z, y, cp.float64(freq), part, cp.int64(n)))
    mod.get_function("solve3")((1,), (32,), (part, cp.int32(blocks), coef))
    if not np.all(np.isfinite(cp.asnumpy(coef))):
        return None
    mod.get_function("apply_deflate")((blocks,), (256,), (z, y, coef, cp.float64(freq), out, cp.int64(n)))
    return out
