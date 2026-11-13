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

from typing import Any, Optional

from pyutilz.performance.gpu_hw_launch import block_size_for

from mlframe._numerical_floors import VARIANCE_FLOOR
from mlframe.feature_selection.filters._kernel_types import render
from mlframe.feature_selection.filters._raw_module_cache import RawModuleCache

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
extern "C" {
__device__ __forceinline__ {ACC} power_term({ACC} sv, {ACC} vv, {ACC} vy, {ACC} n, {ACC} y_ss) {
    const {ACC} v_ss = vv - sv * sv / n;
    const bool ok = (v_ss > 1e-12 * vv) && (v_ss >= @VARIANCE_FLOOR@) && (y_ss >= @VARIANCE_FLOOR@);
    const {ACC} denom = v_ss * y_ss;
    return (ok && denom > 0.0) ? (vy * vy) / denom : 0.0;
}
// Kahan-compensated add of v into acc with the running compensation comp (both in the sample type)
#define KAHAN_ADD(acc, comp, v) { const {REAL} yk_ = (v) - comp; const {REAL} tk_ = acc + yk_; comp = (tk_ - acc) - yk_; acc = tk_; }
__global__ void periodogram_power(const {REAL}* z, const {REAL}* yc, const {ACC}* freqs, {ACC}* out, const long long n, const {ACC} y_ss) {
    const {ACC} w = 6.283185307179586 * freqs[blockIdx.x];
    // Per-thread partials in the SAMPLE type with Kahan compensation, combined in {ACC} below (same change and measurements as the batch kernel: the trig is taken in the sample type, so double
    // partials only paid the card's 1:32 double rate).
    {REAL} ss = 0, sc = 0, vvs = 0, vvc = 0, vys = 0, vyc = 0;
    {REAL} c_ss = 0, c_sc = 0, c_vvs = 0, c_vvc = 0, c_vys = 0, c_vyc = 0;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        {ACC} s_, c_;
        trig(w, z[i], &s_, &c_);
        const {REAL} s = ({REAL})s_, c = ({REAL})c_, y = yc[i];
        KAHAN_ADD(ss, c_ss, s) KAHAN_ADD(sc, c_sc, c) KAHAN_ADD(vvs, c_vvs, s * s) KAHAN_ADD(vvc, c_vvc, c * c) KAHAN_ADD(vys, c_vys, s * y) KAHAN_ADD(vyc, c_vyc, c * y)
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
        out[blockIdx.x] = power_term(tot[0], tot[2], tot[4], ({ACC})n, y_ss) + power_term(tot[1], tot[3], tot[5], ({ACC})n, y_ss);
    }
}
__global__ void sincos_gram(const {REAL}* z, const {REAL}* y, const {ACC} f, {ACC}* part, const long long n) {
    // A = [1, sin, cos]: packed A'A (upper, 6) + A'y (3)
    {ACC} acc[9];
    #pragma unroll
    for (int t = 0; t < 9; ++t) acc[t] = 0.0;
    const {ACC} w = 6.283185307179586 * f;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        {ACC} s, c;
        trig(w, z[i], &s, &c);
        const {ACC} yi = y[i];
        acc[0] += 1.0; acc[1] += s; acc[2] += c; acc[3] += s * s; acc[4] += s * c; acc[5] += c * c;
        acc[6] += yi; acc[7] += s * yi; acc[8] += c * yi;
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
        part[(long long)blockIdx.x * 9 + threadIdx.x] = t;
    }
}
__global__ void solve3(const {ACC}* part, const int nblocks, {ACC}* coef) {
    __shared__ {ACC} tot[9];
    if (threadIdx.x < 9) {
        {ACC} s = 0.0, comp = 0.0;
        for (int b = 0; b < nblocks; ++b) {
            const {ACC} x = part[(long long)b * 9 + threadIdx.x];
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
    if (bad) { coef[0] = nan(""); coef[1] = nan(""); coef[2] = nan(""); return; }
    for (int r = 2; r >= 0; --r) { {ACC} s = M[r][3]; for (int k = r + 1; k < 3; ++k) s -= M[r][k] * coef[k]; coef[r] = s / M[r][r]; }
}
__global__ void apply_deflate(const {REAL}* z, const {REAL}* y, const {ACC}* coef, const {ACC} f, {ACC}* out, const long long n) {
    const {ACC} w = 6.283185307179586 * f;
    const {ACC} c0 = coef[0], c1 = coef[1], c2 = coef[2];
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        {ACC} s, c;
        trig(w, z[i], &s, &c);
        out[i] = y[i] - (c0 + c1 * s + c2 * c);
    }
}
}
"""


# The variance floor is filled in from the shared constant so the kernel and the host agree.
_SRC = render(_SRC_TEMPLATE.replace("@VARIANCE_FLOOR@", repr(VARIANCE_FLOOR)))  # Intentional float64: {ACC} is float64; {REAL} stays for the per-type modules
# One module per input element type: the samples arrive in the critical dtype (float32 by default) and the kernels read them as that type; the sums accumulate in double.
_MODULES_BY_REAL = {"double": RawModuleCache(_SRC.replace("{REAL}", "double")), "float": RawModuleCache(_SRC.replace("{REAL}", "float"))}


def _modules_for(z) -> Any:
    """The module compiled for the element type of the sample buffer ``z``."""
    return _MODULES_BY_REAL["float" if z.dtype.itemsize == 4 else "double"]


def power_sincos(cp: Any, z: Any, yc: Any, y_ss: float, freqs: Any) -> Any:
    """Resident ``(F,)`` periodogram power of ``yc`` at each frequency in ``freqs`` (one block per frequency). ``z`` and ``yc`` are double precision ``(n,)``."""
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    f = cp.ascontiguousarray(freqs, dtype=cp.float64)
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    out = cp.empty(int(f.shape[0]), dtype=cp.float64)
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    power = _modules_for(z).get(cp).get_function("periodogram_power")
    power(
        (int(f.shape[0]),), (block_size_for(power, work_items=int(z.shape[0]), max_width=1024),), (z, yc, f, out, cp.int64(z.shape[0]), cp.float64(y_ss))
    )  # Intentional float64: the power kernel accumulates in double
    return out


def deflate_sincos(cp: Any, z: Any, y: Any, freq: float) -> Optional[Any]:
    """``y`` minus its least-squares projection onto ``[1, sin(2 pi f z), cos(2 pi f z)]`` as a new double precision array, or ``None`` when the 3x3 system is singular (caller falls back).

    The three coefficients come back in one small transfer: whether the system was solvable has to be known before the result is used."""
    import numpy as np

    n = int(z.shape[0])
    mod = _modules_for(z).get(cp)
    gram = mod.get_function("sincos_gram")
    deflate = mod.get_function("apply_deflate")
    gram_threads = block_size_for(gram, work_items=n, max_width=1024)  # one partial per warp in a shared array of 32 rows: any warp multiple up to 1024
    blocks = max(1, min(64, (n + gram_threads - 1) // gram_threads))
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    part = cp.empty((blocks, 9), dtype=cp.float64)
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    coef = cp.empty(3, dtype=cp.float64)
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    out = cp.empty(n, dtype=cp.float64)
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    gram((blocks,), (gram_threads,), (z, y, cp.float64(freq), part, cp.int64(n)))
    mod.get_function("solve3")(
        (1,), (32,), (part, cp.int32(blocks), coef)
    )  # launch-ok: one warp: threads 0-8 fold the nine partial sums, thread 0 solves the 3x3 system; a wider block adds nothing
    if not np.all(np.isfinite(cp.asnumpy(coef))):
        return None
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    deflate((blocks,), (block_size_for(deflate, work_items=n),), (z, y, coef, cp.float64(freq), out, cp.int64(n)))
    return out
