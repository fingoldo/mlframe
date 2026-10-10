"""CUDA twin of ``_offset_product_kernels.scan_offset_products``: one block per (column pair, unary pair) task, nothing stored per candidate.

A block (1) fits the shifts of ``(u + s) * (v + t)`` on the even rows (fixed-order tree reductions of the 14 normal-equation sums, a 4x4 solve and the 2x2 main-effect solve on thread 0),
(2) for each of the six features (the shifted product and the five shift-free baselines) regenerates the value at 2048 sampled even rows, sorts them in shared memory and takes the
interior quantile edges, (3) makes one pass over all rows that regenerates the six values, bins them with the edges and counts a (feature, split, bin, class) joint histogram with
integer shared-memory atomics (order independent, so deterministic), (4) turns the counts into plug-in MI with the CPU loop order. The arithmetic is compiled with ``--fmad=false`` and the
device and host edge rules are the same, so the selection matches the CPU scan; the sums differ from the CPU only by the reduction order (about 1e-12 relative).

Returns ``None`` when the joint histograms do not fit in shared memory (many target classes) so the caller falls back to the CPU scan.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

import numpy as np

from ._offset_product_kernels import N_BASELINES, N_EDGE_SUBSAMPLE

__all__ = ["scan_offset_products_gpu", "scan_offset_products_device", "DEVICE_UNARIES"]

DEVICE_UNARIES = frozenset({"identity", "abs", "sqr", "reciproc", "sqrt", "log", "sin"})  # the unary maps with a device form below; any other name sends the caller to the host path

_THREADS = 256
_SHARED_BUDGET = 44 * 1024  # dynamic shared memory per block; with the ~3.6 KB of static arrays this stays inside the 48 KB default limit
_N_FEATURES = 1 + N_BASELINES
_LOCK = threading.Lock()
_KERNEL: Optional[Any] = None

_SRC = r"""
#define NSUB __NSUB__
#define NF __NF__
#define NACC 14
#define SING_TOL 1e-12
#define MIN_SPAN 1e-300
#define THREADS __THREADS__
#define D_NAN __longlong_as_double(0x7ff8000000000000LL)
#define D_INF __longlong_as_double(0x7ff0000000000000LL)

__device__ __forceinline__ double block_sum(double v, double* red) {
    __syncthreads();
    red[threadIdx.x] = v;
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const double r = red[0];
    __syncthreads();
    return r;
}

// 4x4 normal equations (upper triangle in acc[0..9], rhs acc[10..13]); NaN coefficients when singular. Mirrors _offset_product_kernels._solve4.
__device__ void solve4(const double* acc, double* coef) {
    double M[4][5];
    int t = 0;
    for (int j = 0; j < 4; ++j) for (int k = j; k < 4; ++k) { M[j][k] = acc[t]; M[k][j] = acc[t]; ++t; }
    for (int j = 0; j < 4; ++j) M[j][4] = acc[10 + j];
    double scale = 0.0;
    for (int j = 0; j < 4; ++j) scale = fmax(scale, fabs(M[j][j]));
    for (int col = 0; col < 4; ++col) {
        int piv = col; double best = fabs(M[col][col]);
        for (int r = col + 1; r < 4; ++r) if (fabs(M[r][col]) > best) { best = fabs(M[r][col]); piv = r; }
        if (!(best > SING_TOL * scale) || !isfinite(best)) { for (int k = 0; k < 4; ++k) coef[k] = D_NAN; return; }
        if (piv != col) for (int k = 0; k < 5; ++k) { const double tmp = M[col][k]; M[col][k] = M[piv][k]; M[piv][k] = tmp; }
        for (int r = col + 1; r < 4; ++r) {
            const double f = M[r][col] / M[col][col];
            for (int k = col; k < 5; ++k) M[r][k] -= f * M[col][k];
        }
    }
    for (int r = 3; r >= 0; --r) {
        double s = M[r][4];
        for (int k = r + 1; k < 4; ++k) s -= M[r][k] * coef[k];
        coef[r] = s / M[r][r];
    }
}

__device__ __forceinline__ double feature(const int q, const double u, const double v, const double s, const double t, const double wb, const double wc) {
    if (q == 0) return (u + s) * (v + t);
    if (q == 1) return u * v;
    if (q == 2) return u + v;
    if (q == 3) return u;
    if (q == 4) return v;
    return wb * u + wc * v;
}

__device__ void bitonic_sort(double* a, const int n) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < n; i += THREADS) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const bool up = ((i & k) == 0);
                    const double x = a[i], y = a[ixj];
                    if ((x > y) == up) { a[i] = y; a[ixj] = x; }
                }
            }
            __syncthreads();
        }
    }
}

extern "C" __global__ void scan_offset_products(
    const double* __restrict__ U, const int* __restrict__ tasks, const double* __restrict__ yr, const int* __restrict__ ycodes,
    const double* __restrict__ clips, const int m, const int nu, const long long n, const int ky, const int nb,
    double* __restrict__ out_shift, double* __restrict__ out_mi) {
    extern __shared__ unsigned char smem_raw[];
    double* sub = reinterpret_cast<double*>(smem_raw);                       // NSUB sort buffer
    int* counts = reinterpret_cast<int*>(sub + NSUB);                        // NF * 2 * nb * ky joint counts
    __shared__ double red[THREADS];
    __shared__ double sh[4];            // s, t, wb, wc ; NaN when the shifts could not be fitted
    __shared__ double edges[NF][32];    // interior edges per feature (nb - 1 <= 31)
    const int k = blockIdx.x;
    const int cu = tasks[4 * k], cv = tasks[4 * k + 1], ui = tasks[4 * k + 2], vi = tasks[4 * k + 3];
    const double* u = U + ((long long)cu * nu + ui) * n;
    const double* v = U + ((long long)cv * nu + vi) * n;
    const double ulo = clips[2 * (cu * nu + ui)], uhi = clips[2 * (cu * nu + ui) + 1];
    const double vlo = clips[2 * (cv * nu + vi)], vhi = clips[2 * (cv * nu + vi) + 1];
    const long long n_even = (n + 1) / 2;
    const int tid = threadIdx.x;

    // phase A: means over the even rows
    double a0 = 0.0, a1 = 0.0, a2 = 0.0;
    for (long long i = 2LL * tid; i < n; i += 2LL * THREADS) {
        a0 += fmin(fmax(u[i], ulo), uhi);
        a1 += fmin(fmax(v[i], vlo), vhi);
        a2 += yr[i];
    }
    const double mu = block_sum(a0, red) / (double)n_even;
    const double mv = block_sum(a1, red) / (double)n_even;
    const double my = block_sum(a2, red) / (double)n_even;
    // phase A: the 14 normal-equation sums
    double acc[NACC];
    for (int j = 0; j < NACC; ++j) acc[j] = 0.0;
    for (long long i = 2LL * tid; i < n; i += 2LL * THREADS) {
        const double x = fmin(fmax(u[i], ulo), uhi) - mu, w = fmin(fmax(v[i], vlo), vhi) - mv, yy = yr[i] - my, f3 = x * w;
        acc[0] += 1.0; acc[1] += x; acc[2] += w; acc[3] += f3; acc[4] += x * x; acc[5] += x * w; acc[6] += x * f3;
        acc[7] += w * w; acc[8] += w * f3; acc[9] += f3 * f3; acc[10] += yy; acc[11] += x * yy; acc[12] += w * yy; acc[13] += f3 * yy;
    }
    double tot[NACC];
    for (int j = 0; j < NACC; ++j) tot[j] = block_sum(acc[j], red);
    if (tid == 0) {
        sh[0] = D_NAN; sh[1] = D_NAN; sh[2] = D_NAN; sh[3] = D_NAN;
        if (n_even >= 8) {
            double coef[4];
            solve4(tot, coef);
            const double det = tot[4] * tot[7] - tot[5] * tot[5];
            double wb = D_NAN, wc = D_NAN;
            if (det > SING_TOL * tot[4] * tot[7]) {
                wb = (tot[11] * tot[7] - tot[12] * tot[5]) / det;
                wc = (tot[12] * tot[4] - tot[11] * tot[5]) / det;
            }
            const double d = coef[3];
            if (isfinite(d) && d != 0.0) {
                const double s = coef[2] / d - mu, t = coef[1] / d - mv;
                const double span_u = fmax(uhi - ulo, MIN_SPAN), span_v = fmax(vhi - vlo, MIN_SPAN);
                sh[0] = fmin(fmax(s, -uhi - span_u), -ulo + span_u);
                sh[1] = fmin(fmax(t, -vhi - span_v), -vlo + span_v);
                sh[2] = wb; sh[3] = wc;
            }
        }
    }
    __syncthreads();
    const double s = sh[0], t = sh[1], wb = sh[2], wc = sh[3];
    if (tid == 0) { out_shift[2 * k] = s; out_shift[2 * k + 1] = t; }
    const bool ok = isfinite(s) && isfinite(t) && isfinite(wb) && isfinite(wc);
    if (!ok) {
        if (tid < 2 * NF) out_mi[(long long)k * 2 * NF + tid] = D_NAN;
        return;
    }

    // phase B: interior quantile edges of every feature from NSUB sampled even rows
    const long long msub = n_even < NSUB ? n_even : NSUB;
    for (int q = 0; q < NF; ++q) {
        for (int j = tid; j < NSUB; j += THREADS) {
            if (j < msub) {
                const long long i = 2LL * ((j * n_even) / msub);
                sub[j] = feature(q, u[i], v[i], s, t, wb, wc);
            } else {
                sub[j] = D_INF;
            }
        }
        __syncthreads();
        bitonic_sort(sub, NSUB);
        if (tid < nb - 1) {
            long long pos = ((long long)(tid + 1) * msub) / nb;
            if (pos > msub - 1) pos = msub - 1;
            edges[q][tid] = sub[pos];
        }
        __syncthreads();
    }

    // phase C: one pass over all rows regenerates the six features and counts the joint histograms
    const int per_feat = 2 * nb * ky;
    for (int j = tid; j < NF * per_feat; j += THREADS) counts[j] = 0;
    __syncthreads();
    for (long long i = tid; i < n; i += THREADS) {
        const int sp = (int)(i & 1LL), c = ycodes[i];
        const double ui_ = u[i], vi_ = v[i];
        for (int q = 0; q < NF; ++q) {
            const double x = feature(q, ui_, vi_, s, t, wb, wc);
            int lo = 0, hi = nb - 1;
            while (lo < hi) { const int mid = (lo + hi) >> 1; if (edges[q][mid] <= x) lo = mid + 1; else hi = mid; }
            atomicAdd(&counts[q * per_feat + (sp * nb + lo) * ky + c], 1);
        }
    }
    __syncthreads();

    // phase D: plug-in MI per (feature, split); thread q*2+sp accumulates in the CPU loop order
    if (tid < 2 * NF) {
        const int q = tid >> 1, sp = tid & 1;
        const int* cnt = counts + q * per_feat + sp * nb * ky;
        const double n_tot = (double)(sp == 0 ? n_even : n - n_even);
        double mi = 0.0;
        if (n_tot > 0.0) {
            for (int b = 0; b < nb; ++b) {
                double px = 0.0;
                for (int c = 0; c < ky; ++c) px += (double)cnt[b * ky + c];
                for (int c = 0; c < ky; ++c) {
                    const double kk = (double)cnt[b * ky + c];
                    if (kk > 0.0) {
                        double py = 0.0;
                        for (int bb = 0; bb < nb; ++bb) py += (double)cnt[bb * ky + c];
                        mi += (kk / n_tot) * log(kk * n_tot / (px * py));
                    }
                }
            }
        }
        out_mi[(long long)k * 2 * NF + sp * NF + q] = mi;
    }
}
"""


def _shared_bytes(nb: int, ky: int) -> int:
    """Dynamic shared memory of one block: the sort buffer plus the joint counts of every feature and split."""
    return N_EDGE_SUBSAMPLE * 8 + _N_FEATURES * 2 * nb * ky * 4


def _kernel(cp: Any) -> Any:
    """Compile the scan kernel once (``--fmad=false`` keeps the products and sums as separately rounded as the CPU reference)."""
    global _KERNEL
    with _LOCK:
        if _KERNEL is None:
            src = _SRC.replace("__NSUB__", str(N_EDGE_SUBSAMPLE)).replace("__NF__", str(_N_FEATURES)).replace("__THREADS__", str(_THREADS))
            _KERNEL = cp.RawKernel(src, "scan_offset_products", options=("--fmad=false", "-std=c++14"))
        return _KERNEL


def scan_offset_products_gpu(U: Any, tasks: np.ndarray, yr: Any, ycodes: Any, ky: int, nb: int, clips: Any) -> Optional[tuple]:
    """Score every task on the device; returns host ``(out_shift (T, 2), out_mi (T, 2, 1 + N_BASELINES))`` like the CPU scan, or ``None`` when it does not apply.

    ``U``: ``(m, nu, n)`` float64 (device or host); ``tasks``: ``(T, 4)`` ints ``(col_u, col_v, unary_u, unary_v)``; ``yr``: rank-scaled target; ``ycodes``: class codes in ``[0, ky)``;
    ``clips``: ``(m, nu, 2)`` winsorisation bounds. ``None`` for a target with so many classes that the six joint histograms exceed the shared-memory budget or ``nb`` above 32."""
    import cupy as cp

    if nb > 32 or nb < 2 or _shared_bytes(nb, ky) > _SHARED_BUDGET:
        return None
    Ud = cp.ascontiguousarray(cp.asarray(U, dtype=cp.float64))
    m, nu, n = (int(d) for d in Ud.shape)
    T = int(tasks.shape[0])
    td = cp.asarray(np.ascontiguousarray(tasks, dtype=np.int32))
    yrd = cp.ascontiguousarray(cp.asarray(yr, dtype=cp.float64))
    ycd = cp.ascontiguousarray(cp.asarray(ycodes, dtype=cp.int32))
    cd = cp.ascontiguousarray(cp.asarray(clips, dtype=cp.float64))
    out_shift = cp.empty((T, 2), dtype=cp.float64)
    out_mi = cp.empty((T, 2, _N_FEATURES), dtype=cp.float64)
    _kernel(cp)(
        (T,), (_THREADS,),
        (Ud, td, yrd, ycd, cd, np.int32(m), np.int32(nu), np.int64(n), np.int32(ky), np.int32(nb), out_shift, out_mi),
        shared_mem=_shared_bytes(nb, ky),
    )
    return cp.asnumpy(out_shift), cp.asnumpy(out_mi)


def _device_unary(cp: Any, name: str, x: Any, log_shift: float) -> Any:
    """Device form of one unary map, matching the registry's host form (``reciproc`` uses its fixed 1e9 ceiling at zero, ``log`` the frozen anchor instead of a batch minimum)."""
    if name == "identity":
        return x
    if name == "abs":
        return cp.abs(x)
    if name == "sqr":
        return x * x
    if name == "reciproc":
        zero = x == 0.0
        return cp.where(zero, 1e9, 1.0 / cp.where(zero, 1.0, x))
    if name == "sqrt":
        return cp.sqrt(cp.abs(x))
    if name == "log":
        return cp.log(x + log_shift) if log_shift != 0.0 else cp.log(x)
    if name == "sin":
        return cp.sin(x)
    raise ValueError(f"no device form for the unary {name!r}")


def _column_quantiles(cp: Any, a: Any, qs: "tuple[float, ...]") -> Any:
    """Linear-interpolation quantiles ``qs`` along axis 1 of the ``(k, h)`` array ``a`` (same rule as ``np.quantile``), shape ``(k, len(qs))``."""
    srt = cp.sort(a, axis=1)
    h = int(a.shape[1])
    cols = []
    for q in qs:
        pos = q * (h - 1)
        lo = int(np.floor(pos))
        hi = min(lo + 1, h - 1)
        cols.append(srt[:, lo] + (srt[:, hi] - srt[:, lo]) * (pos - lo))
    return cp.stack(cols, axis=1)


def scan_offset_products_device(
    xs: "list[np.ndarray]", unaries: "tuple[str, ...]", rows: np.ndarray, tasks: np.ndarray, yr: np.ndarray, codes: np.ndarray, ky: int, nb: int,
    log_shifts: "list[float]", winsor_q: "tuple[float, float]",
) -> Optional[tuple]:
    """The scan with its inputs built on the device: only the scan rows of each pooled column are uploaded, the unary outputs, their non-finite fills and the winsorisation bounds are made
    there, and the ``(columns, unaries, rows)`` block is never copied back. Returns host ``(out_mi, clips)`` (``clips`` is needed later for the refit), or ``None`` when a unary has no
    device form or the joint histograms do not fit (the caller then takes the host path)."""
    import cupy as cp

    if not set(unaries) <= DEVICE_UNARIES:
        return None
    m, nu, n = len(xs), len(unaries), len(rows)
    U = cp.empty((m, nu, n), dtype=cp.float64)
    clips = cp.empty((m, nu, 2), dtype=cp.float64)
    for ci, x in enumerate(xs):
        xd = cp.asarray(np.ascontiguousarray(x[rows], dtype=np.float64))
        for ui, name in enumerate(unaries):
            u = _device_unary(cp, name, xd, log_shifts[ci])
            fin = cp.isfinite(u)
            count = fin.sum()
            fill = cp.where(count > 0, cp.where(fin, u, 0.0).sum() / cp.maximum(count, 1), 0.0)
            U[ci, ui] = cp.where(fin, u, fill)
        clips[ci] = _column_quantiles(cp, U[ci][:, ::2], winsor_q)
    res = scan_offset_products_gpu(U, tasks, yr, codes, ky, nb, clips)
    if res is None:
        return None
    return res[1], cp.asnumpy(clips)
