"""One-pass versions of the binary-target root-split kernel of the device honest null (``root_split_stats_k2`` is version 2 and stays).

DISPATCHED BY DEFAULT is version 5 (below, selection-equivalent; ``MLFRAME_HONEST_ROOT_SPLIT=k4|k2`` selects the exact versions). Version 4 (``get_kernel_k4``, exact): v2 scans the sorted rows twice (pass 1 the smallest float32 entropy score, pass 2 the exact double gain of the candidates within 0.5 of it). Here one pass keeps the running
minimum and evaluates the exact gain of every candidate within 0.5 of the RUNNING minimum, with the ``x ln x`` table covering EVERY count up to ``n`` (``xlogx_table(n)``): the ``log((double)v)`` that v2 takes for counts of 32768 and
more is a double-precision call on almost every step at 100000 rows, and that, not the double adds, made the first attempt of this version 2.3 times slower than v2. Measured on the GTX 1050 Ti, 60 columns x 200
permutations x 100000 rows: v2 861 ms, v2 with the full table 770 ms, v4 535 ms; outputs equal to v2 (index and candidate count identical, gain within 2e-15, the device ``log`` against numpy's).
Why the outputs are v2's: the running minimum never goes below the final minimum, so the candidate with the smallest exact entropy (within the float error of about 0.02 of the final minimum) is always admitted at its own
step; a candidate admitted early but outside the final window has a gain lower than the best one by about 0.5 / n or more, so it is never the first strictly greater gain.

Other versions:
* v4f (``get_kernel_k4f``): the same scan with the gain in ``{REAL}`` = float. 577 ms, but the cancellation of the large ``x ln x`` terms gives a gain error up to 2.4e-6 against gains of 1e-4: too coarse for the significance test.
* v5 (``get_kernel_k5``, DEFAULT): one pass in float over the stable entropy score only, the exact double gain taken once at its minimum. 287 ms, gain error up to 1.8e-7 and 7% of the split indices differ from v2 (near ties of the float score);
  enough for the honest null: it only decides which (column, permutation) pairs have an accepted root split, the accepted ones are redone exactly on the host, so the error can only miss a pair whose gain sits within 1e-7 of the significance threshold.
"""

from __future__ import annotations

from typing import Any

# Intentional float64: tab/out_gain hold statistic tables and partial gain sums whose rounding would change split selection vs the CPU twin.
_SOURCE_K4F = r"""
#define XL(v) (((v) < 32768) ? ({REAL})tab[(v)] : ({REAL})(v) * logf(({REAL})(v)))
#define ENT2(m, c) (((c) > 0.f && (c) < (m)) ? -((c) * __logf((c) / (m)) + ((m) - (c)) * __logf(((m) - (c)) / (m))) : 0.f)
// {REAL}: the type of the exact-gain arithmetic (float: the double path runs at 1/32 of the float rate on consumer cards); the stored gain is converted to double.
extern "C" __global__ void root_split_stats_k4(const int* __restrict__ order, const unsigned char* __restrict__ tie, const unsigned char* __restrict__ ybits,
                                               const double* __restrict__ tab, const long long* __restrict__ ctot, const int n, const int B,
                                               const int Kc, const int min_split, const double h_full, const int words, double* __restrict__ out_gain,
                                               int* __restrict__ out_ncand, int* __restrict__ out_idx) {
    const long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= (long long)B * Kc) return;
    const int j = (int)(idx / B);
    const int b = (int)(idx - (long long)j * B);
    const int* ord = order + (long long)j * n;
    const unsigned char* tj = tie + (long long)j * n;
    const unsigned int* yb = (const unsigned int*)ybits;
    const {REAL} n_f = ({REAL})n;
    const {REAL} h_r = ({REAL})h_full;
    const int TILE = 8;
    int ybuf[TILE + 1];
    float w_min = 3.0e38f;
    int ncand = 0;
    long long l0 = 0, l1 = 0, r0 = ctot[0], r1 = ctot[1];
    {REAL} best_gain = -3.0e38f;
    int best_idx = -1;
    for (int i0 = 0; i0 < n - 1; i0 += TILE) {
        const int cnt = min(TILE + 1, n - i0);
#pragma unroll
        for (int t = 0; t < TILE + 1; ++t) {
            if (t < cnt) { const int pos = ord[i0 + t]; ybuf[t] = (yb[(long long)(pos >> 5) * B + b] >> (pos & 31)) & 1; }
        }
        for (int t = 0; t < TILE && i0 + t < n - 1; ++t) {
            const int i = i0 + t;
            const int yi = ybuf[t];
            if (yi) { ++l1; --r1; } else { ++l0; --r0; }
            const int n_l = i + 1, n_r = n - n_l;
            if (n_l >= min_split && n_r >= min_split && yi != ybuf[t + 1] && !tj[i]) {
                const float w_f = ENT2((float)n_l, (float)l1) + ENT2((float)n_r, (float)r1);
                ++ncand;
                if (w_f < w_min) w_min = w_f;
                if (w_f <= w_min + 0.5f) {
                    const {REAL} gain = h_r - (XL(n_l) - (XL(l0) + XL(l1)) + XL(n_r) - (XL(r0) + XL(r1))) / n_f;
                    if (gain > best_gain) { best_gain = gain; best_idx = i; }
                }
            }
        }
    }
    out_gain[idx] = best_idx < 0 ? -1e300 : (double)best_gain;
    out_ncand[idx] = ncand;
    out_idx[idx] = best_idx;
}
"""

_kernel_k4f = None


def get_kernel_k4f(cp: Any) -> Any:
    """Compile v4 once per process (no fused multiply-add: the gains must round as the host's)."""
    global _kernel_k4f
    if _kernel_k4f is None:
        _kernel_k4f = cp.RawKernel(_SOURCE_K4F.replace("{REAL}", "float"), "root_split_stats_k4", options=("--fmad=false",))
    return _kernel_k4f


_SOURCE_K5 = r"""
#define XL(v) (((v) < 32768) ? tab[(v)] : (double)(v) * log((double)(v)))
#define ENT2(m, c) (((c) > 0.f && (c) < (m)) ? -((c) * __logf((c) / (m)) + ((m) - (c)) * __logf(((m) - (c)) / (m))) : 0.f)
// One pass in {REAL} (float): the stable entropy score w_f (every term below n ln 2) is tracked for its minimum and the position; the exact double gain is evaluated ONCE, at that position.
extern "C" __global__ void root_split_stats_k5(const int* __restrict__ order, const unsigned char* __restrict__ tie, const unsigned char* __restrict__ ybits,
                                               const double* __restrict__ tab, const long long* __restrict__ ctot, const int n, const int B,
                                               const int Kc, const int min_split, const double h_full, const int words, double* __restrict__ out_gain,
                                               int* __restrict__ out_ncand, int* __restrict__ out_idx) {
    const long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= (long long)B * Kc) return;
    const int j = (int)(idx / B);
    const int b = (int)(idx - (long long)j * B);
    const int* ord = order + (long long)j * n;
    const unsigned char* tj = tie + (long long)j * n;
    const unsigned int* yb = (const unsigned int*)ybits;
    const int TILE = 8;
    int ybuf[TILE + 1];
    {REAL} w_min = 3.0e38f;
    int ncand = 0, best_i = -1;
    long long l0 = 0, l1 = 0, r0 = ctot[0], r1 = ctot[1];
    long long b_l0 = 0, b_l1 = 0, b_r0 = 0, b_r1 = 0;
    for (int i0 = 0; i0 < n - 1; i0 += TILE) {
        const int cnt = min(TILE + 1, n - i0);
#pragma unroll
        for (int t = 0; t < TILE + 1; ++t) {
            if (t < cnt) { const int pos = ord[i0 + t]; ybuf[t] = (yb[(long long)(pos >> 5) * B + b] >> (pos & 31)) & 1; }
        }
        for (int t = 0; t < TILE && i0 + t < n - 1; ++t) {
            const int i = i0 + t;
            const int yi = ybuf[t];
            if (yi) { ++l1; --r1; } else { ++l0; --r0; }
            const int n_l = i + 1, n_r = n - n_l;
            if (n_l >= min_split && n_r >= min_split && yi != ybuf[t + 1] && !tj[i]) {
                const {REAL} w_f = ({REAL})(ENT2((float)n_l, (float)l1) + ENT2((float)n_r, (float)r1));
                ++ncand;
                if (w_f < w_min) { w_min = w_f; best_i = i; b_l0 = l0; b_l1 = l1; b_r0 = r0; b_r1 = r1; }
            }
        }
    }
    double best_gain = -1e300;
    if (best_i >= 0) {
        const int n_l = best_i + 1, n_r = n - n_l;
        best_gain = h_full - (XL(n_l) - (XL(b_l0) + XL(b_l1)) + XL(n_r) - (XL(b_r0) + XL(b_r1))) / (double)n;
    }
    out_gain[idx] = best_gain;
    out_ncand[idx] = ncand;
    out_idx[idx] = best_i;
}
"""

_kernel_k5 = None


def get_kernel_k5(cp: Any) -> Any:
    """Compile v5 once per process (no fused multiply-add: the gain must round as the host's)."""
    global _kernel_k5
    if _kernel_k5 is None:
        _kernel_k5 = cp.RawKernel(_SOURCE_K5.replace("{REAL}", "float"), "root_split_stats_k5", options=("--fmad=false",))
    return _kernel_k5


_SOURCE_K4 = r"""
// The table ``tab`` holds x ln x for EVERY count up to n, so the gain needs no log call.
#define XL(v) (tab[(v)])
#define ENT2(m, c) (((c) > 0.f && (c) < (m)) ? -((c) * __logf((c) / (m)) + ((m) - (c)) * __logf(((m) - (c)) / (m))) : 0.f)
extern "C" __global__ void root_split_stats_k4(const int* __restrict__ order, const unsigned char* __restrict__ tie, const unsigned char* __restrict__ ybits,
                                               const double* __restrict__ tab, const long long* __restrict__ ctot, const int n, const int B,
                                               const int Kc, const int min_split, const double h_full, const int words, double* __restrict__ out_gain,
                                               int* __restrict__ out_ncand, int* __restrict__ out_idx) {
    const long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= (long long)B * Kc) return;
    const int j = (int)(idx / B);
    const int b = (int)(idx - (long long)j * B);
    const int* ord = order + (long long)j * n;
    const unsigned char* tj = tie + (long long)j * n;
    const unsigned int* yb = (const unsigned int*)ybits;
    const double n_f = (double)n;
    const int TILE = 8;
    int ybuf[TILE + 1];
    float w_min = 3.0e38f;
    int ncand = 0;
    long long l0 = 0, l1 = 0, r0 = ctot[0], r1 = ctot[1];
    double best_gain = -1e300;
    int best_idx = -1;
    for (int i0 = 0; i0 < n - 1; i0 += TILE) {
        const int cnt = min(TILE + 1, n - i0);
#pragma unroll
        for (int t = 0; t < TILE + 1; ++t) {
            if (t < cnt) { const int pos = ord[i0 + t]; ybuf[t] = (yb[(long long)(pos >> 5) * B + b] >> (pos & 31)) & 1; }
        }
        for (int t = 0; t < TILE && i0 + t < n - 1; ++t) {
            const int i = i0 + t;
            const int yi = ybuf[t];
            if (yi) { ++l1; --r1; } else { ++l0; --r0; }
            const int n_l = i + 1, n_r = n - n_l;
            if (n_l >= min_split && n_r >= min_split && yi != ybuf[t + 1] && !tj[i]) {
                const float w_f = ENT2((float)n_l, (float)l1) + ENT2((float)n_r, (float)r1);
                ++ncand;
                if (w_f < w_min) w_min = w_f;
                if (w_f <= w_min + 0.5f) {
                    const double gain = h_full - (XL(n_l) - (XL(l0) + XL(l1)) + XL(n_r) - (XL(r0) + XL(r1))) / n_f;
                    if (gain > best_gain) { best_gain = gain; best_idx = i; }
                }
            }
        }
    }
    out_gain[idx] = best_gain;
    out_ncand[idx] = ncand;
    out_idx[idx] = best_idx;
}
"""

_kernel_k4 = None


def get_kernel_k4(cp: Any) -> Any:
    """Compile v4 once per process (no fused multiply-add: the gains must round as the host's). It needs ``xlogx_table(n)`` (every count up to ``n``) as its table."""
    global _kernel_k4
    if _kernel_k4 is None:
        _kernel_k4 = cp.RawKernel(_SOURCE_K4, "root_split_stats_k4", options=("--fmad=false",))
    return _kernel_k4
