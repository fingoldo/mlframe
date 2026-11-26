"""Acceptance of the root split of a binary-target honest null by a sorted-space bitset, a popcount scan and a provable prune.

What the caller needs: for every (column, permuted target) pair only the boolean "the best root split is accepted", i.e. ``p(g) * max(1, ncand) < alpha`` with ``p(g) = erfc(sqrt(n g))`` (the G-test tail of a 2 x 2 table), where ``g`` is the best gain
over the valid cuts and ``ncand`` the number of valid cuts. ``p`` falls with ``g``, so the pair is accepted exactly when ``g`` exceeds ``thr(ncand) = erfcinv(alpha / ncand) ** 2 / n``. The exact gain of the vast majority of pairs, which are far from
that bar, is never needed.

Per pair one block: the labels of the rows in SORTED order are read through the column's sort order from a shared row-space bit table and packed into a shared bitset indexed by sorted position with warp ballots (no atomics: a
shared-memory ``atomicOr`` scatter was bound by atomic throughput, 4.9% shared efficiency), so the label sequence along the sorted column is ``n / 32`` words and the scan after it needs no per-row work. The valid cuts are the label changes (``bits ^ (bits >> 1)``) that are not ties and keep ``min_split`` rows on both sides, so ``ncand`` is one AND and one popcount per word.
A word is scanned exactly only if it can hold a cut above the bar: the information gain of a split equals its mutual information with the target, which is at most ``log(1 + phi^2)`` with ``phi^2 = D^2 / (n_l n_r pi (1 - pi))`` and
``D = l1 - n_l pi``; inside a word ``l1`` moves within 32 of the running ones count and ``n_l`` within 32 of the word start, which bounds ``phi^2`` from the word's prefix alone (computed in single precision with a 1e-3 margin; ``phi^2 >= log(1 + phi^2) >= MI``). The bar carries a 1e-6 margin so the prune only drops words
that cannot hold an accepted cut. Surviving words are scanned with the exact double-precision gain (the full ``x ln x`` table, the k2 / k4 expression and term order, the first strictly greater gain wins), so the outputs ``(gain, ncand, idx)`` of an accepted pair
equal the full scan's, and a pair without a surviving word reports ``idx = -1`` (rejected, as the full scan would reject it).

Handled when the sorted-space bitset fits the shared budget (about 380000 rows) and every permutation holds the same number of label-1 rows; otherwise the caller keeps the k5 / k4 kernel.
"""

from __future__ import annotations

import functools
import logging

import numpy as np

from ._kernel_arch_macros import ARCH_THREADS_HEADER
from ._kernel_types import render
from ._honest_null_device import _pack_bits_cached, label_one_counts
from ._raw_module_cache import RawModuleCache
from typing import Any
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

THREADS = 256
SHARED_BUDGET_BYTES = 22 * 1024  # per table; the kernel holds two of them (row-space and sorted-space labels)
_MARGIN = 1e-6

_SRC = ARCH_THREADS_HEADER + render(r"""
#define QCAP 512
extern "C" __global__ void __launch_bounds__(512, MLF_THREADS_PER_SM / 512) root_split_accept(const int* __restrict__ order, const unsigned int* __restrict__ valid, const unsigned int* __restrict__ ybits,
                                             const {ACC}* __restrict__ tab, const {ACC}* __restrict__ thr, const long long n, const int W, const int YW, const int m,
                                             const int B, const {ACC} h_full, const {ACC} pi,
                                             {ACC}* __restrict__ out_gain, int* __restrict__ out_ncand, int* __restrict__ out_idx
#ifdef ACCEPT_GLOBAL
                                             , unsigned int* __restrict__ scratch, const long long total
#endif
#ifdef ACCEPT_PRE
                                             , const unsigned int* __restrict__ pre, const int WP
#endif
                                             ) {
#ifdef ACCEPT_PRE
    // the sorted-space label sequences of all (column, permutation) pairs were built beforehand (``sorted_label_bits``): bit table word w of pair (j, b) is pre[(j * W + w) * B + b]
#define BITW(k) pre[((long long)(j * B + b)) * WP + (k)]   // the words of one pair are contiguous (row stride WP, a multiple of 8)
#else
#define BITW(k) bits[k]
#endif
#ifdef ACCEPT_PRE
    unsigned int* bits = 0;
#elif defined(ACCEPT_GLOBAL)
    // beyond the shared budget (about 180000 rows) the sorted-space bit table lives in a per-block global scratch row and the row-space labels are read from global memory (L2); a fixed grid of
    // resident blocks walks all the pairs, so the scratch is grid x W words instead of one row per pair
    unsigned int* bits = scratch + (long long)blockIdx.x * W;
#else
    extern __shared__ unsigned int sh[];              // YW words of row-space labels, then W words: bit i = label of the row at sorted position i
    unsigned int* ysh = sh;
    unsigned int* bits = sh + YW;
#endif
    __shared__ int warp_tot[32];
    __shared__ int s_ncand;
#ifndef ACCEPT_NOQUEUE
    __shared__ int s_qn;                               // surviving words of the pair: their exact scans are shared out over the warps of the block
    __shared__ int q_w[QCAP];
    __shared__ int q_l[QCAP];
#endif
    __shared__ {ACC} best_g[32];
    __shared__ int best_i[32];
    const int tid = threadIdx.x;
#ifdef ACCEPT_GLOBAL
    for (long long pair = blockIdx.x; pair < total; pair += gridDim.x) {
#else
    {
        const long long pair = blockIdx.x;
#endif
    const int j = (int)(pair / B);
    const int b = (int)(pair - (long long)j * B);
#ifndef ACCEPT_PRE
    const unsigned int* yb = ybits + (long long)b * YW;
#ifdef ACCEPT_GLOBAL
    const unsigned int* ysh = yb;
#else
    for (int w = tid; w < YW; w += blockDim.x) ysh[w] = yb[w];
#endif
#endif
    if (tid == 0) { s_ncand = 0;
#ifndef ACCEPT_NOQUEUE
        s_qn = 0;
#endif
    }
    __syncthreads();
#ifndef ACCEPT_PRE
    const int* ord = order + (long long)j * n;
    for (long long base = 0; base < n; base += 4LL * blockDim.x) {  // four independent 256-position steps per iteration: the loads and the table lookups overlap
        int row[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const long long i = base + (long long)u * blockDim.x + tid;
            row[u] = (i < n) ? ord[i] : -1;
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const long long i = base + (long long)u * blockDim.x + tid;
            const int lab = (row[u] >= 0) ? (int)((ysh[row[u] >> 5] >> (row[u] & 31)) & 1u) : 0;
            const unsigned int word = __ballot_sync(0xffffffffu, lab);
            if ((tid & 31) == 0 && (i >> 5) < W) bits[i >> 5] = word;
        }
    }
#endif
    __syncthreads();
    const int cw = (W + blockDim.x - 1) / blockDim.x;
    const int w0 = min(W, tid * cw);
    const int w1 = min(W, w0 + cw);
    const unsigned int* vj = valid + (long long)j * W;
    int pc = 0, nc = 0;
    for (int w = w0; w < w1; ++w) {
        const unsigned int cur = BITW(w);
        const unsigned int nxt = (w + 1 < W) ? BITW(w + 1) : 0u;
        const unsigned int bnd = cur ^ ((cur >> 1) | (nxt << 31));
        pc += __popc(cur);
        nc += __popc(bnd & vj[w]);
    }
    atomicAdd(&s_ncand, nc);
    // exclusive scan of the per-thread ones counts: warp-level inclusive scan by shuffles, then the 8 warp totals
    int v = pc;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const int t = __shfl_up_sync(0xffffffffu, v, o); if ((tid & 31) >= o) v += t; }
    if ((tid & 31) == 31) warp_tot[tid >> 5] = v;
    __syncthreads();
    if (tid == 0) {
        int run = 0;
        for (int w = 0; w < (int)(blockDim.x >> 5); ++w) { const int t = warp_tot[w]; warp_tot[w] = run; run += t; }
    }
    __syncthreads();
    int L = warp_tot[tid >> 5] + v - pc;               // ones before this thread's first word
    const int ncand = s_ncand;
    {ACC} bg = -1e300;
    int bi = -1;
    if (ncand > 0) {
        const {ACC} th = thr[ncand];
        const {REAL} pif = ({REAL})pi;
        const {REAL} pi1mf = ({REAL})(pi * (1.0 - pi));
        const {REAL} nnf = ({REAL})n;
        const {ACC} nn = ({ACC})n;
        for (int w = w0; w < w1; ++w) {
            const unsigned int cur = BITW(w);
            const unsigned int nxt = (w + 1 < W) ? BITW(w + 1) : 0u;
            const unsigned int cand = (cur ^ ((cur >> 1) | (nxt << 31))) & vj[w];
            if (cand) {
                // The bound in single precision ({ACC} rate is 1/32 of {REAL} on the target cards): MI <= log(1 + phi^2) <= phi^2, with 0.5 added to |D| and a 1e-3 relative margin covering the {REAL} rounding.
                const {REAL} i0 = ({REAL})w * 32.0f;
                const {REAL} dlo = ({REAL})L - (i0 + 32.0f) * pif;
                const {REAL} dhi = ({REAL})L + 32.0f - (i0 + 1.0f) * pif;
                const {REAL} d = fmaxf(fabsf(dlo), fabsf(dhi)) + 0.5f;
                const {REAL} nl1 = i0 + 1.0f, nl2 = fminf(i0 + 32.0f, nnf - 1.0f);
                const {REAL} fmn = fminf(nl1 * (nnf - nl1), nl2 * (nnf - nl2));
                const {ACC} bound = ({ACC})(d * d / (fmn * pi1mf)) * 1.001 + 1e-12;
                if (bound >= th) {
#ifndef ACCEPT_NOQUEUE
                    const int slot = atomicAdd(&s_qn, 1);
                    if (slot < QCAP) { q_w[slot] = w; q_l[slot] = L; }
                    else
#endif
                    {
                        unsigned int c = cand;
                        while (c) {
                            const int bpos = __ffs(c) - 1;
                            c &= c - 1;
                            const long long i = (long long)w * 32 + bpos;
                            const unsigned int upto = (bpos == 31) ? 0xFFFFFFFFu : ((2u << bpos) - 1u);
                            const long long l1 = (long long)L + __popc(cur & upto);
                            const long long n_l = i + 1, n_r = (long long)n - n_l, l0 = n_l - l1;
                            const long long r1 = (long long)m - l1, r0 = ((long long)n - m) - l0;
                            const {ACC} gain = h_full - (tab[n_l] - (tab[l0] + tab[l1]) + tab[n_r] - (tab[r0] + tab[r1])) / nn;
                            if (gain > bg || (gain == bg && i < bi)) { bg = gain; bi = (int)i; }
                        }
                    }
                }
            }
            L += __popc(cur);
        }
    }
#ifndef ACCEPT_NOQUEUE
    __syncthreads();                                   // the queue is complete
    {
        // the exact scans of the surviving words, one warp per word and one lane per bit: the gain formula and the (larger gain, then smaller index) rule are the thread-serial scan's, so the winner is the same
        const int nq = min(s_qn, QCAP);
        const int lane = tid & 31;
        const {ACC} nn2 = ({ACC})n;
        for (int e = tid >> 5; e < nq; e += (int)(blockDim.x >> 5)) {
            const int w = q_w[e];
            const int Lw = q_l[e];
            const unsigned int cur = BITW(w);
            const unsigned int nxt = (w + 1 < W) ? BITW(w + 1) : 0u;
            const unsigned int cand = (cur ^ ((cur >> 1) | (nxt << 31))) & vj[w];
            if ((cand >> lane) & 1u) {
                const long long i = (long long)w * 32 + lane;
                const unsigned int upto = (lane == 31) ? 0xFFFFFFFFu : ((2u << lane) - 1u);
                const long long l1 = (long long)Lw + __popc(cur & upto);
                const long long n_l = i + 1, n_r = (long long)n - n_l, l0 = n_l - l1;
                const long long r1 = (long long)m - l1, r0 = ((long long)n - m) - l0;
                const {ACC} gain = h_full - (tab[n_l] - (tab[l0] + tab[l1]) + tab[n_r] - (tab[r0] + tab[r1])) / nn2;
                if (gain > bg || (gain == bg && i < bi)) { bg = gain; bi = (int)i; }
            }
        }
    }
#endif
    // best (gain, index) of the block: shuffle reduction inside each warp, then the 8 warp winners; ties go to the smaller index
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
        const {ACC} og = __shfl_down_sync(0xffffffffu, bg, o);
        const int oi = __shfl_down_sync(0xffffffffu, bi, o);
        if (oi >= 0 && (bi < 0 || og > bg || (og == bg && oi < bi))) { bg = og; bi = oi; }
    }
    if ((tid & 31) == 0) { best_g[tid >> 5] = bg; best_i[tid >> 5] = bi; }
    __syncthreads();
    if (tid == 0) {
        {ACC} G = -1e300;
        int I = -1;
        for (int t = 0; t < (int)(blockDim.x >> 5); ++t)
            if (best_i[t] >= 0 && (best_g[t] > G || (best_g[t] == G && best_i[t] < I))) { G = best_g[t]; I = best_i[t]; }
        const long long idx = pair;
        out_gain[idx] = G;
        out_ncand[idx] = ncand;
        out_idx[idx] = I;
    }
    __syncthreads();                                   // the next pair of this block reuses bits / s_ncand
    }
}
""", real=np.float32)

_MODULE = RawModuleCache(_SRC)
_MODULE_NOQUEUE = RawModuleCache("#define ACCEPT_NOQUEUE 1\n" + _SRC)  # v1: the exact scan of a surviving word stays with the thread that owns it (kept as a version; the queue variant is the default)
_MODULE_G = RawModuleCache("#define ACCEPT_GLOBAL 1\n" + _SRC)
_MODULE_PRE = RawModuleCache("#define ACCEPT_PRE 1\n" + _SRC)

#: The acceptance scan over the precomputed sorted-space tables with COALESCED loads (ACCEPT_PRE, ``root_split_accept`` above, is kept as the reference). In that kernel every thread owns a contiguous run of
#: 61 words (500000 rows), so a warp load touches 32 sectors: nvprof gld_efficiency 13 percent, 17.4 transactions per request, 25 million local loads, IPC 0.32 and two thirds of the stalls waiting for memory,
#: 43.2 ms for 24 columns x 128 permutations. Here a warp walks a contiguous SEGMENT of the words 32 at a time (one 128-byte load per step); the ones before a word come from a shuffle scan in the warp plus a carry,
#: the segments' starting counts from one scan of the warp totals. Outputs (gain, ncand, idx) equal the reference kernel's: ``ncand`` is an integer sum, the exact gain is the same expression, and the winner is the
#: larger gain, the smaller index on a tie, whatever order the words are visited in. 11.2 ms for the same shape (3.85 times). A loop unrolled four times with all loads issued first was no faster (11.8 ms).
#: Launch bounds (512, 2): the reference's (512, MLF_THREADS_PER_SM / 512) caps the kernel at 32 registers on sm_61 and it spills (48 bytes of local memory, flagged by the offline ptxas check); this one needs 56 and does not spill, at the same time.
_SRC_PRE_CO = ARCH_THREADS_HEADER + render(r"""
#define QCAP 512
extern "C" __global__ void __launch_bounds__(512, 2) root_split_accept_pre_co(const int* __restrict__ order, const unsigned int* __restrict__ valid, const unsigned int* __restrict__ ybits,
                                             const {ACC}* __restrict__ tab, const {ACC}* __restrict__ thr, const long long n, const int W, const int YW, const int m,
                                             const int B, const {ACC} h_full, const {ACC} pi,
                                             {ACC}* __restrict__ out_gain, int* __restrict__ out_ncand, int* __restrict__ out_idx,
                                             const unsigned int* __restrict__ pre, const int WP) {
    __shared__ int warp_pc[32];
    __shared__ int s_ncand;
    __shared__ int s_qn;
    __shared__ int q_w[QCAP];
    __shared__ int q_l[QCAP];
    __shared__ {ACC} best_g[32];
    __shared__ int best_i[32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = (int)(blockDim.x >> 5);
    const long long pair = blockIdx.x;
    const int j = (int)(pair / B);
    const int b = (int)(pair - (long long)j * B);
    const unsigned int* bw = pre + ((long long)(j * B + b)) * WP;
    const unsigned int* vj = valid + (long long)j * W;
    const int seg = (((W + nw - 1) / nw) + 31) & ~31;      // words per warp, a multiple of 32
    const int s0 = min(W, warp * seg), s1 = min(W, s0 + seg);
    if (tid == 0) { s_ncand = 0; s_qn = 0; }
    __syncthreads();
    // pass 1: ones and valid label changes of the segment, one coalesced load of 32 words per step
    int pc = 0, nc = 0;
    for (int base = s0; base < s1; base += 32) {
        const int w = base + lane;
        unsigned int cur = 0u, v = 0u;
        if (w < s1) { cur = bw[w]; v = vj[w]; }
        unsigned int nxt = __shfl_down_sync(0xffffffffu, cur, 1);
        if (lane == 31 || w + 1 >= s1) nxt = (w + 1 < W) ? bw[w + 1] : 0u;
        if (w < s1) {
            const unsigned int bnd = cur ^ ((cur >> 1) | (nxt << 31));
            pc += __popc(cur);
            nc += __popc(bnd & v);
        }
    }
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) { pc += __shfl_down_sync(0xffffffffu, pc, o); nc += __shfl_down_sync(0xffffffffu, nc, o); }
    if (lane == 0) { warp_pc[warp] = pc; atomicAdd(&s_ncand, nc); }
    __syncthreads();
    if (tid == 0) {
        int run = 0;
        for (int w = 0; w < nw; ++w) { const int t = warp_pc[w]; warp_pc[w] = run; run += t; }
    }
    __syncthreads();
    int L = warp_pc[warp];                              // ones before the first word of this warp's segment
    const int ncand = s_ncand;
    {ACC} bg = -1e300;
    int bi = -1;
    if (ncand > 0) {
        const {ACC} th = thr[ncand];
        const {REAL} pif = ({REAL})pi;
        const {REAL} pi1mf = ({REAL})(pi * (1.0 - pi));
        const {REAL} nnf = ({REAL})n;
        const {ACC} nn = ({ACC})n;
        for (int base = s0; base < s1; base += 32) {
            const int w = base + lane;
            unsigned int cur = 0u, v = 0u;
            if (w < s1) { cur = bw[w]; v = vj[w]; }
            unsigned int nxt = __shfl_down_sync(0xffffffffu, cur, 1);
            if (lane == 31 || w + 1 >= s1) nxt = (w + 1 < W) ? bw[w + 1] : 0u;
            const int x = (w < s1) ? __popc(cur) : 0;
            int incl = x;                                // inclusive scan of the ones counts of the 32 words
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) { const int t = __shfl_up_sync(0xffffffffu, incl, o); if (lane >= o) incl += t; }
            const int Lw = L + incl - x;                  // ones before word w
            L += __shfl_sync(0xffffffffu, incl, 31);
            if (w < s1) {
                const unsigned int cand = (cur ^ ((cur >> 1) | (nxt << 31))) & v;
                if (cand) {
                    const {REAL} i0 = ({REAL})w * 32.0f;
                    const {REAL} dlo = ({REAL})Lw - (i0 + 32.0f) * pif;
                    const {REAL} dhi = ({REAL})Lw + 32.0f - (i0 + 1.0f) * pif;
                    const {REAL} d = fmaxf(fabsf(dlo), fabsf(dhi)) + 0.5f;
                    const {REAL} nl1 = i0 + 1.0f, nl2 = fminf(i0 + 32.0f, nnf - 1.0f);
                    const {REAL} fmn = fminf(nl1 * (nnf - nl1), nl2 * (nnf - nl2));
                    const {ACC} bound = ({ACC})(d * d / (fmn * pi1mf)) * 1.001 + 1e-12;
                    if (bound >= th) {
                        const int slot = atomicAdd(&s_qn, 1);
                        if (slot < QCAP) { q_w[slot] = w; q_l[slot] = Lw; }
                        else {
                            unsigned int c = cand;
                            while (c) {
                                const int bpos = __ffs(c) - 1;
                                c &= c - 1;
                                const long long i = (long long)w * 32 + bpos;
                                const unsigned int upto = (bpos == 31) ? 0xFFFFFFFFu : ((2u << bpos) - 1u);
                                const long long l1 = (long long)Lw + __popc(cur & upto);
                                const long long n_l = i + 1, n_r = (long long)n - n_l, l0 = n_l - l1;
                                const long long r1 = (long long)m - l1, r0 = ((long long)n - m) - l0;
                                const {ACC} gain = h_full - (tab[n_l] - (tab[l0] + tab[l1]) + tab[n_r] - (tab[r0] + tab[r1])) / nn;
                                if (gain > bg || (gain == bg && i < bi)) { bg = gain; bi = (int)i; }
                            }
                        }
                    }
                }
            }
        }
    }
    __syncthreads();                                   // the queue is complete
    {
        const int nq = min(s_qn, QCAP);
        const {ACC} nn2 = ({ACC})n;
        for (int e = warp; e < nq; e += nw) {
            const int w = q_w[e];
            const int Lw = q_l[e];
            const unsigned int cur = bw[w];
            const unsigned int nxt = (w + 1 < W) ? bw[w + 1] : 0u;
            const unsigned int cand = (cur ^ ((cur >> 1) | (nxt << 31))) & vj[w];
            if ((cand >> lane) & 1u) {
                const long long i = (long long)w * 32 + lane;
                const unsigned int upto = (lane == 31) ? 0xFFFFFFFFu : ((2u << lane) - 1u);
                const long long l1 = (long long)Lw + __popc(cur & upto);
                const long long n_l = i + 1, n_r = (long long)n - n_l, l0 = n_l - l1;
                const long long r1 = (long long)m - l1, r0 = ((long long)n - m) - l0;
                const {ACC} gain = h_full - (tab[n_l] - (tab[l0] + tab[l1]) + tab[n_r] - (tab[r0] + tab[r1])) / nn2;
                if (gain > bg || (gain == bg && i < bi)) { bg = gain; bi = (int)i; }
            }
        }
    }
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
        const {ACC} og = __shfl_down_sync(0xffffffffu, bg, o);
        const int oi = __shfl_down_sync(0xffffffffu, bi, o);
        if (oi >= 0 && (bi < 0 || og > bg || (og == bg && oi < bi))) { bg = og; bi = oi; }
    }
    if (lane == 0) { best_g[warp] = bg; best_i[warp] = bi; }
    __syncthreads();
    if (tid == 0) {
        {ACC} G = -1e300;
        int I = -1;
        for (int t = 0; t < nw; ++t)
            if (best_i[t] >= 0 && (best_g[t] > G || (best_g[t] == G && best_i[t] < I))) { G = best_g[t]; I = best_i[t]; }
        out_gain[pair] = G;
        out_ncand[pair] = ncand;
        out_idx[pair] = I;
    }
}
""", real=np.float32)
_MODULE_PRE_CO = RawModuleCache(_SRC_PRE_CO)

#: Builds the sorted-space label sequences of every (column, permutation) pair of a chunk from the row-major label bit matrix ``yt`` (row r: one word per 32 permutations, bit b = label of row r under
#: permutation b): a warp owns 32 sorted positions, gathers the B-bit vector of the row at each position (one sector per position instead of one lookup per position and permutation) and transposes the
#: 32 x 32 bit tile with ballots, so thread t ends with the 32 positions' bits of permutation ``pw * 32 + t``.
_SRC_SORTED_BITS = r"""
extern "C" __global__ void sorted_label_bits(const int* __restrict__ order, const unsigned int* __restrict__ yt, unsigned int* __restrict__ pre,
                                             const long long n, const int W, const int PW, const int B) {
    const int w = blockIdx.x;
    const int pw = blockIdx.y;
    const int jj = blockIdx.z;
    const int t = threadIdx.x;
    const long long i = (long long)w * 32 + t;
    const int r = (i < n) ? order[(long long)jj * n + i] : -1;
    const unsigned int v = (r >= 0) ? yt[(long long)r * PW + pw] : 0u;
    unsigned int mine = 0u;
    for (int bp = 0; bp < 32; ++bp) {
        const unsigned int c = __ballot_sync(0xffffffffu, (v >> bp) & 1u);
        if (t == bp) mine = c;
    }
    const int b = pw * 32 + t;
    if (b < B) pre[((long long)jj * W + w) * B + b] = mine;
}
"""
_MODULE_SORTED = RawModuleCache(_SRC_SORTED_BITS)

#: ``sorted_label_bits`` writing the tables pair-major: ``pre[(j * B + b) * WP + w]`` with the row stride ``WP`` a multiple of 8. A warp owns 8 words (256 sorted positions) of one block of 32 permutations:
#: the lane of permutation ``b`` ends with its 8 words in registers and stores them as two 16-byte vectors, so the writes fill whole sectors; the scan then reads the words of a pair contiguously.
_SRC_SORTED_BITS8 = r"""
extern "C" __global__ void sorted_label_bits8(const int* __restrict__ order, const uint4* __restrict__ yt, unsigned int* __restrict__ pre,
                                              const long long n, const int WP, const int B) {
    // yt: one uint4 a row, the labels of the row under permutations 0..127 of the chunk (bit b of word q = permutation 32 q + b)
    const int w0 = blockIdx.x * 8;
    const int jj = blockIdx.y;
    const int t = threadIdx.x;
    unsigned int mine[4][8];
#pragma unroll
    for (int ww = 0; ww < 8; ++ww) {
        const long long i = (long long)(w0 + ww) * 32 + t;
        const int r = (i < n) ? order[(long long)jj * n + i] : -1;
        const uint4 v4 = (r >= 0) ? yt[r] : make_uint4(0u, 0u, 0u, 0u);   // one sector for the four words of the row
        const unsigned int vq[4] = {v4.x, v4.y, v4.z, v4.w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            // 32 x 32 bit transpose across the warp (lane = row, bit = column): five butterfly steps, each swapping the off-diagonal j x j blocks with the xor partner; afterwards lane t holds
            // bit t of every lane's word, which is what 32 ballots gave one lane at a time
            unsigned int x = vq[q];
#pragma unroll
            for (int j = 16, mk = 0x0000FFFF; j != 0; j >>= 1, mk ^= (mk << j)) {
                const unsigned int y = __shfl_xor_sync(0xffffffffu, x, j);
                x = (t & j) ? (((y >> j) & (unsigned int)mk) | (x & ~(unsigned int)mk)) : ((x & (unsigned int)mk) | ((y & (unsigned int)mk) << j));
            }
            mine[q][ww] = x;
        }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int b = q * 32 + t;
        if (b < B) {
            uint4* dst = (uint4*)(pre + ((long long)jj * B + b) * WP + w0);
            dst[0] = make_uint4(mine[q][0], mine[q][1], mine[q][2], mine[q][3]);
            dst[1] = make_uint4(mine[q][4], mine[q][5], mine[q][6], mine[q][7]);
        }
    }
}
"""
_MODULE_SORTED8 = RawModuleCache(_SRC_SORTED_BITS8)

#: ``(n, pw)`` row-major label words from the ``(permutations, words)`` row-space bit rows, one 32 x 32 bit transpose per warp (word ``w`` of 32 permutations -> the 32 rows ``32 w .. 32 w + 31``): lane t reads
#: word ``w`` of permutation ``32 pw + t`` (bit j = the label of row ``32 w + j``), the five butterfly steps leave lane t the bits of row ``32 w + t`` (bit p = the label under permutation ``32 pw + p``). The cupy chain it replaces
#: (unpackbits to bytes, a transposed byte copy, packbits) took 18.5 ms per call at 500000 rows x 128 permutations, 405 calls in a 500k fit (nvprof: cupy_copy uint8 5.2 s, unpackbits 1.6 s, packbits 0.8 s).
_SRC_LABEL_TRANSPOSE = r"""
extern "C" __global__ void label_bits_transpose(const unsigned int* __restrict__ yb, unsigned int* __restrict__ yt, const long long n, const int YW, const int nb, const int PW) {
    const int w = blockIdx.x;
    const int pw = blockIdx.y;
    const int t = threadIdx.x;
    const int perm = pw * 32 + t;
    unsigned int x = (perm < nb) ? yb[(long long)perm * YW + w] : 0u;
#pragma unroll
    for (int j = 16, mk = 0x0000FFFF; j != 0; j >>= 1, mk ^= (mk << j)) {
        const unsigned int y = __shfl_xor_sync(0xffffffffu, x, j);
        x = (t & j) ? (((y >> j) & (unsigned int)mk) | (x & ~(unsigned int)mk)) : ((x & (unsigned int)mk) | ((y & (unsigned int)mk) << j));
    }
    const long long row = (long long)w * 32 + t;
    if (row < n) yt[row * PW + pw] = x;
}
"""
_MODULE_LABEL_TRANSPOSE = RawModuleCache(_SRC_LABEL_TRANSPOSE)
_PERM_CHUNK = 128  # permutations per pass: the unpacked label matrix is rows x 128 bytes
_PRE_BYTES = 192 * 1024 * 1024  # sorted-space bit tables held at a time


@functools.lru_cache(maxsize=8)
def threshold_table(n: int, alpha: float) -> np.ndarray:
    """``thr[c]``: the gain a pair with ``c`` valid cuts must exceed to be accepted (``p(g) * max(1, c) < alpha`` with ``p = erfc(sqrt(n g))``), scaled down by the prune margin; ``thr[0]`` is infinite (no candidate).

    A function of ``(n, alpha)`` only, asked at every acceptance call of a fit (92 of them at 500000 rows, 57 ms of ``erfcinv`` each): remembered, and returned read-only."""
    from scipy.special import erfcinv

    c = np.maximum(1, np.arange(n + 1, dtype=np.float64))  # Intentional float64: the acceptance bar in double precision
    thr = erfcinv(np.minimum(alpha / c, 1.0)) ** 2 / float(n) * (1.0 - _MARGIN)
    thr[0] = np.inf
    thr.flags.writeable = False
    return np.asarray(thr)


def valid_cut_bits(tie: np.ndarray, n: int, min_split: int) -> np.ndarray:
    """``(columns, words)`` uint32: bit ``i`` set when cut ``i`` is valid - not a tie with the next value and ``min_split`` rows on both sides (and ``i <= n - 2``)."""
    k = int(tie.shape[0])
    words = (n + 31) // 32
    pos = np.arange(n)
    in_range = (pos <= n - 2) & (pos + 1 >= min_split) & (n - pos - 1 >= min_split)
    mask = (np.asarray(tie) == 0) & in_range[None, :]
    padded = np.zeros((k, words * 32), dtype=np.uint8)
    padded[:, :n] = mask
    return np.ascontiguousarray(np.packbits(padded, axis=1, bitorder="little").view(np.uint32))


def accept_fits(n: int, y_raw_perm: np.ndarray, ones: "np.ndarray | None" = None) -> bool:
    """Whether the bitset path applies: the bitset fits the shared budget and every permutation has the same number of label-1 rows (``ones``: those counts when the caller has them)."""
    words = (int(n) + 31) // 32
    if words * 4 > SHARED_BUDGET_BYTES:
        return False
    ones = label_one_counts(y_raw_perm) if ones is None else ones
    return bool(ones.size and (ones == ones[0]).all() and 0 < int(ones[0]) < int(n))


def accept_fits_global(n: int, y_raw_perm: np.ndarray, ones: "np.ndarray | None" = None) -> bool:
    """Whether the bitset path applies with its tables in global memory (any row count): every permutation has the same number of label-1 rows (``ones``: those counts when the caller has them)."""
    ones = label_one_counts(y_raw_perm) if ones is None else ones
    return bool(ones.size and (ones == ones[0]).all() and 0 < int(ones[0]) < int(n))


def _block_width(kernel, shared_bytes: int, n: int) -> int:
    """Threads per block from the device's limits (``pyutilz.performance.gpu_hw_launch.fixed_shared_block_size``: the table is fixed per block, so the width that keeps the most threads resident on an SM), ``THREADS`` without device information; ``MLFRAME_GPU_BLOCK_HONEST_ACCEPT`` overrides."""
    import os

    forced = os.environ.get("MLFRAME_GPU_BLOCK_HONEST_ACCEPT", "").strip()
    if forced.isdigit() and 32 <= int(forced) <= 1024 and int(forced) % 32 == 0:
        return int(forced)
    from pyutilz.performance.gpu_hw_launch import fixed_shared_block_size

    try:
        width = fixed_shared_block_size(shared_bytes, regs_per_thread=int(kernel.num_regs), static_smem=int(kernel.shared_size_bytes), min_width=64, max_width=512, work_items=n)
    except Exception as exc:  # best-effort: only the launch width changes, never a result
        log_throttle(logger, "_honest_null_accept_bits.block_width", logging.INFO, "honest-null accept block width: device limits unavailable (%s)", exc)
        width = 0
    return width if width > 0 else THREADS


def _launch_global(cp: Any, d_order: Any, valid: Any, ybits: Any, tab_full: Any, thr: Any, n: int, words: int, m: int, n_perm: int, kc: int, h_full: float, d_gain: Any, d_ncand: Any, d_idx: Any) -> bool:
    """The scan with the sorted-space bit table in a per-block global scratch row: a fixed grid of resident blocks walks every (column, permutation) pair, so the row count is not limited by the
    shared-memory budget (the shared variant stops at about 180000 rows, where the k5 kernel's one thread per pair took over: 1.06 s for 8 columns x 256 permutations at 1000000 rows)."""
    kernel = _MODULE_G.get(cp).get_function("root_split_accept")
    width = _block_width(kernel, 0, n)
    sms = int(cp.cuda.Device().attributes.get("MultiProcessorCount", 6))
    total = int(kc) * int(n_perm)
    grid = max(1, min(total, sms * max(1, 1024 // width) * 2))
    scratch = cp.empty(grid * words, dtype=cp.uint32)
    kernel(
        (grid,), (width,),
        # Intentional float64: the kernel signature declares double
        (d_order, valid, ybits, tab_full, thr, np.int64(n), np.int32(words), np.int32(ybits.shape[1]), np.int32(m), np.int32(n_perm), np.float64(h_full), np.float64(m / float(n)), d_gain, d_ncand, d_idx, scratch, np.int64(total)),
    )
    return True


def _transposed_label_bits_kernel(cp: Any, yb: Any, n: int, pw: int) -> Any:
    """:func:`_transposed_label_bits` by one bit-transpose kernel (a warp per 32 rows x 32 permutations) instead of unpack, byte transpose and pack: 0.2 ms against 18.5 ms at 500000 rows x 128 permutations."""
    nb, yw = int(yb.shape[0]), int(yb.shape[1])
    out = cp.zeros((n, pw), dtype=cp.uint32)
    _MODULE_LABEL_TRANSPOSE.get(cp).get_function("label_bits_transpose")((yw, pw), (32,), (cp.ascontiguousarray(yb), out, np.int64(n), np.int32(yw), np.int32(nb), np.int32(pw)))  # launch-ok: one warp per block by design, a warp transposes a 32 x 32 bit tile with shuffles
    return out


def label_transpose_kernel_enabled() -> bool:
    """Whether the label words are transposed by the kernel (default; ``MLFRAME_HONEST_ACCEPT_LABEL_TRANSPOSE=cupy`` keeps the cupy chain), read on every call."""
    import os

    return os.environ.get("MLFRAME_HONEST_ACCEPT_LABEL_TRANSPOSE", "kernel").strip().lower() != "cupy"


def _transposed_label_bits(cp: Any, yb: Any, n: int, pw: int) -> Any:
    """``(n, pw)`` uint32: row ``r`` holds bit ``b`` = the label of row ``r`` under permutation ``b`` of ``yb`` (``(permutations, words)`` row-space bits), 32 permutations per word."""
    nb = int(yb.shape[0])
    bits = cp.unpackbits(cp.ascontiguousarray(yb).view(cp.uint8).reshape(-1), bitorder="little").reshape(nb, -1)[:, :n]  # cupy packs and unpacks flat arrays only
    padded = cp.zeros((n, pw * 32), dtype=cp.uint8)
    padded[:, :nb] = bits.T
    return cp.packbits(padded.reshape(-1), bitorder="little").view(cp.uint32).reshape(n, pw)


def coalesced_pre_scan_enabled() -> bool:
    """Whether the scan over the precomputed tables runs the coalesced kernel (default; ``MLFRAME_HONEST_ACCEPT_PRE_SCAN=block`` runs the reference thread-per-run kernel), read on every call."""
    import os

    return os.environ.get("MLFRAME_HONEST_ACCEPT_PRE_SCAN", "coalesced").strip().lower() != "block"


def _launch_pre(cp: Any, d_order: Any, valid: Any, ybits: Any, tab_full: Any, thr: Any, n: int, words: int, m: int, n_perm: int, kc: int, h_full: float, d_gain: Any, d_ncand: Any, d_idx: Any) -> bool:
    """The scan over sorted-space label sequences built beforehand by gathering one B-bit vector per sorted position (``sorted_label_bits``): the scan kernel does no label lookup per (position,
    permutation), whatever the row count. Permutations go in chunks of ``_PERM_CHUNK`` and columns in chunks that keep the tables under ``_PRE_BYTES``."""
    build = _MODULE_SORTED8.get(cp).get_function("sorted_label_bits8")
    scan = _MODULE_PRE_CO.get(cp).get_function("root_split_accept_pre_co") if coalesced_pre_scan_enabled() else _MODULE_PRE.get(cp).get_function("root_split_accept")
    width = _block_width(scan, 0, n)
    wp = ((words + 7) // 8) * 8
    g2, c2, i2 = d_gain.reshape(kc, n_perm), d_ncand.reshape(kc, n_perm), d_idx.reshape(kc, n_perm)
    for b0 in range(0, n_perm, _PERM_CHUNK):
        bc = min(_PERM_CHUNK, n_perm - b0)
        yt = (_transposed_label_bits_kernel if label_transpose_kernel_enabled() else _transposed_label_bits)(cp, ybits[b0 : b0 + bc], n, 4)  # padded to four words a row (a chunk is at most 128 permutations): one 16-byte gather per row
        col_chunk = max(1, _PRE_BYTES // (wp * bc * 4))
        for j0 in range(0, kc, col_chunk):
            jc = min(col_chunk, kc - j0)
            pre = cp.empty((jc, bc, wp), dtype=cp.uint32)
            build((wp // 8, jc), (32,), (d_order[j0 : j0 + jc], yt, pre, np.int64(n), np.int32(wp), np.int32(bc)))  # launch-ok: one warp per block, the kernel ballots across the 32 lanes
            tg, tc, ti = cp.empty(jc * bc, dtype=d_gain.dtype), cp.empty(jc * bc, dtype=cp.int32), cp.empty(jc * bc, dtype=cp.int32)
            scan(
                (jc * bc,), (width,),
                # Intentional float64: the kernel signature declares double
                (d_order[j0 : j0 + jc], valid[j0 : j0 + jc], ybits, tab_full, thr, np.int64(n), np.int32(words), np.int32(ybits.shape[1]), np.int32(m), np.int32(bc), np.float64(h_full), np.float64(m / float(n)), tg, tc, ti, pre, np.int32(wp)),
            )
            g2[j0 : j0 + jc, b0 : b0 + bc] = tg.reshape(jc, bc)
            c2[j0 : j0 + jc, b0 : b0 + bc] = tc.reshape(jc, bc)
            i2[j0 : j0 + jc, b0 : b0 + bc] = ti.reshape(jc, bc)
    return True


def root_split_accept_gpu(cp: Any, order: Any, tie: Any, y_raw_perm: Any, n: int, n_perm: int, kc: int, min_split: int, alpha: float, h_full: float, tab_full: Any, d_gain: Any, d_ncand: Any, d_idx: Any) -> bool:
    """Fill ``d_gain`` / ``d_ncand`` / ``d_idx`` (index ``column * n_perm + permutation``) with the pruned acceptance scan of every pair; ``False`` when the path does not apply (nothing written)."""
    import os

    # Above the shared budget (about 180000 rows) the scan runs over sorted-space label sequences built beforehand (``_launch_pre``: 3.7 times faster than the k5 kernel that used to take over there,
    # 0.29 s against 1.05 s for 1000000 rows x 8 columns x 256 permutations). MLFRAME_HONEST_ACCEPT_MODE: ``off`` declines above the budget (k5 as before), ``global`` is the variant that keeps the
    # tables in global memory and looks the labels up per element (exact, but slower than k5: 1.93 s against 1.06 s), ``pre`` forces the precomputed variant at any row count.
    mode = os.environ.get("MLFRAME_HONEST_ACCEPT_MODE", "").strip().lower()
    if not mode and os.environ.get("MLFRAME_HONEST_ACCEPT_GLOBAL", "").strip() == "1":
        mode = "global"
    ones = label_one_counts(y_raw_perm)  # one pass over the permutations for the shape tests and the row total
    shared_ok = accept_fits(n, y_raw_perm, ones)
    use_global = mode == "global"
    # the precomputed sorted-space tables are the default at every row count (``block`` keeps the kernel that fills a shared bit table per pair, the reference): at 100000 rows x 120 columns x 999
    # permutations the fill phase was 69 percent of that kernel's cycles (random shared-memory lookups, 24 percent shared efficiency), and building the tables 32 permutations per gather is
    # 2 times faster overall with the same outputs
    use_pre = mode in ("pre", "")
    if mode == "off" and not shared_ok:
        return False
    if not shared_ok and not accept_fits_global(n, y_raw_perm, ones):
        return False
    words = (int(n) + 31) // 32
    m = int(ones[0])
    d_order = cp.asarray(np.ascontiguousarray(order, dtype=np.int32))
    valid = cp.asarray(valid_cut_bits(tie, n, int(min_split)))
    thr = cp.asarray(threshold_table(int(n), float(alpha)))
    ybits = cp.asarray(np.ascontiguousarray(_pack_bits_cached(y_raw_perm)))  # (permutations, words): the row-space labels of every permutation
    if use_pre:
        return _launch_pre(cp, d_order, valid, ybits, tab_full, thr, n, words, m, n_perm, kc, h_full, d_gain, d_ncand, d_idx)
    if use_global:
        return _launch_global(cp, d_order, valid, ybits, tab_full, thr, n, words, m, n_perm, kc, h_full, d_gain, d_ncand, d_idx)
    kernel = _MODULE.get(cp).get_function("root_split_accept")
    shared = (words + int(ybits.shape[1])) * 4
    width = _block_width(kernel, shared, n)
    kernel(
        (kc * n_perm,), (width,),
        (d_order, valid, ybits, tab_full, thr, np.int64(n), np.int32(words), np.int32(ybits.shape[1]), np.int32(m), np.int32(n_perm), np.float64(h_full), np.float64(m / float(n)), d_gain, d_ncand, d_idx),  # Intentional float64: the kernel signature declares double
        shared_mem=shared,
    )
    return True


__all__ = ["root_split_accept_gpu", "threshold_table", "valid_cut_bits", "accept_fits", "accept_fits_global"]
