"""Version 3 of the binary-target root-split kernel of the device honest null (``root_split_stats_k2`` is version 2 and stays).
BENCH-ATTEMPT-REJECTED (2026-11-01, GTX 1050 Ti, outputs bit-identical to v2): v3 is 1.4 to 1.8 times SLOWER than v2 on every shape tried (60 columns x 200 permutations x 100000 rows: 790 -> 1107 ms;
256 x 100 x 100000: 1590 -> 2332 ms; 250 x 400 x 100000: 6489 -> 9692 ms; 250 x 400 x 50000: 2844 -> 5255 ms). Why: in v2 the threads of a warp are consecutive permutations of one column, so the label read of a
step is already ONE coalesced 128-byte transaction per warp; moving the bitset to shared memory replaces it with a per-thread random shared read (bank conflicts, about 3 to 4 way on random words) and caps the
resident blocks by the 12.5 KB of shared memory per block (448 to 896 threads per SM instead of up to 2048), so latency is hidden worse. Retry condition: a card with a larger shared-memory budget per SM or
a much smaller L2 than the table (then the scattered line fetches of v2 cost more than the bank conflicts); `tests/feature_selection/fe/test_honest_null_k3.py` keeps the equivalence so it can be switched on by a
tuner. Not wired into the dispatch.


Win axis: v2 is a thread per (column, permutation) that scans the sorted rows and reads, at every step, one label bit from a global ``(words, permutations)`` table: a random gather over a table that is larger
than the L2 cache (measured: the gather is 52% of the scan, the kernel does ~2.1 G scan steps per second whatever the tiling). Here a block belongs to ONE permutation: it copies that permutation's whole label
bitset (``n / 8`` bytes: 12.5 KB at 100000 rows) to shared memory once, and each of its threads (one per column) then gathers from shared memory. It wins when the bitset fits the shared-memory budget
(``n`` up to ~380000 rows on 48 KB) and there are enough columns per permutation to fill a block's threads; it loses nothing else, the arithmetic and the tie rule are v2's, so the outputs are identical.

``{ACC}`` is double for the exact gain and the entropy table, ``{REAL}`` float for the pass-1 score, exactly as in v2 (the kernel states them as ``double`` / ``float`` because the float pass-1 score is its design, not
a precision choice of the caller). Keep v2 for the shapes that do not fit.
"""

from __future__ import annotations

from ._kernel_types import render

from typing import Any

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for

MAX_SHARED_BYTES = 48 * 1024

_SOURCE_K3 = render(r"""
#define XL(v) (((v) < 32768) ? tab[(v)] : ({ACC})(v) * log(({ACC})(v)))
#define ENT2(m, c) (((c) > 0.f && (c) < (m)) ? -((c) * __logf((c) / (m)) + ((m) - (c)) * __logf(((m) - (c)) / (m))) : 0.f)
// Block (x, b): the label bitset of permutation b in shared memory, thread j = blockIdx.x * blockDim.x + threadIdx.x scans column j. Same two passes and tie rule as root_split_stats_k2.
extern "C" __global__ void root_split_stats_k3(const int* __restrict__ order, const unsigned char* __restrict__ tie, const unsigned int* __restrict__ yb,
                                               const {ACC}* __restrict__ tab, const long long* __restrict__ ctot, const int n, const int B,
                                               const int Kc, const int min_split, const {ACC} h_full, const int words, {ACC}* __restrict__ out_gain,
                                               int* __restrict__ out_ncand, int* __restrict__ out_idx) {
    extern __shared__ unsigned int sy[];
    const int b = blockIdx.y;
    for (int w = threadIdx.x; w < words; w += blockDim.x) sy[w] = yb[(long long)w * B + b];
    __syncthreads();
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= Kc) return;
    const long long idx = (long long)j * B + b;
    const int* ord = order + (long long)j * n;
    const unsigned char* tj = tie + (long long)j * n;
    const {ACC} n_f = ({ACC})n;
    const int TILE = 8;
    int ybuf[TILE + 1];
    {REAL} w_min = 3.0e38f;
    int ncand = 0;
    for (int pass = 0; pass < 2; ++pass) {
        long long l0 = 0, l1 = 0, r0 = ctot[0], r1 = ctot[1];
        {ACC} best_gain = -1e300;
        int best_idx = -1;
        for (int i0 = 0; i0 < n - 1; i0 += TILE) {
            const int cnt = min(TILE + 1, n - i0);
#pragma unroll
            for (int t = 0; t < TILE + 1; ++t) {
                if (t < cnt) { const int pos = ord[i0 + t]; ybuf[t] = (sy[pos >> 5] >> (pos & 31)) & 1; }
            }
            for (int t = 0; t < TILE && i0 + t < n - 1; ++t) {
                const int i = i0 + t;
                const int yi = ybuf[t];
                if (yi) { ++l1; --r1; } else { ++l0; --r0; }
                const int n_l = i + 1, n_r = n - n_l;
                if (n_l >= min_split && n_r >= min_split && yi != ybuf[t + 1] && !tj[i]) {
                    const {REAL} w_f = ENT2(({REAL})n_l, ({REAL})l1) + ENT2(({REAL})n_r, ({REAL})r1);
                    if (pass == 0) {
                        ++ncand;
                        if (w_f < w_min) w_min = w_f;
                    } else if (w_f <= w_min + 0.5f) {
                        const {ACC} gain = h_full - (XL(n_l) - (XL(l0) + XL(l1)) + XL(n_r) - (XL(r0) + XL(r1))) / n_f;
                        if (gain > best_gain) { best_gain = gain; best_idx = i; }
                    }
                }
            }
        }
        if (pass == 1) {
            out_gain[idx] = best_gain;
            out_ncand[idx] = ncand;
            out_idx[idx] = best_idx;
        }
    }
}
""", real=np.float32)

_kernel_k3 = None


def fits_shared(n: int) -> bool:
    """Whether one permutation's label bitset for ``n`` rows fits the shared-memory budget."""
    return 4 * ((int(n) + 31) // 32) <= MAX_SHARED_BYTES


def get_kernel_k3(cp: Any) -> Any:
    """Compile v3 once per process (no fused multiply-add: the gains must round as the host's)."""
    global _kernel_k3
    if _kernel_k3 is None:
        _kernel_k3 = cp.RawKernel(_SOURCE_K3, "root_split_stats_k3", options=("--fmad=false",))
    return _kernel_k3


def launch_k3(cp: Any, order: Any, tie: Any, ybits_wb: Any, tab: Any, counts: Any, n: int, n_perm: int, kc: int, min_split: int, h_full: float, words: int, d_gain: Any, d_ncand: Any, d_idx: Any) -> None:
    """Launch v3: ``ybits_wb`` is the ``(words, permutations)`` uint32 table (the layout v2 reads), the outputs are the ``(kc * permutations)`` vectors v2 writes."""
    kernel = get_kernel_k3(cp)
    threads = block_size_for(kernel, dyn_smem=4 * int(words), work_items=int(kc))
    blocks_x = (int(kc) + threads - 1) // threads
    kernel(
        (blocks_x, int(n_perm)),
        (threads,),
        (order, tie, ybits_wb, tab, counts, np.int32(n), np.int32(n_perm), np.int32(kc), np.int32(min_split), np.float64(h_full), np.int32(words), d_gain, d_ncand, d_idx),  # Intentional float64: the kernel signature declares double
        shared_mem=4 * int(words),
    )
