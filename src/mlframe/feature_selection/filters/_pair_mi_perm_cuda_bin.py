"""The pair maxT null floor against K permuted BINARY targets: counts the outcomes of several permutations with one shared atomic per row.

The package kernel (``_pair_mi_perm_cuda._SRC``) counts, for every row of a pair and every one of K permuted targets, one shared-memory atomic: 25 atomics and 25 target-byte unpackings per row for K = 25 (nvprof,
500000 rows x 300 pairs: 2.18e9 warp instructions, IPC 3.1, issue slots 77 percent, instruction-bound). For a binary target the class of a row under a permutation is one bit. A pass takes ``kb`` permutations
(at most 8): their ``kb`` bits are one byte of a plane of the packed targets, and ONE atomic per row counts the outcome of all of them together in ``hist[cell][outcome]``. The class-1 count of permutation q in a cell
is the sum of the outcomes with bit q set, the cell total the sum of all outcomes: both come out of one in-place superset-sum transform of the ``2 ** kb`` outcomes of a cell (``kb`` parallel steps), and the
cell totals no longer need a pass of their own. The MI of a (cell, class) is the same ``jf * log(jf / (px * py))`` of the same integer counts, summed by one warp per permutation (deterministic order).
nvprof, the same shape: 3.0e8 warp instructions (7 times fewer), 29 ms against 143 ms for 600 pairs (4.9 times), 94 GB/s of the 112 GB/s DRAM peak: what is left is the traffic of the two code columns and one
target byte per row and pass. The values differ from the package kernel by rounding only (about 5e-19).

The number of permutations per pass is limited by the shared memory: ``joint << kb`` counters of 4 bytes per pass. Pairs are therefore grouped by the largest ``kb`` their own joint table allows and each group is
launched with its own ``kb`` (a pair of 5 x 5 bins takes 8 permutations per pass, a pair of 30 x 30 bins 3), so one wide pair does not slow the others.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for

from ._column_gather import gather_columns_cm
from ._kernel_types import render

#: Dynamic shared memory the outcome table may use (the 48 KB every device grants, less the driver's reserve): the same budget as the package kernel.
SHARED_BUDGET_BYTES = 44 * 1024
#: Permutations counted by one atomic: the bits of one byte of a plane.
MAX_PERMS_PER_PASS = 8
#: Rows of a column are padded to this many bytes: the kernel reads 16 rows per load.
ROW_ALIGN = 16

_SRC = render(r"""
extern "C" __global__
void pair_mi_perm_bin(const unsigned char* __restrict__ cols, const long long* __restrict__ pair_a, const long long* __restrict__ pair_b,
                      const int* __restrict__ nbins, const unsigned char* __restrict__ yplanes, const {ACC}* __restrict__ freqs_y,
                      const long long n, const long long stride, const int n_pairs, const int K, const int kb_pass, const long long cap, const {ACC} inv_n,
                      {ACC}* __restrict__ mi_out) {
    extern __shared__ int hist[];
    const int p = blockIdx.x;
    const int tid = threadIdx.x, nt = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, n_warps = nt >> 5;
    const long long a = pair_a[p], b = pair_b[p];
    const long long nb_a = nbins[a], nb_b = nbins[b];
    const bool bad = (nb_a <= 0 || nb_b <= 0 || nb_b > cap / 2 || nb_a > cap / (nb_b * 2));
    if (bad) {
        for (int k = tid; k < K; k += nt) mi_out[(long long)k * n_pairs + p] = 0.0;
        return;
    }
    const int joint = (int)(nb_a * nb_b);
    const unsigned char* ca = cols + a * stride;
    const unsigned char* cb = cols + b * stride;
    const int nbb = (int)nb_b;
    const unsigned int* ca4 = (const unsigned int*)ca;
    const unsigned int* cb4 = (const unsigned int*)cb;
    const long long n4 = n >> 2;
    const {ACC} py0 = freqs_y[0], py1 = freqs_y[1];
    for (int k0 = 0, pass = 0; k0 < K; k0 += kb_pass, ++pass) {
        const int kb = (K - k0 < kb_pass) ? (K - k0) : kb_pass;
        const int outcomes = 1 << kb;
        const unsigned char* yp = yplanes + (long long)pass * stride;
        const unsigned int* yp4 = (const unsigned int*)yp;
        for (int i = tid; i < joint * outcomes; i += nt) hist[i] = 0;
        __syncthreads();
        const long long n16 = n >> 4;
        const uint4* ca16 = (const uint4*)ca;
        const uint4* cb16 = (const uint4*)cb;
        const uint4* yp16 = (const uint4*)yp;
        for (long long g = tid; g < n16; g += nt) {
            const uint4 va = ca16[g], vb = cb16[g], vy = yp16[g];
            const unsigned int wa[4] = {va.x, va.y, va.z, va.w}, wb[4] = {vb.x, vb.y, vb.z, vb.w}, wy[4] = {vy.x, vy.y, vy.z, vy.w};
#pragma unroll
            for (int w = 0; w < 4; ++w) {
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int jc = (int)((wa[w] >> (8 * u)) & 0xffu) * nbb + (int)((wb[w] >> (8 * u)) & 0xffu);
                    atomicAdd(&hist[(jc << kb) + (int)((wy[w] >> (8 * u)) & 0xffu)], 1);
                }
            }
        }
        for (long long g = (n16 << 2) + tid; g < n4; g += nt) {
            const unsigned int ra = ca4[g], rb = cb4[g], yw = yp4[g];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int jc = (int)((ra >> (8 * u)) & 0xffu) * nbb + (int)((rb >> (8 * u)) & 0xffu);
                atomicAdd(&hist[(jc << kb) + (int)((yw >> (8 * u)) & 0xffu)], 1);
            }
        }
        for (long long i = (n & ~3LL) + tid; i < n; i += nt) {
            const int jc = (int)ca[i] * nbb + (int)cb[i];
            atomicAdd(&hist[(jc << kb) + (int)yp[i]], 1);
        }
        __syncthreads();
        // superset sums: afterwards hist[cell][0] is the cell total and hist[cell][1 << q] the rows of the cell whose permuted class under q is 1
        for (int q = 0; q < kb; ++q) {
            const int half = outcomes >> 1;
            for (int idx = tid; idx < joint * half; idx += nt) {
                const int c = idx / half, r = idx - c * half;
                const int o = ((r >> q) << (q + 1)) | (r & ((1 << q) - 1));
                hist[(c << kb) + o] += hist[(c << kb) + (o | (1 << q))];
            }
            __syncthreads();
        }
        for (int q = warp; q < kb; q += n_warps) {
            {ACC} loc = 0.0;
            for (int ci = lane; ci < joint; ci += 32) {
                const int fxc = hist[ci << kb];
                if (fxc == 0) continue;
                const int h1 = hist[(ci << kb) + (1 << q)];
                const {ACC} px = ({ACC})fxc * inv_n;
                const int jc0 = fxc - h1;
                if (jc0 != 0 && py0 > 0.0) { const {ACC} jf = ({ACC})jc0 * inv_n; loc += jf * log(jf / (px * py0)); }
                if (h1 != 0 && py1 > 0.0) { const {ACC} jf = ({ACC})h1 * inv_n; loc += jf * log(jf / (px * py1)); }
            }
            for (int off = 16; off > 0; off >>= 1) loc += __shfl_down_sync(0xffffffffu, loc, off);
            if (lane == 0) mi_out[(long long)(k0 + q) * n_pairs + p] = loc;
        }
        __syncthreads();
    }
}
""")

_KERNEL: Any = None
_KERNEL_LOCK = threading.Lock()


def _kernel(cp: Any) -> Any:
    """The compiled kernel, built once per process."""
    global _KERNEL
    with _KERNEL_LOCK:
        if _KERNEL is None:
            _KERNEL = cp.RawModule(code=_SRC).get_function("pair_mi_perm_bin")
        return _KERNEL


def perms_per_pass(joint: "np.ndarray | int", k: int) -> "np.ndarray | int":
    """The largest ``kb`` in ``1 .. min(8, k)`` whose outcome table (``joint << kb`` counters) fits the shared budget, for one joint cardinality or an array of them (0 where not even ``kb = 1`` fits)."""
    j = np.asarray(joint, dtype=np.int64)
    best = np.zeros(j.shape, dtype=np.int64)
    for kb in range(1, min(MAX_PERMS_PER_PASS, max(1, int(k))) + 1):
        best = np.where((j << kb) * 4 <= SHARED_BUDGET_BYTES, kb, best)
    return int(best) if np.ndim(joint) == 0 else best


def pack_planes(y_perms: np.ndarray, kb: int, stride: int) -> np.ndarray:
    """``(passes, stride)`` uint8: bit q of row i of plane p is the class (0 or 1) of row i under permutation ``p * kb + q``; the rows past the data are zero."""
    k_total, n = y_perms.shape
    out = np.zeros((-(-k_total // kb), stride), dtype=np.uint8)
    for k in range(k_total):
        out[k // kb, :n] |= np.asarray(y_perms[k], dtype=np.uint8) << np.uint8(k % kb)
    return out


def launch_pairs_binary(
    cp: Any,
    factors_data: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    nb: np.ndarray,
    y_perms: np.ndarray,
    freqs_y: np.ndarray,
    cap: int,
    columns: "np.ndarray | None" = None,
) -> np.ndarray:
    """The ``(K, n_pairs)`` plug-in joint MI of every pair against each of the K permuted binary targets (zeros for a pair whose joint cardinality exceeds ``cap``).

    ``a`` and ``b`` index into ``columns`` (the columns of ``factors_data`` that go up; ``None`` is every column), ``nb`` is the bin count of each of them. The pairs are grouped by their ``kb`` and launched one group at a time."""
    n = int(factors_data.shape[0])
    k_total, n_pairs = int(y_perms.shape[0]), int(a.shape[0])
    stride = -(-n // ROW_ALIGN) * ROW_ALIGN
    nb64 = np.asarray(nb, dtype=np.int64)
    joint = nb64[a] * nb64[b]
    valid = (nb64[a] > 0) & (nb64[b] > 0) & (nb64[b] <= cap // 2) & (nb64[a] <= cap // np.maximum(nb64[b] * 2, 1))
    kb_of = np.where(valid, perms_per_pass(np.where(valid, joint, 1), k_total), 1)
    if int(kb_of.min(initial=1)) < 1:
        raise ValueError("a pair joint table does not fit the shared memory of the outcome-code kernel")
    d_cols = cp.asarray(gather_columns_cm(factors_data, np.arange(factors_data.shape[1]) if columns is None else columns, pad_to=ROW_ALIGN))
    d_nb = cp.asarray(nb64.astype(np.int32))
    d_fy = cp.asarray(np.ascontiguousarray(freqs_y, dtype=np.float64))  # Intentional float64: frequency vector of the MI formula (kernel contract double*)
    kernel = _kernel(cp)
    out = np.zeros((k_total, n_pairs), dtype=np.float64)  # Intentional float64: MI matrix returned to the host (kernel contract double*)
    for kb in np.unique(kb_of).tolist():
        pick = np.flatnonzero(kb_of == kb)
        shared = (int(joint[pick][valid[pick]].max(initial=1)) << int(kb)) * 4
        d_planes = cp.asarray(pack_planes(y_perms, int(kb), stride))
        d_a, d_b = cp.asarray(np.ascontiguousarray(a[pick], dtype=np.int64)), cp.asarray(np.ascontiguousarray(b[pick], dtype=np.int64))
        d_out = cp.empty((k_total, pick.shape[0]), dtype=cp.float64)  # Intentional float64: MI output buffer (kernel contract double*)
        threads = block_size_for(kernel, dyn_smem=shared, work_items=max(1, n // ROW_ALIGN), max_width=1024)
        kernel(
            (int(pick.shape[0]),),
            (threads,),
            (d_cols, d_a, d_b, d_nb, d_planes, d_fy, np.int64(n), np.int64(stride), np.int32(pick.shape[0]), np.int32(k_total), np.int32(kb), np.int64(cap), np.float64(1.0 / n), d_out),  # Intentional float64: scalar of a double-typed kernel parameter
            shared_mem=shared,
        )
        out[:, pick] = cp.asnumpy(d_out)
    return out
