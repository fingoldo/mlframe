"""Per-permutation maximum candidate MI of a family-wise null, every permutation of a block in one launch.

``_binned_numeric_agg_resident._binagg_null_calibrated_floor`` permutes the target 39 times and, per permutation, runs the fused bin + histogram + MI kernel of
``_fe_batched_mi.binned_mi_from_values_gpu`` over every candidate (1016 candidates x 100000 rows in a 100000 x 100 fit: 1.04 s). Each launch re-reads and re-bins every
value although the candidate bins do not depend on the target. Here a block of permutations is scored by one launch: one block per candidate bins each value once (the
same f64 interior edges, the same upper-bound search) and adds it to the shared ``(bin, class)`` table of every permutation, with the permuted labels gathered once
into a ``(P, n)`` matrix. Each permutation's MI is then summed by its own thread in the order ``mi_from_values`` sums it, so the null maxima equal the per-permutation
launches'.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

import numpy as np

from ._kernel_types import render
from ._raw_module_cache import RawModuleCache
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

_SHARED_BUDGET_BYTES = 32 * 1024  # the permutation tables of one launch

_SRC = r"""
extern "C" __global__ void null_mi_perms(const {XT}* __restrict__ X, const {XT}* __restrict__ edges, const unsigned char* __restrict__ yp, const long long n,
                                         const int K, const int nbins, const int Ky, const int P, const {ACC} inv_n, {ACC}* __restrict__ out) {
    extern __shared__ int sh[];                       // P tables of (nbins * Ky) counts
    const int c = blockIdx.x;
    const int ne = nbins - 1, M = nbins * Ky;
    const int tid = threadIdx.x, nt = blockDim.x;
    for (int s = tid; s < P * M; s += nt) sh[s] = 0;
    __shared__ {XT} se[256];   // this column's interior edges, read per value by the search
    const bool cached = ne <= 256;
    if (cached) for (int s = threadIdx.x; s < ne; s += blockDim.x) se[s] = edges[(long long)s * K + c];
    __syncthreads();
    const {XT}* Xc = X + (long long)c * n;
    for (long long i = tid; i < n; i += nt) {
        const {XT} v = Xc[i];
        int lo = 0, hi = ne;                          // upper_bound: count of interior edges <= v
        if (v != v) lo = hi;                        // NaN sorts last: the last bin, as cp.searchsorted does
        while (lo < hi) { int mid = (lo + hi) >> 1; if ((cached ? se[mid] : edges[(long long)mid * K + c]) <= v) lo = mid + 1; else hi = mid; }
        const long long base = (long long)lo * Ky;
        for (int p = 0; p < P; ++p) atomicAdd(&sh[p * M + base + yp[(long long)p * n + i]], 1);
    }
    __syncthreads();
    for (int p = tid; p < P; p += nt) {
        const int* t = sh + p * M;
        {ACC} mi = 0.0;
        for (int xx = 0; xx < nbins; ++xx) {
            long long rx = 0;
            for (int yy = 0; yy < Ky; ++yy) rx += t[xx * Ky + yy];
            if (rx == 0) continue;
            {ACC} px = ({ACC})rx * inv_n;
            for (int yy = 0; yy < Ky; ++yy) {
                long long nxy = t[xx * Ky + yy];
                if (nxy == 0) continue;
                long long ry = 0;
                for (int xx2 = 0; xx2 < nbins; ++xx2) ry += t[xx2 * Ky + yy];
                mi += ({ACC})nxy * inv_n * log((({ACC})nxy * inv_n) / (px * (({ACC})ry * inv_n)));
            }
        }
        out[(long long)p * K + c] = mi > 0.0 ? mi : 0.0;
    }
}
#define PMAX_LANE 2
#define NBMAX 32
extern "C" __global__ void null_mi_perms_bits(const {XT}* __restrict__ X, const {XT}* __restrict__ edges_t, const unsigned int* __restrict__ ybits, const long long n,
                                              const int W, const int K, const int nbins, const int P, const {ACC} inv_n, {ACC}* __restrict__ out) {
    // Binary target: a warp bins 32 consecutive rows and keeps one ballot mask per bin; lane l counts the label-1 rows of permutation p = l, l + 32, ... in each bin
    // as popc(mask & label word). The bin totals do not depend on the permutation. Counts stay in registers and reach shared memory once per lane.
    extern __shared__ int sh[];                       // ones[P * nbins], then tot[nbins]
    int* ones = sh;
    int* tot = sh + P * nbins;
    const int c = blockIdx.x;
    const int ne = nbins - 1;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    for (int s = tid; s < P * nbins + nbins; s += blockDim.x) sh[s] = 0;
    __shared__ {XT} se[256];   // this column's interior edges, read per value by the search
    const bool cached = ne <= 256;
    if (cached) for (int s = threadIdx.x; s < ne; s += blockDim.x) se[s] = edges_t[(long long)c * ne + s];
    __syncthreads();
    int cnt[PMAX_LANE][NBMAX];
    int tl[NBMAX];
    for (int q = 0; q < PMAX_LANE; ++q) for (int b = 0; b < nbins; ++b) cnt[q][b] = 0;
    for (int b = 0; b < nbins; ++b) tl[b] = 0;
    const {XT}* Xc = X + (long long)c * n;
    for (int w = warp; w < W; w += nw) {
        const long long i = (long long)w * 32 + lane;
        int code = -1;
        if (i < n) {
            const {XT} v = Xc[i];
            int lo = 0, hi = ne;                      // upper_bound: count of interior edges <= v
            if (v != v) lo = hi;                        // NaN sorts last: the last bin, as cp.searchsorted does
            while (lo < hi) { int mid = (lo + hi) >> 1; if ((cached ? se[mid] : edges_t[(long long)c * ne + mid]) <= v) lo = mid + 1; else hi = mid; }
            code = lo;
        }
        unsigned int mb[NBMAX];
        for (int b = 0; b < nbins; ++b) mb[b] = __ballot_sync(0xffffffffu, code == b);
        if (lane == 0) for (int b = 0; b < nbins; ++b) tl[b] += __popc(mb[b]);
        for (int q = 0; q < PMAX_LANE; ++q) {
            const int p = lane + 32 * q;
            if (p >= P) break;
            const unsigned int yw = ybits[(long long)w * P + p];  // (W, P): the lanes are permutations, so one word of all of them is contiguous
            for (int b = 0; b < nbins; ++b) cnt[q][b] += __popc(mb[b] & yw);
        }
    }
    for (int q = 0; q < PMAX_LANE; ++q) {
        const int p = lane + 32 * q;
        if (p >= P) break;
        for (int b = 0; b < nbins; ++b) if (cnt[q][b]) atomicAdd(&ones[p * nbins + b], cnt[q][b]);
    }
    if (lane == 0) for (int b = 0; b < nbins; ++b) if (tl[b]) atomicAdd(&tot[b], tl[b]);
    __syncthreads();
    for (int p = tid; p < P; p += blockDim.x) {
        // the (bin, class) table of permutation p is (tot - ones, ones); the MI is summed as mi_from_values sums it (bins outer, classes 0 then 1)
        {ACC} mi = 0.0;
        long long ry[2] = {0, 0};
        for (int xx = 0; xx < nbins; ++xx) { const long long o = ones[p * nbins + xx]; ry[1] += o; ry[0] += tot[xx] - o; }
        for (int xx = 0; xx < nbins; ++xx) {
            const long long o = ones[p * nbins + xx];
            const long long cell[2] = {tot[xx] - o, o};
            const long long rx = cell[0] + cell[1];
            if (rx == 0) continue;
            {ACC} px = ({ACC})rx * inv_n;
            for (int yy = 0; yy < 2; ++yy) {
                const long long nxy = cell[yy];
                if (nxy == 0) continue;
                mi += ({ACC})nxy * inv_n * log((({ACC})nxy * inv_n) / (px * (({ACC})ry[yy] * inv_n)));
            }
        }
        out[(long long)c * P + p] = mi > 0.0 ? mi : 0.0;  // (K, P): consecutive permutations of a thread group write contiguous words
    }
}
"""

# ``{XT}`` is the storage type of the candidate values, one compiled module per type (CUDA.md, placeholder substitution): float32 for the binned numeric-aggregate
# null, double where the caller uploads its matrix in float64 (the STRICT null of the local MI gate). The edges take the value type: for float32 values they are
# the double edges rounded up to float32 (``e <= x`` iff ``up(e) <= x``, so the bins are the double comparison's); the MI terms and sum stay {ACC} (a statistic
# thresholded against the observed maximum).
_MODULES_BY_X = {xt: RawModuleCache(render(_SRC.replace("{XT}", xt))) for xt in ("float", "double")}


def _bits_width(kernel: Any, shared_bytes: int, n: int) -> int:
    """Threads per block of the bitset kernel from the device limits (``fixed_shared_block_size``: the count tables are fixed per block), 256 without device information."""
    from pyutilz.performance.gpu_hw_launch import fixed_shared_block_size

    try:
        width = fixed_shared_block_size(shared_bytes, regs_per_thread=int(kernel.num_regs), static_smem=int(kernel.shared_size_bytes), min_width=64, max_width=1024, work_items=(n + 31) // 32 * 32)
    except Exception as exc:  # best-effort: only the launch width changes, never a result
        log_throttle(logger, "_null_max_perm_gpu.block_width", logging.INFO, "null max bitset kernel: device limits unavailable (%s)", exc)
        width = 0
    return width if width > 0 else 256


class PermutationBlocks:
    """The ``count`` row permutations of a null, handed out as consecutive ``(rows, n)`` int32 blocks instead of one ``(count, n)`` int64 stack (200 permutations of 10 million
    rows were 16 GB of host memory). ``draw(i)`` gives permutation ``i``; the blocks must be asked for in order, so a generator-backed ``draw`` is consumed in the same order as the
    eager stack consumed it. ``drain()`` draws the ones not asked for, leaving a generator where the eager stack left it."""

    def __init__(self, draw: Any, count: int) -> None:
        self._draw = draw
        self._count = int(count)
        self._next = 0

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, sl: slice) -> np.ndarray:
        start, stop = int(sl.start or 0), min(int(sl.stop if sl.stop is not None else self._count), self._count)
        if start != self._next:
            raise IndexError(f"permutation blocks are read in order: asked for {start}, next is {self._next}")
        block = np.stack([np.asarray(self._draw(i), dtype=np.int32) for i in range(start, stop)]) if stop > start else np.empty((0, 0), dtype=np.int32)
        self._next = stop
        return block

    def one(self, i: int) -> Any:
        """Permutation ``i`` as ``draw`` gives it (a device array stays on the device); the permutations are asked for in order."""
        if int(i) != self._next:
            raise IndexError(f"permutation blocks are read in order: asked for {i}, next is {self._next}")
        self._next = int(i) + 1
        return self._draw(int(i))

    def drawn(self) -> bool:
        """Whether any permutation has been asked for."""
        return self._next > 0

    def drain(self) -> None:
        """Draw the permutations not asked for yet (the generator ends where the eager stack left it)."""
        for i in range(self._next, self._count):
            self._draw(i)
        self._next = self._count


#: Packed permuted labels of the last ``labels_key`` (one key at a time, about 49 MB for 39 permutations of 10 million rows): a caller that scores many candidate blocks
#: against the same permutations of the same labels (the binned-agg null redraws the same seeded permutations for every block) packs them once.
_LABEL_BITS_CACHE: dict = {}
_LABEL_BITS_LOCK = threading.Lock()


def _cached_label_bits(key: Any, p0: int) -> Any:
    """The packed block starting at permutation ``p0`` stored for ``key``, or ``None``."""
    if key is None:
        return None
    with _LABEL_BITS_LOCK:
        entry = _LABEL_BITS_CACHE.get("latest")
        if entry is None or entry[0] != key:
            return None
        return entry[1].get(int(p0))


def _store_label_bits(key: Any, p0: int, bits: Any) -> None:
    """Keep the packed block of ``key`` starting at ``p0``; a new key replaces the previous one."""
    if key is None:
        return
    with _LABEL_BITS_LOCK:
        entry = _LABEL_BITS_CACHE.get("latest")
        if entry is None or entry[0] != key:
            _LABEL_BITS_CACHE.clear()  # evict-ok: only the latest labels are kept; a miss repacks them
            blocks: "dict[int, Any]" = {}
            entry = (key, blocks)
            _LABEL_BITS_CACHE["latest"] = entry
        entry[1][int(p0)] = bits


def _permuted_labels(cp: Any, y8: Any, perms: Any, p0: int, p1: int, width: "int | None" = None) -> Any:
    """``(p1 - p0, width)`` uint8 device matrix of the labels ``y8`` under permutations ``p0 .. p1 - 1`` (columns past ``n`` zero; ``width`` defaults to ``n``), filled one
    permutation at a time: the host holds one row of row ids (40 MB at 10 million rows), not the whole block (64 x 10 million ids were 2.5 GB at int32)."""
    n = int(y8.shape[0])
    w = n if width is None else int(width)
    out = cp.zeros((p1 - p0, w), dtype=cp.uint8) if w > n else cp.empty((p1 - p0, n), dtype=cp.uint8)
    for r, p in enumerate(range(p0, p1)):
        perm = perms.one(p) if isinstance(perms, PermutationBlocks) else perms[p]
        out[r, :n] = y8[perm if isinstance(perm, cp.ndarray) else cp.asarray(np.ascontiguousarray(perm))]
    return out


def null_max_mi_perms_gpu(cp: Any, x_cm: Any, interior: Any, y_shifted: Any, perms: "np.ndarray | PermutationBlocks", nbins: int, n_classes: int, labels_key: Any = None) -> Optional[np.ndarray]:
    """``(P,)`` maximum over the candidates of the plug-in MI against ``y_shifted[perms[p]]``, for the float32 or double ``(K, n)`` C-order candidate block ``x_cm`` binned by
    ``interior`` (``(nbins - 1, K)``, the edges ``binned_mi_from_values_gpu`` uses); ``None`` when the path does not apply (the caller keeps one launch per permutation)."""
    if x_cm.dtype not in (cp.float32, cp.float64) or not x_cm.flags.c_contiguous or int(n_classes) > 255:  # Intentional float64: the kernel reads float32 or float64 candidates
        return None
    module = _MODULES_BY_X["float" if x_cm.dtype == cp.float32 else "double"]
    k, n = int(x_cm.shape[0]), int(x_cm.shape[1])
    m = int(nbins) * int(n_classes)
    per_launch = max(1, _SHARED_BUDGET_BYTES // (4 * m))
    kernel = module.get(cp).get_function("null_mi_perms")
    from ._fe_batched_mi import _edges_rounded_up_f32

    edges64 = cp.ascontiguousarray(interior.astype(cp.float64, copy=False))  # Intentional float64: the double edges the float32 ones are rounded from
    edges = _edges_rounded_up_f32(cp, edges64) if x_cm.dtype == cp.float32 else edges64
    y8 = y_shifted.astype(cp.uint8)
    out = np.empty(len(perms), dtype=np.float64)  # Intentional float64: null maxima (statistic vector)
    if int(n_classes) == 2 and int(nbins) <= 32:
        bits_kernel = module.get(cp).get_function("null_mi_perms_bits")
        words = (n + 31) // 32
        edges_t = cp.ascontiguousarray(edges.T)  # (K, nbins - 1): a block's interior edges are one contiguous run, read coalesced by the bits kernel
        for p0 in range(0, len(perms), 64):  # PMAX_LANE x 32 permutations per launch
            block_len = min(64, len(perms) - p0)
            ybits = _cached_label_bits(labels_key, p0)
            if ybits is None:
                padded = _permuted_labels(cp, y8, perms, p0, p0 + block_len, width=words * 32)  # (P, 32 words) permuted 0/1 labels, zero padded, built on the device
                # (P, words): bit i of word w = label of row 32 w + i. A row is a whole number of bytes, so packing the flat matrix little-endian packs each row on its own;
                # the uint32 view of 4 packed bytes is that word (the former shift-and-sum widened the block to 4 bytes a label: 2.5 GB for 64 permutations of 10 million rows)
                ybits = cp.ascontiguousarray(cp.packbits(padded.ravel(), bitorder="little").view(cp.uint32).reshape(block_len, words))
                del padded
                _store_label_bits(labels_key, p0, ybits)
            res = cp.empty((k, block_len), dtype=cp.float64)  # Intentional float64: MI accumulator, as the per-permutation kernel's; (K, P) from the bits kernel
            ybits_t = cp.ascontiguousarray(ybits.T)  # (words, block_len): the bits kernel's lanes read one word of every permutation at once
            shared = (block_len * int(nbins) + int(nbins)) * 4
            bits_kernel((k,), (_bits_width(bits_kernel, shared, n),), (x_cm, edges_t, ybits_t, np.int64(n), np.int32(words), np.int32(k), np.int32(nbins), np.int32(block_len), np.float64(1.0 / float(max(1, n))), res), shared_mem=shared)  # Intentional float64: the kernel reads float32 or float64 candidates
            out[p0 : p0 + block_len] = cp.asnumpy(cp.nanmax(res, axis=0))
            del ybits, res
        return out
    for p0 in range(0, len(perms), per_launch):
        block_len = min(per_launch, len(perms) - p0)
        yp = _permuted_labels(cp, y8, perms, p0, p0 + block_len)  # (P, n) permuted labels, built on the device one permutation at a time
        res = cp.empty((block_len, k), dtype=cp.float64)  # Intentional float64: MI accumulator, as the per-permutation kernel's
        kernel((k,), (_bits_width(kernel, block_len * m * 4, n),), (x_cm, edges, yp, np.int64(n), np.int32(k), np.int32(nbins), np.int32(n_classes), np.int32(block_len), np.float64(1.0 / float(max(1, n))), res), shared_mem=block_len * m * 4)
        out[p0 : p0 + block_len] = cp.asnumpy(cp.nanmax(res, axis=1))
        del yp, res
    return out


__all__ = ["null_max_mi_perms_gpu"]
