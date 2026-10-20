"""Device twin of the bit-set family null (``_family_maxt_null_v2_njit.null_max_bitsets``).

The joint count of bin ``a`` and class ``c`` is ``popcount(bin_a & class_c)`` over 64-row words; the last class is the bin total minus the others. One thread handles one (permutation, column):
it walks the words of the column's bin bit sets (stored word-major, so the threads of a warp read consecutive columns: coalesced) against the permutation's class bit sets (the same words for every thread of the block, a broadcast read), so the
histogram's scattered increments become independent AND / population-count / add operations. The column bit sets (``K x bins x words`` uint64) are uploaded once per family and reused by every
block of permutations; only the ``B x classes x words`` class bit sets of a block cross the bus.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

_SOURCE = r"""
extern "C" __global__ void family_null_bits(const unsigned long long* __restrict__ bits, const long long* __restrict__ n_x, const unsigned long long* __restrict__ cls,
                                            const {REAL}* __restrict__ table, const {REAL}* __restrict__ c_a, const long long* __restrict__ px,
                                            const {REAL}* __restrict__ const_b, const long long n, const int cap, const int words, const int n_y, const int K, const int B,
                                            {REAL}* __restrict__ out) {
    const long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= (long long)B * K) return;
    const int b = (int)(idx / K);
    const int j = (int)(idx - (long long)b * K);
    double s = 0.0;
    const int nb = (int)n_x[j];
    for (int a = 0; a < nb; ++a) {
        const unsigned long long* row = bits + (long long)a * words * K + j;
        long long rest = px[(long long)j * cap + a];
        for (int c = 0; c < n_y - 1; ++c) {
            const unsigned long long* cl = cls + ((long long)b * n_y + c) * words;
            int cnt = 0;
            for (int w = 0; w < words; ++w) cnt += __popcll(row[(long long)w * K] & cl[w]);
            s += (double)table[cnt];
            rest -= cnt;
        }
        s += (double)table[rest];
    }
    out[idx] = ({REAL})((s + (double)const_b[b] - (double)c_a[j]) / (double)n);
}
"""

_kernels: dict = {}
_KERNEL_CACHE_LOCK = threading.RLock()
_THREADS = 128

# One block per (permutation, tile of columns). The permutation's class bit sets sit in shared memory (every thread of the block reads the same word: a broadcast), and each thread reads a word of its
# column's bin bit set ONCE and ANDs it against every class, accumulating the joint counts in registers. The first kernel re-read the bin words once per class and the class words from global memory;
# this one reads the bin words ``n_y - 1`` times less. The sum is accumulated in the same order as the first kernel (per bin: classes in order, then the remainder), so the result is bit-identical.
_SOURCE_V2 = r"""
extern "C" __global__ void family_null_bits_v2(const unsigned long long* __restrict__ bits, const long long* __restrict__ n_x, const unsigned long long* __restrict__ cls,
                                               const {REAL}* __restrict__ table, const {REAL}* __restrict__ c_a, const long long* __restrict__ px,
                                               const {REAL}* __restrict__ const_b, const long long n, const int cap, const int words, const int n_y, const int K, const int B,
                                               {REAL}* __restrict__ out) {
    extern __shared__ unsigned long long sh_cls[];
    const int b = blockIdx.x;
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    for (int i = threadIdx.x; i < NY1 * words; i += blockDim.x) sh_cls[i] = cls[((long long)b * n_y) * words + i];
    __syncthreads();
    if (j >= K) return;
    double s = 0.0;
    const int nb = (int)n_x[j];
    for (int a = 0; a < nb; ++a) {
        const unsigned long long* row = bits + (long long)a * words * K + j;
        int cnt[NY1];
        #pragma unroll
        for (int c = 0; c < NY1; ++c) cnt[c] = 0;
        for (int w = 0; w < words; ++w) {
            const unsigned long long r = row[(long long)w * K];
            #pragma unroll
            for (int c = 0; c < NY1; ++c) cnt[c] += __popcll(r & sh_cls[c * words + w]);
        }
        long long rest = px[(long long)j * cap + a];
        #pragma unroll
        for (int c = 0; c < NY1; ++c) { s += (double)table[cnt[c]]; rest -= cnt[c]; }
        s += (double)table[rest];
    }
    out[(long long)b * K + j] = ({REAL})((s + (double)const_b[b] - (double)c_a[j]) / (double)n);
}
"""

_kernels_v2: dict = {}
_THREADS_V2 = 128
_MAX_CLASSES_V2 = 24  # registers hold ``n_y - 1`` counters per thread
_SHARED_MARGIN = 1024


def _get_kernel_v2(cp, ny1: int, real: str):
    """Compile the shared-memory kernel for ``ny1 = n_y - 1`` stored classes and one element type (one binary per pair)."""
    with _KERNEL_CACHE_LOCK:
        kern = _kernels_v2.get((ny1, real))
        if kern is None:
            kern = cp.RawModule(code=_SOURCE_V2.replace("{REAL}", real), options=(f"-DNY1={int(ny1)}",)).get_function("family_null_bits_v2")
            _kernels_v2[(ny1, real)] = kern
    return kern


def _v2_applies(cp, n_y: int, words: int) -> bool:
    """Whether the shared-memory kernel can hold the class bit sets of one permutation: at least two classes, a bounded count, and a block's worth of shared memory."""
    if n_y < 2 or n_y - 1 > _MAX_CLASSES_V2:
        return False
    try:
        limit = int(cp.cuda.Device().attributes.get("MaxSharedMemoryPerBlock", 48 * 1024))
    except Exception as e:
        logger.warning("Could not read MaxSharedMemoryPerBlock (%s: %s); assuming 48 KiB", type(e).__name__, e)
        limit = 48 * 1024
    return (n_y - 1) * words * 8 <= limit - _SHARED_MARGIN


def _get_kernel(cp, real: str):
    """Compile the kernel once per element type (``"float"`` or ``"double"``)."""
    with _KERNEL_CACHE_LOCK:
        kern = _kernels.get(real)
        if kern is None:
            kern = cp.RawKernel(_SOURCE.replace("{REAL}", real), "family_null_bits")
            _kernels[real] = kern
    return kern


class DeviceBitsets:
    """A family's column bit sets (and the tables the MI needs) resident on the device, grouped within the memory budget."""

    def __init__(self, cp, prep, n_x: np.ndarray, n: int):
        table, c_a, px, groups = prep
        self.cp = cp
        self.n = int(n)
        self.table = cp.asarray(table)
        self.c_a = cp.asarray(c_a, dtype=self.table.dtype)
        self.px = cp.asarray(px)
        self.n_x = cp.asarray(np.ascontiguousarray(n_x, dtype=np.int64))
        self.cap = int(groups[0][2].shape[1])
        self.words = int(groups[0][2].shape[2])
        # word-major per group: (bins, words, columns), so the threads of a warp (consecutive columns) read consecutive addresses
        self.groups = [(lo, hi, cp.ascontiguousarray(cp.asarray(bits).transpose(1, 2, 0))) for lo, hi, bits in groups]


def null_max_bits_device(dev: DeviceBitsets, cls_host: np.ndarray, py_const: float) -> np.ndarray:
    """Per-permutation maximum MI over the family. ``cls_host`` is ``(B, n_y, words)`` uint64; ``py_const = n log n - sum_b xlogx(py_b)``."""
    # Intentional float64: label-entropy constant of the MI formula (exact permutation ties)
    const_b = np.full(cls_host.shape[0], float(py_const), dtype=np.float64)
    return _run(dev, cls_host, const_b, reduce_max=True)


def mi_matrix_bits_device(dev: DeviceBitsets, cls_host: np.ndarray, const_b: np.ndarray) -> np.ndarray:
    """``(B, K)`` plug-in MI of every column against every label vector; ``const_b[b] = n log n - sum_c xlogx(count_c)`` of label vector ``b`` (they differ when each vector has its own class counts)."""
    return _run(dev, cls_host, const_b, reduce_max=False)


def _run(dev: DeviceBitsets, cls_host: np.ndarray, const_b: np.ndarray, reduce_max: bool) -> np.ndarray:
    """Launch the bit-set kernel over every column group: the per-permutation maximum, or the full ``(B, K)`` matrix.

    The kernel's element type follows the resident MI table (``dev.table``); the constants and the output are built in that type, the sums are accumulated in double.
    """
    cp = dev.cp
    real = "float" if dev.table.dtype == cp.float32 else "double"
    B, n_y, _ = cls_host.shape
    cls_d = cp.asarray(cls_host)
    # Intentional float64: label-entropy constants of the MI formula, converted to the table's element type below
    const_d = cp.asarray(np.ascontiguousarray(const_b, dtype=np.float64)).astype(dev.table.dtype, copy=False)
    best = cp.zeros(B, dtype=cp.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    parts = []
    use_v2 = _v2_applies(cp, n_y, dev.words)
    kern = _get_kernel_v2(cp, n_y - 1, real) if use_v2 else _get_kernel(cp, real)
    for lo, hi, bits in dev.groups:
        k = hi - lo
        out = cp.empty((B, k), dtype=dev.table.dtype)
        if use_v2:
            kern(
                (B, (k + _THREADS_V2 - 1) // _THREADS_V2),
                (_THREADS_V2,),
                (
                    bits,
                    dev.n_x[lo:hi],
                    cls_d,
                    dev.table,
                    dev.c_a[lo:hi],
                    dev.px[lo:hi],
                    const_d,
                    np.int64(dev.n),
                    np.int32(dev.cap),
                    np.int32(dev.words),
                    np.int32(n_y),
                    np.int32(k),
                    np.int32(B),
                    out,
                ),
                shared_mem=(n_y - 1) * dev.words * 8,
            )
        else:
            kern(
                ((B * k + _THREADS - 1) // _THREADS,),
                (_THREADS,),
                (
                    bits,
                    dev.n_x[lo:hi],
                    cls_d,
                    dev.table,
                    dev.c_a[lo:hi],
                    dev.px[lo:hi],
                    const_d,
                    np.int64(dev.n),
                    np.int32(dev.cap),
                    np.int32(dev.words),
                    np.int32(n_y),
                    np.int32(k),
                    np.int32(B),
                    out,
                ),
            )
        if reduce_max:
            best = cp.maximum(best, out.max(axis=1))
        else:
            parts.append(out)
    if reduce_max:
        host: np.ndarray = cp.asnumpy(best)
        return host
    return np.asarray(cp.asnumpy(parts[0] if len(parts) == 1 else cp.concatenate(parts, axis=1)))
