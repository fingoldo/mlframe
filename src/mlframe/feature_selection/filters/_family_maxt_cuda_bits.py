"""Device twin of the bit-set family null (``_family_maxt_null_v2_njit.null_max_bitsets``).

The joint count of bin ``a`` and class ``c`` is ``popcount(bin_a & class_c)`` over 64-row words; the last class is the bin total minus the others. One thread handles one (permutation, column):
it walks the words of the column's bin bit sets (stored word-major, so the threads of a warp read consecutive columns: coalesced) against the permutation's class bit sets (the same words for every thread of the block, a broadcast read), so the
histogram's scattered increments become independent AND / population-count / add operations. The column bit sets (``K x bins x words`` uint64) are uploaded once per family and reused by every
block of permutations; only the ``B x classes x words`` class bit sets of a block cross the bus.
"""

from __future__ import annotations

import numpy as np

_SOURCE = r"""
extern "C" __global__ void family_null_bits(const unsigned long long* __restrict__ bits, const long long* __restrict__ n_x, const unsigned long long* __restrict__ cls,
                                            const double* __restrict__ table, const double* __restrict__ c_a, const long long* __restrict__ px,
                                            const double* __restrict__ const_b, const long long n, const int cap, const int words, const int n_y, const int K, const int B,
                                            double* __restrict__ out) {
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
            s += table[cnt];
            rest -= cnt;
        }
        s += table[rest];
    }
    out[idx] = (s + const_b[b] - c_a[j]) / (double)n;
}
"""

_kernel = None
_THREADS = 128


def _get_kernel(cp):
    """Compile the kernel once per process."""
    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(_SOURCE, "family_null_bits")
    return _kernel


class DeviceBitsets:
    """A family's column bit sets (and the tables the MI needs) resident on the device, grouped within the memory budget."""

    def __init__(self, cp, prep, n_x: np.ndarray, n: int):
        table, c_a, px, groups = prep
        self.cp = cp
        self.n = int(n)
        self.table = cp.asarray(table)
        self.c_a = cp.asarray(c_a)
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
    """Launch the bit-set kernel over every column group: the per-permutation maximum, or the full ``(B, K)`` matrix."""
    cp = dev.cp
    kern = _get_kernel(cp)
    B, n_y, _ = cls_host.shape
    cls_d = cp.asarray(cls_host)
    const_d = cp.asarray(np.ascontiguousarray(const_b, dtype=np.float64))  # Intentional float64: label-entropy constants of the MI formula
    best = cp.zeros(B, dtype=cp.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    parts = []
    for lo, hi, bits in dev.groups:
        k = hi - lo
        out = cp.empty((B, k), dtype=cp.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
        total = B * k
        kern(
            ((total + _THREADS - 1) // _THREADS,),
            (_THREADS,),
            (bits, dev.n_x[lo:hi], cls_d, dev.table, dev.c_a[lo:hi], dev.px[lo:hi], const_d, np.int64(dev.n), np.int32(dev.cap), np.int32(dev.words), np.int32(n_y), np.int32(k), np.int32(B), out),
        )
        if reduce_max:
            best = cp.maximum(best, out.max(axis=1))
        else:
            parts.append(out)
    if reduce_max:
        host: np.ndarray = cp.asnumpy(best)
        return host
    return np.asarray(cp.asnumpy(parts[0] if len(parts) == 1 else cp.concatenate(parts, axis=1)))
