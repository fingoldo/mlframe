"""Device twin of the host permutation draw behind the family null's class bit sets.

The host draws ``m`` Fisher-Yates permutations of the row index (``_family_maxt_gate._fill_permutations``: permutation ``r`` is seeded ``base_seed + r`` with the splitmix64
finaliser and an xorshift64 stream) and packs the labels read through them into one bit per row and class (``label_bitsets_from_index``). At a million rows that is 20 ms of host
time per permutation with the device idle. Here one thread owns one permutation (the same integer stream, so the same permutation) and a second kernel packs the bits, so the
class bit sets are equal bit for bit and the host draw is not needed.
"""
from __future__ import annotations

import numpy as np

_FY_SRC = r"""
extern "C" __global__
void fy_permutations(int* __restrict__ idx, const long long n, const unsigned long long base_seed, const int m) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m) return;
    int* row = idx + (long long)r * n;
    for (long long i = 0; i < n; ++i) row[i] = (int)i;
    unsigned long long z = base_seed + (unsigned long long)r + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    unsigned long long x = z ^ (z >> 31);
    if (x == 0ULL) x = 1ULL;
    for (long long i = n - 1; i > 0; --i) {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        const unsigned long long rr = (x * 2685821657736338717ULL) >> 32;
        const long long j = (long long)((rr * (unsigned long long)(i + 1)) >> 32);
        const int t = row[i];
        row[i] = row[j];
        row[j] = t;
    }
}

extern "C" __global__
void pack_class_bits(const int* __restrict__ idx, const long long* __restrict__ y, unsigned long long* __restrict__ out,
                     const long long n, const int m, const int n_y, const int words) {
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)m * words) return;
    const int b = (int)(t / words);
    const int w = (int)(t - (long long)b * words);
    unsigned long long acc[16];
    for (int c = 0; c < 16; ++c) acc[c] = 0ULL;
    const long long lo = (long long)w * 64;
    const long long hi = lo + 64 < n ? lo + 64 : n;
    const int* row = idx + (long long)b * n;
    for (long long i = lo; i < hi; ++i) {
        const int c = (int)y[row[i]];
        acc[c] |= 1ULL << (unsigned)(i - lo);
    }
    for (int c = 0; c < n_y; ++c) out[((long long)b * n_y + c) * words + w] = acc[c];
}
"""
_KERNELS: dict = {}


def _kernels(cp):
    """The compiled ``(fy_permutations, pack_class_bits)`` pair."""
    if not _KERNELS:
        module = cp.RawModule(code=_FY_SRC)
        _KERNELS["fy"] = module.get_function("fy_permutations")
        _KERNELS["pack"] = module.get_function("pack_class_bits")
    return _KERNELS["fy"], _KERNELS["pack"]


def device_label_bits(y: np.ndarray, m: int, base_seed: int, n_y: int, words: int) -> "np.ndarray | None":
    """``(m, n_y, words)`` uint64 class bit sets of the ``m`` permutations seeded ``base_seed + r`` of the labels ``y`` (int64 codes below ``n_y`` <= 16), or ``None`` when the device cannot do it
    (no cupy, out of memory): the caller then draws on the host."""
    if n_y > 16 or m <= 0:
        return None
    try:
        import cupy as cp

        fy, pack = _kernels(cp)
        n = int(y.shape[0])
        idx = cp.empty((m, n), dtype=cp.int32)
        fy(((m + 31) // 32,), (32,), (idx, np.int64(n), np.uint64(base_seed), np.int32(m)))
        y_d = cp.asarray(np.ascontiguousarray(y, dtype=np.int64))
        out = cp.empty((m, n_y, words), dtype=cp.uint64)
        total = m * words
        pack(((total + 127) // 128,), (128,), (idx, y_d, out, np.int64(n), np.int32(m), np.int32(n_y), np.int32(words)))
        return np.asarray(cp.asnumpy(out))
    except Exception:  # best-effort: no device / out of memory / compile failure: the host draw gives the same bits
        return None
