"""Device target shuffles without a sort: ``out[k, i] = y[pi_k(i)]`` where ``pi_k`` is a keyed Feistel permutation of ``0 .. n - 1``, one launch for all ``nperm`` rows.

The order-1 maxT floor needs ``nperm`` shuffles of the target. The device generator argsorted a ``(nperm, n)`` double key matrix (``_permutation_null_resident.gen_target_shuffles_cupy_sorted``): 43.5, 80.9 and 33.6 ms at
``(25, 100000)``, ``(50, 100000)`` and ``(100, 30000)`` on the development host, thrust's segmented sort being synchronous (a bare 2-D ``cp.argsort`` keeps a counter thread at 9 to 34 percent of its free pace, against
94 percent while a thread sleeps: it holds the GIL for most of its run) and the safe wrapper falling back to one sort per vector, whose GIL round trips each cost a switch interval while the other lanes of the fit are busy. A Feistel network is a bijection on ``[0, 2^(2h))`` for any round function; cycle-walking (apply it again while the image is ``>= n``) restricts it to a bijection on ``[0, n)``. Every thread computes its own
image: no sort, no scratch memory, an asynchronous launch.

Quality: eight rounds of a 32-bit multiply-xor-shift round function (the murmur3 finalizer) keyed per (row, round) from a splitmix64 stream of the seed. Eight rounds is twice the four a Luby-Rackoff pseudorandom permutation
needs; the tests check that every row is a permutation, the uniformity of the image of every position, the fixed-point count and the descent count against the values of a uniform permutation. The rows are pseudorandom
permutations, not draws from all of ``S_n``: the maxT floor only needs shuffles that are exchangeable with the labels for the statistic at hand, the same statistical-equivalence contract as the host generators.

Win axis: any ``n`` and ``nperm``; the walk length is ``2^(2h) / n`` on average (2.6 at ``n = 100000``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for

from ._batch_mi_noise_gate_bits import _C_TYPES, typed_module

ROUNDS = 8

_SRC = r"""
__device__ __forceinline__ unsigned int mix32(unsigned int x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
// coalescing-ok: the threads of a warp write adjacent columns of one output row; the gather y[image] is data-dependent (a permutation of a few hundred KB that stays in L2)
extern "C" __global__ void feistel_shuffle(const {DT}* __restrict__ y, const unsigned int* __restrict__ keys, const unsigned int n, const unsigned int half_bits, const int rounds,
                                           const int nperm, {DT}* __restrict__ out) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y;
    if (i >= n || k >= nperm) return;
    const unsigned int mask = (1u << half_bits) - 1u;
    const unsigned int* key = keys + (size_t)k * rounds;
    unsigned int x = i;
    do {  // cycle-walking: the network is a bijection on [0, 2^(2 half_bits)), repeating it until the image is below n makes a bijection on [0, n)
        unsigned int l = (x >> half_bits) & mask;
        unsigned int r = x & mask;
        for (int t = 0; t < rounds; ++t) {
            const unsigned int f = mix32(r ^ key[t]) & mask;
            const unsigned int nl = r;
            r = l ^ f;
            l = nl;
        }
        x = (l << half_bits) | r;
    } while (x >= n);
    out[(size_t)k * n + i] = y[x];
}
"""


def _round_keys(nperm: int, seed: int) -> np.ndarray:
    """``(nperm, ROUNDS)`` uint32 keys: the low 32 bits of the splitmix64 output at the counters ``1 .. nperm * ROUNDS`` of ``seed``."""
    count = int(nperm) * ROUNDS
    with np.errstate(over="ignore"):
        z = np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF) + np.uint64(0x9E3779B97F4A7C15) * np.arange(1, count + 1, dtype=np.uint64)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return np.asarray((z & np.uint64(0xFFFFFFFF)).astype(np.uint32).reshape(int(nperm), ROUNDS))


def half_bits_for(n: int) -> int:
    """Bits of one Feistel half: the domain ``2^(2 * bits)`` is the smallest even power of two that is at least ``n`` (and at least 4)."""
    total = max(2, int(max(int(n), 2) - 1).bit_length())
    return (total + 1) // 2


def shuffled_rows(cp: Any, d_y: Any, nperm: int, seed: int) -> Any:
    """The ``(nperm, n)`` device matrix whose row ``k`` is ``d_y`` permuted by the keyed Feistel permutation ``k`` of ``seed`` (``d_y`` a 1-D device array of an integer label type)."""
    n = int(d_y.shape[0])
    out = cp.empty((int(nperm), n), dtype=d_y.dtype)
    if n == 0 or nperm == 0:
        return out
    keys = cp.asarray(_round_keys(nperm, seed))
    kernel = typed_module(cp, _SRC, _C_TYPES[d_y.dtype]).get_function("feistel_shuffle")
    block = block_size_for(kernel, work_items=n)
    kernel(((n + block - 1) // block, int(nperm)), (block,), (d_y, keys, np.uint32(n), np.uint32(half_bits_for(n)), np.int32(ROUNDS), np.int32(nperm), out))
    return out
