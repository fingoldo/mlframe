"""Popcount-bitmap pairwise Symmetric-Uncertainty kernel for ShapProxiedFS.

Sibling of ``_shap_proxy_cluster_su.py``'s scalar joint-counts kernel
``_pairwise_su_edges``. Same input layout (column-major ``bins_packed``,
per-column marginals, entropies, constant mask, threshold). Same output
contract (upper-triangle ``flags`` matrix). What changes is the per-pair inner
loop:

  per-pair scalar path:
      for k in n_samples: joint[bins[i,k], bins[j,k]] += 1     # 1 increment / sample
      total ops per pair: O(n_samples)

  per-pair bitmap path:
      onehot[f, b, word] = packed bitmask of samples where bins[f, sample] == b
      joint[a, b] = popcount(onehot[i, a] AND onehot[j, b]) per word, summed
      total ops per pair: O(n_bins^2 * ceil(n_samples / 64))

The bitmap path looks worse in raw op count at n_bins=10 / n_samples=1500
(100*24=2400 vs 1500), BUT each operation is a 64-wide AND+POPCNT instead of a
single scalar increment with a data-dependent store. On x86-64 with SSE4.2 (POPCNT)
that is one instruction per 64 samples and the joint accumulator stays in registers,
so the wall-clock ratio collapses by an order of magnitude. Even better at lower
``n_bins`` where the n_bins^2 factor shrinks faster than n_samples/64.

When to route here (driven by the dispatcher in ``cluster_correlated_features_su``):
  * max ``n_bins <= 16``: bitmap memory is ``n_features * n_bins * ceil(n_samples/64)
    * 8`` bytes. At n_bins=16 / width=2000 / n_samples=1500 = 7.5 MB, fits L2 per
    feature subset.
  * ``n_features >= bitmap_min_features`` (default 200, kernel_tuning_cache-tunable):
    below that the pack-onehot setup cost dominates the saved per-pair work.

Returns ``flags`` ``(n_features, n_features)`` uint8 upper-triangle, identical
in value to ``_pairwise_su_edges`` on shared inputs (parity test in iter73).
"""

from __future__ import annotations

import logging
import math

import numpy as np
from numba import njit, prange, types
from numba.extending import intrinsic
from llvmlite import ir

logger = logging.getLogger(__name__)


@intrinsic
def _popcount_u64(typingctx, x):
    """LLVM ``ctpop.i64`` intrinsic. Compiles to POPCNT on x86-64 / SSE4.2+.

    Wrapped through ``@intrinsic`` so the LLVM-IR call survives numba lowering
    and gets the hardware POPCNT instruction picked up by the backend's
    instruction selector. Falls back to a software ctpop on CPUs without SSE4.2
    (rare on anything from the last ~12 years).
    """
    sig = types.uint64(types.uint64)

    def codegen(context, builder, signature, args):
        fnty = ir.FunctionType(ir.IntType(64), [ir.IntType(64)])
        fn = builder.module.declare_intrinsic("llvm.ctpop.i64", fnty=fnty)
        return builder.call(fn, [args[0]])

    return sig, codegen


# Memory cap for the bitmap allocation. Above this the dispatcher MUST fall back
# to the scalar kernel - we don't want surprise multi-GB allocs on the hot path.
# n_features * n_bins * ceil(n_samples / 64) * 8 bytes.
_BITMAP_MAX_BYTES_DEFAULT = 256 * 1024 * 1024  # 256 MB hard ceiling


def _bitmap_bytes(n_features: int, max_n_bins: int, n_samples: int) -> int:
    """Bitmap allocation size in bytes (uint64 words: ceil(n_samples/64) per (f, bin))."""
    n_words = (int(n_samples) + 63) // 64
    return int(n_features) * int(max_n_bins) * int(n_words) * 8


def _resolve_bitmap_min_features(default: int = 200) -> int:
    """Smallest ``n_features`` at which the bitmap kernel beats the scalar prange kernel.

    Below this width the one-time onehot-pack cost (single sweep over
    ``n_features * n_samples`` int32 reads writing ``n_features * n_bins * ceil(n_samples/64)``
    uint64 cells) dominates the saved per-pair work. Above it the f^2 pair count amortizes
    the pack and the per-pair POPCNT speedup wins. Dispatcher-tunable per HW via
    ``pyutilz.performance.kernel_tuning.cache`` (key
    ``mlframe.shap_proxied_fs.cluster_su.bitmap_min_features``).
    """
    try:
        from pyutilz.performance.kernel_tuning import cache as kernel_tuning_cache

        value = kernel_tuning_cache.get(
            "mlframe.shap_proxied_fs.cluster_su.bitmap_min_features", default=default,
        )
        return int(value)
    except Exception:
        return default


def _resolve_bitmap_max_n_bins(default: int = 12) -> int:
    """Upper ``max(n_bins)`` at which the bitmap kernel still wins.

    The per-pair work scales as ``n_bins^2 * ceil(n_samples/64)``. iter73 bench at
    n_samples=1500 / 8 numba threads:

      n_bins   scalar   bitmap   speedup
      4        1588 ms   265 ms   6.0x
      8        2088 ms  1047 ms   1.99x
      10       2243 ms  1542 ms   1.45x
      12       1517 ms  1177 ms   1.29x
      14       1656 ms  1492 ms   1.11x
      16       3753 ms  3911 ms   0.96x (regression - scalar wins)

    The default ceiling is 12: above that the n_bins^2 POPCNT work dominates the
    parallel-friendly memory access savings. The dispatcher routes back to the
    scalar prange kernel beyond this threshold. Tunable via
    ``mlframe.shap_proxied_fs.cluster_su.bitmap_max_n_bins``.
    """
    try:
        from pyutilz.performance.kernel_tuning import cache as kernel_tuning_cache

        value = kernel_tuning_cache.get(
            "mlframe.shap_proxied_fs.cluster_su.bitmap_max_n_bins", default=default,
        )
        return int(value)
    except Exception:
        return default


def _resolve_bitmap_min_samples(default: int = 256) -> int:
    """Smallest ``n_samples`` at which the bitmap pack overhead amortizes.

    Below this the constant pack cost (and the SWAR / POPCNT pipeline startup) dwarfs
    the per-pair savings. Tunable via
    ``mlframe.shap_proxied_fs.cluster_su.bitmap_min_samples``; default 256.
    """
    try:
        from pyutilz.performance.kernel_tuning import cache as kernel_tuning_cache

        value = kernel_tuning_cache.get(
            "mlframe.shap_proxied_fs.cluster_su.bitmap_min_samples", default=default,
        )
        return int(value)
    except Exception:
        return default


def should_route_bitmap(
    n_features: int,
    n_samples: int,
    max_n_bins: int,
    *,
    bitmap_min_features: int | None = None,
    bitmap_max_n_bins: int | None = None,
    bitmap_min_samples: int | None = None,
    bitmap_max_bytes: int = _BITMAP_MAX_BYTES_DEFAULT,
) -> bool:
    """Three gates: feature width, bin cap, sample floor + memory cap.

    All must pass for the dispatcher to call ``_pairwise_su_edges_bitmap``.
    """
    fmin = bitmap_min_features if bitmap_min_features is not None else _resolve_bitmap_min_features()
    bmax = bitmap_max_n_bins if bitmap_max_n_bins is not None else _resolve_bitmap_max_n_bins()
    smin = bitmap_min_samples if bitmap_min_samples is not None else _resolve_bitmap_min_samples()
    if int(n_features) < int(fmin):
        return False
    if int(max_n_bins) > int(bmax):
        return False
    if int(n_samples) < int(smin):
        return False
    if _bitmap_bytes(n_features, max_n_bins, n_samples) > int(bitmap_max_bytes):
        return False
    return True


@njit(parallel=True, nogil=True, cache=True, fastmath=False)
def _pack_onehot_bitmap(
    bins_packed: np.ndarray,
    nbins: np.ndarray,
    max_nb: int,
    n_words: int,
) -> np.ndarray:
    """Pack ``bins_packed`` (n_features, n_samples) int32 into a one-hot bitmap.

    Output shape: ``(n_features, max_nb, n_words)`` uint64. Bit ``k`` of
    ``bitmap[f, b, w]`` is 1 iff ``bins_packed[f, w*64 + k] == b``. Padded bin
    slots (``b >= nbins[f]``) remain all-zero. Padding bits beyond ``n_samples``
    in the last word stay zero too (np.zeros init).
    """
    n_features = bins_packed.shape[0]
    n_samples = bins_packed.shape[1]
    bitmap = np.zeros((n_features, max_nb, n_words), dtype=np.uint64)
    for f in prange(n_features):
        nb_f = nbins[f]
        for k in range(n_samples):
            b = bins_packed[f, k]
            # Defensive: bin id should already be < nb_f by construction,
            # but ignore overflowing ids rather than corrupt memory.
            if 0 <= b < nb_f:
                w = k >> 6  # k // 64
                bit = k & 63  # k % 64
                bitmap[f, b, w] |= np.uint64(1) << np.uint64(bit)
    return bitmap


# bench-attempt-rejected (2026-05-31, iter74): transposed bitmap layout
# (max_nb, n_features, n_words) hoping LLVM would vectorize the inner word
# loop across features. Measured at n_samples=1500 / n_threads=8 / threshold=0.4:
#
#   width  legacy_ms  transposed_ms  speedup
#   100      4.2        4.6           0.91x
#   500     86.7       87.5           0.99x
#   1500   868.4      852.7           1.02x
#   2000  1491.9     1505.7           0.99x
#
#   n_bins=4   252.6  266.2  0.95x
#   n_bins=8   942.8  961.6  0.98x
#   n_bins=10 1469.6 1517.1  0.97x
#   n_bins=12 2241.0 2348.8  0.95x
#
# Parity held across all configs. The expected SIMD-across-features win does
# not materialise in the per-pair (i, j) loop structure: within one
# (a, b, w) triple the kernel only reads ONE word from each of two features,
# so adjacent feature rows on cache lines are not reused before eviction.
# A wider-AND-vectorization path would require restructuring to an outer
# (a, b, w) loop accumulating ALL pair joints simultaneously, which needs
# O(n_features^2 * max_nb^2) intermediate storage (1.8 GB at width=1500,
# nbins=10) - blows past the 256 MB memory cap. Don't re-attempt without
# blocking the (a, b) sweep into tiles that fit L2.


@njit(parallel=True, nogil=True, cache=True, fastmath=False)
def _pairwise_su_edges_bitmap(
    bitmap: np.ndarray,
    nbins: np.ndarray,
    freqs_packed: np.ndarray,
    freqs_offsets: np.ndarray,
    h_marginals: np.ndarray,
    constant_mask: np.ndarray,
    n_samples: int,
    threshold: float,
) -> np.ndarray:
    """Pairwise SU edges via popcount(AND) joint-counts.

    For each pair (i, j) with i < j:
      * For every (a, b) in nb_i x nb_j: ``joint[a, b] = sum_w popcount(bitmap[i, a, w]
        AND bitmap[j, b, w])``.
      * MI uses the precomputed marginals from ``freqs_packed``.
      * SU = 2 MI / (h_i + h_j). Edge iff ``SU >= threshold``.

    The outer ``i`` loop is parallel (prange). Each thread keeps a single
    ``(max_nb, max_nb)`` int64 joint scratch reused per ``j`` partner (cleared
    only over the ``(nb_i, nb_j)`` window we'll touch). The inner popcount uses
    the hardware POPCNT instruction via the LLVM ctpop intrinsic above.
    """
    n_features = bitmap.shape[0]
    max_nb = bitmap.shape[1]
    n_words = bitmap.shape[2]
    flags = np.zeros((n_features, n_features), dtype=np.uint8)
    inv_n = 1.0 / n_samples if n_samples > 0 else 0.0
    for i in prange(n_features):
        if constant_mask[i]:
            continue
        nb_i = nbins[i]
        h_i = h_marginals[i]
        off_i = freqs_offsets[i]
        # Thread-local joint scratch; sized to max_nb across all features so each
        # thread allocates once on its stack and clears just the (nb_i, nb_j) window.
        joint = np.zeros((max_nb, max_nb), dtype=np.int64)
        for j in range(i + 1, n_features):
            if constant_mask[j]:
                continue
            nb_j = nbins[j]
            # Loop order (a, b, w): the inner w-sweep accumulates a single popcount
            # total in a register before storing to joint[a, b]. Hoisting the 2D
            # row slices ``bitmap[i, a]`` and ``bitmap[j, b]`` out of the inner w
            # loop lets numba/LLVM lift the (n_features * max_nb * n_words)
            # base-pointer recomputation that the 3D indexing otherwise emits
            # on every iteration.
            #
            # bench-attempt-rejected (2026-05-31, iter73): marginal-derivation
            # trick (compute (nb_i-1)*(nb_j-1) cells via popcount, derive last
            # row/col from row/col sums via round(px*n) - partial_sum). Saved
            # 19% of popcount sweeps at nb=10 but slowed bitmap from 1.41x to
            # 1.36x at width=1000: the derivation arithmetic (np.int64(round(...)))
            # plus the extra row_sum/col_sum reductions cost more than the
            # popcount work they replaced. Hardware POPCNT throughput is high
            # enough that "skip work" is not worth the branch+arithmetic price.
            for a in range(nb_i):
                row_i = bitmap[i, a]
                for b in range(nb_j):
                    row_j = bitmap[j, b]
                    cnt = np.uint64(0)
                    for w in range(n_words):
                        cnt += _popcount_u64(row_i[w] & row_j[w])
                    joint[a, b] = np.int64(cnt)
            mi = 0.0
            off_j = freqs_offsets[j]
            for a in range(nb_i):
                px = freqs_packed[off_i + a]
                if px <= 0.0:
                    continue
                for b in range(nb_j):
                    jc = joint[a, b]
                    if jc == 0:
                        continue
                    py = freqs_packed[off_j + b]
                    if py <= 0.0:
                        continue
                    jf = jc * inv_n
                    mi += jf * math.log(jf / (px * py))
            denom = h_i + h_marginals[j]
            if denom <= 1e-12:
                continue
            su = 2.0 * mi / denom
            if su >= threshold:
                flags[i, j] = 1
    return flags


def pairwise_su_edges_bitmap(
    bins_packed: np.ndarray,
    nbins: np.ndarray,
    freqs_packed: np.ndarray,
    freqs_offsets: np.ndarray,
    h_marginals: np.ndarray,
    constant_mask: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Public entry point: pack the bitmap then run the prange popcount kernel.

    Mirrors ``_pairwise_su_edges``'s signature so the dispatcher can swap in
    transparently. The pack and the pair scan are two sequential prange
    kernels; the bitmap buffer lives only for the scope of this call.
    """
    n_features, n_samples = bins_packed.shape
    if n_features == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    max_nb = int(nbins.max()) if nbins.size else 0
    if max_nb == 0:
        return np.zeros((n_features, n_features), dtype=np.uint8)
    n_words = (int(n_samples) + 63) // 64
    bitmap = _pack_onehot_bitmap(bins_packed, nbins, max_nb, n_words)
    return _pairwise_su_edges_bitmap(
        bitmap, nbins, freqs_packed, freqs_offsets,
        h_marginals, constant_mask, int(n_samples), float(threshold),
    )
