"""Count-table variant of the family-wise permutation null (``family_null_max_njit`` keeps the per-cell logarithm reference).

For a fixed column the marginal counts ``px`` are the same under every label permutation, and so are the label counts ``py``. The plug-in MI therefore needs only the sum of ``c * log c`` over the
joint cells,

    MI = (sum_ab xlogx(n_ab) + n log n - sum_a xlogx(px_a) - sum_b xlogx(py_b)) / n,

with ``xlogx`` a lookup table over the integers ``0..n`` (``xlogx(0) = 0`` needs no branch). The per-permutation work is one histogram pass and a table lookup per cell instead of a logarithm and a
division per occupied cell and a second pass over the cells for every marginal. Two-class labels are counted in a single array (the class-1 count per bin; class 0 is the bin total minus it).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange, types
from numba.extending import intrinsic, overload


@njit(cache=True)
def xlogx_table(n: int) -> np.ndarray:
    """``t[c] = c * log(c)`` for ``c = 0..n`` (``t[0] = 0``)."""
    t = np.zeros(n + 1, dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    for c in range(2, n + 1):
        t[c] = c * np.log(c)
    return t


@njit(cache=True, parallel=True)
def column_marginals(cols: np.ndarray, n_x: np.ndarray, cap: int, table: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """``(px, c_a)``: per-column bin counts ``(k, cap)`` and ``sum_a xlogx(px_a)`` ``(k,)``."""
    k = cols.shape[0]
    n = cols.shape[1]
    px = np.zeros((k, cap), dtype=np.int64)
    c_a = np.zeros(k, dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    for j in prange(k):
        for i in range(n):
            px[j, cols[j, i]] += 1
        s = 0.0
        for a in range(n_x[j]):
            s += table[px[j, a]]
        c_a[j] = s
    return px, c_a


@njit(cache=True)
def mi_column_counts(col: np.ndarray, n_x: int, y: np.ndarray, n_y: int, joint: np.ndarray, table: np.ndarray, const: float) -> float:
    """Plug-in MI of one column with labels ``y`` given ``const = n log n - sum_a xlogx(px_a) - sum_b xlogx(py_b)``; ``joint`` is a caller-owned int64 buffer of ``n_x * n_y`` cells."""
    n = y.shape[0]
    cells = n_x * n_y
    for c in range(cells):
        joint[c] = 0
    for i in range(n):
        joint[col[i] * n_y + y[i]] += 1
    s = 0.0
    for c in range(cells):
        s += table[joint[c]]
    return float((s + const) / n)


@njit(cache=True, parallel=True)
def family_null_max_counts(cols: np.ndarray, n_x: np.ndarray, y_perms: np.ndarray, n_y: int, table: np.ndarray, c_a: np.ndarray) -> np.ndarray:
    """Maximum MI over the columns of ``cols`` for each row of ``y_perms`` (label permutations of one label vector); parallel over permutations."""
    b_count = y_perms.shape[0]
    k = cols.shape[0]
    n = y_perms.shape[1]
    out = np.zeros(b_count, dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    cap = 1
    for j in range(k):
        if n_x[j] > cap:
            cap = n_x[j]
    py = np.zeros(n_y, dtype=np.int64)
    for i in range(n):
        py[y_perms[0, i]] += 1
    c_b = 0.0
    for b in range(n_y):
        c_b += table[py[b]]
    nlogn = table[n]
    consts = np.empty(k, dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    for j in range(k):
        consts[j] = nlogn - c_a[j] - c_b
    for b in prange(b_count):
        best = 0.0
        joint = np.zeros(cap * n_y, dtype=np.int64)
        for j in range(k):
            mi = mi_column_counts(cols[j], n_x[j], y_perms[b], n_y, joint, table, consts[j])
            if mi > best:
                best = mi
        out[b] = best
    return out


@intrinsic
def _ctpop64(typingctx, x):
    """Population count of a uint64 (the LLVM ``ctpop`` intrinsic, a single ``popcnt`` instruction on x86)."""
    sig = types.uint64(types.uint64)

    def codegen(context, builder, signature, args):
        """Emit ``llvm.ctpop.i64``."""
        fn = builder.module.declare_intrinsic("llvm.ctpop", [args[0].type])
        return builder.call(fn, args)

    return sig, codegen


def _popcount64(x):
    """Population count of a uint64: ``_ctpop64`` inside numba, ``int.bit_count`` when JIT is disabled (an intrinsic cannot run as Python)."""
    return int(x).bit_count()


@overload(_popcount64)
def _popcount64_overload(x):
    """Route the numba-compiled call to the ``ctpop`` intrinsic for uint64 arguments."""
    if x == types.uint64:

        def impl(x):
            """Emit the intrinsic popcount."""
            return _ctpop64(x)

        return impl
    return None


@njit(cache=True, parallel=True)
def column_bin_bitsets(cols: np.ndarray, n_x: np.ndarray, cap: int) -> np.ndarray:
    """``(k, cap, words)`` uint64 bitsets: bit ``i`` of ``[j, a]`` is set when ``cols[j, i] == a``."""
    k = cols.shape[0]
    n = cols.shape[1]
    words = (n + 63) // 64
    out = np.zeros((k, cap, words), dtype=np.uint64)
    for j in prange(k):
        for i in range(n):
            out[j, cols[j, i], i >> 6] |= np.uint64(1) << np.uint64(i & 63)
    return out


@njit(cache=True)
def label_bitsets(y: np.ndarray, n_y: int, words: int) -> np.ndarray:
    """``(n_y, words)`` uint64 bitsets of the classes of one label vector."""
    out = np.zeros((n_y, words), dtype=np.uint64)
    for i in range(y.shape[0]):
        out[y[i], i >> 6] |= np.uint64(1) << np.uint64(i & 63)
    return out


@njit(cache=True, parallel=True)
def label_bitsets_batch(y_perms: np.ndarray, n_y: int, words: int) -> np.ndarray:
    """``(permutations, n_y, words)`` uint64 class bit sets of every row of ``y_perms``."""
    b_count = y_perms.shape[0]
    out = np.zeros((b_count, n_y, words), dtype=np.uint64)
    for b in prange(b_count):
        for i in range(y_perms.shape[1]):
            out[b, y_perms[b, i], i >> 6] |= np.uint64(1) << np.uint64(i & 63)
    return out


@njit(cache=True, parallel=True)
def label_bitsets_from_index(y: np.ndarray, perm_idx: np.ndarray, n_y: int, words: int) -> np.ndarray:
    """:func:`label_bitsets_batch` of ``y[perm_idx]`` without building the ``(permutations, n)`` label matrix: row ``b`` uses the labels ``y[perm_idx[b]]``."""
    b_count = perm_idx.shape[0]
    out = np.zeros((b_count, n_y, words), dtype=np.uint64)
    for b in prange(b_count):
        for i in range(perm_idx.shape[1]):
            out[b, y[perm_idx[b, i]], i >> 6] |= np.uint64(1) << np.uint64(i & 63)
    return out


@njit(cache=True, parallel=True)
def family_null_max_bitsets(bits: np.ndarray, n_x: np.ndarray, y_perms: np.ndarray, n_y: int, table: np.ndarray, c_a: np.ndarray, px: np.ndarray) -> np.ndarray:
    """Same result as :func:`family_null_max_counts` with the joint counts taken as popcounts of ``bin & class`` bitsets.

    The class-``b`` count of bin ``a`` is ``popcount(bins[a] & classes[b])``; the last class is the bin total minus the others, so a two-class label needs one AND per bin.
    """
    b_count = y_perms.shape[0]
    k = bits.shape[0]
    words = bits.shape[2]
    n = y_perms.shape[1]
    out = np.zeros(b_count, dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    py = np.zeros(n_y, dtype=np.int64)
    for i in range(n):
        py[y_perms[0, i]] += 1
    c_b = 0.0
    for b in range(n_y):
        c_b += table[py[b]]
    nlogn = table[n]
    for b in prange(b_count):
        cls = label_bitsets(y_perms[b], n_y, words)
        best = 0.0
        for j in range(k):
            s = 0.0
            for a in range(n_x[j]):
                rest = px[j, a]
                for c in range(n_y - 1):
                    cnt = 0
                    for w in range(words):
                        cnt += _popcount64(bits[j, a, w] & cls[c, w])
                    s += table[cnt]
                    rest -= cnt
                s += table[rest]
            mi = (s + nlogn - c_a[j] - c_b) / n
            if mi > best:
                best = mi
        out[b] = best
    return out


@njit(cache=True, parallel=True)
def family_null_max_bitsets_classes(bits: np.ndarray, n_x: np.ndarray, cls_all: np.ndarray, n: int, table: np.ndarray, c_a: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """:func:`family_null_max_bitsets` from the class bit sets ``cls_all`` ``(permutations, n_y, words)`` of the permuted labels, four permutations per pass over the column bitsets.

    The kernel is bound by reading the bitsets (every permutation streams all of them); a block of four permutations loads each bitset word once and ANDs it with the class words of all four, so the
    memory traffic is a quarter and the per-permutation arithmetic, and so every result, is unchanged (the sums run in the same order). The last block is padded with empty permutations that are not written."""
    b_count = cls_all.shape[0]
    n_y = cls_all.shape[1]
    k = bits.shape[0]
    words = bits.shape[2]
    out = np.zeros(b_count, dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    c_b = 0.0
    for b in range(n_y):
        c_b += table[py[b]]
    nlogn = table[n]
    n_blocks = (b_count + 3) // 4
    for blk in prange(n_blocks):
        b0 = blk * 4
        cls0 = cls_all[b0]
        cls1 = cls_all[b0 + 1] if b0 + 1 < b_count else np.zeros((n_y, words), dtype=np.uint64)
        cls2 = cls_all[b0 + 2] if b0 + 2 < b_count else np.zeros((n_y, words), dtype=np.uint64)
        cls3 = cls_all[b0 + 3] if b0 + 3 < b_count else np.zeros((n_y, words), dtype=np.uint64)
        best0 = 0.0
        best1 = 0.0
        best2 = 0.0
        best3 = 0.0
        for j in range(k):
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            s3 = 0.0
            for a in range(n_x[j]):
                rest0 = px[j, a]
                rest1 = px[j, a]
                rest2 = px[j, a]
                rest3 = px[j, a]
                for c in range(n_y - 1):
                    cnt0 = 0
                    cnt1 = 0
                    cnt2 = 0
                    cnt3 = 0
                    for w in range(words):
                        x = bits[j, a, w]
                        cnt0 += _popcount64(x & cls0[c, w])
                        cnt1 += _popcount64(x & cls1[c, w])
                        cnt2 += _popcount64(x & cls2[c, w])
                        cnt3 += _popcount64(x & cls3[c, w])
                    s0 += table[cnt0]
                    s1 += table[cnt1]
                    s2 += table[cnt2]
                    s3 += table[cnt3]
                    rest0 -= cnt0
                    rest1 -= cnt1
                    rest2 -= cnt2
                    rest3 -= cnt3
                s0 += table[rest0]
                s1 += table[rest1]
                s2 += table[rest2]
                s3 += table[rest3]
            mi0 = (s0 + nlogn - c_a[j] - c_b) / n
            mi1 = (s1 + nlogn - c_a[j] - c_b) / n
            mi2 = (s2 + nlogn - c_a[j] - c_b) / n
            mi3 = (s3 + nlogn - c_a[j] - c_b) / n
            if mi0 > best0:
                best0 = mi0
            if mi1 > best1:
                best1 = mi1
            if mi2 > best2:
                best2 = mi2
            if mi3 > best3:
                best3 = mi3
        out[b0] = best0
        if b0 + 1 < b_count:
            out[b0 + 1] = best1
        if b0 + 2 < b_count:
            out[b0 + 2] = best2
        if b0 + 3 < b_count:
            out[b0 + 3] = best3
    return out


def family_null_max_bitsets_blocked(bits: np.ndarray, n_x: np.ndarray, y_perms: np.ndarray, n_y: int, table: np.ndarray, c_a: np.ndarray, px: np.ndarray) -> np.ndarray:
    """:func:`family_null_max_bitsets_classes` for permuted label vectors ``y_perms``: the class bit sets are packed first."""
    words = bits.shape[2]
    py = np.bincount(y_perms[0], minlength=n_y).astype(np.int64)
    out: np.ndarray = family_null_max_bitsets_classes(bits, n_x, label_bitsets_batch(y_perms, n_y, words), y_perms.shape[1], table, c_a, px, py)
    return out


BITSET_MAX_BINS = 32  # columns with more bins than this are counted with the histogram kernel
BITSET_MAX_CLASSES = 16  # label vectors with more classes are counted with the histogram kernel (measured break-even near 24 classes at 10 bins)
BITSET_BUDGET_BYTES = 128 * 1024 * 1024  # column bitsets held at a time


def bitset_groups(cols: np.ndarray, n_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, np.ndarray]]]:
    """``(table, c_a, px, groups)`` for the bitset null: the xlogx table, the per-column ``sum_a xlogx(px_a)`` and bin counts, and ``[(lo, hi, bits)]`` column groups within the memory budget."""
    k, n = cols.shape
    cap = int(n_x.max())
    table = xlogx_table(n)
    px, c_a = column_marginals(cols, n_x, cap, table)
    words = (n + 63) // 64
    per_group = max(1, BITSET_BUDGET_BYTES // (cap * words * 8))
    groups = [(lo, min(lo + per_group, k), column_bin_bitsets(cols[lo : lo + per_group], n_x[lo : lo + per_group], cap)) for lo in range(0, k, per_group)]
    return table, c_a, px, groups


def null_max_bitsets(prep: tuple, n_x: np.ndarray, y_perms: np.ndarray, n_y: int) -> np.ndarray:
    """Per-permutation maximum MI over every column, from the prepared bitset groups (see :func:`bitset_groups`)."""
    table, c_a, px, groups = prep
    out = np.zeros(y_perms.shape[0], dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    for lo, hi, bits in groups:
        np.maximum(out, family_null_max_bitsets_blocked(bits, n_x[lo:hi], y_perms, int(n_y), table, c_a[lo:hi], px[lo:hi]), out=out)
    return out


def null_max_bitsets_from_classes(prep: tuple, n_x: np.ndarray, cls_all: np.ndarray, n: int, py: np.ndarray) -> np.ndarray:
    """:func:`null_max_bitsets` from the class bit sets of the permuted labels (see :func:`label_bitsets_from_index`) and the class counts ``py`` of the label vector."""
    table, c_a, px, groups = prep
    out = np.zeros(cls_all.shape[0], dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    for lo, hi, bits in groups:
        np.maximum(out, family_null_max_bitsets_classes(bits, n_x[lo:hi], cls_all, n, table, c_a[lo:hi], px[lo:hi], py), out=out)
    return out


@njit(cache=True, parallel=True)
def mi_all_columns_counts(cols: np.ndarray, n_x: np.ndarray, y_codes: np.ndarray, n_y: int) -> np.ndarray:
    """Plug-in MI (nats) of every row of ``cols`` with ``y_codes`` in the count-table arithmetic of the null kernels, so an observed value and a null value of the same column and labels are bit-equal."""
    k = cols.shape[0]
    n = y_codes.shape[0]
    cap = 1
    for j in range(k):
        if n_x[j] > cap:
            cap = n_x[j]
    table = xlogx_table(n)
    py = np.zeros(n_y, dtype=np.int64)
    for i in range(n):
        py[y_codes[i]] += 1
    c_b = 0.0
    for b in range(n_y):
        c_b += table[py[b]]
    out = np.zeros(k, dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    for j in prange(k):
        px = np.zeros(n_x[j], dtype=np.int64)
        joint = np.zeros(n_x[j] * n_y, dtype=np.int64)
        for i in range(n):
            px[cols[j, i]] += 1
            joint[cols[j, i] * n_y + y_codes[i]] += 1
        c_a = 0.0
        for a in range(n_x[j]):
            c_a += table[px[a]]
        s = 0.0
        for c in range(n_x[j] * n_y):
            s += table[joint[c]]
        out[j] = (s + table[n] - c_a - c_b) / n
    return out
