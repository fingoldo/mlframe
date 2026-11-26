"""Radix-selected quantile edges for float32 columns (the FE buffer dtype).

:func:`_kernels._quantile_edges_2d_njit` picks the two order statistics each quantile interpolates between with ``multiselect_inplace``.
For the 2000 x 7668 float32 chunk the kernel spends ~50-90 us per column there, which is slower than a plain LSD radix sort of the
column (4 passes of 8 bits over an order-preserving uint32 key, ~40 us). A full sort yields the same order-statistic values, so the edges
are identical; the only difference that can appear is the sign of an exact zero (-0.0 vs +0.0 compare equal, so the bins they define are
identical). Input must be NaN-free, as for the multiselect kernel.
"""

from __future__ import annotations

import math
import os

import numpy as np
from numba import njit, prange


@njit(nogil=True, cache=True)
def _radix_pass_8bit(src: np.ndarray, dst: np.ndarray, cnt: np.ndarray, shift: int, n: int) -> None:
    """One stable LSD counting-sort pass of 8 bits at ``shift``: scatter ``src`` into ``dst`` (``cnt`` is a 256-slot int64 scratch)."""
    cnt[:] = 0
    for i in range(n):
        cnt[(src[i] >> shift) & 0xFF] += 1
    s = 0
    for b in range(256):
        c = cnt[b]
        cnt[b] = s
        s += c
    for i in range(n):
        v = src[i]
        b = (v >> shift) & 0xFF
        dst[cnt[b]] = v
        cnt[b] += 1


@njit(nogil=True, cache=True)
def _radix_sort_f32(col: np.ndarray, keys: np.ndarray, tmp: np.ndarray, cnt: np.ndarray) -> None:
    """Sort the NaN-free float32 vector ``col`` in place (ascending). ``keys``/``tmp`` are uint32 scratch of ``col``'s length, ``cnt`` an
    int64 scratch of 256. The float bits are mapped to order-preserving unsigned keys (flip all bits for negatives, set the sign bit for
    non-negatives), sorted by four LSD passes of 8 bits (an even count, so the result is back in ``keys``), and mapped back."""
    n = col.shape[0]
    u = col.view(np.uint32)
    for i in range(n):
        x = u[i]
        if x & 0x80000000:
            keys[i] = ~x
        else:
            keys[i] = x | 0x80000000
    for shift in range(0, 32, 16):
        _radix_pass_8bit(keys, tmp, cnt, shift, n)
        _radix_pass_8bit(tmp, keys, cnt, shift + 8, n)
    for i in range(n):
        x = keys[i]
        if x & 0x80000000:
            u[i] = x & 0x7FFFFFFF
        else:
            u[i] = ~x


@njit(nogil=True, cache=True)
def _unique_counts_f32_njit(x: np.ndarray):
    """``(values, counts)`` of the NaN-free float32 vector ``x`` as ``np.unique(x, return_counts=True)`` gives them: one radix sort of a copy and one pass over the sorted run.

    ``nogil`` and allocation-local, so the column pool of the adaptive binner runs it from threads. The values are the same numbers; the sign of an exact zero is the one radix order puts first (-0.0), which compares equal to +0.0.
    """
    n = x.shape[0]
    col = x.copy()
    keys = np.empty(n, dtype=np.uint32)
    tmp = np.empty(n, dtype=np.uint32)
    cnt = np.empty(256, dtype=np.int64)
    _radix_sort_f32(col, keys, tmp, cnt)
    m = 1
    for i in range(1, n):
        if col[i] != col[i - 1]:
            m += 1
    vals = np.empty(m, dtype=col.dtype)
    counts = np.empty(m, dtype=np.int64)
    j = 0
    vals[0] = col[0]
    counts[0] = 1
    for i in range(1, n):
        if col[i] != col[i - 1]:
            j += 1
            vals[j] = col[i]
            counts[j] = 1
        else:
            counts[j] += 1
    return vals, counts


# Below this size ``np.unique`` is as fast as the sort-and-scan (and the JIT dispatch is not worth it): measured 6.8 against 1.1 ms for 100000 continuous float32 values.
_UNIQUE_COUNTS_MIN_SIZE = 4096


@njit(nogil=True, cache=True)
def _f64_order_keys(x: np.ndarray, keys: np.ndarray) -> None:
    """Order-preserving unsigned 64-bit keys of the NaN-free double-precision vector ``x`` (flip all bits of a negative, set the sign bit of a non-negative)."""
    u = x.view(np.uint64)
    sign = np.uint64(0x8000000000000000)
    for i in range(x.shape[0]):
        v = u[i]
        if v & sign:
            keys[i] = ~v
        else:
            keys[i] = v | sign


@njit(nogil=True, cache=True)
def _radix_argsort_f64(x: np.ndarray):
    """``(sorted keys, order)`` of the NaN-free double-precision vector ``x``: the stable ascending order by eight LSD passes of 8 bits over the order-preserving keys (an even pass count, so the
    result is back in the first buffers). Equal values keep their input order; ``-0.0`` sorts before ``+0.0`` (they compare equal as values)."""
    n = x.shape[0]
    keys = np.empty(n, dtype=np.uint64)
    k2 = np.empty(n, dtype=np.uint64)
    idx = np.empty(n, dtype=np.int64)
    i2 = np.empty(n, dtype=np.int64)
    cnt = np.empty(256, dtype=np.int64)
    _f64_order_keys(x, keys)
    for i in range(n):
        idx[i] = i
    mask = np.uint64(0xFF)
    for pass_no in range(8):
        shift = np.uint64(8 * pass_no)
        cnt[:] = 0
        for i in range(n):
            cnt[(keys[i] >> shift) & mask] += 1
        s = 0
        for b in range(256):
            c = cnt[b]
            cnt[b] = s
            s += c
        for i in range(n):
            k = keys[i]
            b = (k >> shift) & mask
            p = cnt[b]
            k2[p] = k
            i2[p] = idx[i]
            cnt[b] = p + 1
        keys, k2 = k2, keys
        idx, i2 = i2, idx
    return keys, idx


@njit(nogil=True, cache=True)
def _unique_counts_f64_njit(x: np.ndarray):
    """``(values, counts)`` of the NaN-free double-precision vector ``x`` as ``np.unique(x, return_counts=True)`` gives them: a keys-only radix sort of a copy (eight passes of 8 bits, no index carried) and one
    pass over the sorted run (the sign of an exact zero is the one thing the radix order may put differently)."""
    n = x.shape[0]
    keys = np.empty(n, dtype=np.uint64)
    tmp = np.empty(n, dtype=np.uint64)
    cnt = np.empty(256, dtype=np.int64)
    _f64_order_keys(x, keys)
    for pass_no in range(4):
        _radix_pass_8bit(keys, tmp, cnt, np.uint64(16 * pass_no), n)
        _radix_pass_8bit(tmp, keys, cnt, np.uint64(16 * pass_no + 8), n)
    sign = np.uint64(0x8000000000000000)
    srt = np.empty(n, dtype=np.uint64)  # the sorted values' bit patterns, decoded from the keys
    for i in range(n):
        k = keys[i]
        if k & sign:
            srt[i] = k ^ sign
        else:
            srt[i] = ~k
    col = srt.view(np.float64)  # Intentional float64: the sort keys are reinterpreted as double bits, not narrowed
    m = 1
    for i in range(1, n):
        if col[i] != col[i - 1]:
            m += 1
    vals = np.empty(m, dtype=x.dtype)
    counts = np.empty(m, dtype=np.int64)
    j = 0
    vals[0] = col[0]
    counts[0] = 1
    for i in range(1, n):
        if col[i] != col[i - 1]:
            j += 1
            vals[j] = col[i]
            counts[j] = 1
        else:
            counts[j] += 1
    return vals, counts


@njit(nogil=True, cache=True)
def _sort_xy_f64_njit(x: np.ndarray, y: np.ndarray):
    """``(x[order], y[order])`` for the stable ascending order of the NaN-free double-precision vector ``x`` (the double-precision twin of :func:`_sort_xy_f32_njit`)."""
    n = x.shape[0]
    _, idx = _radix_argsort_f64(x)
    xs = np.empty(n, dtype=np.float64)  # Intentional float64: the MDLP cut search runs at the production binning width
    ys = np.empty(n, dtype=y.dtype)
    for i in range(n):
        j = idx[i]
        xs[i] = x[j]
        ys[i] = y[j]
    return xs, ys


@njit(nogil=True, cache=True)
def _sorted_runs_njit(xs: np.ndarray):
    """``(distinct values, their counts)`` of a non-decreasing vector in one pass: what ``np.unique(xs, return_counts=True)`` returns for a sorted input."""
    n = xs.shape[0]
    vals = np.empty(n, dtype=xs.dtype)
    cnts = np.empty(n, dtype=np.int64)
    m = 0
    i = 0
    while i < n:
        j = i + 1
        while j < n and xs[j] == xs[i]:
            j += 1
        vals[m] = xs[i]
        cnts[m] = j - i
        m += 1
        i = j
    return vals[:m].copy(), cnts[:m].copy()


def sorted_unique_with_counts(xs: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """:func:`unique_with_counts` of a vector that is already sorted ascending and NaN-free (the sorted column an MDLP search holds): one linear pass instead of a second radix sort."""
    runs: "tuple[np.ndarray, np.ndarray]" = _sorted_runs_njit(np.ascontiguousarray(xs))
    return runs


def unique_with_counts(values: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """``np.unique(values, return_counts=True)``: the radix sort-and-scan for a large NaN-free float32 vector (6x faster), numpy for anything else."""
    if values.dtype == np.float32 and values.ndim == 1 and values.size >= _UNIQUE_COUNTS_MIN_SIZE:
        radix: "tuple[np.ndarray, np.ndarray]" = _unique_counts_f32_njit(np.ascontiguousarray(values))
        return radix
    if values.dtype == np.float64 and values.ndim == 1 and values.size >= _UNIQUE_COUNTS_MIN_SIZE and not np.isnan(values).any():  # Intentional float64: the engineered-column dtype
        radix64: "tuple[np.ndarray, np.ndarray]" = _unique_counts_f64_njit(np.ascontiguousarray(values))
        return radix64
    return np.unique(values, return_counts=True)


@njit(nogil=True, cache=True)
def _sorted_quantiles_f32_njit(x: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """``np.quantile(x.astype(double precision), quantiles)`` (method ``linear``) of a NaN-free float32 vector from one radix sort of a copy.

    The virtual index is ``(n - 1) * q``, the interpolation is numpy's ``_lerp`` in double precision (``a + (b - a) * t``, and ``b - (b - a) * (1 - t)`` from ``t >= 0.5``), so the double precision results are the ones numpy gives bit for bit.
    """
    n = x.shape[0]
    col = x.copy()
    keys = np.empty(n, dtype=np.uint32)
    tmp = np.empty(n, dtype=np.uint32)
    cnt = np.empty(256, dtype=np.int64)
    _radix_sort_f32(col, keys, tmp, cnt)
    out = np.empty(quantiles.shape[0], dtype=np.float64)  # Intentional float64: double-precision accumulator / output buffer: sums and statistics keep full precision
    for qi in range(quantiles.shape[0]):
        v = quantiles[qi] * (n - 1)
        lo = math.floor(v)
        if lo >= n - 1:
            out[qi] = np.float64(col[n - 1])  # Intentional float64: the interpolation runs in double, as numpy's on the widened column
        else:
            a = np.float64(col[lo])  # Intentional float64: operands widened exactly before the interpolation
            b = np.float64(col[lo + 1])  # Intentional float64: operands widened exactly before the interpolation
            t = v - lo
            diff = b - a
            if t >= 0.5:
                out[qi] = b - diff * (1.0 - t)
            else:
                out[qi] = a + diff * t
    return out


@njit(nogil=True, cache=True)
def _quantiles_of_sorted_njit(xs: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """:func:`_sorted_quantiles_f32_njit` for a vector that is already sorted ascending (any float width): numpy's linear quantile with its double precision ``_lerp``, no sort."""
    n = xs.shape[0]
    out = np.empty(quantiles.shape[0], dtype=np.float64)  # Intentional float64: double-precision accumulator / output buffer: sums and statistics keep full precision
    for qi in range(quantiles.shape[0]):
        v = quantiles[qi] * (n - 1)
        lo = math.floor(v)
        if lo >= n - 1:
            out[qi] = np.float64(xs[n - 1])  # Intentional float64: the interpolation runs in double, as numpy's
        else:
            a = np.float64(xs[lo])  # Intentional float64: operands widened exactly before the interpolation
            b = np.float64(xs[lo + 1])  # Intentional float64: operands widened exactly before the interpolation
            t = v - lo
            diff = b - a
            if t >= 0.5:
                out[qi] = b - diff * (1.0 - t)
            else:
                out[qi] = a + diff * t
    return out


#: Vectors shorter than this are sorted by numpy (the four radix passes and their scratch cost more than its sort there).
_RADIX_SORT_MIN_SIZE = 4096


def radix_sorted_f32(values: np.ndarray) -> np.ndarray:
    """A sorted copy of a NaN-free float32 vector: the LSD radix sort of :func:`_radix_sort_f32` for a long one (4.4 ms for 500000 values against numpy's 36 ms), numpy's sort for a short one.

    The order is numpy's except that -0.0 and +0.0, which compare equal, may stand in either order."""
    col = np.array(values, dtype=np.float32, copy=True, order="C").reshape(-1)
    n = col.shape[0]
    if n < _RADIX_SORT_MIN_SIZE:
        col.sort()
        return col
    _radix_sort_f32(col, np.empty(n, dtype=np.uint32), np.empty(n, dtype=np.uint32), np.empty(256, dtype=np.int64))
    return col


def quantiles_of_sorted(xs: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Float64 quantiles (fractions in ``[0, 1]``) of a NaN-free vector already sorted ascending, as ``np.quantile`` gives them (method ``linear``)."""
    found: np.ndarray = _quantiles_of_sorted_njit(np.ascontiguousarray(xs), np.ascontiguousarray(quantiles, dtype=np.float64))  # Intentional float64: statistic / reference vector in double precision
    return found


def sorted_quantiles(values: np.ndarray, quantiles: np.ndarray) -> "np.ndarray | None":
    """Float64 quantiles (fractions in ``[0, 1]``) of a large NaN-free float32 vector by the radix route, or ``None`` when it does not apply (another dtype, a small vector): the caller then uses numpy.

    A double-precision vector whose values are all exactly float32 (a float32 column widened, the form the edge search reads its columns in) takes the same route on its narrowed copy: the order statistics
    and their interpolation are those of the widened values, so the edges are the ones ``np.percentile`` gives (about 15 ms of partition for 300000 values against 3 ms of radix sort)."""
    if values.ndim != 1 or values.size < _UNIQUE_COUNTS_MIN_SIZE:
        return None
    q64 = np.ascontiguousarray(quantiles, dtype=np.float64)  # Intentional float64: statistic / reference vector in double precision (compared exactly with its host twin)
    if values.dtype == np.float32:
        found: np.ndarray = _sorted_quantiles_f32_njit(np.ascontiguousarray(values), q64)
        return found
    if values.dtype == np.float64:  # Intentional float64: the test of the incoming vector's dtype, no double array is made
        exact, narrow = _f64_as_f32_njit(np.ascontiguousarray(values))
        if exact:
            widened: np.ndarray = _sorted_quantiles_f32_njit(narrow, q64)
            return widened
    return None


@njit(nogil=True, cache=True)
def _run_of_rank(cum: np.ndarray, rank: int) -> int:
    """Index of the run holding the 0-based sorted position ``rank``: the first run whose cumulative count exceeds it."""
    lo = 0
    hi = cum.shape[0] - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cum[mid] > rank:
            hi = mid
        else:
            lo = mid + 1
    return lo


@njit(nogil=True, cache=True)
def _quantiles_from_runs_njit(uniq: np.ndarray, counts: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """``np.quantile`` (method ``linear``) of the vector described by its ascending distinct values and their counts, in the arithmetic order of ``_edge_of_sorted``: the order statistics are read off the cumulative counts, no sort."""
    m = uniq.shape[0]
    cum = np.empty(m, dtype=np.int64)
    total = 0
    for j in range(m):
        total += counts[j]
        cum[j] = total
    out = np.empty(quantiles.shape[0], dtype=np.float64)  # Intentional float64: the edges are doubles
    for k in range(quantiles.shape[0]):
        virtual = (total - 1) * quantiles[k]
        if virtual >= total - 1:
            out[k] = np.float64(uniq[m - 1])  # Intentional float64: the order statistic widened exactly, the interpolation is done in double precision
            continue
        i0 = 0 if virtual < 0 else int(np.floor(virtual))
        i1 = i0 if virtual < 0 else i0 + 1
        gamma = virtual - i0
        a = np.float64(uniq[_run_of_rank(cum, i0)])  # Intentional float64: operands widened exactly before the interpolation (a float32 difference would round)
        b = np.float64(uniq[_run_of_rank(cum, i1)])  # Intentional float64: operands widened exactly before the interpolation
        diff = b - a
        res = a + diff * gamma
        if gamma >= 0.5:
            res = b - diff * (1.0 - gamma)
        out[k] = res
    return out


def quantiles_from_runs(uniq: np.ndarray, counts: np.ndarray, quantiles: np.ndarray) -> "np.ndarray | None":
    """Double-precision quantiles (fractions in ``[0, 1]``) of the NaN-free vector whose ascending distinct values are ``uniq`` with ``counts``: the order statistics come off the cumulative counts, so a column whose
    distinct values and counts are known (the edge search computes them for every column) is not sorted a second time. ``None`` when there is nothing to read (no runs, or counts that do not add up)."""
    if uniq.ndim != 1 or uniq.shape[0] == 0 or uniq.shape[0] != counts.shape[0] or uniq.dtype.kind != "f":
        return None
    q64 = np.ascontiguousarray(quantiles, dtype=np.float64)  # Intentional float64: statistic / reference vector in double precision
    found: np.ndarray = _quantiles_from_runs_njit(np.ascontiguousarray(uniq), np.ascontiguousarray(counts, dtype=np.int64), q64)
    return found


@njit(nogil=True, cache=True)
def _sort_xy_f32_njit(x: np.ndarray, y: np.ndarray):
    """``(x[order].astype(double precision), y[order])`` for the stable ascending order of the NaN-free float32 vector ``x``: a radix argsort (the order-preserving key of each float with its index riding along, four 8-bit LSD passes) and one gather of both arrays.

    Equal ``x`` keep their input order (``np.argsort``'s quicksort leaves them in an unspecified one); the sorted ``x`` is identical, and so are the per-value label counts that the MDLP split search reads.
    """
    n = x.shape[0]
    u = x.view(np.uint32)
    keys = np.empty(n, dtype=np.uint32)
    idx = np.empty(n, dtype=np.int64)
    k2 = np.empty(n, dtype=np.uint32)
    i2 = np.empty(n, dtype=np.int64)
    cnt = np.empty(256, dtype=np.int64)
    for i in range(n):
        v = u[i]
        if v & 0x80000000:
            keys[i] = ~v
        else:
            keys[i] = v | 0x80000000
        idx[i] = i
    for shift in (0, 8, 16, 24):
        cnt[:] = 0
        for i in range(n):
            cnt[(keys[i] >> shift) & 0xFF] += 1
        s = 0
        for b in range(256):
            c = cnt[b]
            cnt[b] = s
            s += c
        for i in range(n):
            k = keys[i]
            b = (k >> shift) & 0xFF
            p = cnt[b]
            k2[p] = k
            i2[p] = idx[i]
            cnt[b] = p + 1
        keys, k2 = k2, keys
        idx, i2 = i2, idx
    xs = np.empty(n, dtype=np.float64)  # Intentional float64: the MDLP cut search runs at the production binning width
    ys = np.empty(n, dtype=y.dtype)
    for i in range(n):
        j = idx[i]
        xs[i] = x[j]
        ys[i] = y[j]
    return xs, ys


_KEY_SIGN = np.uint32(0x80000000)
_KEY_LOW = np.uint32(0x7FFFFFFF)


@njit(nogil=True, cache=True)
def _sort_xy_pairs_f32_njit(x: np.ndarray, y: np.ndarray):
    """``(x sorted ascending as double precision, y in that order)`` for the NaN-free float32 vector ``x``: the same stable order as :func:`_sort_xy_f32_njit`, with the label itself as the radix payload.

    The index-carrying sort moves 12 bytes an element through each of four passes and then gathers ``x`` and ``y`` at random; here the payload is the label (one byte for the usual int8 target), the four
    digit histograms are taken in the one pass that builds the keys, and the sorted values are rebuilt from the keys (the key transform is a bijection, so the sign of a zero survives). Eight threads sorting
    500000-row columns spent 55 thread-ms a sort with the index kernel on float32 and 102 on double precision, 28 with this one."""
    n = x.shape[0]
    u = x.view(np.uint32)
    keys = np.empty(n, dtype=np.uint32)
    k2 = np.empty(n, dtype=np.uint32)
    p1 = y.copy()
    p2 = np.empty_like(y)
    cnt = np.zeros((4, 256), dtype=np.int64)
    for i in range(n):
        v = u[i]
        if v & _KEY_SIGN:
            k = ~v
        else:
            k = v | _KEY_SIGN
        keys[i] = k
        cnt[0, k & 0xFF] += 1
        cnt[1, (k >> 8) & 0xFF] += 1
        cnt[2, (k >> 16) & 0xFF] += 1
        cnt[3, (k >> 24) & 0xFF] += 1
    for d in range(4):
        s = 0
        for b in range(256):
            c = cnt[d, b]
            cnt[d, b] = s
            s += c
    for d in range(4):
        shift = np.uint32(8 * d)
        for i in range(n):
            k = keys[i]
            b = (k >> shift) & 0xFF
            pos = cnt[d, b]
            k2[pos] = k
            p2[pos] = p1[i]
            cnt[d, b] = pos + 1
        keys, k2 = k2, keys
        p1, p2 = p2, p1
    bits = np.empty(n, dtype=np.uint32)
    for i in range(n):
        k = keys[i]
        if k & _KEY_SIGN:
            bits[i] = k & _KEY_LOW
        else:
            bits[i] = ~k
    f32 = bits.view(np.float32)
    xs = np.empty(n, dtype=np.float64)  # Intentional float64: the MDLP cut search runs at the production binning width
    for i in range(n):
        xs[i] = f32[i]
    return xs, p1


@njit(nogil=True, cache=True)
def _f64_as_f32_njit(x: np.ndarray):
    """``(True, x as float32)`` when every value of the double-precision ``x`` is exactly a float32 (a float32 column widened), else ``(False, an empty vector)``; stops at the first value that is not."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        v = np.float32(x[i])
        if np.float64(v) != x[i]:  # Intentional float64: the comparison is against the widened float32
            return False, out[:0]
        out[i] = v
    return True, out


@njit(nogil=True, cache=True)
def _edge_of_sorted(s: np.ndarray, q: float) -> np.float64:  # Intentional float64: the edge is a double
    """``np.quantile(s, q)`` (method ``linear``) of the ascending vector ``s``, in numpy's arithmetic order (the interpolation of ``_edge_value``)."""
    n = s.shape[0]
    virtual = (n - 1) * q
    if virtual >= n - 1:
        return np.float64(s[n - 1])  # Intentional float64: the edges are computed in double precision from the exactly widened order statistics
    i0 = 0 if virtual < 0 else int(np.floor(virtual))
    i1 = i0 if virtual < 0 else i0 + 1
    gamma = virtual - i0
    a = np.float64(s[i0])  # Intentional float64: operands widened exactly before the interpolation
    b = np.float64(s[i1])  # Intentional float64: operands widened exactly before the interpolation
    diff = b - a
    out = a + diff * gamma
    if gamma >= 0.5:
        out = b - diff * (1.0 - gamma)
    return np.float64(out)  # Intentional float64: the edge is a double


@njit(parallel=True, nogil=True, cache=True)
def quantile_bin_block_radix(v: np.ndarray, n_bins: int, codes: np.ndarray, two_class: bool = True) -> None:
    """:func:`quantile_bin_block` for a finite float32 block ``v`` ``(columns, n)``: the same codes, with the quantile edges read off a radix-sorted copy of the column instead of a quickselect.

    The quickselect of the roughly twenty order statistics of the edges took 22 ms for 500000 values single-threaded (64 columns: 1.44 s on one thread, 0.30 s on eight); the LSD radix sort of the column takes 4.4 ms
    and puts the same value at every position, so the edges (numpy's linear-interpolation quantiles with the repeated ones dropped) and therefore the codes are the same (a zero of either sign compares equal);
    the binning pass counts the inner edges below the value instead of searching for them."""
    k = v.shape[0]
    n = v.shape[1]
    step = 1.0 / n_bins
    for j in prange(k):
        s = v[j].copy()
        keys = np.empty(n, dtype=np.uint32)
        tmp = np.empty(n, dtype=np.uint32)
        cnt = np.empty(256, dtype=np.int64)
        _radix_sort_f32(s, keys, tmp, cnt)
        edges = np.empty(n_bins + 1, dtype=np.float64)  # Intentional float64: quantile edges in numpy's arithmetic (equal codes to the numpy path)
        m = 0
        for b in range(n_bins + 1):
            q = 1.0 if b == n_bins else b * step
            e = _edge_of_sorted(s, q)
            if m == 0 or e != edges[m - 1]:
                edges[m] = e
                m += 1
        if m <= 2:
            for i in range(n):
                codes[j, i] = 1 if (two_class and m == 2 and v[j, i] >= edges[1]) else 0
            continue
        # the code is the number of inner edges that are <= the value (the edges are distinct and ascending): a branch-free count over the handful of edges is 1.8 ms for 500000 values, the binary
        # search of the quickselect kernel 8.3 ms (its branches miss on random data)
        inner = edges[1 : m - 1].copy()
        n_inner = m - 2
        for i in range(n):
            x = v[j, i]
            c = 0
            for t in range(n_inner):
                c += 1 if inner[t] <= x else 0
            codes[j, i] = c


def quantile_bin_block_fast(v: np.ndarray, n_bins: int, codes: np.ndarray, two_class: bool = True) -> None:
    """:func:`quantile_bin_block` through the radix route for a float32 block of at least ``_RADIX_SORT_MIN_SIZE`` values a column (``MLFRAME_QBIN_RADIX=0`` keeps the quickselect kernel), the same codes."""
    if v.dtype == np.float32 and v.ndim == 2 and v.shape[1] >= _RADIX_SORT_MIN_SIZE and os.environ.get("MLFRAME_QBIN_RADIX", "1").strip() != "0":
        quantile_bin_block_radix(v, n_bins, codes, two_class)
        return
    from .._numba_utils import quantile_bin_block

    quantile_bin_block(v, n_bins, codes, two_class)


def _pairs_route_enabled() -> bool:
    """On unless ``MLFRAME_SORT_XY_PAIRS=0`` (the index-carrying kernels then sort, as before)."""
    return os.environ.get("MLFRAME_SORT_XY_PAIRS", "1").strip() != "0"


def sort_xy(x: np.ndarray, y: np.ndarray) -> "tuple[np.ndarray, np.ndarray] | None":
    """``(x sorted ascending as double precision, y in that order)`` by the radix route for a large NaN-free float32 ``x`` with a 1-D integer ``y`` of the same length, else ``None`` (the caller sorts with numpy)."""
    if x.dtype == np.float32 and x.ndim == 1 and x.size >= _UNIQUE_COUNTS_MIN_SIZE and y.ndim == 1 and y.shape[0] == x.shape[0] and y.dtype.kind in "iu":
        if _pairs_route_enabled():
            pairs: "tuple[np.ndarray, np.ndarray]" = _sort_xy_pairs_f32_njit(np.ascontiguousarray(x), np.ascontiguousarray(y))
            return pairs
        pair: "tuple[np.ndarray, np.ndarray]" = _sort_xy_f32_njit(np.ascontiguousarray(x), np.ascontiguousarray(y))
        return pair
    if x.dtype == np.float64 and x.ndim == 1 and x.size >= _UNIQUE_COUNTS_MIN_SIZE and y.ndim == 1 and y.shape[0] == x.shape[0] and y.dtype.kind in "iu" and not np.isnan(x).any():  # Intentional float64: the engineered-column dtype
        x64 = np.ascontiguousarray(x)
        if _pairs_route_enabled():
            exact, x32 = _f64_as_f32_njit(x64)  # a float32 column held in double precision (the usual FE frame) sorts on 4-byte keys: the same order and the same widened values
            if exact:
                narrow: "tuple[np.ndarray, np.ndarray]" = _sort_xy_pairs_f32_njit(x32, np.ascontiguousarray(y))
                return narrow
        pair64: "tuple[np.ndarray, np.ndarray]" = _sort_xy_f64_njit(x64, np.ascontiguousarray(y))
        return pair64
    return None


@njit(parallel=True, nogil=True, cache=True)
def _quantile_edges_2d_radix_njit(arr2d: np.ndarray, quantiles: np.ndarray, edges_out: np.ndarray) -> None:
    """Per-column quantile edges of a NaN-free float32 ``arr2d`` (n_rows x n_cols), written into ``edges_out`` (n_q x n_cols, double).

    Same interpolation as :func:`_kernels._quantile_edges_2d_njit` (the ``lo``/``lo + 1`` order statistics with the ``n - 1`` clamp and the
    same lerp), with the order statistics read from a full radix sort of each column. prange over columns: each iteration owns its scratch.
    """
    n_rows = arr2d.shape[0]
    n_cols = arr2d.shape[1]
    n_q = quantiles.shape[0]
    if n_rows == 0 or n_cols == 0:
        return
    for j in prange(n_cols):
        col = np.empty(n_rows, dtype=np.float32)
        for r in range(n_rows):
            col[r] = arr2d[r, j]
        keys = np.empty(n_rows, dtype=np.uint32)
        tmp = np.empty(n_rows, dtype=np.uint32)
        cnt = np.empty(256, dtype=np.int64)
        _radix_sort_f32(col, keys, tmp, cnt)
        for qi in range(n_q):
            v = (quantiles[qi] / 100.0) * (n_rows - 1)
            lo = math.floor(v)
            if lo >= n_rows - 1:
                edges_out[qi, j] = col[n_rows - 1]
            else:
                a = float(col[lo])
                b = float(col[lo + 1])
                t = v - lo
                diff_b_a = b - a
                if t >= 0.5:
                    edges_out[qi, j] = b - diff_b_a * (1.0 - t)
                else:
                    edges_out[qi, j] = a + diff_b_a * t


_TILE_COLS = 64


def tile_cols_for(n_cols: int, threads: int) -> int:
    """Tile width for ``n_cols`` columns on ``threads`` threads: ``_TILE_COLS`` unless that leaves threads without a tile, then ``ceil(n_cols / threads)``, never under 16 (one 64-byte line of float32 per row read)."""
    return int(max(16, min(_TILE_COLS, -(-int(n_cols) // max(1, int(threads))))))


@njit(parallel=True, nogil=True, cache=True)
def _quantile_edges_2d_radix_tiled_njit(arr2d: np.ndarray, quantiles: np.ndarray, edges_out: np.ndarray, tile_cols: int = _TILE_COLS) -> None:
    """Tiled twin of :func:`_quantile_edges_2d_radix_njit`: BYTE-IDENTICAL edges. Each prange tile gathers ``_TILE_COLS`` adjacent columns
    with row-contiguous reads (``arr2d[r, j0:j1]`` is one cache line run), then radix-sorts each column of the tile in place. The
    per-column gather from the C-ordered buffer otherwise costs one cache miss per element.

    ``tile_cols`` is the tile width (columns are independent, so it changes the schedule and never an edge): see :func:`tile_cols_for`, which narrows it so that every thread gets a tile."""
    n_rows = arr2d.shape[0]
    n_cols = arr2d.shape[1]
    n_q = quantiles.shape[0]
    if n_rows == 0 or n_cols == 0:
        return
    n_tiles = (n_cols + tile_cols - 1) // tile_cols
    for t in prange(n_tiles):
        j0 = t * tile_cols
        j1 = min(n_cols, j0 + tile_cols)
        w = j1 - j0
        tile = np.empty((w, n_rows), dtype=np.float32)
        for r in range(n_rows):
            for jj in range(w):
                tile[jj, r] = arr2d[r, j0 + jj]
        keys = np.empty(n_rows, dtype=np.uint32)
        tmp = np.empty(n_rows, dtype=np.uint32)
        cnt = np.empty(256, dtype=np.int64)
        for jj in range(w):
            col = tile[jj]
            _radix_sort_f32(col, keys, tmp, cnt)
            j = j0 + jj
            for qi in range(n_q):
                v = (quantiles[qi] / 100.0) * (n_rows - 1)
                lo = math.floor(v)
                if lo >= n_rows - 1:
                    edges_out[qi, j] = col[n_rows - 1]
                else:
                    a = float(col[lo])
                    b = float(col[lo + 1])
                    tt = v - lo
                    diff_b_a = b - a
                    if tt >= 0.5:
                        edges_out[qi, j] = b - diff_b_a * (1.0 - tt)
                    else:
                        edges_out[qi, j] = a + diff_b_a * tt


__all__ = ["_quantile_edges_2d_radix_njit", "_quantile_edges_2d_radix_tiled_njit", "_radix_sort_f32"]
