"""Root-split decisions and label packing for the binary MDLP search, hosted with the ``_numba_utils`` family.

``root_decision_*`` decide the root split of a column from its labels alone, and the ``pack_*`` / ``dedup_*`` helpers build
the bit-packed label streams those decisions read. ``_numba_utils`` re-exports every name here.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ._internals import _entropy_from_counts_njit
from ._numba_utils_cross_module import _analytic_p_value
from .supervised_binning import _xlogx

REJECTED = 0
ACCEPTED = 1
UNDECIDED = -1


@njit(cache=True)
def column_tie_runs(xsrt: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """``(starts, ends)`` of the maximal runs of equal values (two or more points) in the sorted column ``xsrt``; ``ends`` is exclusive."""
    n = xsrt.shape[0]
    n_runs = 0
    i = 0
    while i < n - 1:
        if xsrt[i] == xsrt[i + 1]:
            n_runs += 1
            j = i + 1
            while j < n and xsrt[j] == xsrt[i]:
                j += 1
            i = j
        else:
            i += 1
    starts = np.empty(n_runs, dtype=np.int64)
    ends = np.empty(n_runs, dtype=np.int64)
    r = 0
    i = 0
    while i < n - 1:
        if xsrt[i] == xsrt[i + 1]:
            j = i + 1
            while j < n and xsrt[j] == xsrt[i]:
                j += 1
            starts[r] = i
            ends[r] = j
            r += 1
            i = j
        else:
            i += 1
    return starts, ends


@njit(cache=True)
def _dedup_binary_stream(ys: np.ndarray, run_start: np.ndarray, run_end: np.ndarray, yd: np.ndarray, forb: np.ndarray) -> "tuple[int, int]":
    """Write the labels of the deduplicated points into ``yd`` and return ``(points, forbidden)``.

    Within a run of tied values the distinct labels are emitted ascending, as ``_presorted_dedup`` does. ``forb[:forbidden]`` holds the stream positions ``i`` whose boundary to ``i + 1`` lies inside
    a run (the ``0`` and ``1`` point of one value): equal values never bound a candidate cut."""
    m = 0
    nf = 0
    p = 0
    for r in range(run_start.shape[0]):
        a = run_start[r]
        e = run_end[r]
        yd[m : m + (a - p)] = ys[p:a]
        m += a - p
        ones = 0
        for i in range(a, e):
            ones += ys[i]
        zeros = (e - a) - ones
        if zeros > 0:
            yd[m] = 0
            m += 1
        if ones > 0:
            if zeros > 0:
                forb[nf] = m - 1
                nf += 1
            yd[m] = 1
            m += 1
        p = e
    n = ys.shape[0]
    yd[m : m + (n - p)] = ys[p:n]
    m += n - p
    return m, nf


@njit(cache=True)
def _tail_threshold(p_star: float) -> float:
    """The largest ``x`` (to bisection accuracy) with ``erfc(sqrt(x)) > p_star``, the chi-square(1) tail, so the analytic test of a gain ``g`` passes only if ``n * g`` exceeds it."""
    if p_star >= 1.0:
        return 0.0
    lo = 0.0
    hi = 1.0
    while math.erfc(math.sqrt(hi)) > p_star and hi < 1e7:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if math.erfc(math.sqrt(mid)) > p_star:
            lo = mid
        else:
            hi = mid
    return lo


@njit(cache=True)
def _count_stream_candidates(yd: np.ndarray, forb: np.ndarray, n_forb: int, lo: int, hi: int) -> int:
    """Candidate cut positions in ``lo..hi``: label changes of the stream, minus the boundaries that lie inside a run of equal values."""
    n_cand = 0
    for i in range(lo, hi + 1):
        if yd[i] != yd[i + 1]:
            n_cand += 1
    for k in range(n_forb):
        if lo <= forb[k] <= hi:
            n_cand -= 1
    return n_cand


@njit(cache=True)
def _best_gain_over_bound(yd: np.ndarray, forb: np.ndarray, n_forb: int, lo: int, hi: int, n: int, tot1: int, h_full: float, scale: float) -> float:
    """Largest exact split gain among the candidates whose bound ``ln(1 + phi^2)`` reaches the acceptance threshold (``-1e300`` when none does).

    A candidate's gain is the mutual information of its 2 x 2 table, bounded by ln(1 + phi^2) with phi^2 = d^2 / (n_l n_r n_1 n_0), d = c1l * n - n_l * n_1 (KL <= ln(1 + chi^2)); ``scale`` is
    phi_min^2 * n_1 * n_0, so a candidate is skipped when d^2 < scale * n_l * n_r."""
    c1l = 0
    for i in range(lo):
        c1l += yd[i]
    # Usual outcome on a permuted label: no position reaches the bound. A branch-light scan over all positions (a superset of the candidates) settles that before the per-candidate work.
    run1 = c1l
    reached = False
    fn = float(n)
    ft1 = float(tot1)
    for i in range(lo, hi + 1):
        run1 += yd[i]
        fl = float(i + 1)
        dd = float(run1) * fn - fl * ft1
        if dd * dd >= scale * fl * (fn - fl):
            reached = True
            break
    if not reached:
        return -1e300
    best_gain = -1e300
    inv_n = 1.0 / n
    fb = 0
    for i in range(lo, hi + 1):
        y_i = yd[i]
        c1l += y_i
        n_l = i + 1
        n_r = n - n_l
        d = float(c1l * n - n_l * tot1)
        if d * d < scale * float(n_l) * float(n_r):
            continue
        if y_i == yd[i + 1]:
            continue
        while fb < n_forb and forb[fb] < i:
            fb += 1
        if fb < n_forb and forb[fb] == i:
            continue
        c0l = n_l - c1l
        c1r = tot1 - c1l
        c0r = n_r - c1r
        h_split = (_xlogx(n_l) - _xlogx(c0l) - _xlogx(c1l) + _xlogx(n_r) - _xlogx(c0r) - _xlogx(c1r)) * inv_n
        gain = h_full - h_split
        if gain > best_gain:
            best_gain = gain
    return best_gain


@njit(cache=True)
def root_decision_binary(
    ys: np.ndarray,
    run_start: np.ndarray,
    run_end: np.ndarray,
    yd: np.ndarray,
    forb: np.ndarray,
    min_split_size: int,
    alpha: float,
    analytic_min_n: int,
    analytic_min_cell: float,
) -> int:
    """Whether the root split of the sorted column with 0/1 labels ``ys`` is accepted, as ``_mdlp_dfs`` decides it: ACCEPTED, REJECTED, or UNDECIDED when the root is not judged by the analytic test.

    ``yd`` (int8, ``len(ys)``) and ``forb`` (int64, one slot per tie run) are scratch buffers. The root is accepted when p(best gain) * n_cand < alpha, i.e. when some candidate's gain exceeds the
    threshold g*; only the positions whose bound reaches g* get the exact gain."""
    n, n_forb = _dedup_binary_stream(ys, run_start, run_end, yd, forb)
    if n < 2 * min_split_size:
        return REJECTED
    if not (n >= analytic_min_n and float(n) / max(1.0, 2.0 * 2) >= analytic_min_cell):
        return UNDECIDED
    tot1 = 0
    for i in range(n):
        tot1 += yd[i]
    lo = min_split_size - 1  # first scanned position: n_l = lo + 1 >= min_split_size
    hi = n - min_split_size - 1  # last scanned position: n_r = n - hi - 1 >= min_split_size
    n_cand = _count_stream_candidates(yd, forb, n_forb, lo, hi)
    verdict: int = _root_verdict(yd, forb, n_forb, n, tot1, n_cand, lo, hi, alpha)
    return verdict


@njit(cache=True)
def _root_verdict(yd: np.ndarray, forb: np.ndarray, n_forb: int, n: int, tot1: int, n_cand: int, lo: int, hi: int, alpha: float) -> int:
    """ACCEPTED or REJECTED for a stream of ``n`` points with ``tot1`` ones and ``n_cand`` candidate cuts among positions ``lo..hi``."""
    counts_total = np.empty(2, dtype=np.int64)
    counts_total[0] = n - tot1
    counts_total[1] = tot1
    h_full = _entropy_from_counts_njit(counts_total, n)
    if h_full <= 0.0:
        return REJECTED
    x_star = _tail_threshold(alpha / max(1, n_cand)) * (1.0 - 1e-6)
    scale = math.expm1(x_star / n) * float(tot1) * float(n - tot1)
    best_gain = _best_gain_over_bound(yd, forb, n_forb, lo, hi, n, tot1, h_full, scale)
    if best_gain <= 0.0:
        return REJECTED
    p_value = min(1.0, _analytic_p_value(best_gain, n, 2) * max(1, n_cand))
    return ACCEPTED if p_value < alpha else REJECTED


@njit(cache=True)
def gather_binary_stream(yr_b: np.ndarray, order_row: np.ndarray, ys: np.ndarray) -> "tuple[int, int]":
    """``ys[i] = yr_b[order_row[i]]`` in one pass that also counts the ones and the label changes between neighbours (``i`` and ``i + 1``)."""
    ones = 0
    trans = 0
    prev = yr_b[order_row[0]]
    ys[0] = prev
    ones += prev
    for i in range(1, ys.shape[0]):
        v = yr_b[order_row[i]]
        ys[i] = v
        ones += v
        if v != prev:
            trans += 1
        prev = v
    return ones, trans


@njit(cache=True)
def root_decision_tie_free(ys: np.ndarray, ones: int, trans: int, min_split_size: int, alpha: float, analytic_min_n: int, analytic_min_cell: float) -> int:
    """``root_decision_binary`` for a column without tied values, from the label stream ``ys`` and the ``ones`` / ``trans`` counted by ``gather_binary_stream``: no tie run, so the stream is the
    deduplicated one and no boundary is forbidden."""
    n = ys.shape[0]
    if n < 2 * min_split_size:
        return REJECTED
    if not (n >= analytic_min_n and float(n) / max(1.0, 2.0 * 2) >= analytic_min_cell):
        return UNDECIDED
    lo = min_split_size - 1
    hi = n - min_split_size - 1
    n_cand = trans
    for i in range(lo):
        if ys[i] != ys[i + 1]:
            n_cand -= 1
    for i in range(hi + 1, n - 1):
        if ys[i] != ys[i + 1]:
            n_cand -= 1
    verdict: int = _root_verdict(ys, np.empty(0, dtype=np.int64), 0, n, ones, n_cand, lo, hi, alpha)
    return verdict


@njit(cache=True)
def scrub_nonfinite_1d(a: np.ndarray) -> None:
    """Set every NaN and infinity of ``a`` to 0 in place (one pass; ``np.nan_to_num(a, nan=0, posinf=0, neginf=0)``)."""
    for i in range(a.shape[0]):
        v = a[i]
        if not (v - v == 0):  # NaN and +-inf: the difference is not 0
            a[i] = 0


@njit(cache=True)
def scrub_nonfinite_2d(a: np.ndarray) -> None:
    """Set every NaN and infinity of the 2-D ``a`` to 0 in place, row by row."""
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            v = a[i, j]
            if not (v - v == 0):
                a[i, j] = 0


# ---- bit-packed binary label streams: the root test of the honest null on 64 labels per word ----


@njit(cache=True)
def _pc64(x):
    """Population count of a uint64 (SWAR)."""
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=True)
def pack_sorted_labels(yr_b: np.ndarray, order_row: np.ndarray, raw_bits: np.ndarray) -> int:
    """Pack the 0/1 labels read through ``order_row`` (``yr_b[order_row[i]]``, the labels in the column's sorted order) into ``raw_bits`` (bit ``i`` of the stream is bit ``i & 63`` of word ``i >> 6``) and return the
    number of ones. ``raw_bits`` is overwritten, bits past the end of the stream are zero."""
    n = order_row.shape[0]
    words = (n + 63) >> 6
    ones = 0
    for w in range(words):
        base = w << 6
        count = min(64, n - base)
        acc = np.uint64(0)
        for k in range(count):
            acc |= np.uint64(yr_b[order_row[base + k]]) << np.uint64(k)
        raw_bits[w] = acc
        ones += _pc64(acc)
    return ones


@njit(cache=True)
def _copy_bits(src: np.ndarray, start: int, stop: int, dst: np.ndarray, offset: int) -> int:
    """Copy bits ``[start, stop)`` of the stream ``src`` to ``dst`` from bit ``offset`` on (``dst`` bits there must be zero); returns the offset after the copy."""
    s = start
    off = offset
    while s < stop:
        w = s >> 6
        sh = s & 63
        take = min(64 - sh, stop - s)
        val = (src[w] >> np.uint64(sh)) & (np.uint64(0xFFFFFFFFFFFFFFFF) >> np.uint64(64 - take))
        dw = off >> 6
        dsh = off & 63
        dst[dw] |= val << np.uint64(dsh)
        if dsh + take > 64:
            dst[dw + 1] |= val >> np.uint64(64 - dsh)
        s += take
        off += take
    return off


@njit(cache=True)
def dedup_label_bits(raw_bits: np.ndarray, n: int, run_start: np.ndarray, run_end: np.ndarray, out_bits: np.ndarray, forb: np.ndarray) -> "tuple[int, int]":
    """The deduplicated stream (see ``_dedup_binary_stream``) of the packed stream ``raw_bits`` of length ``n``, packed into ``out_bits`` (overwritten): rows outside tie runs are copied, a run of tied values
    emits its distinct labels ascending. Returns ``(points, forbidden)``; ``forb[:forbidden]`` are the stream positions whose boundary to the next point lies inside a run."""
    for w in range(out_bits.shape[0]):
        out_bits[w] = np.uint64(0)
    m = 0
    nf = 0
    p = 0
    for r in range(run_start.shape[0]):
        a = run_start[r]
        e = run_end[r]
        m = _copy_bits(raw_bits, p, a, out_bits, m)
        ones = 0
        for i in range(a, e):
            ones += int((raw_bits[i >> 6] >> np.uint64(i & 63)) & np.uint64(1))
        zeros = (e - a) - ones
        if zeros > 0:
            m += 1  # a 0 point: the bit stays zero
        if ones > 0:
            if zeros > 0:
                forb[nf] = m - 1
                nf += 1
            out_bits[m >> 6] |= np.uint64(1) << np.uint64(m & 63)
            m += 1
        p = e
    m = _copy_bits(raw_bits, p, n, out_bits, m)
    return m, nf


@njit(cache=True)
def _count_ones_bits(bits: np.ndarray, start: int, stop: int) -> int:
    """Ones among stream positions ``[start, stop)``."""
    if stop <= start:
        return 0
    total = 0
    w0 = start >> 6
    w1 = (stop - 1) >> 6
    for w in range(w0, w1 + 1):
        word = bits[w]
        lo_bit = start - (w << 6) if w == w0 else 0
        hi_bit = stop - (w << 6) if w == w1 else 64  # exclusive
        if lo_bit > 0:
            word = word >> np.uint64(lo_bit) << np.uint64(lo_bit)
        if hi_bit < 64:
            word = word & ((np.uint64(1) << np.uint64(hi_bit)) - np.uint64(1))
        total += _pc64(word)
    return total


@njit(cache=True)
def _candidate_count_bits(bits: np.ndarray, forb: np.ndarray, n_forb: int, lo: int, hi: int) -> int:
    """``_count_stream_candidates`` on the packed stream: label changes between positions ``i`` and ``i + 1`` for ``i`` in ``lo..hi``, minus the boundaries that lie inside a run of equal values."""
    n_words = bits.shape[0]
    n_cand = 0
    for w in range(lo >> 6, (hi >> 6) + 1):
        cur = bits[w]
        nxt = bits[w + 1] if w + 1 < n_words else np.uint64(0)
        change = cur ^ ((cur >> np.uint64(1)) | (nxt << np.uint64(63)))
        base = w << 6
        k0 = max(lo - base, 0)
        k1 = min(hi - base, 63)
        if k0 > 0:
            change = change >> np.uint64(k0) << np.uint64(k0)
        if k1 < 63:
            change = change & ((np.uint64(1) << np.uint64(k1 + 1)) - np.uint64(1))
        n_cand += _pc64(change)
    for k in range(n_forb):
        if lo <= forb[k] <= hi:
            n_cand -= 1
    return n_cand


@njit(cache=True)
def root_decision_packed(bits: np.ndarray, m: int, forb: np.ndarray, n_forb: int, min_split_size: int, alpha: float, analytic_min_n: int, analytic_min_cell: float) -> int:
    """``root_decision_binary`` for the packed deduplicated stream ``bits`` of ``m`` points (see ``dedup_label_bits``): the same verdict.

    The candidate count is a popcount of the label changes. The scan for a position whose bound reaches the acceptance threshold skips every word of 64 positions that cannot reach it: from the
    value ``d`` at the start of the word, ``|d|`` moves by at most ``max(n_1, n_0)`` per position, so the word is skipped when even that largest value stays below the smallest threshold in the word.
    A word that may reach it is scanned position by position with the arithmetic of the unpacked scan, and when a position does reach it the unpacked stream goes to ``_best_gain_over_bound``."""
    n = m
    if n < 2 * min_split_size:
        return REJECTED
    if not (n >= analytic_min_n and float(n) / max(1.0, 2.0 * 2) >= analytic_min_cell):
        return UNDECIDED
    tot1 = _count_ones_bits(bits, 0, n)
    counts_total = np.empty(2, dtype=np.int64)
    counts_total[0] = n - tot1
    counts_total[1] = tot1
    h_full = _entropy_from_counts_njit(counts_total, n)
    if h_full <= 0.0:
        return REJECTED
    lo = min_split_size - 1
    hi = n - min_split_size - 1
    n_cand = _candidate_count_bits(bits, forb, n_forb, lo, hi)
    x_star = _tail_threshold(alpha / max(1, n_cand)) * (1.0 - 1e-6)
    scale = math.expm1(x_star / n) * float(tot1) * float(n - tot1)
    fn = float(n)
    ft1 = float(tot1)
    step = float(max(tot1, n - tot1))
    reached = False
    c1_before = 0
    for w in range(0, (hi >> 6) + 1):
        base = w << 6
        word = bits[w]
        first = max(base, lo)
        last = min(base + 63, hi)
        if last >= first:
            d_prev = float(c1_before) * fn - float(base) * ft1
            bound = (abs(d_prev) + 64.0 * step) ** 2
            f_first = scale * float(first + 1) * (fn - float(first + 1))
            f_last = scale * float(last + 1) * (fn - float(last + 1))
            if bound >= 0.999999 * min(f_first, f_last):
                c1 = c1_before
                for i in range(base, last + 1):
                    c1 += int((word >> np.uint64(i - base)) & np.uint64(1))
                    if i >= first:
                        fl = float(i + 1)
                        dd = float(c1) * fn - fl * ft1
                        if dd * dd >= scale * fl * (fn - fl):
                            reached = True
                            break
                if reached:
                    break
        c1_before += _pc64(word)
    if not reached:
        return REJECTED
    yd = np.empty(n, dtype=np.int8)
    for i in range(n):
        yd[i] = np.int8((bits[i >> 6] >> np.uint64(i & 63)) & np.uint64(1))
    best_gain = _best_gain_over_bound(yd, forb, n_forb, lo, hi, n, tot1, h_full, scale)
    if best_gain <= 0.0:
        return REJECTED
    p_value = min(1.0, _analytic_p_value(best_gain, n, 2) * max(1, n_cand))
    return ACCEPTED if p_value < alpha else REJECTED


@njit(cache=True, parallel=True)
def pack_label_rows(rows: np.ndarray) -> np.ndarray:
    """``(rows, words)`` uint64: every row of the 0/1 matrix ``rows`` packed 64 labels to a word (bit ``i`` of a row is bit ``i & 63`` of word ``i >> 6``)."""
    n_rows = rows.shape[0]
    n = rows.shape[1]
    words = (n + 63) >> 6
    out = np.zeros((n_rows, words), dtype=np.uint64)
    for b in prange(n_rows):
        for w in range(words):
            base = w << 6
            count = min(64, n - base)
            acc = np.uint64(0)
            for k in range(count):
                acc |= np.uint64(rows[b, base + k]) << np.uint64(k)
            out[b, w] = acc
    return out


@njit(cache=True)
def pack_sorted_labels_from_bits(label_bits: np.ndarray, order_row: np.ndarray, raw_bits: np.ndarray) -> int:
    """``pack_sorted_labels`` for labels held packed in ``label_bits``: the random reads of the gather hit a row of n / 8 bytes (cache resident) instead of n."""
    n = order_row.shape[0]
    words = (n + 63) >> 6
    ones = 0
    for w in range(words):
        base = w << 6
        count = min(64, n - base)
        acc = np.uint64(0)
        for k in range(count):
            r = order_row[base + k]
            acc |= ((label_bits[r >> 6] >> np.uint64(r & 63)) & np.uint64(1)) << np.uint64(k)
        raw_bits[w] = acc
        ones += _pc64(acc)
    return ones


@njit(cache=True, parallel=True)
def within_stratum_shuffles(keys: np.ndarray, z_rank: np.ndarray, order: np.ndarray, x_sorted: np.ndarray, out: np.ndarray) -> None:
    """Row ``i`` of ``out`` is the candidate column shuffled within the strata of the support: ``out[i][order[j]] = x_sorted[w[j]]`` with ``w`` the stable argsort of ``z_rank + keys[i]``.

    ``order`` sorts the rows by stratum, ``x_sorted`` is the candidate in that order and ``z_rank`` the dense stratum number of each sorted row, so adding a key in ``[0, 1)`` sorts the rows of a stratum
    by their key: a uniform shuffle inside each stratum. The rows are independent, hence one parallel task per shuffle; the result is that of ``np.argsort(z_rank + keys[i], kind="stable")`` row by row."""
    for i in prange(keys.shape[0]):
        within = np.argsort(z_rank + keys[i], kind="mergesort")
        for j in range(within.shape[0]):
            out[i, order[j]] = x_sorted[within[j]]


@njit(cache=True)
def rows_to_columns(rows: np.ndarray) -> np.ndarray:
    """``(n, k)`` C-ordered copy of the ``(k, n)`` matrix ``rows`` (``out[r, j] = rows[j, r]``), by tiles of rows so that every tile of the output is written while it is cache resident.

    Writing the candidate columns of a block one by one into an ``(n, k)`` matrix touches a cache line and a page per element (a stride of ``k`` items); filling contiguous rows and transposing once is the same
    data movement without that pattern."""
    k, n = rows.shape
    out = np.empty((n, k), dtype=rows.dtype)
    tile = 64
    for r0 in range(0, n, tile):
        r1 = min(n, r0 + tile)
        for j in range(k):
            for r in range(r0, r1):
                out[r, j] = rows[j, r]
    return out


@njit(cache=True, parallel=True)
def column_fingerprints(bits: np.ndarray) -> np.ndarray:
    """``(k, 2)`` uint64 content fingerprints of the columns of the ``(n, k)`` integer matrix ``bits`` (the raw bit patterns of a float matrix): two independent FNV-1a style lanes over the column in row order, so
    two columns with equal values in the same order get equal fingerprints and a different value or order changes both. Reads the columns in place (no per-column copy of a strided column)."""
    n, k = bits.shape
    out = np.empty((k, 2), dtype=np.uint64)
    for j in prange(k):
        h1 = np.uint64(14695981039346656037)
        h2 = np.uint64(0x9E3779B97F4A7C15)
        for i in range(n):
            v = np.uint64(bits[i, j])
            h1 = (h1 ^ v) * np.uint64(1099511628211)
            h2 = (h2 ^ (v + np.uint64(0x632BE59BD9B4E019))) * np.uint64(0xC2B2AE3D27D4EB4F)
            h2 ^= h2 >> np.uint64(29)
        out[j, 0] = h1
        out[j, 1] = h2
    return out


@njit(cache=True)
def rows_to_columns_into(rows: np.ndarray, out: np.ndarray, col0: int) -> None:
    """``out[r, col0 + j] = rows[j, r]``: the ``(k, n)`` block ``rows`` written, tile by tile, into columns ``col0 .. col0 + k`` of the ``(n, K)`` matrix ``out`` (see ``rows_to_columns``)."""
    k, n = rows.shape
    tile = 64
    for r0 in range(0, n, tile):
        r1 = min(n, r0 + tile)
        for j in range(k):
            for r in range(r0, r1):
                out[r, col0 + j] = rows[j, r]
