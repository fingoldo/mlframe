"""Multidimensional Adaptive Histogram (MAH) MI estimator.

Python port of Marx, Yang, van Leeuwen (2021), "Estimating Conditional Mutual
Information for Discrete-Continuous Mixtures using Multi-Dimensional Adaptive
Histograms", SIAM Int. Conf. Data Mining (SDM 2021). arXiv:2101.05009.

Algorithm overview (marginal 1-D MI variant; the paper's main contribution
is the multi-d CMI version, but the 1-D marginal estimator inherits the same
stochastic-complexity-of-histogram bin-merge logic):

  1. Initial fine quantile binning of X and Y (max K bins each).
  2. Greedy bin merging on the joint table guided by REFINED MDL (Normalised
     Maximum Likelihood, NML) regret over a multinomial code:

         L_NML(joint | M) = -log Sum_{n_1..n_K} prod_i n_i! / n!  (Shtarkov)

     equivalently the regret-of-regret term has a closed form via the
     multinomial NML regret (Mononen-Myllymaeki 2008):

         R(K, N) ~ ((K-1)/2) * log(N / (2 * pi)) + log Gamma(K/2)

     The estimator MERGES bins whenever the NML code shortens (the regret
     reduction outweighs the empirical-likelihood reduction).

  3. After merging, the joint counts give plug-in MI in nats, but the SCI
     score (stochastic complexity index) is what the paper uses for
     hypothesis-test calibration:

         SCI(X; Y) = L_NML(X | flat marginal Y) - L_NML(X | refined joint)

     Under H_0 (X indep Y) the expected SCI is 0; large SCI signals real
     dependence. We return SCI / N - a regret-normalised proxy for I(X; Y)
     in nats per sample, with the same 0-under-independence property.

Reference: https://arxiv.org/abs/2101.05009
R code (canonical): https://github.com/ylincen/CMI-adaptive-hist
Background: Marx & Vreeken (2019), "Testing Conditional Independence on
Discrete Data using Stochastic Complexity", AISTATS 2019, arXiv:1903.04829.

Implementation notes (per README.md optimization methodology):
  * njit-compiled NML regret + greedy merge inner loop.
  * Vectorised initial quantile binning via numpy.
  * No external deps beyond numpy/scipy/numba.
"""
from __future__ import annotations

from typing import cast
import math
import threading
import weakref
from collections import OrderedDict

import numpy as np
from numba import njit
import logging

from mlframe._array_buffer import array_buffer

logger = logging.getLogger(__name__)

# Y-BINNING CACHE: ``per_feature_edges``/``mah_mi`` dispatchers call ``mah_bin_edges``/``mah_mi`` ONCE PER
# COLUMN (up to 500 columns for one fit) with the SAME ``y`` object every time - yet each call re-derives y's
# quantile/label binning (``np.unique``/``np.quantile``/``np.searchsorted``) from scratch. y is fixed for the whole
# dispatch, so cache the binning keyed on ``(id(y), K)`` with a weakref identity guard (mirrors the ``_cmi_cuda``
# resident-cache pattern): a recycled id (y GC'd, allocator reuses the address) fails the ``ref() is y`` check and
# recomputes, never returning a stale binning. Small bound (a handful of concurrent fits/folds) since each entry is
# only an O(n) int array, not a big buffer.
_Y_BINNING_CACHE: "OrderedDict[tuple, tuple[weakref.ReferenceType, tuple[np.ndarray, int]]]" = OrderedDict()
_Y_BINNING_CACHE_MAX_ENTRIES = 8
_Y_BINNING_LOCK = threading.Lock()


def _compute_y_binning(y: np.ndarray, K: int) -> tuple[np.ndarray, int]:
    """(yb, K_y): genuinely low-cardinality y (<=K distinct values, any dtype) label-encodes exactly;
    everything else K-quantile-bins, regardless of dtype.

    Bug fix: the prior condition was ``dtype.kind in "iub" OR unique.size <= K``, which
    routed EVERY integer-dtype y (bool/int8..int64) through the exact label-encode branch unconditionally
    - including a high-cardinality but integer-typed target (e.g. a rounded price/timestamp/count with
    thousands of distinct values, which is effectively continuous). That set ``K_y`` to the FULL
    cardinality instead of capping at K, and the (K_x, K_y) joint table then fed ``_greedy_merge_bins``'s
    per-step O(rows * cols) candidate-matrix rebuild across O(cols) candidates, repeated as cols shrinks by
    1 each step - an O(rows * K_y^3)-ish blowup. Measured: n=3000, K_y~2000 (int64 target) hangs past 40s
    (pre-fix code, verified via ``timeout 40 python ...`` -> exit 124); this fix keeps it fast because the
    quantile branch caps ``K_y <= K`` (16 by default) regardless of dtype. Any real MRMR user picking
    ``binning_scheme='mah'`` with an integer-coded continuous-like target hit this DoS silently
    (no error, no warning - the fit just never returns from this column's edge computation).
    """
    uniq_y = np.unique(y)
    if uniq_y.size <= K:
        yb = np.searchsorted(uniq_y, y).astype(np.int64)
    else:
        qy = np.quantile(y.astype(np.float64), np.linspace(0, 1, K + 1))  # Intentional float64: quantile edges of the target for the MAH statistic
        qy_unique = np.unique(qy)
        if qy_unique.size < 3:
            return np.array([], dtype=np.int64), 0
        yb = np.searchsorted(qy_unique[1:-1], y.astype(np.float64), side="right").astype(np.int64)  # Intentional float64: target bin search for the MAH statistic
    K_y = int(yb.max()) + 1 if yb.size else 0
    return yb, K_y


def _get_y_binning(y: np.ndarray, K: int) -> tuple[np.ndarray, int]:
    """Cached ``(yb, K_y)`` for ``y``+``K``, computed once per distinct ``y`` identity (see module docstring above)."""
    ident = _view_identity(y, allow_writable=True)  # as before, the target is not written into while its binning is cached
    anchor = ident[0] if ident is not None else y  # a view of a read-only owner is named by the owner and its position: ``ravel()`` of the same target is a new object every call
    key = (id(anchor), ident[1] if ident is not None else None, int(K))
    with _Y_BINNING_LOCK:
        ent = _Y_BINNING_CACHE.get(key)
        if ent is not None and ent[0]() is anchor:
            _Y_BINNING_CACHE.move_to_end(key)
            return ent[1]
        dead = [k for k, v in _Y_BINNING_CACHE.items() if v[0]() is None]
        for k in dead:
            del _Y_BINNING_CACHE[k]
        by_multiset = _y_binning_by_multiset(y, K)
        result = by_multiset if by_multiset is not None else _compute_y_binning(y, K)
        try:
            ref = weakref.ref(anchor)
        except TypeError as _exc:
            logger.debug("_get_y_binning fell back to its default after %s", type(_exc).__name__)
            return result  # y doesn't support weakref (unlikely for ndarray) - skip caching, still correct
        _Y_BINNING_CACHE[key] = (ref, result)
        if len(_Y_BINNING_CACHE) > _Y_BINNING_CACHE_MAX_ENTRIES:
            # evict-ok: memo; a miss recomputes the value
            _Y_BINNING_CACHE.popitem(last=False)
        return result


def clear_mah_y_binning_cache() -> None:
    """Drop the resident y-binning cache (call at FE-step teardown alongside the other resident caches)."""
    # Hold the same lock _get_y_binning takes to mutate/iterate this OrderedDict, else a teardown clear
    # racing a concurrent _get_y_binning raises "dictionary changed size during iteration".
    with _Y_BINNING_LOCK:
        _Y_BINNING_CACHE.clear()


def _view_identity(a: np.ndarray, *, allow_writable: bool = False) -> "tuple[object, tuple] | None":
    """``(owner array, position of the view in it)`` when ``a`` is a view of an owner that cannot change (read-only; with ``allow_writable`` any owner, on the caller's promise not to write into it while cached), else ``None``.

    ``np.asarray(y).ravel()`` makes a NEW array object of the same buffer on every call, so a cache keyed by ``id`` of the argument never hit; the owner and the view's offset, length and stride name the
    same values for as long as the owner lives, and a read-only owner cannot change under the cache."""
    root = a
    while isinstance(root.base, np.ndarray):
        root = root.base
    if root.flags.writeable and not allow_writable:
        return None
    offset = a.__array_interface__["data"][0] - root.__array_interface__["data"][0]
    return root, (offset, a.shape, a.strides, a.dtype.str)


# X-SIDE PLAN CACHE: the quantile cut points of a column and its bin index depend on the column alone, yet the honest null of a supervised ``binning_scheme`` asks for the edges of the SAME column under
# every permutation of the target. Keyed by the content digest of the column and ``K``; bounded (a plan is an int16 vector of the column's length plus a short edge vector).
_X_PLAN_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_X_PLAN_CACHE_MAX_ENTRIES = 512
_X_PLAN_LOCK = threading.Lock()


def _x_plan(x: np.ndarray, K: int) -> "tuple[np.ndarray, np.ndarray, int] | None":
    """``(initial_inner, xb, K_x)`` of the float64 column ``x`` (its K-quantile inner cuts, the bin index of every row, the number of bins), or ``None`` when fewer than three distinct cuts exist.

    The numbers are those ``mah_bin_edges`` derives inline; they are memoised by the content of the column."""
    import hashlib

    ident = _view_identity(x)
    key: tuple
    if ident is not None:
        key = (id(ident[0]), ident[1], int(K))  # a read-only owner: the view names the values, no hash of the column needed
    else:
        key = (hashlib.blake2b(array_buffer(x), digest_size=16).digest(), int(x.size), int(K))
    with _X_PLAN_LOCK:
        hit = _X_PLAN_CACHE.get(key)
        if hit is not None and (ident is None or hit[3]() is ident[0]):
            _X_PLAN_CACHE.move_to_end(key)
            return hit[:3] if hit[2] > 0 else None
    qx = np.quantile(x, np.linspace(0, 1, K + 1))
    qx_unique = np.unique(qx)
    if qx_unique.size < 3:
        plan = (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int16), 0)
    else:
        initial_inner = qx_unique[1:-1].astype(np.float64)  # Intentional float64: initial inner edges for the MAH statistic
        xb = np.searchsorted(initial_inner, x, side="right")
        plan = (initial_inner, xb.astype(np.int16), int(xb.max()) + 1)
    owner_ref = weakref.ref(ident[0]) if ident is not None else (lambda: None)
    with _X_PLAN_LOCK:
        _X_PLAN_CACHE[key] = (*plan, owner_ref)
        while len(_X_PLAN_CACHE) > _X_PLAN_CACHE_MAX_ENTRIES:
            _X_PLAN_CACHE.popitem(last=False)  # evict-ok: a miss recomputes the plan
    return plan if plan[2] > 0 else None


# CONTENT-KEYED Y-BINNING for integer targets: the binning of ``y`` (label encode when at most K distinct values, else K-quantile bins) is a function of the MULTISET of its values, and a permuted target
# has the multiset of the raw one. For an integer target of small range the exact multiset is its bincount, so the value -> bin map is kept per bincount and a permutation costs one searchsorted.
_Y_MAP_CACHE: "OrderedDict[tuple, tuple[np.ndarray, bool, int]]" = OrderedDict()
_Y_MAP_CACHE_MAX_ENTRIES = 16
_Y_MAP_MAX_RANGE = 1 << 20
_Y_MAP_LOCK = threading.Lock()  # its own lock: ``_get_y_binning`` calls this function while it holds ``_Y_BINNING_LOCK``


def _y_binning_by_multiset(y: np.ndarray, K: int) -> "tuple[np.ndarray, int] | None":
    """``_compute_y_binning(y, K)`` through a map kept per multiset of an integer ``y``, or ``None`` when ``y`` is not an integer vector of small range."""
    if y.dtype.kind not in "iub" or y.size == 0:
        return None
    lo, hi = int(y.min()), int(y.max())
    if hi - lo >= _Y_MAP_MAX_RANGE:
        return None
    counts = np.bincount(y.astype(np.int64) - lo)
    key = (counts.tobytes(), lo, int(K))
    with _Y_MAP_LOCK:
        hit = _Y_MAP_CACHE.get(key)
        if hit is not None:
            _Y_MAP_CACHE.move_to_end(key)
    if hit is None:
        _, K_y = _compute_y_binning(y, K)
        if K_y == 0:
            hit = (np.empty(0, dtype=np.float64), False, 0)
        else:
            uniq_y = np.unique(y)
            if uniq_y.size <= K:
                hit = (uniq_y, True, K_y)  # label encode: the rank of the value among the distinct values
            else:
                qy_unique = np.unique(np.quantile(y.astype(np.float64), np.linspace(0, 1, K + 1)))  # Intentional float64: target quantile edges for the MAH statistic
                hit = (qy_unique[1:-1], False, K_y)
        with _Y_MAP_LOCK:
            _Y_MAP_CACHE[key] = hit
            while len(_Y_MAP_CACHE) > _Y_MAP_CACHE_MAX_ENTRIES:
                _Y_MAP_CACHE.popitem(last=False)  # evict-ok: a miss recomputes the map
    table, label_encode, K_y = hit
    if K_y == 0:
        return np.array([], dtype=np.int64), 0
    if label_encode:
        return np.searchsorted(table, y).astype(np.int64), K_y
    return np.searchsorted(table, y.astype(np.float64), side="right").astype(np.int64), K_y  # Intentional float64: target bin search for the MAH statistic


@njit(nogil=True, cache=True)
def _nml_regret_legacy(K: int, N: int) -> float:
    """The regret formula this module shipped with (kept as the reference): ``(K-1)/2 log(N/(2 pi)) + log Gamma(K/2) - (K/2) log Gamma(1/2)``.

    It is not the multinomial NML regret: the log term has the wrong scale (``N/(2 pi)`` for ``N/2``) and the Gamma terms have the wrong signs, so it understates the regret by about 2.3 nats per
    two-category row at N = 1250. A merge then never pays for itself on a column without signal, and a column keeps all its quantile bins."""
    if K <= 1 or N <= 0:
        return 0.0
    half_K = float(K) / 2.0
    return (K - 1) * 0.5 * math.log(max(N, 1) / (2.0 * math.pi)) + math.lgamma(half_K) - half_K * math.lgamma(0.5)


@njit(nogil=True, cache=True)
def _nml_regret(K: int, N: int) -> float:
    """Multinomial NML regret of K categories on N samples, Rissanen's asymptotic expansion (the first-order form of Mononen and Myllymaki, 2008):

        R(K, N) ~ ((K-1)/2) log(N/2) + log(sqrt(pi) / Gamma(K/2))

    in nats; ``R(1, N) = 0``. At K = 2 and N = 200 it is 2.875 nats against the exact 2.912 (the next term is of order 0.27 / sqrt(N))."""
    if K <= 1 or N <= 0:
        return 0.0
    half_K = float(K) / 2.0
    return (K - 1) * 0.5 * math.log(max(N, 1) / 2.0) + 0.5 * math.log(math.pi) - math.lgamma(half_K)


@njit(nogil=True, cache=True)
def _stochastic_complexity(counts: np.ndarray) -> float:
    """Stochastic complexity (NML code length) of a multinomial:

        L_NML(counts) = -log prod_i (n_i / N)^{n_i}  +  R(K, N)
                     = -sum_i n_i log(n_i / N)        +  R(K, N)
                     =  N * H_plug(counts)            +  R(K, N)
    in nats. Used as the MDL "code length" in Marx 2019/2021 sec. 3.
    """
    N = 0
    K_nonzero = 0
    for c in counts:
        N += int(c)
        if c > 0:
            K_nonzero += 1
    if N <= 0 or K_nonzero <= 1:
        return 0.0
    h = 0.0
    N_f = float(N)
    for c in counts:
        if c > 0:
            p = float(c) / N_f
            h -= p * math.log(p)
    return float(N * h + _nml_regret(K_nonzero, N))


@njit(nogil=True, cache=True)
def _merge_adjacent(cur, i, axis):
    """``cur`` with the slices ``i`` and ``i + 1`` along ``axis`` summed into slice ``i``."""
    R, C = cur.shape
    if axis == 1:
        out = np.empty((R, C - 1), dtype=np.float64)  # Intentional float64: MAH partition accumulator state
        for r in range(R):
            for c in range(C - 1):
                if c < i:
                    out[r, c] = cur[r, c]
                elif c == i:
                    out[r, c] = cur[r, i] + cur[r, i + 1]
                else:
                    out[r, c] = cur[r, c + 1]
    else:
        out = np.empty((R - 1, C), dtype=np.float64)  # Intentional float64: MAH partition accumulator state
        for r in range(R - 1):
            for c in range(C):
                if r < i:
                    out[r, c] = cur[r, c]
                elif r == i:
                    out[r, c] = cur[i, c] + cur[i + 1, c]
                else:
                    out[r, c] = cur[r + 1, c]
    return out


@njit(nogil=True, cache=True)
def _best_adjacent_merge(cur, axis):
    """Index of the adjacent pair along ``axis`` whose merge lowers the stochastic complexity most, or ``-1`` when no merge lowers it."""
    sci_now = _stochastic_complexity(cur.ravel())
    best_delta = 0.0
    best_i = -1
    for i in range(cur.shape[axis] - 1):
        delta = sci_now - _stochastic_complexity(_merge_adjacent(cur, i, axis).ravel())
        if delta > best_delta:
            best_delta = delta
            best_i = i
    return best_i


@njit(nogil=True, cache=True)
def _pair_gain_conditional(cur, i, axis):
    """Code length saved by merging the slices ``i`` and ``i + 1`` along ``axis``, counted on the CONDITIONAL distribution of the other variable: ``SC(a) + SC(b) - SC(a + b)``.

    ``a`` and ``b`` are the count vectors of the two slices over the other axis (the y distribution of two adjacent x bins). Merging two slices whose distributions agree saves one regret term and
    costs nothing; merging two that differ costs ``N`` times their Jensen-Shannon divergence. Positive when the merge shortens the description of the other variable given this one."""
    n_other = cur.shape[1] if axis == 0 else cur.shape[0]
    a = np.empty(n_other)
    b = np.empty(n_other)
    m = np.empty(n_other)
    for t in range(n_other):
        if axis == 0:
            a[t] = cur[i, t]
            b[t] = cur[i + 1, t]
        else:
            a[t] = cur[t, i]
            b[t] = cur[t, i + 1]
        m[t] = a[t] + b[t]
    return _stochastic_complexity(a) + _stochastic_complexity(b) - _stochastic_complexity(m)


@njit(nogil=True, cache=True)
def _best_adjacent_merge_conditional(cur, axis, cut_cost=0.0):
    """Index of the adjacent pair along ``axis`` with the largest positive :func:`_pair_gain_conditional` (plus ``cut_cost``, the code length of naming the cut a merge removes), or ``-1`` when no merge shortens the code."""
    best_gain = 0.0
    best_i = -1
    for i in range(cur.shape[axis] - 1):
        gain = _pair_gain_conditional(cur, i, axis) + cut_cost
        if gain > best_gain:
            best_gain = gain
            best_i = i
    return best_i


@njit(nogil=True, cache=True)
def _greedy_merge_core(joint, conditional=True, merge_columns=True, name_cuts=False):
    """Column merges then row merges of ``joint``: the merged table and the merge positions, as arrays of the number of merges made.

    ``conditional=True`` (the default) merges on the conditional code length of the other variable (:func:`_pair_gain_conditional`) and may merge down to one slice, so a column with no
    dependence on the target ends in a single bin. ``merge_columns=False`` leaves the columns (the target's bins) as they are and merges rows only: the binning of a feature merges the feature's bins, never the labels. ``conditional=False`` is the earlier JOINT criterion (the NML code length of the whole table, floor of two slices): that code length falls
    with every merge of cells, so it merged every column to its first two bins whatever the data; kept as the reference."""
    cur = joint.astype(np.float64)  # Intentional float64: MAH partition accumulator state
    floor = 1 if conditional else 2
    col_merges = np.empty(max(cur.shape[1], 1), dtype=np.int64)
    n_col = 0
    while merge_columns and cur.shape[1] > floor:
        best_i = _best_adjacent_merge_conditional(cur, 1) if conditional else _best_adjacent_merge(cur, 1)
        if best_i < 0:
            break
        col_merges[n_col] = best_i
        n_col += 1
        cur = _merge_adjacent(cur, best_i, 1)
    row_merges = np.empty(max(cur.shape[0], 1), dtype=np.int64)
    n_row = 0
    while cur.shape[0] > floor:
        cut_cost = math.log(max(2.0, float(cur.shape[0] - 1))) if name_cuts else 0.0
        best_i = _best_adjacent_merge_conditional(cur, 0, cut_cost) if conditional else _best_adjacent_merge(cur, 0)
        if best_i < 0:
            break
        row_merges[n_row] = best_i
        n_row += 1
        cur = _merge_adjacent(cur, best_i, 0)
    return cur, row_merges[:n_row], col_merges[:n_col]


@njit(nogil=True, cache=True)
def _greedy_merge_bins(joint):
    """Greedy column- and row-merging of a (K_x, K_y) joint table guided by
    the JOINT NML code length (Marx 2021 sec. 3.2): merge when the regret
    reduction outweighs the log-likelihood increase. Operates on the full
    flat joint matrix, not the marginal - so merging is data-dependent
    (preserves the joint structure even when the marginal would otherwise
    collapse).
    """
    return _greedy_merge_core(joint, True)[0]


@njit(nogil=True, cache=True)
def _mi_from_joint(joint: np.ndarray) -> float:
    """Plug-in MI in nats from a joint count matrix."""
    K_x, K_y = joint.shape
    N = joint.sum()
    if N <= 0:
        return 0.0
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    mi = 0.0
    for i in range(K_x):
        if px[i] <= 0:
            continue
        for j in range(K_y):
            if joint[i, j] <= 0 or py[j] <= 0:
                continue
            p_ij = joint[i, j] / N
            mi += p_ij * math.log(p_ij * N / (px[i] * py[j] / N))
    return mi


def _greedy_merge_with_history(joint: np.ndarray, conditional: bool = True, merge_columns: bool = True, name_cuts: bool = False) -> tuple[np.ndarray, list[int], list[int]]:
    """Identical SCI-greedy merge to ``_greedy_merge_bins`` but ALSO records
    which row/col positions got merged at each step.

    Args:
        joint: ``(K_x, K_y)`` joint count matrix to merge; it is copied, not modified.
        conditional: merge on the conditional code length of the other variable (default); ``False`` is the earlier joint criterion.
        merge_columns: also merge the columns (the target's bins); ``False`` merges rows only.
        name_cuts: charge the code length of naming each kept cut among the remaining ones (a merge that removes a cut saves ``log`` of their number).

    Returns:
        merged (K_x', K_y'): the post-merge joint table.
        row_merges (list[int]): at each step the CURRENT-matrix row index
            that merged with index+1.
        col_merges (list[int]): same for columns.
    """
    merged, row_merges, col_merges = _greedy_merge_core(joint, conditional, merge_columns, name_cuts)
    return np.asarray(merged), row_merges.tolist(), col_merges.tolist()


def _apply_merges_to_edges(initial_inner_edges: np.ndarray, row_merges) -> np.ndarray:
    """Walk through the row-merge sequence and drop the inner edges that
    got swallowed by a merge.

    ``initial_inner_edges`` has length ``K - 1`` (one inner edge between each
    pair of adjacent quantile bins). Each merge at current-matrix position ``i``
    removes the edge between the ``i``-th and ``(i+1)``-th currently-remaining
    bin - which corresponds to a specific position in the dropping list.
    """
    # Track ranges of original-bin indices per current bin.
    K = initial_inner_edges.size + 1
    ranges = [[k, k + 1] for k in range(K)]
    for pos in row_merges:
        if pos + 1 >= len(ranges):
            continue
        # Merge ranges[pos] and ranges[pos+1].
        ranges[pos][1] = ranges[pos + 1][1]
        del ranges[pos + 1]
    # Remaining inner edges: original index = ranges[i].end for i in 0..len-2.
    remaining_positions = [ranges[i][1] - 1 for i in range(len(ranges) - 1)]
    if not remaining_positions:
        return np.array([], dtype=np.float64)  # Intentional float64: empty edge vector keeps the float64 edge contract
    return initial_inner_edges[np.asarray(remaining_positions, dtype=np.int64)]


def mah_bin_edges(x: np.ndarray, y: np.ndarray, *, initial_k: int = 16) -> np.ndarray:
    """Multidimensional Adaptive Histogram bin edges for X (Marx 2021).

    Exposed as a Family-1 ``binning_scheme`` option.
    Returns INNER bin edges suitable for ``np.searchsorted(edges, x, side='right')``.

    Pipeline:
      1. Equal-frequency K-bin x; label-encode (or K-bin) y.
      2. Build (K_x, K_y) joint count matrix.
      3. Greedy 2D bin merging guided by SCI (Marx 2021 sec. 3.2).
      4. Drop the X-axis inner edges that got swallowed by row-merges.

    Args:
        x: 1-D continuous.
        y: 1-D supervised target (discrete preferred per the paper's
            discrete-continuous regime; continuous y is quantile-binned to K).
        initial_k: starting bin count. Marx 2021 uses K=16; greedy merge
            adapts down.

    Returns:
        1-D double array of the surviving inner bin edges of ``x``; empty when fewer than 16 samples or no edge survives.

    Reference: Marx, Yang, van Leeuwen (2021), SDM 2021. arXiv:2101.05009.
    """
    x = np.asarray(x, dtype=np.float64).ravel()  # Intentional float64: bin-edge computation input; edges compared at full precision
    y = np.asarray(y).ravel()
    n = x.size
    if n < 16:
        return np.array([], dtype=np.float64)  # Intentional float64: empty edge vector keeps the float64 edge contract
    K = int(initial_k)
    plan = _x_plan(x, K)
    if plan is None:
        return np.array([], dtype=np.float64)  # Intentional float64: empty edge vector keeps the float64 edge contract
    initial_inner, xb16, K_x = plan
    xb = xb16.astype(np.int64)
    # y is fit-constant across the whole per_feature_edges column dispatch -> cached binning (see _get_y_binning); a permuted integer target reuses the map of its multiset.
    yb, K_y = _get_y_binning(y, K)
    if K_y == 0:
        return cast("np.ndarray", initial_inner)
    if K_x < 2 or K_y < 2:
        return cast("np.ndarray", initial_inner)
    joint = np.bincount(xb * K_y + yb, minlength=K_x * K_y).reshape(K_x, K_y).astype(np.float64)  # Intentional float64: joint-count accumulator for the MAH statistic
    _, row_merges, _ = _greedy_merge_with_history(joint, True, False)  # the labels are not merged: only the bins of x
    # Map row merges to drop the swallowed X edges.
    return _apply_merges_to_edges(initial_inner, row_merges)


def mah_mi(x: np.ndarray, y: np.ndarray, *, initial_k: int = 16, return_sci: bool = False) -> float:
    """MAH/SCI mutual information estimator (Marx 2021, SDM). See module
    docstring for the algorithm. Returns I(X; Y) in nats; for ``return_sci=True``
    returns the SCI difference (regret-normalised hypothesis test statistic).
    """
    x = np.asarray(x, dtype=np.float64).ravel()  # Intentional float64: bin-edge computation input; edges compared at full precision
    y = np.asarray(y).ravel()
    n = x.size
    if n < 16:
        return 0.0
    K = int(initial_k)
    qx = np.quantile(x, np.linspace(0, 1, K + 1))
    qx_unique = np.unique(qx)
    if qx_unique.size < 3:
        return 0.0
    xb = np.searchsorted(qx_unique[1:-1], x, side="right").astype(np.int64)
    # y is fit-constant across the whole per_feature_edges column dispatch -> cached binning (see _get_y_binning).
    yb, K_y = _get_y_binning(y, K)
    if K_y == 0:
        return 0.0
    K_x = int(xb.max()) + 1
    if K_x < 2 or K_y < 2:
        return 0.0
    joint = np.bincount(xb * K_y + yb, minlength=K_x * K_y).reshape(K_x, K_y).astype(np.float64)  # Intentional float64: joint-count accumulator for the MAH statistic
    merged, _, _ = _greedy_merge_with_history(joint)
    if return_sci:
        sci_joint = _stochastic_complexity(merged.ravel())
        marg_x = merged.sum(axis=1)
        marg_y = merged.sum(axis=0)
        sci_marg = _stochastic_complexity(marg_x) + _stochastic_complexity(marg_y)
        return max(0.0, float(sci_marg - sci_joint) / float(n))
    return max(0.0, float(_mi_from_joint(merged)))


__all__ = ["mah_mi", "mah_bin_edges"]
