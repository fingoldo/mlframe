"""Layer 72: Chatterjee's Xi rank-correlation dependence
scorer for the auto-scorer pool (``_orth_auto_scorer_fe.py`` / ``_orthogonal_scorer_auto_fe.py``).

Chatterjee (2021, "A New Coefficient of Correlation", JASA) defines the TIE-CORRECTED estimator::

    xi(X, Y) = 1 - n * sum_i |r_{i+1} - r_i| / (2 * sum_i l_i (n - l_i))

with ``r_i = #{j: y_j <= y_(i)}`` and ``l_i = #{j: y_j >= y_(i)}`` (both counting ties), which reduces to
the familiar no-ties form ``1 - 3 * sum_i |r_{i+1} - r_i| / (n^2 - 1)`` when ``y`` has no ties but stays
unbiased on tied/discrete ``y`` (e.g. any classification target),
where ``r_i`` is the rank of ``y`` reordered by ascending ``x`` (ties in ``x`` broken by a random
permutation, per Chatterjee's own construction, so the estimator stays well-defined on discrete/
low-cardinality ``x``). Xi is asymptotically 0 iff X and Y are independent and 1 iff Y is a
MEASURABLE FUNCTION of X (not merely monotone, unlike Spearman) - a genuinely different
construction from every scorer already in the pool (plug-in MI is quantile-binning-based, KSG is
kNN-distance-based, copula-MI is rank-uniformized-MI, dCor is U-centred-distance-matrix-based,
HSIC is RKHS-kernel-based).

Why this catches a shape the catalog misses: on a smooth but highly-OSCILLATORY target such as
``y = sin(20*x) + noise``, plug-in MI's fixed quantile bins average many oscillation cycles into
each bin (near-zero MI), KSG's kNN balls at moderate k similarly smear across cycles, and dCor/HSIC
(calibrated to a GLOBAL scale) underweight the fine local structure. Xi's sort-then-walk
construction is scale-free and directly sees every local up/down rank flip in y, so it stays high
at oscillation frequencies where every distance/kernel/binning scorer decays toward the null floor.

Cost: O(n log n) (one argsort), not O(n^2) like dCor/HSIC - no subsampling needed even at large n.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit, prange

from mlframe._numba_parallel_guard import parallel_kernel_entry

from ._fe_usability_signal import _crit_np_dtype
from ._numba_utils_shared import radix_order_f32

__all__ = ["xi_correlation", "xi_correlation_batch"]


def _xi_from_order(y_ordered: np.ndarray) -> float:
    """Chatterjee's Xi from ``y`` already reordered by ascending ``x``, using the TIE-CORRECTED estimator.

    ``xi = 1 - n * sum_i |r_{i+1} - r_i| / (2 * sum_i l_i (n - l_i))`` where ``r_i = #{j: y_j <= y_(i)}``
    (right-rank, counts ties) and ``l_i = #{j: y_j >= y_(i)}``. This reduces to the no-ties form
    ``1 - 3*sum|dr|/(n^2-1)`` when y has no ties, but on tied/discrete y (any classification target) the
    no-ties form's forced-distinct ranks and ``n^2-1`` denominator biased Xi toward 0.
    """
    n = y_ordered.size
    if n < 2:
        return 0.0
    _uniq, inv, counts = np.unique(y_ordered, return_counts=True, return_inverse=True)
    le = np.cumsum(counts)  # per unique-group: #{y <= group value}
    ge = np.cumsum(counts[::-1])[::-1]  # per unique-group: #{y >= group value}
    # Intentional float64: exact rank accumulator kept f64
    r = le[inv].astype(np.float64)  # r_i, already in x-induced order (y_ordered is x-sorted)
    # Intentional float64: exact rank accumulator kept f64
    ll = ge[inv].astype(np.float64)  # l_i
    num = float(n) * float(np.sum(np.abs(np.diff(r))))
    den = 2.0 * float(np.sum(ll * (n - ll)))
    return float(1.0 - num / den) if den > 0.0 else 0.0


def xi_correlation(x: np.ndarray, y: np.ndarray, *, random_state: int = 0) -> float:
    """Chatterjee's Xi correlation coefficient between two 1-D arrays.

    Ties in ``x`` are broken by a random permutation (Chatterjee's own construction) so the
    estimator stays well-defined on discrete/low-cardinality ``x`` rather than depending on
    argsort's tie-breaking order. Returns 0.0 on degenerate input (n < 2, or ``y`` constant, since
    a constant y has zero variation to detect regardless of x).
    """
    x = np.asarray(x, dtype=_crit_np_dtype()).ravel()
    # Intentional float64: target labels kept f64 (exact binning / comparison contract)
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    if n < 2 or y.size != n:
        return 0.0
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return 0.0
    if float(np.std(y)) <= 1e-12:
        return 0.0
    # Break x-ties by the counter-hash key of column 0 (Chatterjee 2021, Section 2: a random tie-break keeps the estimator correct in expectation on discrete/repeated-value x); the batch gives
    # its first finite column the same key, so ``xi_correlation(X[:, 0], y)`` equals ``xi_correlation_batch(X, y)[0]``.
    order = np.argsort(x, kind="mergesort")
    _break_ties_by_key(x, order, column_seeds(random_state, 1)[0])
    y_ordered = y[order]
    return _xi_from_order(y_ordered)


_U64 = np.uint64
_MIX_GAMMA = _U64(0x9E3779B97F4A7C15)
_MIX_M1 = _U64(0xBF58476D1CE4E5B9)
_MIX_M2 = _U64(0x94D049BB133111EB)
_SEED_STRIDE = _U64(0xD1B54A32D192ED03)


@njit(cache=True, inline="always")
def _mix64(x):
    """splitmix64 finaliser: a bijective 64-bit mixer (the same arithmetic is in the device kernel ``xi_sort_keys``)."""
    x = (x ^ (x >> _U64(30))) * _MIX_M1
    x = (x ^ (x >> _U64(27))) * _MIX_M2
    return x ^ (x >> _U64(31))


@njit(cache=True, inline="always")
def _tie_key(seed, row):
    """The 32-bit tie-break key of ``row`` under the column seed ``seed``: the high word of the mixed counter, so no permutation is drawn or stored."""
    return _mix64(seed + (_U64(row) + _U64(1)) * _MIX_GAMMA) >> _U64(32)


def column_seeds(random_state: int, k: int) -> np.ndarray:
    """The tie-break seeds of ``k`` columns: column ``j`` gets ``mix64(random_state * stride + j)``; the single-column :func:`xi_correlation` is column 0."""
    j = np.arange(k, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = np.uint64(int(random_state) & 0xFFFFFFFFFFFFFFFF) * _SEED_STRIDE + j
        x = (x ^ (x >> np.uint64(30))) * _MIX_M1
        x = (x ^ (x >> np.uint64(27))) * _MIX_M2
        return np.asarray(x ^ (x >> np.uint64(31)), dtype=np.uint64)


@njit(cache=True)
def _break_ties_by_key(col: np.ndarray, order: np.ndarray, seed) -> None:
    """Reorder every run of equal values of ``col`` (``order`` sorts ``col`` stably) by ``(tie key, row)``: the order the device sorts the packed 64-bit keys into."""
    n = order.shape[0]
    i = 0
    while i < n:
        v = col[order[i]]
        e = i + 1
        while e < n and col[order[e]] == v:
            e += 1
        m = e - i
        if m > 1:
            keys = np.empty(m, dtype=np.uint64)
            for q in range(m):
                row = order[i + q]
                keys[q] = (_tie_key(seed, row) << _U64(32)) | _U64(row)
            keys.sort()
            for q in range(m):
                order[i + q] = np.int64(keys[q] & _U64(0xFFFFFFFF))
        i = e


@njit(cache=True)
def _xi_from_sorted_order(order: np.ndarray, r_row: np.ndarray, l_row: np.ndarray) -> float:
    """Chatterjee's tie-corrected Xi from the row order the column induces (the walk over the target's right and left ranks)."""
    n = order.shape[0]
    sdiff = 0.0
    prev = r_row[order[0]]
    den = l_row[order[0]] * (n - l_row[order[0]])
    for q in range(1, n):
        cur = r_row[order[q]]
        sdiff += abs(cur - prev)
        prev = cur
        lq = l_row[order[q]]
        den += lq * (n - lq)
    den = 2.0 * den
    return 1.0 - float(n) * sdiff / den if den > 0.0 else 0.0


@njit(cache=True, parallel=True)
def _xi_columns_njit(cols: np.ndarray, seeds: np.ndarray, r_row: np.ndarray, l_row: np.ndarray, out: np.ndarray) -> None:
    """``out[j]`` = Xi of column ``cols[j]`` (a row of ``(K, n)``) against the target, the rows ordered by ``(value, tie key, row)``: a stable merge sort by value, then each tie run by key."""
    for j in prange(cols.shape[0]):
        col = cols[j]
        order = np.argsort(col, kind="mergesort")
        _break_ties_by_key(col, order, seeds[j])
        out[j] = _xi_from_sorted_order(order, r_row, l_row)


@njit(cache=True, parallel=True)
def _xi_columns_radix_njit(cols: np.ndarray, seeds: np.ndarray, r_row: np.ndarray, l_row: np.ndarray, out: np.ndarray) -> None:
    """:func:`_xi_columns_njit` for float32 columns, same order and same values: the stable order by value comes from a 4-pass LSD radix sort (``-0.0`` keyed as ``+0.0``, which compares equal)."""
    for j in prange(cols.shape[0]):
        col = cols[j]
        order = radix_order_f32(col)
        _break_ties_by_key(col, order, seeds[j])
        out[j] = _xi_from_sorted_order(order, r_row, l_row)


def _target_ranks(y: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """The tie-corrected right and left ranks of ``y`` per row (double precision integer counts)."""
    _uniq, inv, counts = np.unique(y, return_counts=True, return_inverse=True)
    # Intentional float64: exact rank accumulator kept f64
    r_row = np.cumsum(counts)[inv].astype(np.float64)
    # Intentional float64: exact rank accumulator kept f64
    l_row = np.cumsum(counts[::-1])[::-1][inv].astype(np.float64)
    return r_row, l_row


def xi_correlation_batch(X: np.ndarray, y: np.ndarray, *, random_state: int = 0, device_block: "Any | None" = None) -> np.ndarray:
    """Vectorized ``xi_correlation`` for every column of ``X`` (shape ``(n, K)``) against the same ``y``. Returns a ``(K,)`` double precision array.

    The tie-break key of a row is a counter hash of ``(random_state, finite-column index, row)``: nothing is drawn from a generator, so the cost of a column is its sort. ``device_block``: the same columns, all finite, already on the device as a ``(K, n)`` float32 cupy array (the caller uploaded them for several scorers); it is used when the
    tuner chooses the device for this row count."""
    with parallel_kernel_entry():
        X = np.asarray(X, dtype=_crit_np_dtype())
        if X.ndim == 1:
            X = X[:, None]
        n, k = X.shape
        # Intentional float64: target labels kept f64 (exact binning / comparison contract)
        y = np.asarray(y, dtype=np.float64).ravel()
        # Intentional float64: accumulator output kept f64
        out = np.zeros(k, dtype=np.float64)
        if n < 2 or y.size != n or float(np.std(y)) <= 1e-12:
            return out
        if not np.isfinite(y).all():
            return out
        r_row, l_row = _target_ranks(y)
        finite = np.flatnonzero(np.isfinite(X).all(axis=0))
        if finite.size == 0:
            return out
        seeds = column_seeds(random_state, int(finite.size))
        if device_block is not None and finite.size == k:
            from ._orthogonal_xi_gpu import use_device, xi_columns_device

            on_device = xi_columns_device(device_block, seeds, r_row, l_row) if use_device(n) else None
            if on_device is not None:
                out[:] = on_device
                return out
        block = max(1, min(64, (64 << 20) // (8 * max(n, 1))))
        for s0 in range(0, finite.size, block):
            js = finite[s0 : s0 + block]
            _score_block(X, js, seeds[s0 : s0 + block], r_row, l_row, out)
        return out


def _score_block(X: np.ndarray, js: np.ndarray, seeds: np.ndarray, r_row: np.ndarray, l_row: np.ndarray, out: np.ndarray) -> None:
    """Score the columns ``js`` of ``X`` with their tie seeds ``seeds`` into ``out[js]``."""
    cols = np.ascontiguousarray(X[:, js].T)
    from ._orthogonal_xi_gpu import use_device, xi_columns_device

    if use_device(int(cols.shape[1])):
        on_device = xi_columns_device(cols, seeds, r_row, l_row)
        if on_device is not None:
            out[js] = on_device
            return
    vals = np.empty(len(js), dtype=np.float64)  # Intentional float64: accumulator output kept f64
    kernel = _xi_columns_radix_njit if cols.dtype == np.float32 else _xi_columns_njit
    with parallel_kernel_entry():
        kernel(cols, seeds, r_row, l_row, vals)
    out[js] = vals
