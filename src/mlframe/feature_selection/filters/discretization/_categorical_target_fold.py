"""Target-aware folding of the levels of a categorical column (``categorical_fold_strategy="target_stat"``).

The frequency cap keeps the most frequent levels and drops the rest into one bucket, whatever the target says about them; the ``min_support`` rule folds every level under a fixed number of rows. Both can throw
away a rare level that carries the signal (a merchant seen 12 times, all of them frauds). Here two levels share a code only when the target cannot tell them apart.

The test is the G statistic of the merge. For two groups with class counts ``a`` and ``b`` the merge loses ``dG = 2 (nA + nB) JS(pA, pB)`` of log-likelihood (twice the mutual information the two levels carry about the
target); the merge is accepted when ``dG < (L - 1) ln n`` (the BIC price of the ``L - 1`` extra class probabilities), cheapest merge first. A small level is therefore kept apart exactly when its class mix differs from
every other group's by more than chance can produce at its size: three rows that are all of a 2 % class are kept, three rows of a 50 % class are not. The mapping is built from all rows, so the column's MI is
estimated in sample; the BIC price keeps that optimism small and ``_binning`` marks the column target-dependent, so the permutation null refolds it for every permuted target.
"""

from __future__ import annotations

import numpy as np
from numba import njit

#: At most this many levels enter the agglomeration (the cost is cubic in it); the rest of a wider column join one pool, except the levels that differ from the base rate on their own.
_POOL = 512
#: Classes of a continuous target: its quantile classes.
_TARGET_CLASSES = 8
_MAX_DISCRETE_CLASSES = 32


def target_classes(y: np.ndarray) -> "np.ndarray | None":
    """Class code (0..L-1, -1 for a missing value) of every row of the target ``y``: its own values when it has at most 32 distinct ones, else 8 quantile classes; ``None`` for a constant target."""
    y = np.asarray(y).ravel()
    if y.dtype.kind in "fc":
        finite = np.isfinite(y)
        base = y[finite]
    else:
        finite = np.ones(y.shape[0], dtype=bool)
        base = y
    if base.size == 0:
        return None
    values, inverse = np.unique(base, return_inverse=True)
    if values.size < 2:
        return None
    codes = np.full(y.shape[0], -1, dtype=np.int64)
    if values.size <= _MAX_DISCRETE_CLASSES:
        codes[finite] = inverse.ravel()
    else:
        edges = np.unique(np.quantile(base.astype(np.float64), np.linspace(0.0, 1.0, _TARGET_CLASSES + 1)[1:-1]))
        codes[finite] = np.searchsorted(edges, base, side="right")
    return codes


@njit(cache=True, nogil=True)
def _merge_cost(a, b, na, nb):
    """``dG`` of merging two groups of class counts ``a`` and ``b`` (sizes ``na``, ``nb``): twice the log-likelihood the merge gives up."""
    n = na + nb
    total = 0.0
    for c in range(a.shape[0]):
        s = a[c] + b[c]
        if s > 0.0:
            if a[c] > 0.0:
                total += a[c] * np.log(a[c] * n / (na * s))
            if b[c] > 0.0:
                total += b[c] * np.log(b[c] * n / (nb * s))
    return 2.0 * total


@njit(cache=True, nogil=True)
def _initial_costs(cnt, size, alive):
    """The symmetric ``(k, k)`` merge-cost matrix of the live levels (``inf`` for a dead level or the diagonal)."""
    k = cnt.shape[0]
    cost = np.full((k, k), np.inf)
    for i in range(k):
        if not alive[i]:
            continue
        for j in range(i + 1, k):
            if alive[j]:
                c = _merge_cost(cnt[i], cnt[j], size[i], size[j])
                cost[i, j] = c
                cost[j, i] = c
    return cost


def _cheapest_pair(cost, alive):
    """The live pair ``(best cost, i, j)`` with the lowest merge cost, the first in row-major order on a tie (``(inf, -1, -1)`` when no pair is live)."""
    k = cost.shape[0]
    best = np.inf
    bi = -1
    bj = -1
    for i in range(k):
        if not alive[i]:
            continue
        for j in range(i + 1, k):
            if alive[j] and cost[i, j] < best:
                best = cost[i, j]
                bi = i
                bj = j
    return best, bi, bj


def _merge_levels(cnt, size, alive, label, cost, bi, bj):
    """Merge level ``bj`` into ``bi`` in place: counts, sizes, labels and the cost row and column of ``bi`` are updated, those of ``bj`` are blanked."""
    k = cnt.shape[0]
    cnt[bi] += cnt[bj]
    size[bi] += size[bj]
    alive[bj] = False
    for r in range(k):
        if label[r] == bj:
            label[r] = bi
    for j in range(k):
        cost[bj, j] = np.inf
        cost[j, bj] = np.inf
    for j in range(k):
        if alive[j] and j != bi:
            c = _merge_cost(cnt[bi], cnt[j], size[bi], size[j])
            cost[bi, j] = c
            cost[j, bi] = c


def _agglomerate(counts, threshold, max_groups):
    """Group label of every row of ``counts`` (levels x classes): the cheapest merge goes first, and merging continues while the cheapest costs less than ``threshold`` or more than ``max_groups`` groups remain. A level with no rows keeps its own label."""
    k = counts.shape[0]
    cnt = counts.copy()
    size = np.zeros(k)
    for i in range(k):
        size[i] = cnt[i].sum()
    alive = size > 0.0
    label = np.arange(k)
    cost = _initial_costs(cnt, size, alive)
    n_alive = int(alive.sum())
    while n_alive > 1:
        best, bi, bj = _cheapest_pair(cost, alive)
        if best >= threshold and n_alive <= max_groups:
            break
        _merge_levels(cnt, size, alive, label, cost, bi, bj)
        n_alive -= 1
    return label


def _single_level_g(counts: np.ndarray, base: np.ndarray) -> np.ndarray:
    """``G`` of every level against the base class rates ``base``: ``2 sum_c n_kc ln(n_kc / (n_k p_c))``."""
    n_k = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(counts > 0, counts * np.log(counts / (n_k * base[None, :])), 0.0)
    return np.asarray(2.0 * term.sum(axis=1))


def fold_column_by_target(col: np.ndarray, y_codes: np.ndarray, max_cardinality: int) -> np.ndarray:
    """The column ``col`` (float codes, ``-1`` for a missing level) with its levels merged by the target: levels the class mix cannot tell apart share a code, the codes are dense from 0 in order of group size,
    a missing value stays ``-1``. ``y_codes`` holds the class of each row (``-1`` rows are not counted). The column is returned unchanged when the target has fewer than two classes or no level is merged."""
    mask = col >= 0
    levels = col[mask].astype(np.int64)
    if levels.size == 0:
        return col
    n_levels = int(levels.max()) + 1
    ok = mask & (y_codes >= 0)
    n_classes = int(y_codes.max()) + 1 if (y_codes >= 0).any() else 0
    if n_classes < 2 or int(ok.sum()) < 2:
        return col
    counts = np.zeros((n_levels, n_classes))
    np.add.at(counts, (col[ok].astype(np.int64), y_codes[ok]), 1.0)
    n = float(counts.sum())
    threshold = (n_classes - 1) * float(np.log(n))
    pool_of = np.arange(n_levels)
    if n_levels > _POOL:
        # a column wider than the pool: the most frequent levels and every level that differs from the base rate on its own enter the merge, the rest share one pooled row
        base = counts.sum(axis=0) / n
        sizes = counts.sum(axis=1)
        keep = set(np.argsort(-sizes, kind="stable")[: _POOL - 1].tolist()) | set(np.flatnonzero(_single_level_g(counts, base) > threshold).tolist())
        keep_arr = np.array(sorted(keep), dtype=np.int64)
        rest = np.setdiff1d(np.arange(n_levels), keep_arr)
        pooled = np.vstack([counts[keep_arr], counts[rest].sum(axis=0, keepdims=True)]) if rest.size else counts[keep_arr]
        rows = np.concatenate([keep_arr, rest])
        row_of = np.concatenate([np.arange(keep_arr.size), np.full(rest.size, keep_arr.size)])
        pool_of = np.empty(n_levels, dtype=np.int64)
        pool_of[rows] = row_of
        counts = pooled
    labels = _agglomerate(counts, threshold, max(1, int(max_cardinality) - 1))
    level_group = labels[pool_of]
    row_group = level_group[levels]
    sizes_by_group = np.bincount(row_group, minlength=int(labels.size))
    present = np.unique(row_group)
    order = present[np.argsort(-sizes_by_group[present], kind="stable")]
    dense = np.zeros(int(labels.size), dtype=np.int64)
    dense[order] = np.arange(order.size)
    out = col.copy()
    out[mask] = dense[row_group].astype(col.dtype)
    return np.asarray(out)
