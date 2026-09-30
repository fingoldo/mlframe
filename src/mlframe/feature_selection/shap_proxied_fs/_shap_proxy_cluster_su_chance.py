"""Chance-corrected veto of plug-in SU edges between high-cardinality columns.

Plug-in Symmetric Uncertainty is upward-biased by roughly ``(K_i-1)(K_j-1)/(2n)`` nats of MI, so at n=556k two independent
10k-level columns score SU~0.56 and would be merged at the default 0.5 threshold. The analytic Miller-Madow correction is
invalid in that sparse regime (the table has far more cells than rows), so the null is measured instead: shuffle one column
of the pair and re-score. The link is kept only if the chance-corrected ``(SU - SU_null) / (1 - SU_null)`` still clears the
threshold. Corrections only lower SU, so the plug-in kernels' flagged edges are a superset and only those are re-scored.

Edges whose expected bias is negligible against the column entropy (every ordinary low-cardinality pair) are never touched,
so results there stay bit-identical to the uncorrected scan. Pairs where both columns are near-unique (``K > n/2``) carry no
statistical evidence of dependence at all - an ID column and its recoding look exactly like two independent IDs - and are unlinked.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su_joint import dense_relabel_codes

CHANCE_SCOPE_RATIO: float = 0.1
"""An edge is re-scored when ``(K_i-1)(K_j-1)/(2n) > CHANCE_SCOPE_RATIO * min(H_i, H_j)`` (bias vs entropy, both in nats)."""

CHANCE_N_PERM: int = 2

_INSERTION_SORT_MAX: int = 96
"""a-groups up to this many rows are sorted by inline insertion sort (a slice-sort call per group costs more than the sort itself)."""


@njit(nogil=True, cache=True)
def _entropy(counts: np.ndarray, n: int) -> float:
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            h -= p * math.log(p)
    return h


@njit(parallel=True, nogil=True, cache=True)
def _group_orders(packed: np.ndarray, nbins: np.ndarray, counts: np.ndarray, offs: np.ndarray, major: np.ndarray, orders: np.ndarray) -> None:
    """Stable counting-sort row permutation of each ``major`` column into ``orders[r]`` (rows grouped by ascending code)."""
    n = packed.shape[1]
    for r in prange(major.shape[0]):
        c = major[r]
        col = packed[c]
        nb = nbins[c]
        start = np.zeros(nb, dtype=np.int64)
        acc = 0
        for x in range(nb):
            start[x] = acc
            acc += counts[offs[c] + x]
        for k in range(n):
            x = col[k]
            orders[r, start[x]] = k
            start[x] += 1


@njit(nogil=True, cache=True)
def _mi_grouped(order: np.ndarray, ca: np.ndarray, cb: np.ndarray, b: np.ndarray, scratch: np.ndarray) -> float:
    """Plug-in MI of (a, b) with rows pre-grouped by ``a`` code (``order``); only the b values inside each a-group are sorted.

    Visits distinct (a, b) cells in ascending ``a * nb_b + b`` key order and applies the same per-cell expression as a global key sort,
    so the float accumulation is bit-identical to it.
    """
    n = order.shape[0]
    mi = 0.0
    s = 0
    for x in range(ca.shape[0]):
        g = ca[x]
        for k in range(g):
            scratch[k] = b[order[s + k]]
        s += g
        if g > _INSERTION_SORT_MAX:
            scratch[:g].sort()
        else:
            for k in range(1, g):
                v = scratch[k]
                j = k - 1
                while j >= 0 and scratch[j] > v:
                    scratch[j + 1] = scratch[j]
                    j -= 1
                scratch[j + 1] = v
        cur = scratch[0]
        cnt = 1
        for k in range(1, g + 1):
            if k < g and scratch[k] == cur:
                cnt += 1
                continue
            mi += (cnt / n) * math.log((cnt / n) / ((ca[x] / n) * (cb[cur] / n)))
            if k < g:
                cur = scratch[k]
                cnt = 1
    return mi


@njit(parallel=True, nogil=True, cache=True)
def _score_edges(
    packed: np.ndarray, nbins: np.ndarray, counts: np.ndarray, offs: np.ndarray, ent: np.ndarray,
    orders: np.ndarray, order_row: np.ndarray, ea: np.ndarray, eb: np.ndarray, n_perm: int, seed: int,
) -> np.ndarray:
    """Return ``(n_edges, 2)``: plug-in SU and mean permutation-null SU per edge."""
    m = ea.shape[0]
    n = packed.shape[1]
    out = np.zeros((m, 2))
    for e in prange(m):
        a = ea[e]
        b = packed[eb[e]]
        ca = counts[offs[a] : offs[a] + nbins[a]]
        cb = counts[offs[eb[e]] : offs[eb[e]] + nbins[eb[e]]]
        order = orders[order_row[a]]
        denom = ent[a] + ent[eb[e]]
        scratch = np.empty(n, dtype=np.int32)
        out[e, 0] = 2.0 * _mi_grouped(order, ca, cb, b, scratch) / denom
        shuf = b.copy()
        state = np.uint64(seed) * np.uint64(6364136223846793005) + np.uint64(e + 1) * np.uint64(1442695040888963407)
        acc = 0.0
        for _ in range(n_perm):
            for k in range(n - 1, 0, -1):
                state ^= state << np.uint64(13)
                state ^= state >> np.uint64(7)
                state ^= state << np.uint64(17)
                j = np.int64(state % np.uint64(k + 1))
                t = shuf[k]
                shuf[k] = shuf[j]
                shuf[j] = t
            acc += 2.0 * _mi_grouped(order, ca, cb, shuf, scratch) / denom
        out[e, 1] = acc / n_perm
    return out


def veto_chance_edges(
    arrays: list[np.ndarray],
    ei: np.ndarray,
    ej: np.ndarray,
    threshold: float,
    *,
    scope_ratio: float = CHANCE_SCOPE_RATIO,
    n_perm: int = CHANCE_N_PERM,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop flagged edges that are indistinguishable from chance; return the surviving ``(ei, ej)``."""
    if ei.size == 0:
        return ei, ej
    n = int(arrays[0].shape[0])
    cols = np.unique(np.concatenate([ei, ej]))
    pos = {int(c): k for k, c in enumerate(cols)}
    dense = np.empty((cols.size, n), dtype=np.int32)
    nbins = np.empty(cols.size, dtype=np.int64)
    ent = np.empty(cols.size, dtype=np.float64)
    cnts: list[np.ndarray] = []
    for k, c in enumerate(cols):
        arr = np.ascontiguousarray(arrays[int(c)], dtype=np.int64)
        nbins[k] = dense_relabel_codes(arr, dense[k])
        cc = np.bincount(dense[k], minlength=int(nbins[k])).astype(np.int64)
        cnts.append(cc)
        ent[k] = _entropy(cc, n)
    offs = np.concatenate([[0], np.cumsum(nbins)[:-1]]).astype(np.int64)
    counts = np.concatenate(cnts)

    pa = np.array([pos[int(i)] for i in ei], dtype=np.int64)
    pb = np.array([pos[int(j)] for j in ej], dtype=np.int64)
    ka = nbins[pa].astype(np.float64)
    kb = nbins[pb].astype(np.float64)
    bias = (ka - 1.0) * (kb - 1.0) / (2.0 * n)
    hmin = np.minimum(ent[pa], ent[pb])
    in_scope = bias > scope_ratio * hmin
    keep = np.ones(ei.size, dtype=bool)
    no_evidence = in_scope & (np.minimum(ka, kb) > n / 2.0)
    keep[no_evidence] = False
    todo = np.flatnonzero(in_scope & ~no_evidence)
    if todo.size:
        ta, tb = pa[todo], pb[todo]
        major = np.unique(ta)
        order_row = np.full(cols.size, -1, dtype=np.int64)
        order_row[major] = np.arange(major.size)
        orders = np.empty((major.size, n), dtype=np.int32)
        _group_orders(dense, nbins, counts, offs, major, orders)
        res = _score_edges(dense, nbins, counts, offs, ent, orders, order_row, ta, tb, int(n_perm), int(seed))
        su, null = res[:, 0], res[:, 1]
        adj = (su - null) / np.maximum(1.0 - null, 1e-12)
        keep[todo] = adj >= threshold
    return ei[keep], ej[keep]
