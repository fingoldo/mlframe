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


@njit(nogil=True, cache=True)
def _entropy(counts: np.ndarray, n: int) -> float:
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            h -= p * math.log(p)
    return h


@njit(nogil=True, cache=True)
def _mi_sorted(a: np.ndarray, b: np.ndarray, nb_b: int, ca: np.ndarray, cb: np.ndarray, keys: np.ndarray) -> float:
    n = a.shape[0]
    for k in range(n):
        keys[k] = np.int64(a[k]) * nb_b + b[k]
    keys.sort()
    mi = 0.0
    cur = keys[0]
    cnt = 1
    for k in range(1, n + 1):
        if k < n and keys[k] == cur:
            cnt += 1
            continue
        x = cur // nb_b
        y = cur - x * nb_b
        mi += (cnt / n) * math.log((cnt / n) / ((ca[x] / n) * (cb[y] / n)))
        if k < n:
            cur = keys[k]
            cnt = 1
    return mi


@njit(parallel=True, nogil=True, cache=True)
def _score_edges(
    packed: np.ndarray, nbins: np.ndarray, counts: np.ndarray, offs: np.ndarray, ent: np.ndarray,
    ea: np.ndarray, eb: np.ndarray, n_perm: int, seed: int,
) -> np.ndarray:
    """Return ``(n_edges, 2)``: plug-in SU and mean permutation-null SU per edge."""
    m = ea.shape[0]
    n = packed.shape[1]
    out = np.zeros((m, 2))
    for e in prange(m):
        a = packed[ea[e]]
        b = packed[eb[e]]
        ca = counts[offs[ea[e]]: offs[ea[e]] + nbins[ea[e]]]
        cb = counts[offs[eb[e]]: offs[eb[e]] + nbins[eb[e]]]
        nb_b = nbins[eb[e]]
        denom = ent[ea[e]] + ent[eb[e]]
        keys = np.empty(n, dtype=np.int64)
        out[e, 0] = 2.0 * _mi_sorted(a, b, nb_b, ca, cb, keys) / denom
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
            acc += 2.0 * _mi_sorted(a, shuf, nb_b, ca, cb, keys) / denom
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
        res = _score_edges(dense, nbins, counts, offs, ent, pa[todo], pb[todo], int(n_perm), int(seed))
        su, null = res[:, 0], res[:, 1]
        adj = (su - null) / np.maximum(1.0 - null, 1e-12)
        keep[todo] = adj >= threshold
    return ei[keep], ej[keep]
