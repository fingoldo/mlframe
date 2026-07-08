"""Interaction-aware subset proxy scoring (``proxy_mode="interaction"``).

The default subset proxy (``proxy_mode="additive"``) scores a subset ``S`` as the purely ADDITIVE SHAP
coalition value ``base + sum_{j in S} phi_j``. That folds every pairwise interaction into the two
operands' MAIN effects, so it UNDER-scores a subset whose value comes from a non-additive PAIR (XOR /
multiplicative -- each operand's main effect is ~0) and OVER-scores redundant additive features.

This module adds the interaction-aware proxy: score ``S`` as

    base + sum_{j in S} phi_j + sum_{i<j in S} 2*Phi_ij

where ``Phi_ij`` is the (symmetric, off-diagonal) TreeSHAP INTERACTION value for the pair (the factor 2
folds the symmetric ``Phi_ij + Phi_ji`` the tensor carries). The additive term is the unchanged
main-effect coalition; the pairwise term lets a subset earn credit for joint signal its members
produce together. For tree models the interaction tensor comes from the fast numba TreeSHAP kernel
(``interaction_tensor_numba``); for non-tree models the caller has no tensor and falls back to the
additive proxy.

GATE for tractability: the dense pairwise sum is O(|S|^2). To keep it bounded on wide proxies the
interaction term is restricted to the ``interaction_top_k`` features with the largest mean ``|phi|``;
pairs touching a non-top-k feature contribute only their additive main effect. So the per-subset cost
is O(|S| + k^2) not O(|S| + P^2), and the precomputed pair-sum table is O(k^2) memory, not O(P^2).
This keeps the interaction proxy usable at the same widths as the additive search (default k=30).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_objective import (
    METRIC_CODES,
    resolve_metric,
    score_margin_auto,
)

# Default gate width: the interaction term is summed only over the top-k features by mean |phi|.
# k=30 keeps the O(k^2)=900-cell pair table cheap while covering the informative head on every bed we
# benched (informatives are always inside the |phi| top-30 at the post-prescreen widths). Tunable.
_DEFAULT_INTERACTION_TOP_K = 30


def build_pair_table(Phi: np.ndarray, phi: np.ndarray, interaction_top_k: int):
    """Build the PER-ROW gated interaction tensor used by the interaction proxy scorer.

    Returns ``(pair_rows, in_gate)`` where:
      * ``pair_rows`` is a (P, P, n) float64 tensor with ``pair_rows[i, j] = Phi[:, i, j]`` (the per-row
        off-diagonal interaction values) for both ``i, j`` in the top-k |phi| gate (i != j), zero
        otherwise. Symmetric in (i, j). Row-major over (i, j) so the scorer's per-pair access is a
        contiguous length-n read.
      * ``in_gate`` is a (P,) bool mask of the gated features.

    The interaction term MUST stay PER-ROW: for an XOR pair the row mean of ``Phi_ij`` is ~0 (the sign
    of the interaction flips with the operand quadrant), so collapsing to a scalar would erase exactly
    the signal this proxy exists to capture. Memory is O(k^2 * n) (k=interaction_top_k, not P), the
    whole point of the gate -- a full (P, P, n) tensor would be prohibitive on wide proxies.
    """
    n, P, _ = Phi.shape
    imp = np.abs(phi).mean(axis=0)
    k = min(int(interaction_top_k), P)
    gate_idx = np.argsort(-imp)[:k]
    in_gate = np.zeros(P, dtype=bool)
    in_gate[gate_idx] = True
    pair_rows = np.zeros((P, P, n), dtype=np.float64)
    if k >= 2:
        gi = np.sort(gate_idx)
        block = np.ascontiguousarray(np.transpose(Phi[:, gi][:, :, gi], (1, 2, 0)))  # (k, k, n)
        for a in range(k):
            for b in range(k):
                if a != b:
                    pair_rows[gi[a], gi[b]] = block[a, b]
    return pair_rows, in_gate


@njit(cache=True)
def _interaction_extra(idx: np.ndarray, pair_rows: np.ndarray, out: np.ndarray) -> None:
    """Accumulate ``2 * sum_{i<j in idx} Phi[:, i, j]`` into ``out`` (per-row interaction shift).

    ``pair_rows`` is zero for non-gated features, so iterating only over the subset's own indices
    already realises the top-k gate (a non-gated member contributes 0 to every pair)."""
    n = out.shape[0]
    r = idx.shape[0]
    for a in range(r):
        ia = idx[a]
        for b in range(a + 1, r):
            row = pair_rows[ia, idx[b]]
            for t in range(n):
                out[t] += 2.0 * row[t]


def interaction_proxy_top_n(
    phi: np.ndarray,
    Phi: np.ndarray,
    base: np.ndarray,
    y: np.ndarray,
    *,
    classification: bool,
    metric: str | None = None,
    min_card: int = 1,
    max_card: int | None = None,
    top_n: int = 30,
    interaction_top_k: int = _DEFAULT_INTERACTION_TOP_K,
    candidate_subsets=None,
):
    """Rank feature subsets by the interaction-aware coalition loss; return top-N ``(loss, idx_tuple)``.

    ``phi`` (n, P) main-effect SHAP, ``Phi`` (n, P, P) interaction tensor, ``base`` (n,) per-row offset.
    The margin of a subset ``S`` is ``base + sum_{j in S} phi_j + 2*sum_{i<j in S, both gated} Phi_ij``.

    ``candidate_subsets`` (optional): an iterable of index tuples to RE-SCORE under the interaction
    proxy (e.g. the additive search's top-N) so the interaction term can re-rank them without a fresh
    combinatorial search. When ``None`` a greedy-forward + top single/pair sweep is run so pure
    interaction pairs the additive search missed still surface.
    """
    metric = resolve_metric(classification, metric)
    code = METRIC_CODES.get(metric, 1)
    phi = np.ascontiguousarray(phi, dtype=np.float64)
    base = np.ascontiguousarray(base, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    _n, P = phi.shape
    max_card = P if max_card is None else min(max_card, P)
    phi_T = np.ascontiguousarray(phi.T)
    pair_rows, in_gate = build_pair_table(Phi, phi, interaction_top_k)
    cache: dict[tuple[int, ...], float] = {}

    def _loss(key: tuple[int, ...]) -> float:
        """Cached greedy-search objective for a candidate feature subset ``key`` (empty subset scores ``+inf``, never selected)."""
        if not key:
            return float("inf")
        v = cache.get(key)
        if v is not None:
            return v
        idx = np.asarray(key, dtype=np.int64)
        margin = base + phi_T[idx].sum(axis=0)
        _interaction_extra(idx, pair_rows, margin)
        if metric == "auc":
            from mlframe.feature_selection.shap_proxied_fs._shap_proxy_objective import proxy_loss
            v = proxy_loss(margin, y, "auc")
        elif metric == "rmse":
            from math import sqrt
            v = sqrt(score_margin_auto(margin, y, 1))
        else:
            v = float(score_margin_auto(margin, y, code))
        cache[key] = v
        return v

    # Re-score externally supplied candidates (the additive top-N) under the interaction proxy.
    if candidate_subsets is not None:
        for c in candidate_subsets:
            key = tuple(sorted(int(i) for i in c))
            if min_card <= len(key) <= max_card:
                _loss(key)

    # Always sweep gated singletons + gated pairs so pure-interaction pairs (XOR) surface even when the
    # additive candidates never included them. O(k^2) pairs, k = interaction_top_k.
    gated = [int(i) for i in np.flatnonzero(in_gate)]
    if min_card <= 1:
        for j in gated:
            _loss((j,))
    if max_card >= 2:
        for a in range(len(gated)):
            for b in range(a + 1, len(gated)):
                _loss(tuple(sorted((gated[a], gated[b]))))

    # Greedy-forward over the interaction proxy (recovers larger interacting subsets cheaply).
    current: tuple[int, ...] = ()
    best = float("inf")
    remaining = set(range(P))
    while remaining and len(current) < max_card:
        cand, cl = None, float("inf")
        for j in remaining:
            lo = _loss(tuple(sorted(current + (j,))))
            if lo < cl:
                cl, cand = lo, j
        if cand is None or cl >= best:
            break
        current = tuple(sorted(current + (cand,)))
        best = cl
        remaining.discard(cand)

    items = [(v, k) for k, v in cache.items() if k and np.isfinite(v)]
    items.sort(key=lambda t: t[0])
    return items[:top_n]
