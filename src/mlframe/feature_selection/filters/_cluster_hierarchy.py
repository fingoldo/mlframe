"""Layer 48 (2026-05-31): hierarchical post-hoc clustering for DCD.

DCD is greedy: each feature joins exactly ONE cluster (the first anchor
whose pair SU passes ``tau_cluster``). Genuine data is often
hierarchical: ``temp_sensor_1`` very-closely tracks ``temp_sensor_2``
(sub-cluster), and both moderately track ``humidity_sensor_1``
(super-cluster). The greedy single-anchor rule cannot surface the
super-cluster.

Layer 48 is a read-only POST-FIT analyser. It walks the anchors that
DCD already discovered (``MRMR.dcd_["cluster_anchors_names"]``),
computes pair SU between every pair of level-N anchors via the same
``pair_su`` codepath the live DCD uses, and merges anchors whose
pairwise SU exceeds ``super_tau`` into level-(N+1) super-clusters via
connected-components. The recursion repeats up to ``max_levels``.

The output is a level-keyed dict mapping each super-anchor (the
lexicographically-smallest level-N anchor in its component) to the
list of level-N anchors that merged into it:

    {1: {"temp_sensor_1": ["temp_sensor_2", "humidity_sensor_1"]},
     2: ...}

NO DCD logic changes. The original anchor->member map (level 0) stays
exactly as the greedy pass produced it; Layer 48 only ADDS a
super-structure on top.

Determinism: the analyser is a pure function of
``(dcd_summary, X, super_tau, max_levels)`` -- no RNG, no thread-locals.

Pickleable: only stdlib + numpy types in the returned dict.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _component_medoid(comp: list, pair_sus: dict) -> Any:
    """Pick the cluster representative (super-anchor) as the medoid: the member
    maximising mean within-component SU to the other members, using the
    already-computed ``pair_sus`` dict (keys ordered as (a, b) in the original
    anchor order, so we look up both orderings). Deterministic alphabetical
    tie-break. Audit hierarchy-stability-12: the prior ``comp[0]`` picked the
    lexicographically smallest name, so the cluster label carried no statistical
    meaning. (Mirrors the medoid convention in group_aware / _cluster_aggregate.)
    """
    if len(comp) <= 2:
        return sorted(comp)[0]
    best_member = None
    best_score = -1.0
    for m in comp:
        total = 0.0
        cnt = 0
        for o in comp:
            if o == m:
                continue
            s = pair_sus.get((m, o))
            if s is None:
                s = pair_sus.get((o, m))
            if s is not None:
                total += float(s)
                cnt += 1
        mean_su = (total / cnt) if cnt else 0.0
        # Strictly-greater keeps the first (alphabetical) winner on ties.
        if mean_su > best_score:
            best_score = mean_su
            best_member = m
    return best_member if best_member is not None else sorted(comp)[0]


# Default super-cluster threshold. Lower than the typical Layer-47 / DCD
# ``tau_cluster=0.7`` because the SU between two cluster ANCHORS is by
# construction lower than the within-cluster SU (the anchors were
# selected as representatives of distinct greedy clusters; otherwise
# they would have merged at level 0). 0.5 is the calibrated mid-range
# that surfaces "moderate" cross-cluster ties without re-merging the
# whole feature space into a single super-cluster.
_DEFAULT_SUPER_TAU = 0.5
# Recursion depth cap. 3 levels covers leaves -> sub-clusters ->
# super-clusters -> meta-clusters; deeper structures past 3 are rare in
# tabular ML and the depth cap protects against pathological inputs.
_DEFAULT_MAX_LEVELS = 3


def _quantize_for_su(X) -> tuple:
    """Quantize a DataFrame / ndarray into integer bin codes the same
    way the L47 calibration test does, so the resulting ``factors_data``
    is consumable by ``pair_su`` with ``distance='su'``.

    Returns ``(factors_data, factors_nbins, col_names)``.
    """
    if X is None:
        return None, None, []
    n_bins = 10
    cols: list = []
    nbins: list = []
    names: list = []
    if hasattr(X, "columns"):
        col_iter = list(X.columns)
        for c in col_iter:
            col = X[c].to_numpy(dtype=np.float64, copy=False)
            names.append(str(c))
            edges = np.quantile(col[np.isfinite(col)] if np.isfinite(col).any() else col, np.linspace(0, 1, n_bins + 1))
            edges = np.unique(edges)
            if edges.size < 3:
                binned = np.zeros(col.shape, dtype=np.int32)
                nb = 1
            else:
                binned = np.searchsorted(edges[1:-1], col, side="right").astype(np.int32)
                nb = int(binned.max()) + 1
            cols.append(binned)
            nbins.append(nb)
    else:
        arr = np.asarray(X)
        if arr.ndim != 2:
            return None, None, []
        for j in range(arr.shape[1]):
            col = arr[:, j].astype(np.float64, copy=False)
            names.append(f"col_{j}")
            # 2026-06-03 (audit hierarchy-stability-3): mirror the DataFrame
            # branch's finite-mask. np.quantile over a column containing any
            # NaN/Inf returns NaN edges -> searchsorted assigns a single bin ->
            # SU(this anchor, *)=0, so the anchor silently never merges. The
            # DataFrame path already guards this at the sibling branch above.
            edges = np.quantile(
                col[np.isfinite(col)] if np.isfinite(col).any() else col,
                np.linspace(0, 1, n_bins + 1),
            )
            edges = np.unique(edges)
            if edges.size < 3:
                binned = np.zeros(col.shape, dtype=np.int32)
                nb = 1
            else:
                binned = np.searchsorted(edges[1:-1], col, side="right").astype(np.int32)
                nb = int(binned.max()) + 1
            cols.append(binned)
            nbins.append(nb)
    if not cols:
        return None, None, []
    factors_data = np.column_stack(cols)
    factors_nbins = np.asarray(nbins, dtype=np.int64)
    return factors_data, factors_nbins, names


def _resolve_anchor_indices(anchor_names: list, name_to_idx: dict) -> dict:
    """Map each anchor name to its column index in the quantised matrix.
    Anchors that don't resolve (engineered ``_dcd_pc1_*`` aggregates not
    present in ``X``) are dropped: hierarchy is built only over anchors
    we can actually score.
    """
    resolved: dict = {}
    for name in anchor_names:
        idx = name_to_idx.get(str(name))
        if idx is None:
            continue
        resolved[str(name)] = int(idx)
    return resolved


def _components_from_pair_sus(
    anchor_names: list,
    pair_sus: dict,
    super_tau: float,
) -> list:
    """Connected-components over the graph
    ``{anchors} x {pair_sus[(a,b)] > super_tau}``.

    Returns a list of sorted-by-name anchor lists, one per component.
    Anchors with no super-tau edges form singleton components.
    """
    parent: dict = {a: a for a in anchor_names}

    def find(x: str) -> str:
        # Iterative path compression -- stays pickle/clone-safe (no closures
        # captured in stored data).
        root = x
        while parent[root] != root:
            root = parent[root]
        cur = x
        while parent[cur] != root:
            nxt = parent[cur]
            parent[cur] = root
            cur = nxt
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Deterministic root: lexicographically smaller name wins.
        if ra <= rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for (a, b), su in pair_sus.items():
        if su > float(super_tau):
            union(a, b)
    groups: dict = {}
    for a in anchor_names:
        r = find(a)
        groups.setdefault(r, []).append(a)
    return [sorted(g) for g in groups.values()]


def build_cluster_hierarchy(
    dcd_summary: Optional[dict],
    X: Any,
    super_tau: float = _DEFAULT_SUPER_TAU,
    max_levels: int = _DEFAULT_MAX_LEVELS,
    distance: str = "su",
) -> dict:
    """Build a hierarchical post-hoc cluster map over DCD anchors.

    Parameters
    ----------
    dcd_summary
        The ``MRMR.dcd_`` dict produced by Layer 41+. Must carry
        ``cluster_anchors_names`` (anchor_name -> sorted member_name list).
        ``None`` / missing keys / empty maps -> returns ``{}``.
    X
        The raw feature matrix (DataFrame or ndarray) used at fit time.
        Required to compute SU between anchors -- the analyser does not
        peek into DCD state.
    super_tau
        Pairwise-SU threshold for merging two level-N anchors into a
        level-(N+1) super-cluster. Default 0.5; higher = stricter (fewer
        merges).
    max_levels
        Recursion depth cap. Default 3.
    distance
        Distance metric forwarded to ``pair_su``. ``"su"`` matches the
        live DCD default; ``"auto"`` / ``"vi"`` are accepted for parity
        with Layer 46.

    Returns
    -------
    dict[int, dict[str, list[str]]]
        Maps ``level`` (1..) to a dict of ``{super_anchor: [sub_anchors]}``.
        Level 1 merges level-0 anchors (the DCD-discovered ones); level 2
        merges level-1 super-anchors; etc. Empty dict when:
        - ``dcd_summary`` is None or has no ``cluster_anchors_names``
        - fewer than 2 anchors resolve against ``X``
        - no pair exceeds ``super_tau`` at level 1 (no super-structure)

    Notes
    -----
    Read-only: no mutation of ``dcd_summary`` or ``X``. Deterministic:
    no RNG, no thread-locals. Pickleable: returns only str/int/list/dict.
    """
    if dcd_summary is None:
        return {}
    anchors_map = dcd_summary.get("cluster_anchors_names")
    if not isinstance(anchors_map, dict) or not anchors_map:
        return {}
    level_anchors = sorted(str(a) for a in anchors_map.keys())
    if len(level_anchors) < 2:
        return {}
    if X is None:
        return {}
    factors_data, factors_nbins, col_names = _quantize_for_su(X)
    if factors_data is None or len(col_names) == 0:
        return {}
    name_to_idx = {str(c): i for i, c in enumerate(col_names)}
    # Lazy-import to break the circular dependency with the parent module.
    from ._dynamic_cluster_discovery import DCDState, pair_su
    # Layer 51 (2026-05-31): batched pairwise-SU dispatch for the
    # anchors-cross-anchors O(K^2) sweep at every hierarchy level.
    # Bit-equivalent to looped pair_su; pre-warms the per-column
    # entropy cache for all anchor columns in one sibling-column pass.
    from ._dcd_pair_su_batch import pair_su_batch

    cal_state = DCDState(
        pool_pruned_mask=np.zeros(factors_data.shape[1], dtype=bool),
        factors_data=factors_data,
        factors_nbins=np.asarray(factors_nbins),
        cols=list(col_names),
        nbins=np.asarray(factors_nbins),
        distance=str(distance),
    )

    hierarchy: dict = {}
    # 2026-06-03 (audit hierarchy-stability-7): count SU-dispatch failures so a
    # systematic error stops masquerading as "clusters stabilised". A pair that
    # errors defaults to SU=0 (treated as non-redundant -> never merges), so a
    # silent failure would hide genuine super-structure.
    n_failed_pairs = 0
    max_levels_eff = max(1, int(max_levels))
    for level in range(1, max_levels_eff + 1):
        if len(level_anchors) < 2:
            break
        resolved = _resolve_anchor_indices(level_anchors, name_to_idx)
        # Anchors not present in X are silently dropped from level-N
        # consideration (engineered post-swap aggregates can't be scored
        # against the raw matrix). If fewer than 2 anchors resolve, no
        # super-structure can form at this level -- stop.
        if len(resolved) < 2:
            break
        resolved_names = sorted(resolved.keys())
        pair_sus: dict = {}
        # Layer 51: batched pairwise-SU dispatch. The (a, b) name pairs
        # are mapped to their resolved column indices in one pass; the
        # batch call pre-warms per-column entropies for every anchor
        # column once instead of paying the marginal cost per pair.
        index_pairs: list = []
        name_pairs: list = []
        for i in range(len(resolved_names)):
            for j in range(i + 1, len(resolved_names)):
                a_name, b_name = resolved_names[i], resolved_names[j]
                a_idx = int(resolved[a_name])
                b_idx = int(resolved[b_name])
                index_pairs.append((a_idx, b_idx))
                name_pairs.append((a_name, b_name))
        try:
            scores = pair_su_batch(cal_state, index_pairs)
        except Exception as exc:
            logger.warning(
                "build_cluster_hierarchy: batched pair-SU dispatch failed at "
                "level %d (%r); falling back to per-pair scoring.", level, exc,
            )
            scores = None
        if scores is None:
            for (a_name, b_name), (a_idx, b_idx) in zip(name_pairs, index_pairs):
                try:
                    s = pair_su(cal_state, a_idx, b_idx)
                except Exception as exc:
                    s = 0.0
                    n_failed_pairs += 1
                    if n_failed_pairs == 1:
                        logger.warning(
                            "build_cluster_hierarchy: per-pair SU failed (e.g. "
                            "(%s,%s): %r); affected pairs default to SU=0 "
                            "(treated as non-redundant). Further failures "
                            "suppressed for this build.", a_name, b_name, exc,
                        )
                pair_sus[(a_name, b_name)] = float(s) if np.isfinite(s) else 0.0
        else:
            for (a_name, b_name), s in zip(name_pairs, scores):
                pair_sus[(a_name, b_name)] = float(s) if np.isfinite(s) else 0.0
        components = _components_from_pair_sus(
            resolved_names, pair_sus, super_tau,
        )
        # Keep only non-trivial components (size >= 2) for the level map.
        # Singletons stay implicit: an anchor not present in any level-N
        # value list means it didn't merge with anything at this level.
        level_dict: dict = {}
        for comp in components:
            if len(comp) < 2:
                continue
            # 2026-06-03 (audit hierarchy-stability-12): pick the cluster
            # representative as the medoid (most-central anchor by mean
            # within-component SU) instead of the lexicographically smallest
            # name, so the super-anchor label carries statistical meaning.
            super_anchor = _component_medoid(comp, pair_sus)
            sub_anchors = [a for a in comp if a != super_anchor]
            level_dict[super_anchor] = sub_anchors
        if not level_dict:
            # No merges at this level -- the hierarchy has stabilised.
            break
        hierarchy[int(level)] = level_dict
        # Prepare next level: super-anchors of this level become the
        # candidates for level+1. Anchors not merged at this level drop
        # out (they have no super-partner to test against).
        next_level_anchors = sorted(level_dict.keys())
        if len(next_level_anchors) == len(level_anchors):
            # Degenerate cycle guard: if the next level has as many
            # candidates as the current one, no merging actually
            # happened (every component was size 1 except trivially) --
            # already guarded above by ``not level_dict``, but the cycle
            # check is cheap insurance against pathological inputs.
            break
        level_anchors = next_level_anchors
    return hierarchy


__all__ = [
    "build_cluster_hierarchy",
    "_DEFAULT_SUPER_TAU",
    "_DEFAULT_MAX_LEVELS",
]
