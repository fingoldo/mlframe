"""Layer 48: hierarchical post-hoc clustering for DCD.

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
``(dcd_summary, X, super_tau, max_levels)`` - no RNG, no thread-locals.

Pickleable: only stdlib + numpy types in the returned dict.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)


def _component_medoid(comp: list, pair_sus: dict) -> Any:
    """Pick the cluster representative (super-anchor) as the medoid: the member
    maximising mean within-component SU to the other members, using the
    already-computed ``pair_sus`` dict (keys ordered as (a, b) in the original
    anchor order, so we look up both orderings). Deterministic alphabetical
    tie-break. Audit hierarchy-stability-12: the prior ``comp[0]`` picked the
    lexicographically smallest name, so the cluster label carried no statistical
    meaning. (Mirrors the medoid convention in correlated_features / _cluster_aggregate.)
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


def _su_codes_of_finite_columns(columns: list, n_bins: int) -> list:
    """``(int32 codes, bin count)`` for every all-finite column of ``columns`` (``None`` for the others), binned together in parallel blocks.

    The rule is the quantiser's own: ``np.unique`` of the ``n_bins + 1`` quantile edges, fewer than three distinct edges -> one class, else ``searchsorted(edges[1:-1], x, "right")``."""
    from .discretization import quantile_bin_block_fast as quantile_bin_block

    out: list = [None] * len(columns)
    if not columns:
        return out
    n = int(columns[0].shape[0])
    block = max(1, (128 << 20) // (np.dtype(_crit_np_dtype()).itemsize * max(n, 1)))
    for j0 in range(0, len(columns), block):
        part = columns[j0 : j0 + block]
        v = np.ascontiguousarray(np.stack([np.asarray(c, dtype=_crit_np_dtype()) for c in part]))
        ok = np.isfinite(v).all(axis=1)
        if not ok.any():
            continue
        codes = np.empty((int(ok.sum()), n), dtype=np.int16)
        quantile_bin_block(np.ascontiguousarray(v[ok]), int(n_bins), codes, False)
        r = 0
        for j in range(len(part)):
            if ok[j]:
                binned = codes[r].astype(np.int32)
                out[j0 + j] = (binned, int(binned.max()) + 1 if n else 1)
                r += 1
    return out


def _su_codes_into(columns: list, n_bins: int, factors_data: np.ndarray, nbins: np.ndarray, j0: int) -> list:
    """Quantise the all-finite columns of a block straight into columns ``j0 + k`` of the code matrix and their bin counts into ``nbins``; the positions it left to the per-column rule (a non-finite column).

    The same codes as :func:`_su_codes_of_finite_columns` (the quantiser's own rule), written as one tiled block instead of a column at a time: a column assignment into a row-major byte matrix is
    one write every ``p`` bytes (7.7 s of the 500000 x 1800 quantisation), and the per-column ``astype`` and ``max`` were two more passes."""
    from ._column_gather import scatter_columns_rm
    from .discretization import quantile_bin_block_fast as quantile_bin_block

    n = int(columns[0].shape[0])
    v = np.ascontiguousarray(np.stack([np.asarray(c, dtype=_crit_np_dtype()) for c in columns]))
    ok = np.flatnonzero(np.isfinite(v).all(axis=1))
    if ok.size:
        codes = np.empty((ok.size, n), dtype=np.int16)
        quantile_bin_block(v if ok.size == v.shape[0] else np.ascontiguousarray(v[ok]), int(n_bins), codes, False)  # a block of all-finite columns is binned where it is (the fancy index copied it)
        nbins[j0 + ok] = (codes.max(axis=1).astype(np.int64) + 1) if n else 1
        scatter_columns_rm(factors_data, j0 + ok, codes)
    left = np.ones(len(columns), dtype=bool)
    left[ok] = False
    pending: list = np.flatnonzero(left).tolist()
    return pending


def _su_code_column(col: np.ndarray, n_bins: int) -> tuple:
    """``(int32 codes, bin count)`` of one column that has non-finite values (or needs the plain rule): the quantiser's edges over the finite part, ``searchsorted`` against the inner edges."""
    edges = np.quantile(col[np.isfinite(col)] if np.isfinite(col).any() else col, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return np.zeros(col.shape, dtype=np.int32), 1
    binned = np.searchsorted(edges[1:-1], col, side="right").astype(np.int32)
    return binned, int(binned.max()) + 1


def _su_codes_device(X_dev, n_bins: int) -> tuple:
    """Device twin of the SU quantiser (:func:`_su_codes_of_finite_columns`) for an all-finite ``(n, k)`` device matrix: ``(codes, nbins)`` as device arrays.

    One sort per call: the sorted columns give the quantile edges (the coder's interpolation), the distinct-edge count of the SU rule, and the inner edges the codes are
    counted against (the coder's exclusion rule, with the two-valued column set to one class where the SU rule has fewer than three distinct edges). Equal to the host
    quantiser per column (tested); the shared device coder is not called, so its second sort is not paid.
    """
    import cupy as cp

    n = int(X_dev.shape[0])
    xs = cp.sort(X_dev, axis=0)
    pos = cp.linspace(0.0, 100.0, int(n_bins) + 1) / 100.0 * (n - 1)
    lo = cp.floor(pos).astype(cp.int64)
    hi = cp.minimum(lo + 1, n - 1)
    frac = pos - lo
    diff = xs[hi] - xs[lo]
    edges = cp.where(frac[:, None] >= 0.5, xs[hi] - diff * (1 - frac)[:, None], xs[lo] + diff * frac[:, None])
    distinct = 1 + (edges[1:] != edges[:-1]).sum(axis=0)
    emin = edges[0]
    emax = edges[-1]
    dup = cp.zeros(edges.shape, dtype=bool)
    dup[1:] = edges[1:] == edges[:-1]
    has_interior = cp.any((edges > emin) & (edges < emax), axis=0)
    excl = (edges == emin) | dup | (has_interior[None, :] & (edges == emax))
    e2 = cp.sort(cp.where(excl, cp.inf, edges), axis=0).T  # (k, n_bins + 1), ascending with the +inf tail
    codes = cp.empty((n, int(X_dev.shape[1])), dtype=cp.int64)
    for c in range(int(X_dev.shape[1])):
        # side="right": the count of inner edges <= value, i.e. the coder's code (the +inf tail never counts)
        codes[:, c] = cp.searchsorted(e2[c], X_dev[:, c], side="right")
    one_class = distinct < 3
    codes = cp.where(one_class[None, :], 0, codes)
    nbins = cp.where(one_class, 1, codes.max(axis=0) + 1)
    return codes, nbins


def _quantize_for_su(X) -> tuple:
    """Quantize a DataFrame / ndarray into integer bin codes the same
    way the L47 calibration test does, so the resulting ``factors_data``
    is consumable by ``pair_su`` with ``distance='su'``.

    Returns ``(factors_data, factors_nbins, col_names)``.

    The code matrix (int8) is allocated once and filled a block of columns at a time: the per-column code arrays, the list of them kept for the finite columns and the ``column_stack`` copy of
    that list were three ``(n, p)`` int32 arrays at once (1.2 GB at a million rows x 100 columns).
    """
    if X is None:
        return None, None, []
    n_bins = 10
    if hasattr(X, "columns"):
        col_iter = list(X.columns)
        names = [str(c) for c in col_iter]

        def get(ci: int) -> np.ndarray:
            """Column ``ci`` of the frame in the criterion dtype."""
            return np.asarray(X[col_iter[ci]].to_numpy(dtype=_crit_np_dtype(), copy=False))

        n = len(X)
    else:
        arr = np.asarray(X)
        if arr.ndim != 2:
            return None, None, []
        names = [f"col_{j}" for j in range(arr.shape[1])]

        def get(ci: int) -> np.ndarray:
            """Column ``ci`` of the array in the criterion dtype."""
            return arr[:, ci].astype(_crit_np_dtype(), copy=False)

        n = int(arr.shape[0])
    p = len(names)
    if p == 0:
        return None, None, []
    # the codes are below n_bins (10): one byte each, a quarter of the int32 matrix (400 MB less at a million rows x 100 columns); the SU kernels read the narrow codes the fit's own matrix holds
    factors_data = np.empty((n, p), dtype=np.int8 if n_bins <= 127 else np.int32)
    nbins = np.empty(p, dtype=np.int64)
    # a block of columns is binned together by the parallel block quantiser; the block holds at most 128 MB of values
    block = max(1, (128 << 20) // (np.dtype(_crit_np_dtype()).itemsize * max(n, 1)))
    use_device = _su_device_enabled(n, p)
    for j0 in range(0, p, block):
        cols_block = [get(ci) for ci in range(j0, min(p, j0 + block))]
        if use_device and all(np.isfinite(c).all() for c in cols_block):
            try:
                import cupy as cp

                codes_dev, nb_dev = _su_codes_device(cp.asarray(np.column_stack(cols_block)), n_bins)
                factors_data[:, j0 : j0 + len(cols_block)] = cp.asnumpy(codes_dev.astype(cp.int8)).astype(factors_data.dtype, copy=False)  # narrow on the device: an eighth of the copy back
                nbins[j0 : j0 + len(cols_block)] = cp.asnumpy(nb_dev).astype(np.int64)
                continue
            except Exception:
                log_throttle(logger, "su_quantise_device", logging.WARNING, "SU quantiser: device binning failed, quantising on the CPU", exc_info=True)
        for k in _su_codes_into(cols_block, n_bins, factors_data, nbins, j0):
            binned, nb = _su_code_column(cols_block[k], n_bins)
            factors_data[:, j0 + k] = binned
            nbins[j0 + k] = nb
    return factors_data, nbins, names


def _su_device_enabled(n: int, p: int) -> bool:
    """Whether the SU quantiser bins on the device: only when ``MLFRAME_SU_QUANTISE_GPU=1`` is set.

    Measured on the GTX 1050 Ti (one run each, same input): 100000 x 100, device 0.44 s against CPU 0.24 s; 300000 x 100, device 1.25 s against CPU 0.88 s. The host
    quantiser wins at both shapes (the device route pays the upload and the block sort), so the default stays on the CPU and the twin is opt-in until a shape where it wins
    is measured. The shape arguments are kept for that measurement."""
    import os

    return os.environ.get("MLFRAME_SU_QUANTISE_GPU", "0").strip() == "1" and n >= 0 and p >= 0


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


def _record_pair_su(pair_sus: dict, a_name: str, b_name: str, s: float) -> None:
    """Store a pair's SU only when it was actually measured.

    A failed or non-finite SU used to be stored as 0.0 - "maximally non-redundant" - which is a measurement nobody
    made, and it also dragged down the medoid statistic of any component the pair sat in. Leaving the pair out is
    what the tau-calibration path already does; both consumers here treat a missing pair as "no edge, no vote".
    """
    if s is not None and np.isfinite(s):
        pair_sus[(a_name, b_name)] = float(s)


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
        """Union-find root lookup with iterative path compression."""
        # Iterative path compression - stays pickle/clone-safe (no closures
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
        """Merge the components containing ``a`` and ``b``, always attaching the lexicographically-larger root under the smaller one so the resulting root is deterministic regardless of edge processing order."""
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


def _level_pair_sus(resolved: dict, resolved_names: list, cal_state: Any, pair_su: Any, pair_su_batch: Any, level: int) -> tuple:
    """Pairwise SU of the resolved anchors at one hierarchy level (batched, falling back to per-pair); returns ``(pair_sus, n_failed_pairs)``."""
    n_failed_pairs = 0
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
        log_throttle(
            logger,
            "cluster_hierarchy_batched_pair_su_failed",
            logging.WARNING,
            "build_cluster_hierarchy: batched pair-SU dispatch failed at "
            "level %d (%r); falling back to per-pair scoring.", level, exc,
        )
        scores = None
    if scores is None:
        for (a_name, b_name), (a_idx, b_idx) in zip(name_pairs, index_pairs):
            try:
                s = pair_su(cal_state, a_idx, b_idx)
            except Exception as exc:
                s = float("nan")
                n_failed_pairs += 1
                log_throttle(
                    logger,
                    "cluster_hierarchy_per_pair_su_failed",
                    logging.WARNING,
                    "build_cluster_hierarchy: per-pair SU failed (e.g. "
                    "(%s,%s): %r); affected pairs are left out of this "
                    "level. Further failures suppressed for this build.", a_name, b_name, exc,
                )
            _record_pair_su(pair_sus, a_name, b_name, s)
    else:
        for (a_name, b_name), s in zip(name_pairs, scores):
            _record_pair_su(pair_sus, a_name, b_name, s)
    return pair_sus, n_failed_pairs


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
        Required to compute SU between anchors - the analyser does not
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
    # Layer 51: batched pairwise-SU dispatch for the
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
    # count SU-dispatch failures so a
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
        # super-structure can form at this level - stop.
        if len(resolved) < 2:
            break
        resolved_names = sorted(resolved.keys())
        pair_sus, n_failed_level = _level_pair_sus(resolved, resolved_names, cal_state, pair_su, pair_su_batch, level)
        n_failed_pairs += n_failed_level
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
            # pick the cluster
            # representative as the medoid (most-central anchor by mean
            # within-component SU) instead of the lexicographically smallest
            # name, so the super-anchor label carries statistical meaning.
            super_anchor = _component_medoid(comp, pair_sus)
            sub_anchors = [a for a in comp if a != super_anchor]
            level_dict[super_anchor] = sub_anchors
        if not level_dict:
            # No merges at this level - the hierarchy has stabilised.
            break
        hierarchy[int(level)] = level_dict
        # Prepare next level: super-anchors of this level become the
        # candidates for level+1. Anchors not merged at this level drop
        # out (they have no super-partner to test against).
        next_level_anchors = sorted(level_dict.keys())
        if len(next_level_anchors) == len(level_anchors):
            # Degenerate cycle guard: if the next level has as many
            # candidates as the current one, no merging actually
            # happened (every component was size 1 except trivially) -
            # already guarded above by ``not level_dict``, but the cycle
            # check is cheap insurance against pathological inputs.
            break
        level_anchors = next_level_anchors
    if n_failed_pairs:
        log_throttle(
            logger, "cluster_hierarchy_failed_pair_total", logging.WARNING,
            "build_cluster_hierarchy: %d anchor pair(s) could not be scored and were left out; the hierarchy may be missing merges.", n_failed_pairs,
        )
    return hierarchy


__all__ = [
    "build_cluster_hierarchy",
    "_DEFAULT_SUPER_TAU",
    "_DEFAULT_MAX_LEVELS",
]
