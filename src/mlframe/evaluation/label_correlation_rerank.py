"""``detect_correlated_label_pairs``/``label_correlation_rerank``: correlation-aware multi-label top-K rerank.

Source: 2nd_santander-product-recommendation.md -- MAP@7 simulation study shows that when two labels are
near-perfectly correlated (always co-occur), each label's OWN raw predicted probability is a noisier estimate
of "should this be in the top-K" than the PAIR's joint evidence; the source describes analytically deriving a
combined-probability decision threshold and "closing gaps" between correlated labels' predicted ranks.

Rather than re-deriving the source's analytical MAP@K threshold formula (error-prone to generalize correctly
outside MAP@7's exact combinatorics, as the source's own critique flags), this module uses an empirically
safer mechanism with the same effect: detect near-perfectly co-occurring label pairs from TRAIN co-occurrence
statistics, then at inference average each detected pair's predicted scores together -- pulling a correlated
pair's ranks toward each other (their joint evidence), which is exactly what "closing the gap between
correlated labels' predicted ranks" means, without depending on a metric-specific closed-form derivation.

Extension: real label sets often have GROUPS of 3+ mutually co-occurring labels (e.g. several product
variants of one underlying event), not just pairs. Pairwise detection alone still flags every edge inside
such a group, but ``label_correlation_rerank``'s pairwise loop processes those overlapping pairs sequentially
and the LAST pair touching a label wins (documented above) -- for a 3-way group this silently drops one
member's contribution to the averaged score instead of computing the true group mean. ``detect_correlated_-
label_groups`` merges overlapping pairs (union-find) into groups so the group's mean is computed once, over
all members at once. ``optimize_group_blend_weight`` additionally lets each group's blend weight (how much of
the group mean vs. the label's own score to use) be tuned per-group via a small CV grid search against a real
ranking metric (LRAP by default) instead of always using a fixed full average (weight=1.0).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import label_ranking_average_precision_score


def detect_correlated_label_pairs(y_multilabel: np.ndarray, min_cooccurrence_rate: float = 0.9, min_support: int = 5) -> List[Tuple[int, int]]:
    """Flag label-index pairs that co-occur near-perfectly in a binary multi-label matrix.

    A pair ``(i, j)`` is flagged when, among rows where EITHER label is present, both
    ``P(label_j present | label_i present) >= min_cooccurrence_rate`` and the symmetric converse hold --
    i.e. the two labels are near-interchangeable indicators of the same underlying event.

    Parameters
    ----------
    y_multilabel
        ``(n_samples, n_labels)`` binary (0/1) label matrix.
    min_cooccurrence_rate
        Minimum conditional co-occurrence rate (both directions) to flag a pair.
    min_support
        Minimum number of rows where each label individually is present, to avoid flagging rare-label pairs
        on too little evidence.

    Returns
    -------
    list of (int, int)
        Flagged label-index pairs, ``i < j``.
    """
    y_arr = np.asarray(y_multilabel).astype(np.float64)
    n_labels = y_arr.shape[1]
    counts = y_arr.sum(axis=0)

    # a per-pair Python loop computing (y[:,i] & y[:,j]).sum() separately pays a full-array scan PER PAIR
    # (O(n_labels^2) numpy calls, measured as the dominant cost: 41.6s cProfile total at n_labels=200/50000
    # rows) -- a single matrix multiplication computes every pair's co-occurrence count at once (BLAS-backed,
    # one call), since (y.T @ y)[i, j] is exactly count(label_i AND label_j) for a 0/1 matrix.
    both_counts = y_arr.T @ y_arr

    pairs: List[Tuple[int, int]] = []
    for i in range(n_labels):
        if counts[i] < min_support:
            continue
        for j in range(i + 1, n_labels):
            if counts[j] < min_support:
                continue
            both = both_counts[i, j]
            rate_i_given_j = both / counts[j]
            rate_j_given_i = both / counts[i]
            if rate_i_given_j >= min_cooccurrence_rate and rate_j_given_i >= min_cooccurrence_rate:
                pairs.append((i, j))

    return pairs


def label_correlation_rerank(
    pred_scores: np.ndarray,
    correlated_pairs: List[Tuple[int, int]],
    correlated_groups: Optional[List[Tuple[int, ...]]] = None,
    group_weights: Optional[Dict[Tuple[int, ...], float]] = None,
) -> np.ndarray:
    """Average each detected correlated pair's (and, opt-in, group's) predicted scores together, per row.

    Parameters
    ----------
    pred_scores
        ``(n_samples, n_labels)`` predicted per-label scores/probabilities.
    correlated_pairs
        Label-index pairs from :func:`detect_correlated_label_pairs` (or supplied directly). Always applied
        first, exactly as before -- pass ``[]`` to skip this stage entirely and use only ``correlated_groups``.
    correlated_groups
        Opt-in. Label-index groups (size >= 2) from :func:`detect_correlated_label_groups`, applied AFTER
        ``correlated_pairs``. Unlike the pairwise loop, each group's mean is computed once over ALL its
        members simultaneously, so a 3+-way group is not corrupted by sequential pair overwrites. ``None``
        (the default) leaves behavior bit-identical to the pairs-only path.
    group_weights
        Opt-in, only used when ``correlated_groups`` is given. Maps a group tuple to a blend weight in
        ``[0, 1]``: each member's new score is ``(1 - w) * own_score + w * group_mean``. A group absent from
        this mapping (or when ``group_weights`` is ``None``) defaults to ``w=1.0`` (a full average, matching
        the pairwise stage's behavior). See :func:`optimize_group_blend_weight` to tune ``w`` per group.

    Returns
    -------
    np.ndarray
        ``(n_samples, n_labels)`` reranked scores: labels in a correlated pair are both replaced by their
        row-wise mean (pulling their ranks together); labels not in any pair are unchanged. If a label
        appears in multiple pairs, the LAST pair processed wins (pairs are typically near-disjoint in
        practice; callers with overlapping pair chains should pre-merge them into groups via
        :func:`detect_correlated_label_groups` and pass them as ``correlated_groups`` instead).
    """
    scores = np.asarray(pred_scores, dtype=np.float64).copy()
    for i, j in correlated_pairs:
        pair_mean = (scores[:, i] + scores[:, j]) / 2.0
        scores[:, i] = pair_mean
        scores[:, j] = pair_mean

    if correlated_groups is not None:
        for group in correlated_groups:
            idx = list(group)
            weight = 1.0 if group_weights is None else float(group_weights.get(group, 1.0))
            group_mean = scores[:, idx].mean(axis=1)
            if weight == 1.0:
                for label_idx in idx:
                    scores[:, label_idx] = group_mean
            else:
                for label_idx in idx:
                    scores[:, label_idx] = (1.0 - weight) * scores[:, label_idx] + weight * group_mean

    return np.asarray(scores)


def detect_correlated_label_groups(y_multilabel: np.ndarray, min_cooccurrence_rate: float = 0.9, min_support: int = 5) -> List[Tuple[int, ...]]:
    """Merge pairwise-detected co-occurring label edges into connected GROUPS of 3+ (or 2) mutually correlated labels.

    Runs :func:`detect_correlated_label_pairs` for the pairwise edges, then union-finds overlapping pairs
    into connected components -- e.g. edges ``(0, 1)``, ``(1, 2)``, ``(0, 2)`` all detected pairwise (because
    every pair among 3 mutually co-occurring labels individually clears the threshold) merge into the single
    group ``(0, 1, 2)``, so downstream reranking can average all 3 members at once instead of the sequential
    pairwise loop's "last pair wins" corruption.

    Parameters
    ----------
    y_multilabel
        ``(n_samples, n_labels)`` binary (0/1) label matrix.
    min_cooccurrence_rate
        Forwarded to :func:`detect_correlated_label_pairs`.
    min_support
        Forwarded to :func:`detect_correlated_label_pairs`.

    Returns
    -------
    list of tuple of int
        Groups (size >= 2), each a sorted tuple of label indices, sorted by their first member.
    """
    pairs = detect_correlated_label_pairs(y_multilabel, min_cooccurrence_rate=min_cooccurrence_rate, min_support=min_support)

    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        """Return the root of x's union-find set, path-compressing along the way."""
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        """Merge the union-find sets containing a and b."""
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for i, j in pairs:
        parent.setdefault(i, i)
        parent.setdefault(j, j)
        union(i, j)

    components: Dict[int, List[int]] = {}
    for node in parent:
        root = find(node)
        components.setdefault(root, []).append(node)

    groups = [tuple(sorted(members)) for members in components.values() if len(members) >= 2]
    groups.sort()
    return groups


def _default_ranking_metric(y_true: np.ndarray, scores: np.ndarray) -> float:
    """LRAP restricted to rows with >= 1 true label (LRAP is undefined for all-zero rows)."""
    has_positive = y_true.sum(axis=1) > 0
    if not has_positive.any():
        return 0.0
    return float(label_ranking_average_precision_score(y_true[has_positive], scores[has_positive]))


def optimize_group_blend_weight(
    y_true: np.ndarray,
    pred_scores: np.ndarray,
    correlated_groups: List[Tuple[int, ...]],
    weight_grid: Optional[np.ndarray] = None,
    n_splits: int = 3,
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    random_state: int = 0,
) -> Dict[Tuple[int, ...], float]:
    """Per-group CV grid search for the blend weight in :func:`label_correlation_rerank`'s ``group_weights``.

    A fixed simple average (``w=1.0``) is a reasonable default, but it is not always metric-optimal: when a
    group's individual raw scores are less noisy than the source's simulation study assumed, a partial blend
    (``0 < w < 1``) can rank better than either the raw scores (``w=0``) or the full average (``w=1``). This
    does a small grid search per group, evaluated via K-fold CV on ``metric_fn`` (default: label ranking
    average precision, LRAP) so the chosen weight is not overfit to one row split. Groups are optimized one
    at a time, in the given order, each holding previously-optimized groups' weights fixed (coordinate
    ascent) -- correlated groups are typically disjoint in practice, so this is exact when they don't share
    label indices.

    Parameters
    ----------
    y_true
        ``(n_samples, n_labels)`` binary (0/1) label matrix.
    pred_scores
        ``(n_samples, n_labels)`` predicted per-label scores/probabilities.
    correlated_groups
        Groups from :func:`detect_correlated_label_groups` (or supplied directly) to tune weights for.
    weight_grid
        Candidate weights to search, default ``np.linspace(0.0, 1.0, 11)`` (0.0, 0.1, ..., 1.0).
    n_splits
        Number of random row folds averaged per candidate weight.
    metric_fn
        ``(y_true_fold, scores_fold) -> float``, higher is better. Default: LRAP over rows with >= 1 true label.
    random_state
        Seed for the fold assignment.

    Returns
    -------
    dict
        Maps each input group tuple to its CV-selected blend weight, ready to pass as ``group_weights`` to
        :func:`label_correlation_rerank`.
    """
    if weight_grid is None:
        weight_grid = np.linspace(0.0, 1.0, 11)
    if metric_fn is None:
        metric_fn = _default_ranking_metric

    y_arr = np.asarray(y_true)
    n = y_arr.shape[0]
    rng = np.random.default_rng(random_state)
    fold_ids = rng.integers(0, n_splits, size=n)

    weights: Dict[Tuple[int, ...], float] = {}
    for group in correlated_groups:
        best_weight = 1.0
        best_score = -np.inf
        for w in weight_grid:
            candidate_weights = dict(weights)
            candidate_weights[group] = float(w)
            # Rerank ALL previously-optimized groups too (not just the one under test), each still using
            # its own already-decided weight via `candidate_weights` -- otherwise the CV score being
            # searched here is computed against a scores array where earlier groups are left as raw
            # (unblended) scores, which does not match what `label_correlation_rerank(...,
            # correlated_groups=all_groups, group_weights=weights)` will actually produce once every group
            # is genuinely applied together. This is what makes the search a real coordinate ascent instead
            # of silently degrading to "each group tuned independently".
            candidate_groups = [*weights.keys(), group]
            fold_scores = []
            for fold in range(n_splits):
                mask = fold_ids == fold
                if not mask.any():
                    continue
                reranked = label_correlation_rerank(pred_scores[mask], correlated_pairs=[], correlated_groups=candidate_groups, group_weights=candidate_weights)
                fold_scores.append(metric_fn(y_arr[mask], reranked))
            if fold_scores:
                mean_score = float(np.mean(fold_scores))
                if mean_score > best_score:
                    best_score = mean_score
                    best_weight = float(w)
        weights[group] = best_weight

    return weights


__all__ = [
    "detect_correlated_label_pairs",
    "label_correlation_rerank",
    "detect_correlated_label_groups",
    "optimize_group_blend_weight",
]
