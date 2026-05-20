"""Learning-to-Rank quality-visualisation panels.

Each panel builder takes ``(y_true, y_score, group_ids)`` (1-D arrays
of length N; per-row relevance, per-row predicted score, per-row
query identifier) and returns one ``PanelSpec``.

Token catalogue (all 6):
- ``NDCG_K``        — NDCG@k curve for k=1..max_per_query
- ``NDCG_DIST``     — per-query NDCG@10 distribution (violin)
- ``LIFT``          — cumulative relevance vs rank position (lift / gain
                      curve aggregated over queries)
- ``MRR_DIST``      — per-query reciprocal rank histogram (where the
                      first relevant doc lands per query)
- ``SCORE_BY_REL``  — predicted-score box-plot per relevance grade
                      (well-separated grades = good ranker)
- ``TOP1_BY_QSIZE`` — top-1 accuracy as a function of query size
                      (line: x = bucketed query size, y = % queries
                      where the top-scored doc is the true top doc)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from mlframe.reporting.charts._layout import (
    figsize_for_grid, pack_panels, parse_panel_template,
)
from mlframe.reporting.spec import (
    BarPanelSpec, FigureSpec, HistogramPanelSpec, LinePanelSpec, PanelSpec,
    ViolinPanelSpec,
)


# ----------------------------------------------------------------------------
# Per-token panel builders
# ----------------------------------------------------------------------------


def _per_query_groups(group_ids: np.ndarray) -> List[np.ndarray]:
    """Return a list of index arrays, one per query group."""
    sort_idx = np.argsort(group_ids, kind="stable")
    sorted_groups = group_ids[sort_idx]
    boundaries = np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1
    starts = np.concatenate(([0], boundaries, [len(sorted_groups)]))
    return [sort_idx[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]


def _ndcg_k_panel(y_true, y_score, group_ids) -> LinePanelSpec:
    """Mean NDCG@k across queries, k=1..max_per_query."""
    from mlframe.metrics.ranking import ndcg_at_k

    queries = _per_query_groups(group_ids)
    if not queries:
        return LinePanelSpec(x=np.array([1]), y=np.array([0.0]),
                             title="NDCG@k", xlabel="k", ylabel="Mean NDCG@k")
    max_k = int(max(len(q) for q in queries))
    max_k = min(max_k, 50)  # cap for plot readability
    ks = np.arange(1, max_k + 1)
    ndcgs = np.zeros(len(ks))
    for i, k in enumerate(ks):
        ndcgs[i] = ndcg_at_k(y_true, y_score, group_ids, k=int(k))
    return LinePanelSpec(
        x=ks.astype(np.float64),
        y=ndcgs,
        title=f"NDCG@k curve (max k = {max_k})",
        xlabel="k",
        ylabel="Mean NDCG@k",
    )


def _ndcg_dist_panel(y_true, y_score, group_ids) -> ViolinPanelSpec:
    """Per-query NDCG@10 (or full-query) distribution as a single violin.

    Tail at low NDCG = query types where the model is failing.
    """
    from mlframe.metrics.ranking import _ndcg_one_query

    queries = _per_query_groups(group_ids)
    per_q: List[float] = []
    for q_idx in queries:
        if len(q_idx) < 2:
            continue
        v = _ndcg_one_query(
            np.asarray(y_true, dtype=np.float64)[q_idx],
            np.asarray(y_score, dtype=np.float64)[q_idx],
            10,
        )
        if not np.isnan(v):
            per_q.append(v)
    if not per_q:
        per_q = [0.0]
    return ViolinPanelSpec(
        groups=(np.asarray(per_q),),
        group_labels=(f"all queries (n={len(per_q)})",),
        title=f"Per-query NDCG@10 (mean={np.mean(per_q):.3f})",
        xlabel="",
        ylabel="NDCG@10",
    )


def _lift_panel(y_true, y_score, group_ids) -> LinePanelSpec:
    """Cumulative-relevance lift curve.

    For each rank position 1..max_q, compute the average relevance
    accumulated up to that position across all queries (normalised
    by the per-query best-possible cumulative relevance at that rank).
    """
    queries = _per_query_groups(group_ids)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    max_k = max((len(q) for q in queries), default=1)
    max_k = min(max_k, 50)
    lift = np.zeros(max_k)
    counts = np.zeros(max_k)
    for q_idx in queries:
        if len(q_idx) < 1:
            continue
        scores_q = y_score[q_idx]
        rels_q = y_true[q_idx]
        # Sort by predicted score descending.
        order = np.argsort(-scores_q)
        ordered_rels = rels_q[order]
        cum = np.cumsum(ordered_rels)
        # Normalise by best-possible cumulative at each k.
        ideal_order = np.argsort(-rels_q)
        ideal_cum = np.cumsum(rels_q[ideal_order])
        for k in range(min(len(q_idx), max_k)):
            if ideal_cum[k] > 0:
                lift[k] += cum[k] / ideal_cum[k]
                counts[k] += 1
    counts[counts == 0] = 1
    lift = lift / counts
    return LinePanelSpec(
        x=np.arange(1, max_k + 1, dtype=np.float64),
        y=lift,
        title="Cumulative-relevance lift",
        xlabel="Rank position (1-indexed)",
        ylabel="Cumulative relevance / ideal",
    )


def _mrr_dist_panel(y_true, y_score, group_ids) -> HistogramPanelSpec:
    """Per-query reciprocal rank distribution.

    For each query, find the first relevant doc in the score-sorted
    order; reciprocal of its 1-indexed rank. Queries with no relevant
    doc contribute 0.
    """
    queries = _per_query_groups(group_ids)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    rrs: List[float] = []
    for q_idx in queries:
        if len(q_idx) == 0:
            continue
        rels_q = y_true[q_idx]
        scores_q = y_score[q_idx]
        order = np.argsort(-scores_q)
        first_rel = -1
        for pos, idx in enumerate(order):
            if rels_q[idx] > 0:
                first_rel = pos + 1
                break
        rrs.append(1.0 / first_rel if first_rel > 0 else 0.0)
    if not rrs:
        rrs = [0.0]
    return HistogramPanelSpec(
        values=np.asarray(rrs),
        bins=20,
        title=f"Per-query Reciprocal Rank (MRR={np.mean(rrs):.3f})",
        xlabel="Reciprocal rank (1 = first hit at top)",
        ylabel="Density",
        density=True,
    )


def _score_by_rel_panel(y_true, y_score, group_ids) -> ViolinPanelSpec:
    """Predicted-score distribution per relevance grade.

    Well-separated violins = ranker correctly orders grades. Heavily
    overlapping = ranker is confused (or grades are noisy).

    Continuous-relevance handling: when relevance grades have many
    unique values (continuous regression-style scores leaking into
    LTR, or fine-grained graded relevance > 12 levels), the panel
    BINS the relevance into quartile buckets (Q1..Q4) rather than
    rendering one violin per unique value. Without this, an LTR call
    with continuous relevance produces 50+ overlapping x-tick labels
    that make the chart unreadable.

    Sampling: each group passes through ``subsample_for_density``
    (cap=5000) before ``violinplot``'s ``gaussian_kde``. On 1M-row LTR
    each Q1-Q4 quartile holds ~250k scores and would otherwise spend
    seconds per chart in KDE. Group labels still carry the full
    pre-sampling counts.
    """
    from ._sampling import subsample_for_density
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=np.float64)

    if y_true_arr.size == 0:
        return ViolinPanelSpec(
            groups=(np.array([0.0]),), group_labels=("(no data)",),
            title="Predicted score by relevance grade",
            xlabel="Relevance grade", ylabel="Predicted score",
        )

    # Decide: discrete-grade path or quartile-bin path?
    # - Float dtype with non-integer values → quartile path.
    # - Integer dtype with > 12 unique values → quartile path.
    # - Otherwise → original discrete-grade path.
    is_float_continuous = (
        y_true_arr.dtype.kind == "f"
        and not np.array_equal(y_true_arr, np.round(y_true_arr))
    )
    if is_float_continuous:
        n_unique = int(np.unique(y_true_arr).size)
    else:
        # Cheap path: cap at 13 unique values to avoid full unique scan
        # on N=5M when grades are integer with thousands of levels.
        n_unique = int(np.unique(y_true_arr[:50_000]).size)
        if n_unique > 12:
            # Confirm on full array to be safe
            n_unique = int(np.unique(y_true_arr).size)

    use_quartile_bins = is_float_continuous or n_unique > 12

    groups: List[np.ndarray] = []
    labels: List[str] = []
    if use_quartile_bins:
        # 4 quartile buckets — readable, captures rank monotonicity.
        edges = np.quantile(y_true_arr, [0.0, 0.25, 0.5, 0.75, 1.0])
        # Deduplicate edges in case of heavy ties.
        edges_unique = np.unique(edges)
        if len(edges_unique) < 2:
            # Degenerate (single-value relevance) → one violin
            groups = [subsample_for_density(y_score_arr, seed=0)]
            labels = [f"rel={edges[0]:.3g} (n={len(y_score_arr)})"]
        else:
            n_bins = len(edges_unique) - 1
            for i in range(n_bins):
                lo, hi = edges_unique[i], edges_unique[i + 1]
                if i == n_bins - 1:
                    mask = (y_true_arr >= lo) & (y_true_arr <= hi)
                else:
                    mask = (y_true_arr >= lo) & (y_true_arr < hi)
                if mask.any():
                    full = y_score_arr[mask]
                    groups.append(subsample_for_density(full, seed=i))
                    qlabel = ("Q1", "Q2", "Q3", "Q4")[i] if n_bins == 4 else f"B{i+1}"
                    labels.append(
                        f"{qlabel} [{lo:.3g}..{hi:.3g}] (n={int(mask.sum()):_})"
                    )
    else:
        # Discrete-grade path (original behavior, n_unique <= 12)
        grades = sorted(set(int(g) for g in y_true_arr.tolist()))
        for idx, g in enumerate(grades):
            mask = y_true_arr == g
            if mask.any():
                full = y_score_arr[mask]
                groups.append(subsample_for_density(full, seed=idx))
                labels.append(f"rel={g} (n={int(mask.sum()):_})")

    if not groups:
        groups = [np.array([0.0])]
        labels = ["(no data)"]

    return ViolinPanelSpec(
        groups=tuple(groups),
        group_labels=tuple(labels),
        title="Predicted score by relevance grade",
        xlabel="Relevance grade",
        ylabel="Predicted score",
    )


def _top1_by_qsize_panel(y_true, y_score, group_ids) -> LinePanelSpec:
    """Top-1 accuracy bucketed by query size.

    For each query size bucket [2,3], [4,5], [6,8], [9,15], [16+]:
    the fraction of queries where the top-scored doc has the highest
    relevance. Reveals whether model degrades on tiny / huge queries.
    """
    queries = _per_query_groups(group_ids)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    buckets = [(2, 3), (4, 5), (6, 8), (9, 15), (16, 10**9)]
    correct = [0] * len(buckets)
    counts = [0] * len(buckets)
    for q_idx in queries:
        n = len(q_idx)
        if n < 2:
            continue
        rels_q = y_true[q_idx]
        scores_q = y_score[q_idx]
        if rels_q.max() <= 0:
            continue  # no relevant doc -> degenerate query
        # Wave 21 P2: nan-safe argmax. NaN score picked as top would
        # under-report correct@1 silently.
        _finite = np.isfinite(scores_q)
        if not _finite.any():
            continue  # all-NaN query: cannot pick top
        if _finite.all():
            top_pred_idx = int(np.argmax(scores_q))
        else:
            top_pred_idx = int(np.nanargmax(scores_q))
        # "Correct" = predicted top has the maximum relevance in the query
        # (allow ties: the top-pred relevance equals the max relevance).
        is_correct = rels_q[top_pred_idx] == rels_q.max()
        for b_idx, (lo, hi) in enumerate(buckets):
            if lo <= n <= hi:
                counts[b_idx] += 1
                if is_correct:
                    correct[b_idx] += 1
                break
    accs = np.array([
        (c / n_) if n_ > 0 else np.nan for c, n_ in zip(correct, counts)
    ])
    bucket_labels = [
        f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
        for lo, hi in buckets
    ]
    # Use an integer x for the line plot; xlabels get attached via the
    # bar variant of LinePanelSpec? Simpler: keep numeric x; renderers
    # don't read tick labels from LinePanelSpec. Use the bucket midpoint.
    midpoints = np.array([
        (lo + hi) / 2 if hi < 10**9 else lo + 5 for lo, hi in buckets
    ], dtype=np.float64)
    title = "Top-1 accuracy by query size"
    return LinePanelSpec(
        x=midpoints,
        y=accs,
        title=title + " (buckets: " + " ".join(bucket_labels) + ")",
        xlabel="Query size (bucket midpoint)",
        ylabel="Top-1 correct",
    )


# ----------------------------------------------------------------------------
# Token registry + composer
# ----------------------------------------------------------------------------


_TOKEN_BUILDERS: Dict[str, Callable] = {
    "NDCG_K": _ndcg_k_panel,
    "NDCG_DIST": _ndcg_dist_panel,
    "LIFT": _lift_panel,
    "MRR_DIST": _mrr_dist_panel,
    "SCORE_BY_REL": _score_by_rel_panel,
    "TOP1_BY_QSIZE": _top1_by_qsize_panel,
}

ALLOWED_LTR_PANEL_TOKENS = frozenset(_TOKEN_BUILDERS)


def compose_ltr_figure(
    y_true,
    y_score,
    group_ids,
    *,
    panels_template: str = "NDCG_K NDCG_DIST LIFT MRR_DIST SCORE_BY_REL",
    suptitle: str = "",
    max_cols: int = 2,
    cell_width: float = 6.0,
    cell_height: float = 4.0,
) -> FigureSpec:
    """Build an LTR quality FigureSpec from a panel template.

    Inputs: 1-D arrays of length N (per-row relevance, predicted score,
    query id). At least one query with >=2 docs and graded relevance is
    required; degenerate queries (singleton or single-grade) are
    skipped per-panel.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    if not (y_true.ndim == 1 and y_score.ndim == 1 and group_ids.ndim == 1):
        raise ValueError(
            f"LTR panels require 1-D inputs; got shapes "
            f"{y_true.shape}, {y_score.shape}, {group_ids.shape}"
        )
    if not (len(y_true) == len(y_score) == len(group_ids)):
        raise ValueError(
            f"length mismatch: y_true={len(y_true)}, y_score={len(y_score)}, "
            f"group_ids={len(group_ids)}"
        )

    tokens = parse_panel_template(panels_template)
    unknown = [t for t in tokens if t not in _TOKEN_BUILDERS]
    if unknown:
        raise ValueError(
            f"Unknown LTR panel tokens {unknown}. "
            f"Allowed: {sorted(ALLOWED_LTR_PANEL_TOKENS)}"
        )
    panels: List[PanelSpec] = [
        _TOKEN_BUILDERS[tok](y_true, y_score, group_ids) for tok in tokens
    ]
    grid = pack_panels(panels, max_cols=max_cols)
    n_rows = len(grid)
    n_cols = max_cols if grid else 0
    return FigureSpec(
        suptitle=suptitle,
        panels=grid,
        figsize=figsize_for_grid(n_rows, n_cols,
                                 cell_width=cell_width, cell_height=cell_height),
    )


__all__ = [
    "ALLOWED_LTR_PANEL_TOKENS",
    "compose_ltr_figure",
]
