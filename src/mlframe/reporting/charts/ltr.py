"""Learning-to-Rank quality-visualisation panels.

Each panel builder takes ``(y_true, y_score, group_ids)`` (1-D arrays
of length N; per-row relevance, per-row predicted score, per-row
query identifier) and returns one ``PanelSpec``.

Token catalogue (all 7):
- ``NDCG_K``         — NDCG@k curve for k=1..max_per_query
- ``NDCG_DIST``      — per-query NDCG@10 distribution (violin)
- ``NDCG_BY_QSIZE``  — mean NDCG@10 binned by query size (bar; exposes
                       small-group metric inflation)
- ``LIFT``           — cumulative relevance vs rank position (lift / gain
                       curve aggregated over queries)
- ``MRR_DIST``       — per-query reciprocal rank histogram (where the
                       first relevant doc lands per query)
- ``SCORE_BY_REL``   — predicted-score box-plot per relevance grade
                       (well-separated grades = good ranker)
- ``TOP1_BY_QSIZE``  — top-1 accuracy as a function of query size
                       (line: x = bucketed query size, y = % queries
                       where the top-scored doc is the true top doc)

Heavy per-query work runs through the batched numba kernels in
``mlframe.metrics.ranking`` over the sorted-groups layout; panel builders
share that layout (and the per-query NDCG@10 vector) through the
``shared`` cache that ``compose_ltr_figure`` threads through one figure
build, so the group sort and kernels run once per figure, not per panel.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

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


def _sorted_layout(
    y_true, y_score, group_ids, shared: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group-sorted layout ``(sorted_y_true, sorted_y_score, group_starts, group_sizes)``.

    Cached in ``shared`` (one dict per figure build) so the O(N log N)
    group sort + float64 conversion run once per figure, not per panel.
    """
    if shared is not None and "layout" in shared:
        return shared["layout"]
    from mlframe.metrics.ranking import _iter_group_slices

    sorted_y_true, sorted_y_score, group_starts = _iter_group_slices(
        np.asarray(y_true), np.asarray(y_score, dtype=np.float64),
        np.asarray(group_ids),
    )
    layout = (sorted_y_true, sorted_y_score, group_starts, np.diff(group_starts))
    if shared is not None:
        shared["layout"] = layout
    return layout


def _per_query_ndcg10(
    y_true, y_score, group_ids, shared: Optional[dict] = None,
) -> np.ndarray:
    """Per-query NDCG@10 vector (NaN for degenerate queries), shared by
    NDCG_DIST and NDCG_BY_QSIZE so the kernel runs once per figure."""
    if shared is not None and "ndcg10" in shared:
        return shared["ndcg10"]
    from mlframe.metrics.ranking import _per_query_ndcg_kernel

    sorted_y_true, sorted_y_score, group_starts, _ = _sorted_layout(
        y_true, y_score, group_ids, shared)
    vals = _per_query_ndcg_kernel(sorted_y_true, sorted_y_score, group_starts, 10)
    if shared is not None:
        shared["ndcg10"] = vals
    return vals


DEFAULT_BOOTSTRAP_B: int = 1_000
# Bootstrap over QUERIES is bounded by the per-resample mean over a (B, n_queries) gather; cap n_queries per resample
# so a million-query LTR draws a representative subsample rather than a full (1000, 1e6) gather (4e9 cells / 32 GB).
_BOOTSTRAP_QUERY_CAP: int = 50_000


def bootstrap_ndcg_ci(
    per_query_ndcg: np.ndarray,
    *,
    n_boot: int = DEFAULT_BOOTSTRAP_B,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Bootstrap percentile CI for the mean per-query NDCG, resampling QUERIES (the independent unit).

    Returns ``(mean, lower, upper)`` of the per-query NDCG over the ``1-alpha`` percentile bootstrap. Resampling is the
    correct unit here -- queries, not rows -- because rows within a query are dependent. Fully vectorised: one
    ``(n_boot, n_eff)`` integer gather of resampled query indices, ``mean(axis=1)``, then two percentiles; no python
    bootstrap loop. ``n_eff`` is capped at ``_BOOTSTRAP_QUERY_CAP`` so a huge query count subsamples per resample
    (the CI narrows with the true query count, which the cap preserves up to its bound). NaN-bracket when no valid query.
    """
    vals = np.asarray(per_query_ndcg, dtype=np.float64)
    vals = vals[~np.isnan(vals)]
    nq = vals.size
    if nq == 0:
        return float("nan"), float("nan"), float("nan")
    if nq == 1:
        return float(vals[0]), float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    n_eff = min(nq, _BOOTSTRAP_QUERY_CAP)
    # Resample n_eff query indices WITH replacement per bootstrap; one (n_boot, n_eff) gather, mean across queries.
    idx = rng.integers(0, nq, size=(n_boot, n_eff))
    boot_means = vals[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return float(vals.mean()), lo, hi


def _ndcg_k_panel(y_true, y_score, group_ids, shared: Optional[dict] = None) -> LinePanelSpec:
    """Mean NDCG@k across queries, k=1..max_per_query.

    One batched kernel pass with ``eval_ks=1..max_k`` replaces the prior
    50 independent full ``ndcg_at_k`` calls (each re-sorting all groups).
    """
    from mlframe.metrics.ranking import _summary_batched_kernel

    sorted_y_true, sorted_y_score, group_starts, sizes = _sorted_layout(
        y_true, y_score, group_ids, shared)
    n_groups = len(group_starts) - 1
    if n_groups == 0 or int(sizes.max(initial=0)) < 1:
        return LinePanelSpec(x=np.array([1]), y=np.array([0.0]),
                             title="NDCG@k", xlabel="k", ylabel="Mean NDCG@k")
    max_k = min(int(sizes.max()), 50)  # cap for plot readability
    eval_ks = np.arange(1, max_k + 1, dtype=np.int64)
    ndcg_sums, ndcg_counts, _, _, _, _ = _summary_batched_kernel(
        sorted_y_true, sorted_y_score, group_starts, eval_ks)
    ndcgs = np.where(ndcg_counts > 0, ndcg_sums / np.maximum(ndcg_counts, 1), np.nan)
    return LinePanelSpec(
        x=eval_ks.astype(np.float64),
        y=ndcgs,
        title=f"NDCG@k curve (max k = {max_k})",
        xlabel="k",
        ylabel="Mean NDCG@k",
    )


def _ndcg_dist_panel(y_true, y_score, group_ids, shared: Optional[dict] = None) -> ViolinPanelSpec:
    """Per-query NDCG@10 (or full-query) distribution as a single violin.

    Tail at low NDCG = query types where the model is failing.
    Singleton queries are skipped (their NDCG is trivially 1.0 / NaN).
    """
    _, _, _, sizes = _sorted_layout(y_true, y_score, group_ids, shared)
    ndcg10 = _per_query_ndcg10(y_true, y_score, group_ids, shared)
    per_q = ndcg10[(sizes >= 2) & ~np.isnan(ndcg10)]
    if per_q.size == 0:
        per_q = np.array([0.0])
    # Bootstrap-over-queries 95% CI on the mean: the violin shows the spread, this brackets how well the MEAN is pinned.
    mean, lo, hi = bootstrap_ndcg_ci(per_q)
    return ViolinPanelSpec(
        groups=(per_q,),
        group_labels=(f"all queries (n={per_q.size})",),
        title=f"Per-query NDCG@10 (mean={mean:.3f}, 95% CI [{lo:.3f}, {hi:.3f}])",
        xlabel="",
        ylabel="NDCG@10",
    )


def _ndcg_by_qsize_panel(y_true, y_score, group_ids, shared: Optional[dict] = None) -> BarPanelSpec:
    """Mean NDCG@10 binned by query size (log2-spaced bins) with per-bin query counts.

    Tiny groups score trivially high NDCG (a 1-doc query with any positive item is a guaranteed 1.0), so a high overall mean can be pure
    small-group inflation; this panel makes the size dependence visible. Singleton queries are INCLUDED here on purpose (NDCG_DIST excludes
    them) precisely because their inflated contribution is what the panel exposes.
    """
    _, _, _, sizes = _sorted_layout(y_true, y_score, group_ids, shared)
    ndcg10 = _per_query_ndcg10(y_true, y_score, group_ids, shared)
    valid = ~np.isnan(ndcg10)
    if sizes.size == 0 or not valid.any():
        return BarPanelSpec(
            categories=("(no data)",), values=np.array([0.0]),
            title="Mean NDCG@10 by query size",
            xlabel="Query size (docs per query, log2 bins)",
            ylabel="Mean NDCG@10",
        )
    sizes_v = sizes[valid].astype(np.int64)
    vals_v = ndcg10[valid]
    bin_idx = np.floor(np.log2(sizes_v)).astype(np.int64)  # size 1 -> bin 0, 2-3 -> 1, 4-7 -> 2, ...
    categories: List[str] = []
    means: List[float] = []
    for b in np.unique(bin_idx):
        m = bin_idx == b
        lo_sz, hi_sz = int(2 ** b), int(2 ** (b + 1)) - 1
        label = f"{lo_sz}" if lo_sz == hi_sz else f"{lo_sz}-{hi_sz}"
        # Per-bin bootstrap-over-queries 95% CI: a wide bracket on a sparse bin flags its mean is not yet pinned down.
        bmean, blo, bhi = bootstrap_ndcg_ci(vals_v[m])
        categories.append(f"{label} (n={int(m.sum()):_}, CI[{blo:.2f},{bhi:.2f}])")
        means.append(bmean)
    omean, olo, ohi = bootstrap_ndcg_ci(vals_v)
    return BarPanelSpec(
        categories=tuple(categories),
        values=np.asarray(means),
        title=f"Mean NDCG@10 by query size (small groups inflate NDCG; overall 95% CI [{olo:.3f}, {ohi:.3f}])",
        xlabel="Query size (docs per query, log2 bins)",
        ylabel="Mean NDCG@10",
        xtick_rotation=30.0,
    )


def _lift_panel(y_true, y_score, group_ids, shared: Optional[dict] = None) -> LinePanelSpec:
    """Cumulative-relevance lift curve.

    For each rank position 1..max_q (capped at 50), the average across queries of cumulative relevance accumulated up to that position,
    normalised by the per-query best-possible cumulative relevance at the same position.
    """
    from mlframe.metrics.ranking import _lift_curve_kernel

    sorted_y_true, sorted_y_score, group_starts, sizes = _sorted_layout(
        y_true, y_score, group_ids, shared)
    max_k = min(int(sizes.max(initial=1)), 50)
    if max_k < 1:
        max_k = 1
    lift_sums, counts = _lift_curve_kernel(
        sorted_y_true, sorted_y_score, group_starts, max_k)
    lift = lift_sums / np.maximum(counts, 1)
    return LinePanelSpec(
        x=np.arange(1, max_k + 1, dtype=np.float64),
        y=lift,
        title="Cumulative-relevance lift",
        xlabel="Rank position (1-indexed)",
        ylabel="Cumulative relevance / ideal",
    )


def _mrr_dist_panel(y_true, y_score, group_ids, shared: Optional[dict] = None) -> HistogramPanelSpec:
    """Per-query reciprocal rank distribution.

    For each query, reciprocal of the 1-indexed rank of the first relevant doc in the score-sorted order; queries with no relevant doc
    contribute 0.
    """
    from mlframe.metrics.ranking import _per_query_mrr_kernel

    from ._sampling import prebin_histogram

    sorted_y_true, sorted_y_score, group_starts, _ = _sorted_layout(
        y_true, y_score, group_ids, shared)
    rrs_raw = _per_query_mrr_kernel(sorted_y_true, sorted_y_score, group_starts)
    rrs = np.where(np.isnan(rrs_raw), 0.0, rrs_raw)
    if rrs.size == 0:
        rrs = np.array([0.0])
    mrr = float(np.mean(rrs))
    heights, centers, width = prebin_histogram(rrs, 20, True)
    return HistogramPanelSpec(
        values=heights if centers is not None else rrs,
        bins=20,
        bin_centers=centers,
        bin_width=width,
        title=f"Per-query Reciprocal Rank (MRR={mrr:.3f})",
        xlabel="Reciprocal rank (1 = first hit at top)",
        ylabel="Density",
        density=True,
    )


def _score_by_rel_panel(y_true, y_score, group_ids, shared: Optional[dict] = None) -> ViolinPanelSpec:
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
    unique_vals_full = None
    if is_float_continuous:
        unique_vals_full = np.unique(y_true_arr)
        n_unique = int(unique_vals_full.size)
    else:
        # Cheap path: cap at 13 unique values to avoid full unique scan
        # on N=5M when grades are integer with thousands of levels.
        n_unique = int(np.unique(y_true_arr[:50_000]).size)
        if n_unique > 12:
            # Confirm on full array to be safe
            unique_vals_full = np.unique(y_true_arr)
            n_unique = int(unique_vals_full.size)

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
        # Discrete-grade path (n_unique <= 12). The head probe above may miss grades that only appear later in the array, so the grade list
        # always comes from a full-array unique (vectorised ~20 ms at 2M; the prior python set-comprehension over .tolist() cost ~0.25 s).
        grades_arr = unique_vals_full if unique_vals_full is not None else np.unique(y_true_arr)
        for idx, g_raw in enumerate(grades_arr):
            g = int(g_raw)
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


def _top1_by_qsize_panel(y_true, y_score, group_ids, shared: Optional[dict] = None) -> LinePanelSpec:
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
    "NDCG_BY_QSIZE": _ndcg_by_qsize_panel,
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
    panels_template: str = "NDCG_K NDCG_DIST NDCG_BY_QSIZE LIFT MRR_DIST SCORE_BY_REL",
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
    # One shared cache per figure: the group sort + per-query NDCG kernel results are reused by every panel that needs them.
    shared: Dict[str, object] = {}
    panels: List[PanelSpec] = [
        _TOKEN_BUILDERS[tok](y_true, y_score, group_ids, shared) for tok in tokens
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
    "bootstrap_ndcg_ci",
    "DEFAULT_BOOTSTRAP_B",
]
