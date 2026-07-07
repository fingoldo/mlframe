"""Multi-model leaderboard for a suite that trains several models on the same task.

Three panels answer "which model wins, by how much, and do any of them agree":
- a curve overlay (binary -> ROC, one decimated line per model + chance diagonal; regression / other ->
  a sorted-prediction overlay so the spread of each model's outputs is comparable);
- a metric-bar leaderboard sorted by the headline metric, with an hline at the best (or a supplied baseline);
- a between-model prediction-correlation heatmap (Spearman of model scores on a shared subsample) -- ~1.0 cells
  flag near-redundant models, low cells flag genuinely diverse ones worth ensembling.

Efficiency: every curve is decimated to <=2000 vertices off one descending-score sort per model (reusing the
binary module's shared-sort primitives), and the correlation is computed on a <=20k-row subsample with ranks
obtained by a single vectorized double-argsort over the (n_sub, K) score matrix -- never a per-pair scipy loop.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

import numpy as np

from mlframe.reporting.charts._layout import figsize_for_grid, pack_panels
from mlframe.reporting.charts.binary import _ScoreSort, _decimate, _finite_binary
from mlframe.reporting.spec import (
    AnnotationPanelSpec, BarPanelSpec, FigureSpec, HeatmapPanelSpec, LinePanelSpec, PanelSpec,
)

# Row cap for the between-model prediction-correlation. Spearman on score rank is stable far below this; the cap
# keeps the rank computation (a double-argsort over the (n_sub, K) matrix) cheap at any input size.
CORR_SUBSAMPLE: int = 20_000
_CURVE_VERTEX_CAP: int = 2_000
# Distinct colours cycled across model curves / bars (extended past the count of models by repetition).
_MODEL_COLORS: Tuple[str, ...] = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)


def _model_score(entry: Mapping[str, Any]) -> Optional[np.ndarray]:
    """Pull the per-row scalar prediction from a per-model entry: ``y_score`` preferred, else ``y_pred``."""
    for key in ("y_score", "y_pred"):
        if key in entry and entry[key] is not None:
            return np.asarray(entry[key], dtype=np.float64).ravel()
    return None


def _headline_metric(per_model: Mapping[str, Mapping[str, Any]], metric: Optional[str], task_type: str) -> str:
    """Resolve the leaderboard's headline metric name.

    Explicit ``metric`` wins; otherwise the first metric common to every model's ``metrics`` dict is used, with a
    task-type default (roc_auc for binary, r2 for regression) when present.
    """
    if metric is not None:
        return metric
    metric_dicts = [dict(e.get("metrics", {})) for e in per_model.values()]
    if not metric_dicts or any(not m for m in metric_dicts):
        return ""
    default = {"binary": "roc_auc", "regression": "r2"}.get(task_type)
    if default is not None and all(default in m for m in metric_dicts):
        return default
    common = set(metric_dicts[0])
    for m in metric_dicts[1:]:
        common &= set(m)
    return sorted(common)[0] if common else next(iter(metric_dicts[0]), "")


def _roc_from_sort(sort: _ScoreSort) -> Tuple[np.ndarray, np.ndarray, float]:
    """ROC (fpr, tpr) decimated to the vertex cap + the trapezoid AUC, from one model's shared score sort."""
    tps, fps, _ = sort.distinct_threshold_counts()
    tpr = np.concatenate(([0.0], tps / max(1, sort.n_pos)))
    fpr = np.concatenate(([0.0], fps / max(1, sort.n_neg)))
    auc = float(np.trapezoid(tpr, fpr))
    x_thin, (tpr_thin,) = _decimate(fpr, tpr, cap=_CURVE_VERTEX_CAP)
    return x_thin, tpr_thin, auc


def _roc_overlay_panel(per_model: Mapping[str, Mapping[str, Any]]) -> PanelSpec:
    """ROC overlay: one decimated curve per model (AUC in its legend label) + the chance diagonal."""
    names = list(per_model)
    series_x: List[np.ndarray] = []
    series_y: List[np.ndarray] = []
    labels: List[str] = []
    colors: List[str] = []
    styles: List[str] = []
    for i, name in enumerate(names):
        entry = per_model[name]
        score = _model_score(entry)
        if score is None:
            continue
        yt, ys = _finite_binary(entry["y_true"], score)
        sort = _ScoreSort(yt, ys)
        if sort.n_pos == 0 or sort.n_neg == 0:
            continue
        fpr, tpr, auc = _roc_from_sort(sort)
        series_x.append(fpr)
        series_y.append(tpr)
        labels.append(f"{name} (AUC={auc:.3f})")
        colors.append(_MODEL_COLORS[i % len(_MODEL_COLORS)])
        styles.append("-")
    if not series_x:
        return AnnotationPanelSpec(text="ROC overlay undefined\n(no model has both classes)", title="ROC overlay")
    series_x.append(np.array([0.0, 1.0]))
    series_y.append(np.array([0.0, 1.0]))
    labels.append("chance")
    colors.append("gray")
    styles.append("--")
    return LinePanelSpec(
        x=tuple(np.asarray(x, dtype=np.float64) for x in series_x),
        y=tuple(series_y),
        series_labels=tuple(labels),
        title="ROC overlay (all models)",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        line_styles=tuple(styles),
        colors=tuple(colors),
    )


def _sorted_prediction_overlay_panel(per_model: Mapping[str, Mapping[str, Any]]) -> PanelSpec:
    """Sorted-prediction overlay (non-binary): each model's predictions sorted ascending, on a shared percentile x.

    A model's sorted-output curve summarises its prediction distribution; overlaying them shows which models predict
    a wider / narrower range and where they diverge. Decimated to the vertex cap so a 1M-row prediction stays light.
    """
    names = list(per_model)
    series_x: List[np.ndarray] = []
    series_y: List[np.ndarray] = []
    labels: List[str] = []
    colors: List[str] = []
    for i, name in enumerate(names):
        score = _model_score(per_model[name])
        if score is None:
            continue
        s = np.sort(score[np.isfinite(score)])
        if s.size == 0:
            continue
        pct = np.linspace(0.0, 1.0, s.size)
        x_thin, (y_thin,) = _decimate(pct, s, cap=_CURVE_VERTEX_CAP)
        series_x.append(x_thin)
        series_y.append(y_thin)
        labels.append(name)
        colors.append(_MODEL_COLORS[i % len(_MODEL_COLORS)])
    if not series_x:
        return AnnotationPanelSpec(text="Prediction overlay undefined\n(no finite predictions)", title="Prediction overlay")
    return LinePanelSpec(
        x=tuple(series_x),
        y=tuple(series_y),
        series_labels=tuple(labels),
        title="Sorted-prediction overlay (all models)",
        xlabel="percentile",
        ylabel="prediction",
        line_styles=tuple("-" for _ in series_x),
        colors=tuple(colors),
    )


def _leaderboard_panel(
    per_model: Mapping[str, Mapping[str, Any]], metric: str, higher_is_better: bool, baseline: Optional[float],
) -> PanelSpec:
    """Horizontal metric-bar leaderboard sorted best-first, with an hline at the best score (or supplied baseline).

    Horizontal bars read best-to-worst top-down regardless of model-name length. The hline marks the reference the
    rest are judged against (the best model's score by default, or an external ``baseline`` when given).
    """
    names = list(per_model)
    vals = np.array([float(per_model[n].get("metrics", {}).get(metric, np.nan)) for n in names], dtype=np.float64)
    finite = np.isfinite(vals)
    if not finite.any():
        return AnnotationPanelSpec(text=f"Leaderboard: metric '{metric}' missing on all models", title="Leaderboard")
    order = np.argsort(vals)
    if higher_is_better:
        order = order[::-1]
    # NaN sorts to one end via argsort; drop non-finite from the displayed bar so a metric-less model is not shown.
    order = [int(i) for i in order if finite[i]]
    cats = tuple(names[i] for i in order)
    bar_vals = vals[order]
    ref = baseline if baseline is not None else float(bar_vals[0])
    ref_label = "baseline" if baseline is not None else "best"
    direction = "higher=better" if higher_is_better else "lower=better"
    return BarPanelSpec(
        categories=cats,
        values=bar_vals,
        title=f"Leaderboard: {metric} ({direction})",
        xlabel=metric,
        ylabel="model",
        colors=("#4c78a8",),
        orientation="horizontal",
        hline=(float(ref), "red", ref_label),
    )


def _spearman_corr_matrix(scores: np.ndarray) -> np.ndarray:
    """Spearman correlation between the columns of ``scores`` (n_sub x K) via one vectorized double-argsort.

    Spearman = Pearson on ranks. Ranks are obtained by ``argsort(argsort(col))`` applied across all columns at once
    (average ranks for ties are not used -- ordinal ranks suffice for a diagnostic and avoid a per-column scipy
    ``rankdata`` python loop). Returns a K x K matrix with 1.0 on the diagonal; columns with zero variance correlate
    as NaN, replaced by 0 off-diagonal / 1 on-diagonal.
    """
    n, k = scores.shape
    if n < 2 or k == 0:
        return np.eye(k, dtype=np.float64)
    ranks = np.argsort(np.argsort(scores, axis=0), axis=0).astype(np.float64)
    ranks -= ranks.mean(axis=0, keepdims=True)
    std = np.sqrt((ranks * ranks).sum(axis=0))
    std_safe = np.where(std > 0, std, 1.0)
    normed = ranks / std_safe
    corr = normed.T @ normed
    # Zero-variance columns produce a degenerate row/col; force diag to 1 and clip FP overshoot.
    np.fill_diagonal(corr, 1.0)
    return np.asarray(np.clip(corr, -1.0, 1.0))


def _corr_heatmap_panel(per_model: Mapping[str, Mapping[str, Any]], subsample: int, seed: int) -> PanelSpec:
    """Between-model Spearman prediction-correlation heatmap on a shared <=``subsample`` row block.

    Only models with finite predictions over a common set of rows are included; the rows are subsampled once and the
    same indices are used for every model so the correlation is computed on aligned predictions.
    """
    names = [n for n in per_model if _model_score(per_model[n]) is not None]
    if len(names) < 2:
        return AnnotationPanelSpec(text="Correlation heatmap needs >= 2 models\nwith predictions", title="Prediction correlation")
    cols = [_model_score(per_model[n]) for n in names]
    n_rows = min(c.shape[0] for c in cols)
    cols = [c[:n_rows] for c in cols]
    mat = np.column_stack(cols)
    finite_rows = np.isfinite(mat).all(axis=1)
    mat = mat[finite_rows]
    if mat.shape[0] > subsample:
        idx = np.sort(np.random.default_rng(seed).choice(mat.shape[0], size=subsample, replace=False))
        mat = mat[idx]
    if mat.shape[0] < 2:
        return AnnotationPanelSpec(text="Correlation heatmap: < 2 aligned finite rows", title="Prediction correlation")
    corr = _spearman_corr_matrix(mat)
    # Seriate the square symmetric correlation matrix so correlated model clusters form contiguous blocks (arbitrary
    # model order otherwise hides them); apply the SAME permutation to rows, cols and labels.
    from mlframe.core.matrix_seriation import seriate

    corr, perm = seriate(corr)
    names = [names[int(i)] for i in perm]
    return HeatmapPanelSpec(
        matrix=corr,
        row_labels=tuple(names),
        col_labels=tuple(names),
        title="Between-model prediction correlation (Spearman)",
        colormap="RdBu_r",
        cell_text=corr,
        text_format=".2f",
        colorbar_label="Spearman rho",
    )


def compose_model_comparison_figure(
    per_model: Mapping[str, Mapping[str, Any]],
    task_type: str,
    *,
    metric: Optional[str] = None,
    higher_is_better: Optional[bool] = None,
    baseline: Optional[float] = None,
    corr_subsample: int = CORR_SUBSAMPLE,
    seed: int = 0,
    suptitle: str = "Model comparison",
    cell_width: float = 6.0,
    cell_height: float = 4.5,
) -> FigureSpec:
    """Build a multi-model leaderboard FigureSpec (curve overlay + leaderboard bar + prediction-correlation heatmap).

    Parameters
    ----------
    per_model : mapping ``name -> {"y_true": ..., "y_score"/"y_pred": ..., "metrics": {name: value}}``.
        ``y_true`` is required for the ROC overlay (binary); ``y_score`` is preferred over ``y_pred`` for the score
        read. ``metrics`` feeds the leaderboard bar.
    task_type : "binary" selects a ROC overlay; anything else selects the sorted-prediction overlay.
    metric : headline leaderboard metric; defaults to a task-type metric (roc_auc / r2) or the first metric common
        to every model.
    higher_is_better : leaderboard sort direction. Default None -> derived from the resolved headline metric via the
        canonical ``metric_name_higher_is_better`` table (unknown metric -> higher-is-better), so a lower-is-better
        headline (rmse / log_loss / ece / pinball) sorts best-first instead of inverted. Pass an explicit bool to override.
    baseline : optional external reference drawn as the leaderboard hline (default: the best model's score).
    corr_subsample : row cap for the Spearman prediction-correlation (default 20k).

    Returns a 2x2 (-> packed) FigureSpec: [curve overlay, leaderboard], [correlation heatmap].
    """
    if not per_model:
        return FigureSpec(suptitle=suptitle, panels=((AnnotationPanelSpec(text="compose_model_comparison_figure: no models"),),), figsize=(8.0, 3.0))

    headline = _headline_metric(per_model, metric, task_type)
    if higher_is_better is None:
        from mlframe.training.metrics_registry import metric_name_higher_is_better
        _dir = metric_name_higher_is_better(headline)
        higher_is_better = True if _dir is None else _dir
    if task_type == "binary":
        curve = _roc_overlay_panel(per_model)
    else:
        curve = _sorted_prediction_overlay_panel(per_model)
    leaderboard = _leaderboard_panel(per_model, headline, higher_is_better, baseline)
    corr = _corr_heatmap_panel(per_model, corr_subsample, seed)

    packed = pack_panels([curve, leaderboard, corr], max_cols=2)
    n_rows = len(packed)
    n_cols = 2 if packed else 0
    return FigureSpec(
        suptitle=suptitle,
        panels=packed,
        figsize=figsize_for_grid(n_rows, n_cols, cell_width=cell_width, cell_height=cell_height),
    )


__all__ = [
    "CORR_SUBSAMPLE",
    "compose_model_comparison_figure",
]
