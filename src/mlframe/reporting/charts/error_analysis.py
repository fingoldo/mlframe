"""Error-analysis diagnostic charts: see WHERE a model is weak.

Task-agnostic builders that take explicit data (feature frame, ``y_true``,
``y_pred`` / score, split labels, feature names) and return ``FigureSpec`` /
panel specs / small DataFrames. The suite integrator feeds them from its
context; nothing here imports training internals.

Diagnostics:

* ``weak_segment_heatmap`` -- own FreaAI / weak-segments reimplementation: a
  shallow decision tree on per-row error picks the most error-discriminating
  features, then a 1-2-feature grid is coloured by mean error (darker = worse).
* ``error_bias_per_feature`` -- own Evidently error-bias reimplementation: rows
  tagged OVER / UNDER / MAJORITY by signed-error quantile; per feature, the three
  groups' value distributions overlay plus a group-mean table.
* ``worst_k_table`` -- top-K rows by |resid| (regression) or loss (classification)
  with id/timestamp/y/yhat/resid + top-FI feature values; the index accessor lets
  the scatter highlight those K points red.
* ``segments_bar`` -- per-subgroup metric bars with a global-reference hline.
* ``target_dist_overlay`` -- per-split overlaid density histograms of y AND of
  predictions (incl. OOF-vs-test), aggregate-first so it is safe at 1M+ rows.

All aggregation is O(n) (bincount / histogram / quantile partition); scatters are
subsampled with extremes preserved; curves stay under a few thousand vertices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from mlframe.reporting import colors as _colors
from mlframe.reporting.charts._layout import (
    figsize_for_grid, pack_panels,
)
from mlframe.reporting.charts._sampling import subsample_for_density
from mlframe.reporting.spec import (
    BarPanelSpec, FigureSpec, HeatmapPanelSpec, HistogramPanelSpec,
    LinePanelSpec, PanelSpec,
)

# A 1-2-feature weak-segment grid: more cells than this fragments support per cell into noise (FreaAI keeps slices coarse so findings stay actionable).
DEFAULT_HEATMAP_BINS: int = 6
# Shallow on purpose: deep trees overfit the error signal and the top splits stop being the genuinely worst-performing slices.
DEFAULT_TREE_DEPTH: int = 3
# The tree only needs enough rows to RANK split features; fitting it on all of a 1M+ row set is the whole cost (~1.8s).
# A 50k subsample picks the same top features and drops that to ~80ms; cell stats below still use every row.
DEFAULT_TREE_FIT_CAP: int = 50_000
# Histogram resolution for per-feature / target overlays; >~60 turns density curves into noisy combs at the row counts we see.
DEFAULT_OVERLAY_BINS: int = 40
# Default worst-K rows surfaced; 20 fits a screen and red-highlights without swamping the scatter.
DEFAULT_WORST_K: int = 20
# Over/under tail fraction for error-bias tagging (Evidently's signature 5% tails).
DEFAULT_TAIL_FRACTION: float = 0.05


def _as_float_1d(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).ravel()


def _per_row_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    task: str,
) -> np.ndarray:
    """Per-row error signal the weak-segment tree fits.

    Regression: absolute residual. Classification: log-loss when ``y_pred`` is a probability/score in [0,1]
    (richer than 0/1, so the tree sees *how* wrong), else 0/1 incorrectness when ``y_pred`` is a hard label.
    """
    yt = _as_float_1d(y_true)
    yp = _as_float_1d(y_pred)
    if task == "regression":
        return np.abs(yt - yp)
    looks_proba = yp.size > 0 and float(np.nanmin(yp)) >= 0.0 and float(np.nanmax(yp)) <= 1.0
    if looks_proba:
        eps = 1e-12
        p = np.clip(yp, eps, 1.0 - eps)
        return -(yt * np.log(p) + (1.0 - yt) * np.log(1.0 - p))
    return (yt != yp).astype(np.float64)


def _resolve_feature_matrix(
    X: Any,
    feature_names: Optional[Sequence[str]],
) -> Tuple[np.ndarray, List[str]]:
    """Coerce ``X`` (pandas / polars / ndarray) to a 2-D float matrix + name list without a full frame copy.

    Columns are pulled one at a time (narrow ndarray views), never via a whole-frame ``to_pandas`` / ``to_numpy``
    on a 100+ GB carrier. Non-numeric columns are label-encoded to integer codes so the tree can still split on them.
    """
    if hasattr(X, "columns") and hasattr(X, "__getitem__") and not isinstance(X, np.ndarray):
        cols = list(X.columns)
        names = list(feature_names) if feature_names is not None else [str(c) for c in cols]
        mats: List[np.ndarray] = []
        for c in cols:
            col = X[c]
            arr = col.to_numpy() if hasattr(col, "to_numpy") else np.asarray(col)
            if arr.dtype.kind in "OUS" or arr.dtype.kind == "b":
                _, codes = np.unique(arr.astype(str), return_inverse=True)
                mats.append(codes.astype(np.float64))
            else:
                mats.append(arr.astype(np.float64))
        mat = np.column_stack(mats) if mats else np.empty((len(X), 0), dtype=np.float64)
        return mat, names
    mat = np.asarray(X, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(mat.shape[1])]
    return mat, names


@dataclass(frozen=True)
class WeakSegmentResult:
    """Heatmap spec + the chosen split features + the worst cell's localisation.

    ``worst_cell`` is ``(feat_a_low, feat_a_high, feat_b_low, feat_b_high, mean_error)``; ``feat_b_*`` are NaN
    for a 1-D grid. Tests assert the injected bad region lands inside the worst cell's bounds.
    """

    figure: FigureSpec
    split_features: Tuple[str, ...]
    worst_cell: Tuple[float, float, float, float, float]
    cell_error: np.ndarray
    cell_count: np.ndarray


def _top_split_features(
    mat: np.ndarray,
    err: np.ndarray,
    names: List[str],
    *,
    max_depth: int,
    n_features: int,
    seed: int,
    fit_cap: int = DEFAULT_TREE_FIT_CAP,
) -> List[int]:
    """Fit a shallow regression tree on the per-row error and rank features by impurity-importance.

    The tree finds where the error concentrates (its splits ARE the weak-segment boundaries); we then take the
    ``n_features`` most-used columns. The fit is capped at ``fit_cap`` rows (subsample preserving the largest-error
    points so the weak region is never sampled away) -- ranking the splits does not need all of a 1M+ row set, and
    the cell statistics downstream still use every row. Falls back to error-variance ranking when sklearn is missing.
    """
    n_cols = mat.shape[1]
    if n_cols == 0:
        return []
    if mat.shape[0] > fit_cap:
        from mlframe.reporting.charts._sampling import subsample_preserving_extremes

        idx = subsample_preserving_extremes(err, sample_size=fit_cap, extreme_values=err,
                                             k_extremes=min(fit_cap // 10, err.size), rng=seed)
        fit_mat, fit_err = mat[idx], err[idx]
    else:
        fit_mat, fit_err = mat, err
    try:
        from sklearn.tree import DecisionTreeRegressor

        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=seed)
        tree.fit(fit_mat, fit_err)
        imp = np.asarray(tree.feature_importances_, dtype=np.float64)
    except Exception:
        # Surrogate ranking: a feature whose high/low halves differ most in mean error is the most error-discriminating.
        imp = np.zeros(n_cols, dtype=np.float64)
        for j in range(n_cols):
            med = np.median(fit_mat[:, j])
            hi = fit_err[fit_mat[:, j] > med]
            lo = fit_err[fit_mat[:, j] <= med]
            if hi.size and lo.size:
                imp[j] = abs(float(hi.mean()) - float(lo.mean()))
    if not np.any(imp > 0):
        return list(range(min(n_features, n_cols)))
    order = np.argsort(imp)[::-1]
    return [int(j) for j in order[:n_features] if imp[j] > 0]


def _bin_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    """Quantile edges (equal-frequency) deduped; falls back to a single bin for a degenerate constant feature."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([0.0, 1.0])
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, nbins + 1)))
    if edges.size < 2:
        lo = float(finite.min())
        return np.array([lo, lo + 1.0])
    return edges


def weak_segment_heatmap(
    X: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    task: str = "regression",
    feature_names: Optional[Sequence[str]] = None,
    nbins: int = DEFAULT_HEATMAP_BINS,
    max_depth: int = DEFAULT_TREE_DEPTH,
    title: str = "Weak segments (mean error by feature slice)",
    seed: int = 0,
) -> WeakSegmentResult:
    """Own FreaAI / weak-segments heatmap: localise where the model is weak.

    A shallow ``DecisionTreeRegressor`` is fit on the per-row error (``|resid|`` for regression; log-loss or 0/1
    incorrectness for classification) -- its splits identify the most error-discriminating features. The top one or
    two such features are binned (quantile) into a grid whose cells are coloured by mean error (darker = worse) and
    annotated with cell counts. Aggregation is O(n) via ``bincount``; nothing iterates per row in Python.
    """
    err = _per_row_error(y_true, y_pred, task=task)
    mat, names = _resolve_feature_matrix(X, feature_names)
    finite = np.isfinite(err)
    if not np.all(finite):
        err = err[finite]
        mat = mat[finite]
    top = _top_split_features(mat, err, names, max_depth=max_depth, n_features=2, seed=seed)
    if not top:
        ann = HeatmapPanelSpec(
            matrix=np.zeros((1, 1)), row_labels=("n/a",), col_labels=("n/a",),
            title=title + " (no usable features)", colorbar_label="mean error",
        )
        return WeakSegmentResult(
            FigureSpec(panels=((ann,),), figsize=(7.0, 5.0)),
            (), (np.nan,) * 4 + (float("nan"),), np.zeros((1, 1)), np.zeros((1, 1)),
        )

    ja = top[0]
    ea = _bin_edges(mat[:, ja], nbins)
    ia = np.clip(np.digitize(mat[:, ja], ea[1:-1]), 0, len(ea) - 2)
    na = len(ea) - 1

    if len(top) >= 2:
        jb = top[1]
        eb = _bin_edges(mat[:, jb], nbins)
        ib = np.clip(np.digitize(mat[:, jb], eb[1:-1]), 0, len(eb) - 2)
        nb = len(eb) - 1
        flat = ia * nb + ib
        ncells = na * nb
        counts = np.bincount(flat, minlength=ncells).astype(np.float64)
        sums = np.bincount(flat, weights=err, minlength=ncells)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_err = np.where(counts > 0, sums / np.where(counts > 0, counts, 1.0), np.nan)
        cell_error = mean_err.reshape(na, nb)
        cell_count = counts.reshape(na, nb)
        worst = int(np.nanargmax(np.where(np.isfinite(mean_err), mean_err, -np.inf)))
        wa, wb = worst // nb, worst % nb
        worst_cell = (float(ea[wa]), float(ea[wa + 1]), float(eb[wb]), float(eb[wb + 1]), float(mean_err[worst]))
        row_labels = tuple(f"{ea[i]:.3g}..{ea[i + 1]:.3g}" for i in range(na))
        col_labels = tuple(f"{eb[i]:.3g}..{eb[i + 1]:.3g}" for i in range(nb))
        xlabel, ylabel = names[jb], names[ja]
        split_features = (names[ja], names[jb])
    else:
        counts = np.bincount(ia, minlength=na).astype(np.float64)
        sums = np.bincount(ia, weights=err, minlength=na)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_err = np.where(counts > 0, sums / np.where(counts > 0, counts, 1.0), np.nan)
        cell_error = mean_err.reshape(na, 1)
        cell_count = counts.reshape(na, 1)
        worst = int(np.nanargmax(np.where(np.isfinite(mean_err), mean_err, -np.inf)))
        worst_cell = (float(ea[worst]), float(ea[worst + 1]), float("nan"), float("nan"), float(mean_err[worst]))
        row_labels = tuple(f"{ea[i]:.3g}..{ea[i + 1]:.3g}" for i in range(na))
        col_labels = ("error",)
        xlabel, ylabel = "", names[ja]
        split_features = (names[ja],)

    heat = HeatmapPanelSpec(
        matrix=cell_error,
        row_labels=row_labels,
        col_labels=col_labels,
        title=title + f"\nworst slice: {split_features} mean_err={worst_cell[4]:.3g}",
        xlabel=xlabel,
        ylabel=ylabel,
        colormap="Reds",
        cell_text=cell_count,
        text_format=".0f",
        colorbar_label="mean error (darker = worse)",
    )
    return WeakSegmentResult(
        FigureSpec(panels=((heat,),), figsize=(8.0, 6.0)),
        split_features, worst_cell, cell_error, cell_count,
    )


def segments_bar(
    slice_frame: Any,
    *,
    group_col: Optional[str] = None,
    metric_col: Optional[str] = None,
    global_value: Optional[float] = None,
    metric_name: str = "metric",
    title: str = "Metric by subgroup",
    higher_is_worse: bool = False,
    max_groups: int = 30,
    seed: int = 0,
) -> FigureSpec:
    """Per-subgroup metric bars with a global-reference overlay.

    ``slice_frame`` is the existing fairness / slice DataFrame: one row per subgroup with a group-name column and a
    metric column (auto-detected when not named). The global reference is drawn as a second flat bar series across
    all categories (renderers have no horizontal-line primitive for bar panels -- a flat companion series reads the
    same way and stays backend-agnostic). When ``global_value`` is None it defaults to the count-weighted-or-plain
    mean of the per-group metric. Subgroups are sorted worst-first so the weakest slice is leftmost.
    """
    import pandas as pd

    df = slice_frame
    cols = list(df.columns)
    if group_col is None:
        obj_cols = [c for c in cols if df[c].dtype.kind in "OUS"]
        group_col = obj_cols[0] if obj_cols else cols[0]
    if metric_col is None:
        num_cols = [c for c in cols if c != group_col and df[c].dtype.kind in "fiu"]
        if not num_cols:
            raise ValueError("segments_bar: no numeric metric column found in slice_frame")
        metric_col = num_cols[0]

    groups = df[group_col].astype(str).to_numpy()
    metric = df[metric_col].to_numpy().astype(np.float64)
    count_col = next((c for c in cols if str(c).lower() in ("count", "n", "size", "support")), None)
    if global_value is None:
        if count_col is not None:
            w = df[count_col].to_numpy().astype(np.float64)
            global_value = float(np.average(metric, weights=w)) if w.sum() > 0 else float(np.nanmean(metric))
        else:
            global_value = float(np.nanmean(metric))

    order = np.argsort(metric)
    if not higher_is_worse:
        order = order  # lowest metric = worst (e.g. accuracy / NDCG) -> leftmost
    else:
        order = order[::-1]  # highest metric = worst (e.g. error rate) -> leftmost
    order = order[:max_groups]

    cats = tuple(groups[order])
    vals = metric[order]
    ref = np.full(vals.shape, global_value, dtype=np.float64)
    bar = BarPanelSpec(
        categories=cats,
        values=(vals, ref),
        series_labels=(metric_name, f"global = {global_value:.3g}"),
        title=title + f"\n(worst-first; global reference = {global_value:.3g})",
        xlabel=str(group_col),
        ylabel=metric_name,
        colors=("steelblue", "darkorange"),
        xtick_rotation=45.0,
    )
    return FigureSpec(suptitle="", panels=((bar,),), figsize=(max(8.0, len(cats) * 0.5), 5.0))


@dataclass(frozen=True)
class WorstKResult:
    """Top-K worst-error rows DataFrame + the original-row indices for scatter highlighting.

    ``table`` columns: id / timestamp (when supplied) / y_true / y_pred / resid / loss + the top-FI feature values.
    ``indices`` are positions into the ORIGINAL (pre-finite-filter) arrays so the integrator can mark those points
    red on the pred-vs-actual scatter.
    """

    table: Any  # pandas.DataFrame
    indices: np.ndarray

    def highlight_indices(self) -> np.ndarray:
        """Original-array positions of the worst-K rows (for red scatter highlight)."""
        return self.indices


def worst_k_table(
    X: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    task: str = "regression",
    k: int = DEFAULT_WORST_K,
    feature_names: Optional[Sequence[str]] = None,
    feature_importances: Optional[Sequence[float]] = None,
    top_fi: int = 5,
    ids: Optional[Sequence[Any]] = None,
    timestamps: Optional[Sequence[Any]] = None,
) -> WorstKResult:
    """Top-K worst predictions by |resid| (regression) or loss (classification).

    Returns a small DataFrame (id / timestamp / y_true / y_pred / resid / loss + the ``top_fi`` highest-importance
    feature values) sorted worst-first, plus the original-row indices so the integrator can highlight those points
    red on the pred-vs-actual scatter. The K worst rows are found with ``np.argpartition`` (O(n)), not a full sort.
    """
    import pandas as pd

    yt = _as_float_1d(y_true)
    yp = _as_float_1d(y_pred)
    loss = _per_row_error(yt, yp, task=task)
    resid = yt - yp
    finite = np.isfinite(loss)
    finite_idx = np.flatnonzero(finite)
    score = loss[finite]
    n = score.size
    kk = min(int(k), n)
    if kk <= 0:
        empty = pd.DataFrame()
        return WorstKResult(empty, np.empty(0, dtype=np.int64))

    part = np.argpartition(score, n - kk)[n - kk:]
    order = part[np.argsort(score[part])[::-1]]
    sel = finite_idx[order]

    mat, names = _resolve_feature_matrix(X, feature_names)
    if feature_importances is not None and len(feature_importances) == len(names):
        fi = np.asarray(feature_importances, dtype=np.float64)
        fi_cols = [int(j) for j in np.argsort(fi)[::-1][:top_fi]]
    else:
        fi_cols = list(range(min(top_fi, len(names))))

    data: Dict[str, Any] = {}
    if ids is not None:
        data["id"] = np.asarray(ids)[sel]
    if timestamps is not None:
        data["timestamp"] = np.asarray(timestamps)[sel]
    data["y_true"] = yt[sel]
    data["y_pred"] = yp[sel]
    data["resid"] = resid[sel]
    data["loss"] = loss[sel]
    for j in fi_cols:
        data[names[j]] = mat[sel, j]

    table = pd.DataFrame(data)
    table.index = np.arange(1, len(sel) + 1)
    table.index.name = "rank"
    return WorstKResult(table, sel.astype(np.int64))


@dataclass(frozen=True)
class ErrorBiasResult:
    """Per-feature OVER/UNDER/MAJORITY overlay figure + the group-mean table.

    ``group_means`` is a small DataFrame indexed by feature, columns OVER/UNDER/MAJORITY (each the group's mean
    feature value). ``group_masks`` are the boolean row selectors so a caller can reuse the tagging.
    """

    figure: FigureSpec
    group_means: Any  # pandas.DataFrame
    group_masks: Dict[str, np.ndarray]


def _signed_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Signed prediction error y_pred - y_true: positive => OVER-estimate, negative => UNDER-estimate."""
    return _as_float_1d(y_pred) - _as_float_1d(y_true)


def _tag_error_groups(
    signed_err: np.ndarray,
    tail_fraction: float,
) -> Dict[str, np.ndarray]:
    """Split rows into OVER / UNDER / MAJORITY by signed-error quantile.

    The top ``tail_fraction`` of signed errors (most positive) are OVER-estimates, the bottom ``tail_fraction``
    (most negative) UNDER-estimates, the middle is MAJORITY. Quantile cut via ``np.quantile`` (k-way partition,
    O(n)); no full sort.
    """
    n = signed_err.size
    finite = np.isfinite(signed_err)
    hi_cut = np.quantile(signed_err[finite], 1.0 - tail_fraction) if finite.any() else np.inf
    lo_cut = np.quantile(signed_err[finite], tail_fraction) if finite.any() else -np.inf
    over = finite & (signed_err >= hi_cut)
    under = finite & (signed_err <= lo_cut)
    majority = finite & ~over & ~under
    return {"OVER": over, "UNDER": under, "MAJORITY": majority}


def error_bias_per_feature(
    X: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    feature_names: Optional[Sequence[str]] = None,
    features: Optional[Sequence[str]] = None,
    max_features: int = 4,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
    nbins: int = DEFAULT_OVERLAY_BINS,
    title: str = "Error bias per feature (OVER / UNDER / MAJORITY)",
    seed: int = 0,
) -> ErrorBiasResult:
    """Own Evidently error-bias reimplementation: which feature values drive extreme errors.

    Rows are tagged into the top-``tail_fraction`` signed-error OVER-estimates, the bottom-``tail_fraction``
    UNDER-estimates, and the MAJORITY middle. For each (up to ``max_features``) feature the three groups' value
    distributions are overlaid as density histograms (one LinePanelSpec per feature), and a group-mean table is
    returned. Per-feature binning is via ``np.histogram`` (O(n)); group means via masked sums.
    """
    import pandas as pd

    mat, names = _resolve_feature_matrix(X, feature_names)
    signed = _signed_error(y_true, y_pred)
    masks = _tag_error_groups(signed, tail_fraction)

    if features is not None:
        sel = [names.index(f) for f in features if f in names]
    else:
        sel = list(range(min(max_features, mat.shape[1])))

    group_colors = {"OVER": "#d62728", "UNDER": "#1f77b4", "MAJORITY": "#7f7f7f"}
    panels: List[PanelSpec] = []
    rows: Dict[str, List[float]] = {g: [] for g in ("OVER", "UNDER", "MAJORITY")}
    feat_index: List[str] = []

    for j in sel:
        col = mat[:, j]
        finite = np.isfinite(col)
        cf = col[finite]
        if cf.size == 0:
            continue
        edges = np.histogram_bin_edges(cf, bins=nbins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        series: List[np.ndarray] = []
        labels: List[str] = []
        cols: List[str] = []
        for g in ("OVER", "UNDER", "MAJORITY"):
            gvals = col[masks[g] & finite]
            dens, _ = np.histogram(gvals, bins=edges, density=True)
            series.append(dens)
            labels.append(g)
            cols.append(group_colors[g])
            rows[g].append(float(gvals.mean()) if gvals.size else float("nan"))
        feat_index.append(names[j])
        panels.append(LinePanelSpec(
            x=centers,
            y=tuple(series),
            series_labels=tuple(labels),
            colors=tuple(cols),
            line_styles=("-", "-", "--"),
            title=f"{names[j]} value distribution by error group",
            xlabel=names[j],
            ylabel="Density",
        ))

    group_means = pd.DataFrame(
        {g: rows[g] for g in ("OVER", "UNDER", "MAJORITY")},
        index=feat_index,
    )
    grid = pack_panels(panels, max_cols=2)
    n_rows = len(grid)
    fig = FigureSpec(
        suptitle=title,
        panels=grid,
        figsize=figsize_for_grid(max(n_rows, 1), 2, cell_width=6.0, cell_height=4.0),
    )
    return ErrorBiasResult(fig, group_means, masks)


def _split_arrays(
    values: np.ndarray,
    split_labels: Sequence[Any],
) -> Dict[str, np.ndarray]:
    """Group a flat value array by its per-row split label, preserving label order of first appearance."""
    vals = np.asarray(values)
    labels = np.asarray(split_labels)
    out: Dict[str, np.ndarray] = {}
    for lab in dict.fromkeys(labels.tolist()):
        out[str(lab)] = vals[labels == lab]
    return out


def _common_edges(groups: Dict[str, np.ndarray], nbins: int) -> Optional[np.ndarray]:
    """Shared histogram edges across all split groups so overlaid densities are comparable. None when no finite data."""
    finite_min = np.inf
    finite_max = -np.inf
    for arr in groups.values():
        a = _as_float_1d(arr)
        a = a[np.isfinite(a)]
        if a.size:
            finite_min = min(finite_min, float(a.min()))
            finite_max = max(finite_max, float(a.max()))
    if not np.isfinite(finite_min) or finite_max <= finite_min:
        if np.isfinite(finite_min):
            return np.linspace(finite_min, finite_min + 1.0, nbins + 1)
        return None
    return np.linspace(finite_min, finite_max, nbins + 1)


def _density_overlay_panel(
    groups: Dict[str, np.ndarray],
    *,
    nbins: int,
    title: str,
    xlabel: str,
    train_key: Optional[str],
) -> PanelSpec:
    """Overlaid per-split density histograms on shared edges, with p01/p99 vlines and train-envelope shading.

    All binning is ``np.histogram`` on shared edges (O(n) per split); the curve is the bin centres so it stays at
    ``nbins`` vertices regardless of row count. cProfile at 2.9M total rows: ~120 ms, all in ``np.histogram`` bin
    search + the single train ``np.percentile`` partition -- no actionable speedup, this is the aggregate floor.
    """
    edges = _common_edges(groups, nbins)
    if edges is None:
        from mlframe.reporting.spec import AnnotationPanelSpec
        return AnnotationPanelSpec(text=f"{title}\n(no finite data)", title=title)
    centers = (edges[:-1] + edges[1:]) / 2.0
    series: List[np.ndarray] = []
    labels: List[str] = []
    cols: List[str] = []
    for i, (lab, arr) in enumerate(groups.items()):
        a = _as_float_1d(arr)
        a = a[np.isfinite(a)]
        dens, _ = np.histogram(a, bins=edges, density=True) if a.size else (np.zeros(len(centers)), edges)
        series.append(dens)
        labels.append(f"{lab} (mean={a.mean():.3g})" if a.size else f"{lab} (empty)")
        cols.append(_colors.line_color(i))

    vlines = None
    vspans = None
    if train_key is not None and train_key in groups:
        tr = _as_float_1d(groups[train_key])
        tr = tr[np.isfinite(tr)]
        if tr.size:
            p01, p99 = float(np.percentile(tr, 1)), float(np.percentile(tr, 99))
            vlines = ((p01, "gray", "train p01"), (p99, "gray", "train p99"))
            vspans = ((p01, p99, "gray", 0.08),)
    return LinePanelSpec(
        x=centers,
        y=tuple(series),
        series_labels=tuple(labels),
        colors=tuple(cols),
        title=title,
        xlabel=xlabel,
        ylabel="Density",
        vlines=vlines,
        vspans=vspans,
    )


def _classrate_panel(
    groups: Dict[str, np.ndarray],
    *,
    title: str,
    xlabel: str,
) -> PanelSpec:
    """Per-split class-rate grouped bars: one bar group per class, one bar per split. Aggregate via bincount."""
    classes = np.unique(np.concatenate([np.asarray(a).ravel() for a in groups.values() if len(a)])) \
        if any(len(a) for a in groups.values()) else np.array([0])
    class_index = {c: i for i, c in enumerate(classes)}
    rate_series: List[np.ndarray] = []
    split_labels: List[str] = []
    for lab, arr in groups.items():
        a = np.asarray(arr).ravel()
        rates = np.zeros(len(classes), dtype=np.float64)
        if a.size:
            for c, cnt in zip(*np.unique(a, return_counts=True)):
                rates[class_index[c]] = cnt / a.size
        rate_series.append(rates)
        split_labels.append(str(lab))
    return BarPanelSpec(
        categories=tuple(f"class {c:g}" if np.issubdtype(type(c), np.number) else str(c) for c in classes),
        values=tuple(rate_series),
        series_labels=tuple(split_labels),
        title=title,
        xlabel=xlabel,
        ylabel="Class rate",
    )


def target_dist_overlay(
    y_true_by_split: Dict[str, np.ndarray],
    *,
    pred_by_split: Optional[Dict[str, np.ndarray]] = None,
    task: str = "regression",
    nbins: int = DEFAULT_OVERLAY_BINS,
    train_key: str = "train",
    title: str = "Target & prediction distribution by split",
) -> FigureSpec:
    """Overlaid per-split distributions of y AND of predictions (R-3 / INV-11).

    ``y_true_by_split`` / ``pred_by_split`` map a split name ("train"/"val"/"test"/"oof") to its value array. For
    regression each panel overlays per-split density histograms with the train p01/p99 vlines + a shaded train
    envelope, so a train/test target shift is visible as separated curves. For classification each panel shows
    per-split class-rate grouped bars. The prediction panel naturally carries the OOF-vs-test prediction overlay
    when both keys are present. All binning is ``np.histogram`` / ``bincount`` (O(n)); curves stay at ``nbins``
    vertices regardless of row count.
    """
    panels: List[PanelSpec] = []
    if task == "classification":
        panels.append(_classrate_panel(y_true_by_split, title="Target class rate by split", xlabel="class"))
        if pred_by_split:
            panels.append(_classrate_panel(pred_by_split, title="Prediction class rate by split", xlabel="class"))
    else:
        panels.append(_density_overlay_panel(
            y_true_by_split, nbins=nbins, title="Target (y) distribution by split",
            xlabel="y", train_key=train_key,
        ))
        if pred_by_split:
            panels.append(_density_overlay_panel(
                pred_by_split, nbins=nbins, title="Prediction distribution by split (incl. OOF vs test)",
                xlabel="prediction", train_key=train_key if train_key in pred_by_split else None,
            ))
    grid = pack_panels(panels, max_cols=2)
    return FigureSpec(
        suptitle=title,
        panels=grid,
        figsize=figsize_for_grid(1, max(len(panels), 1), cell_width=7.0, cell_height=4.5),
    )


__all__ = [
    "WeakSegmentResult",
    "ErrorBiasResult",
    "WorstKResult",
    "weak_segment_heatmap",
    "error_bias_per_feature",
    "worst_k_table",
    "segments_bar",
    "target_dist_overlay",
    "DEFAULT_HEATMAP_BINS",
    "DEFAULT_TREE_DEPTH",
    "DEFAULT_TREE_FIT_CAP",
    "DEFAULT_OVERLAY_BINS",
    "DEFAULT_WORST_K",
    "DEFAULT_TAIL_FRACTION",
]
