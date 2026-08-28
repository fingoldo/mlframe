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

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from mlframe.reporting.charts._layout import (
    figsize_for_grid, pack_panels,
)
from ._error_analysis_shared import DEFAULT_OVERLAY_BINS, _as_float_1d
from mlframe.reporting.spec import (
    AnnotationPanelSpec, BarPanelSpec, FigureSpec, HeatmapPanelSpec,
    LinePanelSpec, PanelSpec,
)

# A 1-2-feature weak-segment grid: more cells than this fragments support per cell into noise (FreaAI keeps slices coarse so findings stay actionable).
DEFAULT_HEATMAP_BINS: int = 6
# Shallow on purpose: deep trees overfit the error signal and the top splits stop being the genuinely worst-performing slices.
DEFAULT_TREE_DEPTH: int = 3
# The tree only needs enough rows to RANK split features; fitting it on all of a 1M+ row set is the whole cost (~1.8s).
# A 50k subsample picks the same top features and drops that to ~80ms; cell stats below still use every row.
DEFAULT_TREE_FIT_CAP: int = 50_000
# DEFAULT_OVERLAY_BINS / _DRIFT_Z / _as_float_1d come from ._error_analysis_shared (one definition,
# imported by this module and by the carved-out siblings).
# Default worst-K rows surfaced; 20 fits a screen and red-highlights without swamping the scatter.
DEFAULT_WORST_K: int = 20
# Over/under tail fraction for error-bias tagging (Evidently's signature 5% tails).
DEFAULT_TAIL_FRACTION: float = 0.05


def _per_row_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    task: str,
) -> np.ndarray:
    """Per-row error signal the weak-segment tree fits.

    Regression: absolute residual. Classification: log-loss when ``y_pred`` is a probability/score in [0,1]
    (richer than 0/1, so the tree sees *how* wrong), else 0/1 incorrectness when ``y_pred`` is a hard label.

    The log-loss branch is BINARY, so it only applies when ``y_true`` is in {0, 1}. It was previously chosen off
    ``y_pred``'s range alone, so a multiclass label reached it too: for ``y_true=2, p=0.9`` the binary formula
    returns -2.09, a NEGATIVE "error" that then trained the weak-segment tree and coloured a "darker = worse"
    heatmap. Multiclass labels now take the 0/1 incorrectness branch, which is well defined for any label set.
    """
    yt = _as_float_1d(y_true)
    yp = _as_float_1d(y_pred)
    if task == "regression":
        return np.asarray(np.abs(yt - yp))
    finite_yt = yt[np.isfinite(yt)]
    is_binary_label = finite_yt.size == 0 or bool(np.isin(finite_yt, (0.0, 1.0)).all())
    looks_proba = is_binary_label and yp.size > 0 and float(np.nanmin(yp)) >= 0.0 and float(np.nanmax(yp)) <= 1.0
    if looks_proba:
        eps = 1e-12
        p = np.clip(yp, eps, 1.0 - eps)
        return np.asarray(-(yt * np.log(p) + (1.0 - yt) * np.log(1.0 - p)))
    return np.asarray((yt != yp).astype(np.float64))


def _resolve_feature_matrix(
    X: Any,
    feature_names: Optional[Sequence[str]],
) -> Tuple[np.ndarray, List[str]]:
    """Coerce ``X`` (pandas / polars / ndarray) to a 2-D float matrix + name list without a full frame copy.

    Columns are pulled one at a time (narrow ndarray views), never via a whole-frame ``to_pandas`` / ``to_numpy``
    on a 100+ GB carrier. Non-numeric columns are label-encoded to integer codes so the tree can still split on them.
    Object-dtype columns holding non-scalar elements (e.g. list-valued embedding columns surfaced as pandas
    object dtype) can't be stringified by ``astype(str)`` -- numpy raises "setting an array element with a
    sequence" trying to broadcast the list into a fixed-width string array -- so those columns are dropped
    (a single embedding vector isn't a meaningful scalar split feature anyway).
    """
    if hasattr(X, "columns") and hasattr(X, "__getitem__") and not isinstance(X, np.ndarray):
        cols = list(X.columns)
        all_names = list(feature_names) if feature_names is not None else [str(c) for c in cols]
        if len(all_names) != len(cols):
            # A bare zip() silently truncated to the shorter side, so a short `feature_names` dropped real columns
            # from every weak-segment and error-bias diagnostic with no signal at all -- the caller saw a plausible
            # chart built on a subset it never asked for.
            raise ValueError(
                f"_resolve_feature_matrix: feature_names has {len(all_names)} entries but X has {len(cols)} columns; "
                "they must correspond one-to-one, otherwise columns are silently dropped from the diagnostics."
            )
        mats: List[np.ndarray] = []
        names: List[str] = []
        for c, name in zip(cols, all_names):
            col = X[c]
            arr = col.to_numpy() if hasattr(col, "to_numpy") else np.asarray(col)
            if arr.dtype.kind in "OUS" or arr.dtype.kind == "b":
                if arr.dtype.kind == "O" and any(isinstance(v, (list, tuple, np.ndarray)) for v in arr):
                    continue
                _, codes = np.unique(arr.astype(str), return_inverse=True)
                mats.append(codes.astype(np.float64))
            else:
                mats.append(arr.astype(np.float64))
            names.append(name)
        mat = np.column_stack(mats) if mats else np.empty((len(X), 0), dtype=np.float64)
        return mat, names
    mat = np.asarray(X, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(mat.shape[1])]
    return mat, names


def _support_floor(n_rows: int) -> int:
    """Rows a grid cell needs before its mean error may be ranked as the worst segment.

    ``nanargmax`` over per-cell means with no floor picks whichever cell happens to hold the most extreme few rows:
    on near-collinear features the observed count matrices carry off-diagonal cells with 5-14 rows sitting beside
    650-row cells, and one unlucky outlier in a 5-row cell wins every time. The chosen cell is printed in the title
    as the actionable finding, so this is a wrong answer, not a cosmetic one.
    """
    return max(20, round(0.005 * max(int(n_rows), 0)))


def _worst_supported_cell(mean_err: np.ndarray, counts: np.ndarray, floor: int) -> int:
    """Flat index of the highest-mean-error cell holding at least ``floor`` rows.

    Falls back to the unfiltered argmax when the floor excludes everything, so a small dataset still gets an answer
    rather than an exception -- the title states the floor either way.
    """
    scored = np.where(np.isfinite(mean_err) & (counts >= floor), mean_err, -np.inf)
    if not np.isfinite(scored).any():
        scored = np.where(np.isfinite(mean_err), mean_err, -np.inf)
    return int(np.nanargmax(scored))


def _row_count(X: Any) -> int:
    """Row count of ``X`` (frame ``len`` / ndarray first axis) without materialising it."""
    if _is_frame(X):
        return len(X)
    arr = np.asarray(X)
    return arr.shape[0] if arr.ndim >= 1 else 0


def _is_frame(X: Any) -> bool:
    """True when ``X`` is a pandas / polars frame (has ``columns`` and is indexable) rather than an ndarray."""
    return hasattr(X, "columns") and hasattr(X, "__getitem__") and not isinstance(X, np.ndarray)


def _resolve_feature_names(X: Any, feature_names: Optional[Sequence[str]]) -> List[str]:
    """Feature names WITHOUT densifying the matrix -- for callers that only need a handful of columns at a few rows.

    Mirrors :func:`_resolve_feature_matrix`'s naming, but skips the whole-frame ``column_stack``. Frames expose their
    column labels directly; an ndarray gets positional ``f{i}`` names. Lets ``worst_k_table`` rank importances and pick
    label columns without building the full dense matrix it would immediately discard all but a K x top_fi slice of.
    """
    if _is_frame(X):
        cols = list(X.columns)
        return list(feature_names) if feature_names is not None else [str(c) for c in cols]
    arr = np.asarray(X)
    ncols = 1 if arr.ndim == 1 else arr.shape[1]
    return list(feature_names) if feature_names is not None else [f"f{i}" for i in range(ncols)]


def _pull_columns_at_rows(X: Any, col_indices: Sequence[int], row_idx: np.ndarray) -> Dict[int, np.ndarray]:
    """Densify ONLY ``col_indices`` at ``row_idx`` -- bit-identical to ``_resolve_feature_matrix(X)[row_idx][:, j]``.

    Returns ``{col_index -> float64 values at row_idx}``. Non-numeric frame columns are label-encoded over the FULL
    column (``np.unique`` codes) exactly as :func:`_resolve_feature_matrix` does, then indexed -- so the codes match
    the full-matrix path. Avoids building+discarding the whole dense matrix when only a few columns at a few rows
    are needed.
    """
    out: Dict[int, np.ndarray] = {}
    if _is_frame(X):
        cols = list(X.columns)
        for j in col_indices:
            col = X[cols[j]]
            arr = col.to_numpy() if hasattr(col, "to_numpy") else np.asarray(col)
            if arr.dtype.kind in "OUS" or arr.dtype.kind == "b":
                # An object column holding non-scalar elements (e.g. a list-valued embedding column) can't be
                # stringified by ``astype(str)`` -- numpy raises "setting an array element with a sequence"
                # trying to broadcast the list into a fixed-width string array. Mirror _resolve_feature_matrix's
                # embedding-column handling: substitute NaN rather than crash (a single embedding vector isn't
                # a meaningful scalar table cell anyway).
                if arr.dtype.kind == "O" and any(isinstance(v, (list, tuple, np.ndarray)) for v in arr):
                    out[j] = np.full(len(row_idx), np.nan, dtype=np.float64)
                    continue
                _, codes = np.unique(arr.astype(str), return_inverse=True)
                out[j] = codes.astype(np.float64)[row_idx]
            else:
                out[j] = arr.astype(np.float64)[row_idx]
        return out
    mat = np.asarray(X, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    for j in col_indices:
        out[j] = mat[row_idx, j]
    return out


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

        idx = subsample_preserving_extremes(err, sample_size=fit_cap, extreme_values=err, k_extremes=min(fit_cap // 10, err.size), rng=seed)
        fit_mat, fit_err = mat[idx], err[idx]
    else:
        fit_mat, fit_err = mat, err
    try:
        from sklearn.tree import DecisionTreeRegressor

        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=seed)
        tree.fit(fit_mat, fit_err)
        imp = np.asarray(tree.feature_importances_, dtype=np.float64)
    except (ValueError, ImportError) as e:
        logger.warning(
            "[reporting.charts] weak-segment tree fit failed (%s: %s); falling back to a weaker single-feature " "median-split surrogate ranking.",
            type(e).__name__,
            e,
        )
        # Surrogate ranking: a feature whose high/low halves differ most in mean error is the most error-discriminating.
        imp = np.zeros(n_cols, dtype=np.float64)
        for j in range(n_cols):
            col = fit_mat[:, j]
            finite = np.isfinite(col)
            if not finite.any():
                continue
            # nanmedian (not median): a column with ANY NaN made plain np.median return NaN, which makes
            # BOTH comparison masks (> med, <= med) all-False for every row (NaN comparisons are always
            # False) -- hi/lo came back empty and the column's surrogate importance silently stayed 0.0
            # regardless of its real discriminating power on the non-missing rows.
            med = np.nanmedian(col)
            hi = fit_err[finite & (col > med)]
            lo = fit_err[finite & (col <= med)]
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
    # Inf in any feature column crashes DecisionTreeRegressor.fit (ValueError); NaN feature values are
    # tolerated by modern sklearn and left alone so the tree can still use those rows.
    finite = np.isfinite(err) & ~np.any(np.isinf(mat), axis=1)
    if not np.all(finite):
        err = err[finite]
        mat = mat[finite]
    # Zero usable rows (empty input, or every row dropped by the finite-mask above e.g. all-NaN error):
    # binning/quantile-edge logic downstream assumes at least one row, and some sklearn versions raise
    # a ValueError from DecisionTreeRegressor.fit here that the except-fallback below doesn't itself
    # guard against re-indexing an empty array. Degenerate-out with the same "no usable features" panel
    # the all-zero-split-importance case already uses, before ever reaching the tree fit.
    top = [] if mat.shape[0] == 0 else _top_split_features(mat, err, names, max_depth=max_depth, n_features=2, seed=seed)
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
    # np.digitize sorts NaN after every finite edge, so a NaN feature value would silently land in the
    # HIGHEST bin instead of being excluded -- corrupting the worst-slice localization with rows that
    # were never actually binned by value. Exclude rows with NaN in the feature(s) being binned here
    # (a NaN feature value is still tolerated upstream for the tree fit that CHOSE these features).
    bin_finite = np.isfinite(mat[:, ja])
    if len(top) >= 2:
        bin_finite &= np.isfinite(mat[:, top[1]])
    if not np.all(bin_finite):
        err = err[bin_finite]
        mat = mat[bin_finite]

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
        support_floor = _support_floor(err.size)
        worst = _worst_supported_cell(mean_err, counts, support_floor)
        wa, wb = worst // nb, worst % nb
        worst_cell = (float(ea[wa]), float(ea[wa + 1]), float(eb[wb]), float(eb[wb + 1]), float(mean_err[worst]))
        row_labels = tuple(f"{ea[i]:.3g}..{ea[i + 1]:.3g}" for i in range(na))
        col_labels = tuple(f"{eb[i]:.3g}..{eb[i + 1]:.3g}" for i in range(nb))
        xlabel, ylabel = names[jb], names[ja]
        split_features: Tuple[str, ...] = (names[ja], names[jb])
    else:
        counts = np.bincount(ia, minlength=na).astype(np.float64)
        sums = np.bincount(ia, weights=err, minlength=na)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_err = np.where(counts > 0, sums / np.where(counts > 0, counts, 1.0), np.nan)
        cell_error = mean_err.reshape(na, 1)
        cell_count = counts.reshape(na, 1)
        support_floor = _support_floor(err.size)
        worst = _worst_supported_cell(mean_err, counts, support_floor)
        worst_cell = (float(ea[worst]), float(ea[worst + 1]), float("nan"), float("nan"), float(mean_err[worst]))
        row_labels = tuple(f"{ea[i]:.3g}..{ea[i + 1]:.3g}" for i in range(na))
        col_labels = tuple(["error"])
        xlabel, ylabel = "", names[ja]
        split_features = tuple([names[ja]])

    heat = HeatmapPanelSpec(
        matrix=cell_error,
        row_labels=row_labels,
        col_labels=col_labels,
        title=(
            title
            + f"\nworst slice: {split_features} mean_err={worst_cell[4]:.3g} over {int(counts[worst]):,} rows"
            + f" (global mean {float(np.nanmean(err)):.3g}; cells under {support_floor:,} rows excluded from the ranking)"
        ),
        xlabel=xlabel,
        ylabel=ylabel,
        colormap="Reds",
        cell_text=cell_count,
        text_format=".0f",
        # The colour and the colorbar encode MEAN ERROR; the number printed in each cell is that cell's ROW COUNT.
        # Naming both is what stops a reader taking the annotation for the quantity on the scale.
        colorbar_label="mean error (darker = worse); cell number = rows in cell",
    )
    return WeakSegmentResult(
        FigureSpec(
            panels=((heat,),),
            figsize=(8.0, 6.0),
            caption=(
                "A shallow tree fitted on PER-ROW error picked the most error-discriminating features; the grid bins "
                "them into equal-population slices. Colour = mean error (darker = worse); the number printed in each "
                f"cell is that cell's ROW COUNT, not the error. Cells under {support_floor:,} rows are excluded from "
                "the worst-slice ranking -- a dark cell backed by a handful of rows is an outlier, not a weak "
                "segment. Use the worst slice named in the title as the place to look first, then check its count."
            ),
        ),
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
    """Per-subgroup metric bars with a global-reference hline.

    ``slice_frame`` is the existing fairness / slice DataFrame: one row per subgroup with a group-name column and a
    metric column (auto-detected when not named). The global reference is drawn as a single ``hline`` across the value
    axis (perpendicular to the bars), so each subgroup is one honest bar instead of two interleaved series. When
    ``global_value`` is None it defaults to the count-weighted-or-plain mean of the per-group metric. Subgroups are
    sorted worst-first so the weakest slice is leftmost.
    """
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

    # Worst-first: ascending for a higher-is-better metric (accuracy / NDCG), descending for an error rate.
    order = np.argsort(metric)
    if higher_is_worse:
        order = order[::-1]
    order = order[:max_groups]

    cats = tuple(groups[order])
    vals = metric[order]
    bar = BarPanelSpec(
        categories=cats,
        values=vals,
        series_labels=(metric_name,),
        title=title + f"\n(worst-first; global reference = {global_value:.3g})",
        xlabel=str(group_col),
        ylabel=metric_name,
        colors=("steelblue",),
        xtick_rotation=45.0,
        hline=(float(global_value), "darkorange", f"global = {global_value:.3g}"),
    )
    caption = (
        "One bar per subgroup, sorted worst-first, against the global reference drawn in orange. The bars are raw "
        "point estimates with NO uncertainty attached, so a short bar over a small subgroup may be sample size "
        "rather than a real weakness -- read each bar together with its group size before acting on it."
    )
    return FigureSpec(suptitle="", panels=((bar,),), figsize=(max(8.0, len(cats) * 0.5), 5.0), caption=caption)


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

    part = np.argpartition(score, n - kk)[n - kk :]
    order = part[np.argsort(score[part])[::-1]]
    sel = finite_idx[order]

    names = _resolve_feature_names(X, feature_names)
    if feature_importances is not None and len(feature_importances) == len(names):
        fi = np.asarray(feature_importances, dtype=np.float64)
        fi_cols = [int(j) for j in np.argsort(fi)[::-1][:top_fi]]
    else:
        fi_cols = list(range(min(top_fi, len(names))))

    # Pull ONLY the chosen importance columns at the K worst rows -- avoids densifying the whole feature matrix
    # to read a K x top_fi slice (the full matrix would be built and discarded).
    col_vals = _pull_columns_at_rows(X, fi_cols, sel)

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
        data[names[j]] = col_vals[j]

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
    """Signed residual y_true - y_pred: positive => the model UNDER-predicts here, negative => it OVER-predicts.

    ONE convention for the whole figure. Group tagging previously used ``y_pred - y_true`` while the per-segment
    bias readout in the same figure used ``y_true - y_pred``, so the two halves disagreed about which sign meant
    "over" and a comment existed solely to reconcile them. The suptitle already declares this one.
    """
    return np.asarray(_as_float_1d(y_true) - _as_float_1d(y_pred))


def _tag_error_groups(
    signed_err: np.ndarray,
    tail_fraction: float,
) -> Dict[str, np.ndarray]:
    """Split rows into OVER / UNDER / MAJORITY by signed-error quantile.

    The top ``tail_fraction`` of signed errors (most positive) are OVER-estimates, the bottom ``tail_fraction``
    (most negative) UNDER-estimates, the middle is MAJORITY. Quantile cut via ``np.quantile`` (k-way partition,
    O(n)); no full sort.
    """
    finite = np.isfinite(signed_err)
    hi_cut = np.quantile(signed_err[finite], 1.0 - tail_fraction) if finite.any() else np.inf
    lo_cut = np.quantile(signed_err[finite], tail_fraction) if finite.any() else -np.inf
    if hi_cut == lo_cut:
        # A constant signed error makes both tail tests true for every row, so OVER and UNDER each held ALL rows and
        # MAJORITY held none -- three identical densities, which reads as "no error bias" rather than "no variation".
        return {"OVER": np.zeros_like(finite), "UNDER": np.zeros_like(finite), "MAJORITY": finite}
    # `signed_err` is y_true - y_pred, so the MOST NEGATIVE tail is where the model over-predicts.
    over = finite & (signed_err <= lo_cut)
    under = finite & (signed_err >= hi_cut)
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

    names = _resolve_feature_names(X, feature_names)
    signed = _signed_error(y_true, y_pred)
    masks = _tag_error_groups(signed, tail_fraction)
    resid_signed = signed

    missing_features: List[str] = []
    if features is not None:
        missing_features = [f for f in features if f not in names]
        sel = [names.index(f) for f in features if f in names]
    else:
        sel = list(range(min(max_features, len(names))))
    missing_note = f"\nrequested features not found, skipped: {', '.join(missing_features)}" if missing_features else ""

    # Densify ONLY the selected columns over all rows -- the overlay touches at most ``max_features`` (default 4) of
    # what can be a several-hundred-column frame, so building the whole dense matrix here just to discard all but
    # ``sel`` is wasted O(n*cols) work. Column pull is bit-identical to ``_resolve_feature_matrix(X)[:, j]``.
    all_rows = np.arange(_row_count(X), dtype=np.int64)
    col_vals = _pull_columns_at_rows(X, sel, all_rows)

    group_colors = {"OVER": "#d62728", "UNDER": "#1f77b4", "MAJORITY": "#7f7f7f"}
    panels: List[PanelSpec] = []
    rows: Dict[str, List[float]] = {g: [] for g in ("OVER", "UNDER", "MAJORITY")}
    feat_index: List[str] = []
    # Track the single worst-bias segment across every feature for the figure title: (|mean signed resid|, signed text).
    global_worst_abs = -1.0
    global_worst_text = ""

    for j in sel:
        col = col_vals[j]
        finite = np.isfinite(col)
        cf = col[finite]
        if cf.size == 0:
            continue
        try:
            edges = np.histogram_bin_edges(cf, bins=nbins)
        except ValueError:
            # A near-constant finite column (e.g. an MLP-engineered feature with ~zero variance) has a range so
            # tiny that numpy can't carve it into ``nbins`` distinct finite-precision edges -- raises "Too many
            # bins for data range" (caught live via a fuzz combo with mlp in the model mix). This one feature's
            # bias panel isn't meaningful anyway (no spread to bin), so skip it rather than aborting the whole
            # diagnostic (best-effort, matches every other panel in this dispatcher).
            continue
        if edges.size < 2 or not np.all(np.diff(edges) > 0):
            # numpy doesn't always raise on a near-constant huge-magnitude column -- it can instead
            # return edges collapsed to fewer distinct floats than requested (zero-width bins), which
            # silently produces NaN/inf densities downstream via divide-by-zero in np.histogram's
            # normalization instead of raising. Skip this feature's panel the same way as above.
            continue
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

        # Per-segment signed-residual bias: bin this feature's values, take the mean residual (y_true - y_pred) in each
        # bin. The segment with the largest |mean residual| is the model's worst-bias slice for this feature; its sign
        # tells direction (> 0 UNDER-predict, < 0 OVER-predict). Aggregated O(n) via two bincounts, no per-row Python.
        bin_idx = np.clip(np.digitize(col[finite], edges[1:-1]), 0, len(centers) - 1)
        nbin = len(centers)
        cnt = np.bincount(bin_idx, minlength=nbin).astype(np.float64)
        ssum = np.bincount(bin_idx, weights=resid_signed[finite], minlength=nbin)
        with np.errstate(invalid="ignore", divide="ignore"):
            seg_bias = np.where(cnt > 0, ssum / np.where(cnt > 0, cnt, 1.0), np.nan)
        worst_b = int(np.nanargmax(np.where(np.isfinite(seg_bias), np.abs(seg_bias), -np.inf))) if np.isfinite(seg_bias).any() else -1
        if worst_b >= 0:
            wb = float(seg_bias[worst_b])
            direction = "UNDER-predicts" if wb > 0 else "OVER-predicts"
            seg_lo, seg_hi = float(edges[worst_b]), float(edges[worst_b + 1])
            worst_note = f"worst-bias segment: {names[j]} in [{seg_lo:.3g}, {seg_hi:.3g}] -> model {direction} (mean resid {wb:+.3g})"
            if abs(wb) > global_worst_abs:
                global_worst_abs = abs(wb)
                global_worst_text = f"{names[j]} in [{seg_lo:.3g}, {seg_hi:.3g}] {direction} (resid {wb:+.3g})"
        else:
            worst_note = "worst-bias segment: n/a"

        panels.append(LinePanelSpec(
            x=centers,
            y=tuple(series),
            series_labels=tuple(labels),
            colors=tuple(cols),
            line_styles=("-", "-", "--"),
            title=f"{names[j]} value distribution by error group\n{worst_note}",
            xlabel=names[j],
            ylabel="Density",
        ))

    group_means = pd.DataFrame(
        {g: rows[g] for g in ("OVER", "UNDER", "MAJORITY")},
        index=feat_index,
    )
    if not panels:
        # No usable feature (all-NaN columns, zero features, or none selected); an empty grid would crash the renderer.
        ann = AnnotationPanelSpec(text=f"{title}\n(no usable feature columns){missing_note}", title=title)
        return ErrorBiasResult(FigureSpec(suptitle=title + missing_note, panels=((ann,),), figsize=(8.0, 3.0)), group_means, masks)
    grid = pack_panels(panels, max_cols=2)
    n_rows = len(grid)
    worst_line = f"Worst-bias segment overall: {global_worst_text}" if global_worst_text else "Worst-bias segment overall: n/a"
    suptitle = f"{title}\nSigned residual = y_true - y_pred; > 0 in a segment => model UNDER-predicts there, < 0 => OVER-predicts.\n" f"{worst_line}{missing_note}"
    fig = FigureSpec(
        suptitle=suptitle,
        panels=grid,
        figsize=figsize_for_grid(max(n_rows, 1), 2, cell_width=6.0, cell_height=4.0),
        caption=(
            f"Rows are split by signed residual (y_true - y_pred) into the bottom {tail_fraction:.0%} where the model "
            f"OVER-predicts, the top {tail_fraction:.0%} where it UNDER-predicts, and the middle MAJORITY. Each panel "
            "overlays those three groups' value distributions for one feature: wherever the tail curves pull away "
            "from the majority, that feature's values are what drive the extreme errors. y is a density, so each "
            "curve integrates to 1 and the three groups' heights are NOT comparable as counts."
        ),
    )
    return ErrorBiasResult(fig, group_means, masks)


# The per-split target/prediction overlay family (``target_dist_overlay`` and its helpers) lives in
# ``._error_analysis_splits`` and is re-exported below. Carved out to keep this file under the house
# 1000-LOC limit; it is the most self-contained group here, sharing only the small array helpers above.

# ``from ...error_analysis import target_dist_overlay`` import site keeps resolving after the carve.
from ._error_analysis_splits import (
    _classrate_panel,  # noqa: F401 -- re-exported for import sites predating the carve
    _common_edges,  # noqa: F401 -- re-exported for import sites predating the carve
    _density_overlay_panel,  # noqa: F401 -- re-exported for import sites predating the carve
    _split_arrays,  # noqa: F401 -- re-exported for import sites predating the carve
    _target_drift_verdict,  # noqa: F401 -- re-exported for import sites predating the carve
    target_dist_overlay,  # in __all__, so no noqa needed
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
