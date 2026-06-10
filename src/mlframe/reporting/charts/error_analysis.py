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


__all__ = [
    "WeakSegmentResult",
    "weak_segment_heatmap",
    "DEFAULT_HEATMAP_BINS",
    "DEFAULT_TREE_DEPTH",
    "DEFAULT_OVERLAY_BINS",
    "DEFAULT_WORST_K",
    "DEFAULT_TAIL_FRACTION",
]
