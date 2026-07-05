"""2D calibration-ECE heatmap: condition calibration on a TWO-feature quantile grid, ECE per cell.

A pooled reliability curve -- and even the 1D per-feature views (``calibration_by_feature``) -- can hide a pocket of
miscalibration that only appears at a SPECIFIC corner of joint feature space (e.g. the model is overconfident only when
``f_x`` is high AND ``f_y`` is high; either feature alone averages the pocket away). This composer quantile-bins BOTH
features into an ``n_bins x n_bins`` grid, computes a per-cell ECE, and renders an RdYlGn_r heatmap (green = calibrated,
red = miscalibrated) annotated with each cell's ECE + support, masking under-populated cells. The headline is the worst
cell's ECE and its ``(f_x bin, f_y bin)`` location -- where in feature space to distrust the probabilities.

Per-cell ECE here is the calibration GAP ``|mean(y_score) - mean(y_true)|`` over the cell's rows. With both features
quantile-binned this is the cell-conditional reliability error; it isolates a localized over/under-confidence pocket that
a global binned-ECE pools away, and it is computable in a single O(n) pass (bincount of score+true over the flattened
cell index), no per-cell python loop over the data.

Efficiency: cell assignment is two ``np.searchsorted`` calls on the quantile edges; sums are ``np.bincount`` weighted by
score / true / 1 over the flattened ``row*n_bins + col`` index -- one O(n) pass. Huge inputs are uniformly subsampled to a
cap before the pass so cost stays bounded at n >= 1e6. Edge-safe: NaN rows dropped; a feature with <2 distinct quantile
edges is annotated and the grid degenerates; cells below the support floor are greyed (masked) rather than shown as noise.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from mlframe.reporting.spec import AnnotationPanelSpec, FigureSpec, HeatmapPanelSpec, PanelSpec

# Below this many rows a cell's mean-gap ECE is binomial noise, not a calibration signal; grey (mask) the cell.
_MIN_CELL_ROWS: int = 30
# Uniform subsample cap before the single O(n) pass: cell means are stable far below this, so a cap keeps n>=1e6 bounded
# without materially moving any cell's ECE.
_SUBSAMPLE_CAP: int = 1_000_000
# Worst-cell traffic-light: below green the joint grid is calibrated everywhere; above red there is a clear pocket.
_GREEN: float = 0.05
_RED: float = 0.10


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Unique quantile edges for ``values`` (ties collapse on a low-cardinality / constant feature)."""
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    return np.unique(np.quantile(values, qs))


def _bin_codes(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map each value to a [0, len(edges)-2] bin via the inner edges; O(n) searchsorted."""
    inner = edges[1:-1]
    if inner.size == 0:
        return np.zeros(values.size, dtype=np.int64)
    return np.asarray(np.searchsorted(inner, values, side="right"), dtype=np.int64)


def _edge_labels(edges: np.ndarray) -> Tuple[str, ...]:
    nb = edges.size - 1
    return tuple(f"[{edges[i]:.3g}, {edges[i + 1]:.3g}{')' if i < nb - 1 else ']'}" for i in range(nb))


def _traffic_light(worst: float) -> str:
    if not np.isfinite(worst):
        return "n/a"
    if worst < _GREEN:
        return "green"
    if worst < _RED:
        return "amber"
    return "red"


def compute_calibration_heatmap_2d(
    y_true: np.ndarray,
    y_score: np.ndarray,
    feat_x: np.ndarray,
    feat_y: np.ndarray,
    *,
    n_bins: int = 5,
    random_state: int = 0,
) -> Mapping[str, object]:
    """Headless per-cell ECE grid + worst-cell headline, without building a figure.

    Returns ``{"ece_grid", "support_grid", "x_labels", "y_labels", "worst_ece", "worst_cell", "median_cell_ece",
    "traffic_light", "skipped"}``. ``ece_grid`` is an ``n_y x n_x`` float array (rows = ``feat_y`` quantile bins low->high,
    cols = ``feat_x`` bins) of per-cell ``|mean(score) - mean(true)|``; under-populated cells are NaN. ``worst_cell`` is the
    ``(x_bin, y_bin)`` index pair of the max-ECE populated cell, or ``None`` when no cell clears the support floor. This is
    the metric the biz_value test + the suite consume; the figure composer is the visual surface.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    fx = np.asarray(feat_x, dtype=np.float64).ravel()
    fy = np.asarray(feat_y, dtype=np.float64).ravel()
    n = min(yt.size, ys.size, fx.size, fy.size)
    yt, ys, fx, fy = yt[:n], ys[:n], fx[:n], fy[:n]

    finite = np.isfinite(yt) & np.isfinite(ys) & np.isfinite(fx) & np.isfinite(fy)
    yt, ys, fx, fy = yt[finite], ys[finite], fx[finite], fy[finite]
    skipped: list[str] = []
    if yt.size == 0:
        return {"ece_grid": np.zeros((0, 0)), "support_grid": np.zeros((0, 0), dtype=np.int64),
                "x_labels": (), "y_labels": (), "worst_ece": float("nan"), "worst_cell": None,
                "median_cell_ece": float("nan"), "traffic_light": "n/a",
                "skipped": ["no finite (label, score, feat_x, feat_y) rows"]}

    if yt.size > _SUBSAMPLE_CAP:
        sel = np.random.default_rng(random_state).choice(yt.size, size=_SUBSAMPLE_CAP, replace=False)
        yt, ys, fx, fy = yt[sel], ys[sel], fx[sel], fy[sel]

    ex = _quantile_edges(fx, n_bins)
    ey = _quantile_edges(fy, n_bins)
    if ex.size < 2:
        skipped.append("feat_x has <2 distinct quantile values (degenerate axis)")
    if ey.size < 2:
        skipped.append("feat_y has <2 distinct quantile values (degenerate axis)")
    if ex.size < 2 or ey.size < 2:
        return {"ece_grid": np.zeros((0, 0)), "support_grid": np.zeros((0, 0), dtype=np.int64),
                "x_labels": (), "y_labels": (), "worst_ece": float("nan"), "worst_cell": None,
                "median_cell_ece": float("nan"), "traffic_light": "n/a", "skipped": skipped}

    cx = _bin_codes(fx, ex)
    cy = _bin_codes(fy, ey)
    nx = ex.size - 1
    ny = ey.size - 1
    flat = cy * nx + cx  # row-major over (y_bin, x_bin)
    ncell = nx * ny
    cnt = np.bincount(flat, minlength=ncell).astype(np.float64)
    sum_s = np.bincount(flat, weights=ys, minlength=ncell)
    sum_t = np.bincount(flat, weights=yt, minlength=ncell)

    populated = cnt >= _MIN_CELL_ROWS
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_s = np.where(populated, sum_s / cnt, np.nan)
        mean_t = np.where(populated, sum_t / cnt, np.nan)
    ece_flat = np.abs(mean_s - mean_t)

    ece_grid = ece_flat.reshape(ny, nx)
    support_grid = cnt.reshape(ny, nx).astype(np.int64)

    if not np.any(np.isfinite(ece_flat)):
        skipped.append(f"no cell reached the {_MIN_CELL_ROWS}-row support floor")
        worst_ece, worst_cell, median_ece, light = float("nan"), None, float("nan"), "n/a"
    else:
        worst_idx = int(np.nanargmax(ece_flat))
        worst_ece = float(ece_flat[worst_idx])
        worst_cell = (int(worst_idx % nx), int(worst_idx // nx))  # (x_bin, y_bin)
        median_ece = float(np.nanmedian(ece_flat))
        light = _traffic_light(worst_ece)

    return {"ece_grid": ece_grid, "support_grid": support_grid,
            "x_labels": _edge_labels(ex), "y_labels": _edge_labels(ey),
            "worst_ece": worst_ece, "worst_cell": worst_cell,
            "median_cell_ece": median_ece, "traffic_light": light, "skipped": skipped}


def compose_calibration_heatmap_2d_figure(
    y_true: np.ndarray,
    y_score: np.ndarray,
    feat_x: np.ndarray | Sequence,
    feat_y: np.ndarray | Sequence,
    *,
    feat_x_name: str = "feat_x",
    feat_y_name: str = "feat_y",
    n_bins: int = 5,
    random_state: int = 0,
    figsize: Optional[Tuple[float, float]] = None,
) -> FigureSpec:
    """RdYlGn_r heatmap of per-cell calibration ECE over a quantile grid of (feat_x, feat_y), worst-cell as headline.

    Both features are quantile-binned into an ``n_bins x n_bins`` grid (rows = ``feat_y`` low->high, cols = ``feat_x``); each
    cell shows ``|mean(y_score) - mean(y_true)|`` (the cell-conditional calibration gap) coloured green (calibrated) to red
    (miscalibrated), annotated with the ECE and the cell support ``n``. Cells below the {floor}-row support floor are greyed
    (NaN) rather than shown as noise. The suptitle reports the worst cell's ECE + its ``(f_x bin, f_y bin)`` location and a
    traffic-light (< {green} green, < {red} amber, else red) -- a localized pocket the pooled / 1D views average away. A
    degenerate feature (<2 distinct quantile values) is annotated and the grid is skipped.

    O(n): two ``np.searchsorted`` for cell assignment + three weighted ``np.bincount`` (one pass); huge inputs subsampled.
    """
    res = compute_calibration_heatmap_2d(
        y_true, y_score, feat_x, feat_y, n_bins=n_bins, random_state=random_state,
    )
    title = f"2D calibration ECE: {feat_x_name} x {feat_y_name}"
    skipped = res["skipped"]
    ece_grid = np.asarray(res["ece_grid"], dtype=np.float64)

    if ece_grid.size == 0 or res["worst_cell"] is None:
        note = (f"2D calibration heatmap needs a populated quantile grid (>= {_MIN_CELL_ROWS} rows/cell)."
                + ("  skipped: " + ", ".join(skipped) if skipped else ""))
        return FigureSpec(
            suptitle="",
            panels=((AnnotationPanelSpec(text=note, title=title),),),
            figsize=figsize or (9.0, 4.0),
        )

    x_labels = res["x_labels"]
    y_labels = res["y_labels"]
    support = np.asarray(res["support_grid"], dtype=np.int64)
    wx, wy = res["worst_cell"]
    worst = float(res["worst_ece"])
    median_ece = float(res["median_cell_ece"])
    light = res["traffic_light"]

    cell_text = np.empty(ece_grid.shape, dtype=object)
    for i in range(ece_grid.shape[0]):
        for j in range(ece_grid.shape[1]):
            e = ece_grid[i, j]
            cell_text[i, j] = "n/a" if not np.isfinite(e) else f"{e:.3f}\nn={support[i, j]:,}"

    headline = f"worst cell: {feat_x_name}={x_labels[wx]}, {feat_y_name}={y_labels[wy]}  ECE={worst:.3f}  [{light}]" f"  |  median-cell ECE={median_ece:.3f}"
    skipped_note = ("  skipped: " + ", ".join(skipped)) if skipped else ""

    panel: PanelSpec = HeatmapPanelSpec(
        matrix=ece_grid,
        row_labels=tuple(y_labels),
        col_labels=tuple(x_labels),
        title=headline,
        xlabel=f"{feat_x_name} quantile bin (low -> high)",
        ylabel=f"{feat_y_name} quantile bin (low -> high)",
        colormap="RdYlGn_r",
        cell_text=cell_text,
        text_format="",
        colorbar_label="per-cell ECE (lower = better calibrated)",
    )
    width = figsize[0] if figsize else max(8.0, 1.6 * len(x_labels) + 3.0)
    height = figsize[1] if figsize else max(6.0, 1.4 * len(y_labels) + 2.0)
    return FigureSpec(
        suptitle=f"{title}{skipped_note}",
        panels=((panel,),),
        figsize=(width, height),
    )


__all__ = [
    "compose_calibration_heatmap_2d_figure",
    "compute_calibration_heatmap_2d",
]
