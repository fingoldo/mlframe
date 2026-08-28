"""2D calibration-ECE heatmap: condition calibration on a TWO-feature quantile grid, ECE per cell.

A pooled reliability curve -- and even the 1D per-feature views (``calibration_by_feature``) -- can hide a pocket of
miscalibration that only appears at a SPECIFIC corner of joint feature space (e.g. the model is overconfident only when
``f_x`` is high AND ``f_y`` is high; either feature alone averages the pocket away). This composer quantile-bins BOTH
features into an ``n_bins x n_bins`` grid, computes a per-cell ECE, and renders an RdYlGn_r heatmap (green = calibrated,
red = miscalibrated) annotated with each cell's ECE + support, masking under-populated cells. The headline is the worst
cell's ECE and its ``(f_x bin, f_y bin)`` location -- where in feature space to distrust the probabilities.

Per-cell ECE here is a REAL binned ECE: the cell's rows are bucketed by score into ``_CELL_SCORE_BINS``, and the
support-weighted mean of the per-bucket ``|mean(score) - mean(true)|`` gaps is the cell's value. The cheaper
cell-level mean gap it replaced was not an ECE at all -- opposite-signed miscalibration inside one cell cancels
exactly under it, so a cell holding half its rows at score 0.9 / target 0 and half at 0.1 / target 1 is maximally
miscalibrated yet reported 0.00 and painted green. Still a single O(n) pass: one bincount over the flattened
``cell * score_bins + score_bin`` index, no per-cell python loop over the data.

Both the per-cell values and the worst-cell headline are graded against the sampling-noise floor their own support
implies (``null_ece_scale``), because an ECE is a mean ABSOLUTE deviation and so is bounded away from zero at finite
n. Against a fixed bar the identical calibrated model read red at 40 rows/cell and green at 4,000 -- the verdict
measured the grid resolution rather than the model.

Efficiency: cell assignment is two ``np.searchsorted`` calls on the quantile edges; sums are ``np.bincount`` weighted by
score / true / 1 over the flattened ``row*n_bins + col`` index -- one O(n) pass. Huge inputs are uniformly subsampled to a
cap before the pass so cost stays bounded at n >= 1e6. Edge-safe: NaN rows dropped; a feature with <2 distinct quantile
edges is annotated and the grid degenerates; cells below the support floor are greyed (masked) rather than shown as noise.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from mlframe.reporting.charts._calibration_chart_shared import null_ece_scale
from mlframe.reporting.spec import AnnotationPanelSpec, FigureSpec, HeatmapPanelSpec, PanelSpec

# Below this many rows a cell's ECE is binomial noise, not a calibration signal; grey (mask) the cell.
_MIN_CELL_ROWS: int = 30
# Uniform subsample cap before the single O(n) pass: cell means are stable far below this, so a cap keeps n>=1e6 bounded
# without materially moving any cell's ECE.
_SUBSAMPLE_CAP: int = 1_000_000
# Worst-cell traffic-light: below green the joint grid is calibrated everywhere; above red there is a clear pocket.
_GREEN: float = 0.05
_RED: float = 0.10
# Score bins used INSIDE each grid cell to compute a real ECE. Few enough that a cell at the support floor still
# puts several rows in each bin, many enough that opposite-signed miscalibration inside a cell cannot cancel.
_CELL_SCORE_BINS: int = 5


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
    """Half-open interval labels per bin (``[lo, hi)``), with the last bin closed on the right (``]``) so the max value is included."""
    nb = edges.size - 1
    return tuple(f"[{edges[i]:.3g}, {edges[i + 1]:.3g}{')' if i < nb - 1 else ']'}" for i in range(nb))


def _worst_cell_noise_floor(cell_floor: float, n_populated: int, n_score_bins: int) -> float:
    """The value the WORST of ``n_populated`` calibrated cells is expected to reach, not the value one cell reaches.

    ``null_ece_scale`` gives the ECE a single perfectly calibrated cell shows. The headline here is a MAXIMUM over
    every populated cell, and the max of many draws sits well above any one of them -- with 256 cells a calibrated
    model produced a worst-cell ECE of 0.188 against a per-cell floor of 0.101, so grading the max by the per-cell
    floor still condemned it. Each cell's ECE is a mean of ``n_score_bins`` absolute deviations, so its standard
    deviation is about ``sqrt((pi - 2) / 2 / n_score_bins)`` times its mean, and the expected maximum of ``m``
    such draws is about ``mu + sigma * sqrt(2 * ln m)``.
    """
    if n_populated <= 1 or n_score_bins <= 0 or not np.isfinite(cell_floor):
        return cell_floor
    rel_sd = float(np.sqrt((np.pi - 2.0) / 2.0 / n_score_bins))
    return float(cell_floor * (1.0 + rel_sd * np.sqrt(2.0 * np.log(n_populated))))


def _traffic_light(worst: float, noise_floor: float = 0.0) -> str:
    """Grade the worst cell's ECE against the LARGER of the fixed bar and that cell's own sampling-noise floor.

    A per-cell ECE cannot be graded by a constant: it is a mean absolute deviation, so a perfectly calibrated cell
    still shows a non-zero value that grows as its row count falls. With the fixed bar alone the same calibrated
    model read RED at 40 rows/cell and green at 4,000 -- the verdict measured the grid resolution, not the model.
    ``"n/a"`` when ``worst`` is non-finite (empty / degenerate grid).
    """
    if not np.isfinite(worst):
        return "n/a"
    if worst < max(_GREEN, noise_floor):
        return "green"
    if worst < max(_RED, 2.0 * noise_floor):
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
) -> Mapping[str, Any]:
    """Headless per-cell ECE grid + worst-cell headline, without building a figure.

    Returns ``{"ece_grid", "support_grid", "x_labels", "y_labels", "worst_ece", "worst_cell", "median_cell_ece",
    "traffic_light", "skipped"}``. ``ece_grid`` is an ``n_y x n_x`` float array (rows = ``feat_y`` quantile bins low->high,
    cols = ``feat_x`` bins) of per-cell binned ECE; under-populated cells are NaN. ``worst_cell`` is the
    ``(x_bin, y_bin)`` index pair of the max-ECE populated cell, or ``None`` when no cell clears the support floor. This is
    the metric the biz_value test + the suite consume; the figure composer is the visual surface.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    fx = np.asarray(feat_x, dtype=np.float64).ravel()
    fy = np.asarray(feat_y, dtype=np.float64).ravel()
    n = yt.size
    if not (ys.size == n == fx.size == fy.size):
        raise ValueError(f"calibration_heatmap_2d: y_true ({n}), y_score ({ys.size}), feat_x ({fx.size}), " f"feat_y ({fy.size}) must have equal length")

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

    populated = cnt >= _MIN_CELL_ROWS
    # A REAL binned ECE per cell, not |mean(score) - mean(target)|. The mean gap lets within-cell over- and
    # under-confidence cancel exactly: a cell holding half its rows at score 0.9 / target 0 and half at score 0.1 /
    # target 1 is maximally miscalibrated, yet both means are 0.5 and it reported a gap of 0.00 and painted green.
    # Binning by score inside each cell before taking the absolute gap is what makes that cell visible.
    sb = np.clip((ys * _CELL_SCORE_BINS).astype(np.int64), 0, _CELL_SCORE_BINS - 1)
    sub = flat * _CELL_SCORE_BINS + sb
    nsub = ncell * _CELL_SCORE_BINS
    sub_cnt = np.bincount(sub, minlength=nsub).astype(np.float64)
    sub_s = np.bincount(sub, weights=ys, minlength=nsub)
    sub_t = np.bincount(sub, weights=yt, minlength=nsub)
    with np.errstate(invalid="ignore", divide="ignore"):
        sub_gap = np.abs(np.where(sub_cnt > 0, sub_s / np.where(sub_cnt > 0, sub_cnt, 1.0), 0.0)
                         - np.where(sub_cnt > 0, sub_t / np.where(sub_cnt > 0, sub_cnt, 1.0), 0.0))
    # Support-weighted mean of the per-score-bin gaps within each cell -- the standard ECE, computed per cell.
    weighted = (sub_gap * sub_cnt).reshape(ncell, _CELL_SCORE_BINS).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ece_flat = np.where(populated, weighted / cnt, np.nan)
    # Each cell's own noise floor: a perfectly calibrated cell of this size still shows a non-zero ECE, and the
    # floor falls as 1/sqrt(cell rows). Grading every cell against one constant made a calibrated model read RED
    # at 40 rows/cell and green at 4,000 -- the verdict tracked the grid resolution, not the model.
    prevalence = float(np.mean(yt)) if yt.size else 0.0
    floor_flat = np.array([null_ece_scale(int(c), prevalence, _CELL_SCORE_BINS) for c in cnt])
    # Excess over the floor is what the worst-cell headline should rank on: the raw argmax over cells is a max of
    # noise, and it systematically picks the thinnest populated cell.
    excess_flat = np.where(populated, ece_flat - floor_flat, np.nan)

    ece_grid = ece_flat.reshape(ny, nx)
    support_grid = cnt.reshape(ny, nx).astype(np.int64)

    if not np.any(np.isfinite(ece_flat)):
        skipped.append(f"no cell reached the {_MIN_CELL_ROWS}-row support floor")
        worst_ece, worst_cell, median_ece, light = float("nan"), None, float("nan"), "n/a"
    else:
        worst_idx = int(np.nanargmax(excess_flat))
        worst_ece = float(ece_flat[worst_idx])
        worst_cell = (int(worst_idx % nx), int(worst_idx // nx))  # (x_bin, y_bin)
        median_ece = float(np.nanmedian(ece_flat))
        light = _traffic_light(worst_ece, _worst_cell_noise_floor(float(floor_flat[worst_idx]), int(populated.sum()), _CELL_SCORE_BINS))

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
        y_true, y_score, np.asarray(feat_x), np.asarray(feat_y), n_bins=n_bins, random_state=random_state,
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
