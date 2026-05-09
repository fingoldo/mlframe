"""Quantile-regression quality-visualisation panels.

Each panel builder takes ``(y_true, preds_NK, alphas)`` and returns
one ``PanelSpec``. Inputs are 1-D ``y`` of length N + 2-D ``preds``
of shape ``(N, K)`` where ``K = len(alphas)``.

Token catalogue (5 standard panels):
- ``RELIABILITY``    -- empirical coverage vs nominal alpha (diagonal
                        = perfect calibration). The most important
                        single calibration view.
- ``PINBALL_BY_ALPHA`` - mean pinball loss per alpha (line plot vs
                        alphas; reveals which tail the model is bad at)
- ``INTERVAL_BAND``  -- per-row q_lo / q_median / q_hi over sample
                        index sorted by median (filled band view of
                        the prediction interval). Caps at 1000 points
                        for plot readability.
- ``WIDTH_DIST``     -- histogram of interval widths (q_hi - q_lo);
                        sharpness diagnostic.
- ``PIT_HIST``       -- histogram of probability-integral-transform
                        values; uniform = well-calibrated.
                        Skipped with a placeholder when K < 3 (PIT
                        requires K >= 3 alphas to interpolate).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

from mlframe.reporting.charts._layout import (
    figsize_for_grid, pack_panels, parse_panel_template,
)
from mlframe.reporting.spec import (
    FigureSpec, HistogramPanelSpec, LinePanelSpec, PanelSpec,
)


# ----------------------------------------------------------------------------
# Per-token panel builders
# ----------------------------------------------------------------------------


def _reliability_panel(y_true, preds_NK, alphas) -> LinePanelSpec:
    """Empirical coverage vs nominal alpha.

    For each alpha_k, the empirical coverage is the fraction of rows
    where y <= q_k (the cumulative coverage at that quantile level).
    Perfect calibration draws on the diagonal (y = x).
    """
    from mlframe.quantile_metrics import _fast_coverage  # noqa: F401 (import for jit warmup)

    y = np.asarray(y_true, dtype=np.float64).ravel()
    P = np.asarray(preds_NK, dtype=np.float64)
    a_arr = np.asarray(alphas, dtype=np.float64)
    K = a_arr.shape[0]
    emp = np.zeros(K)
    for k in range(K):
        emp[k] = float(np.mean(y <= P[:, k]))
    diag = a_arr.copy()
    return LinePanelSpec(
        x=a_arr,
        y=tuple([diag, emp]),
        series_labels=("perfect", "empirical"),
        title="Reliability: empirical vs nominal coverage",
        xlabel="Nominal alpha",
        ylabel="Empirical P(y <= q_alpha)",
        line_styles=(":", "-"),
        colors=("green", "steelblue"),
    )


def _pinball_by_alpha_panel(y_true, preds_NK, alphas) -> LinePanelSpec:
    """Pinball loss per alpha as a line plot. Lower = better."""
    from mlframe.quantile_metrics import pinball_loss_per_alpha
    losses_dict = pinball_loss_per_alpha(y_true, preds_NK, alphas)
    a_arr = np.asarray(alphas, dtype=np.float64)
    losses = np.array([losses_dict[float(a)] for a in alphas], dtype=np.float64)
    return LinePanelSpec(
        x=a_arr,
        y=losses,
        title=f"Pinball loss by alpha (mean={losses.mean():.4f})",
        xlabel="Alpha",
        ylabel="Pinball loss",
        line_styles=("-",),
        colors=("crimson",),
    )


def _interval_band_panel(y_true, preds_NK, alphas) -> LinePanelSpec:
    """Per-row prediction band: median + lo / hi quantiles.

    Sorts rows by predicted median so the band reads top-to-bottom in
    score-rank order. Caps at the first 1000 sorted rows for plot
    readability on large datasets. y_true is overlaid as a scatter
    series so the operator sees how often it leaves the band.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    P = np.asarray(preds_NK, dtype=np.float64)
    a_arr = list(alphas)
    K = len(a_arr)

    # Pick lo/median/hi columns. Lo = first alpha; hi = last; median =
    # closest to 0.5 (skip if no alpha is in [0.4, 0.6] -> use middle col).
    lo_col = 0
    hi_col = K - 1
    median_col = min(range(K), key=lambda j: abs(a_arr[j] - 0.5))

    n = len(y)
    cap = 1000
    if n > cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=cap, replace=False)
    else:
        idx = np.arange(n)
    P_sub = P[idx]
    y_sub = y[idx]
    order = np.argsort(P_sub[:, median_col])
    P_sorted = P_sub[order]
    y_sorted = y_sub[order]
    x_axis = np.arange(P_sorted.shape[0], dtype=np.float64)

    return LinePanelSpec(
        x=x_axis,
        y=tuple([
            P_sorted[:, lo_col],
            P_sorted[:, median_col],
            P_sorted[:, hi_col],
            y_sorted,
        ]),
        series_labels=tuple([
            f"q_alpha={a_arr[lo_col]}",
            f"q_alpha={a_arr[median_col]}",
            f"q_alpha={a_arr[hi_col]}",
            "y_true",
        ]),
        title=f"Prediction interval (sorted by median, n={len(x_axis)})",
        xlabel="Sample index (sorted by predicted median)",
        ylabel="y / quantile",
        line_styles=("--", "-", "--", ":"),
        colors=("steelblue", "darkblue", "steelblue", "darkorange"),
    )


def _width_dist_panel(y_true, preds_NK, alphas) -> HistogramPanelSpec:
    """Histogram of interval widths (q_hi - q_lo).

    Uses the OUTERMOST alpha pair (first vs last column) for the width;
    that's the widest interval the model is committing to. Concentrated
    histogram = uniform sharpness; right-skewed = some inputs cause
    the model to widen drastically.
    """
    P = np.asarray(preds_NK, dtype=np.float64)
    widths = P[:, -1] - P[:, 0]
    # Degenerate: all rows have the same width (e.g. linear quantile
    # regressor that just adds a constant offset per alpha). numpy
    # raises ``Too many bins for data range`` when bins>=2 and
    # max==min. Clamp bins to a safe value -- numpy still emits a 1-bin
    # histogram which renders as a single bar (faithful representation).
    n_unique = int(np.unique(widths).size)
    bins = min(30, max(1, n_unique))
    return HistogramPanelSpec(
        values=widths,
        bins=bins,
        title=(
            f"Width(q_{alphas[-1]} - q_{alphas[0]}) "
            f"(mean={widths.mean():.3f}, max={widths.max():.3f})"
        ),
        xlabel=f"Interval width (q_{alphas[-1]} - q_{alphas[0]})",
        ylabel="Density",
        density=True,
    )


def _pit_hist_panel(y_true, preds_NK, alphas) -> HistogramPanelSpec:
    """PIT histogram. Uniform => calibrated.

    Requires K >= 3 alphas for interpolation; with K < 3 we emit a
    placeholder histogram of zeros so the panel slot stays consistent
    with the requested template.
    """
    from mlframe.quantile_metrics import pit_values

    if len(alphas) < 3:
        return HistogramPanelSpec(
            values=np.array([0.0]),
            bins=10,
            title="PIT (skipped: requires K >= 3 alphas)",
            xlabel="PIT value",
            ylabel="Density",
            density=True,
        )
    pit = pit_values(y_true, preds_NK, alphas)
    return HistogramPanelSpec(
        values=pit,
        bins=20,
        title=(
            f"PIT histogram (uniform = calibrated; "
            f"mean={pit.mean():.3f})"
        ),
        xlabel="PIT value",
        ylabel="Density",
        density=True,
    )


# ----------------------------------------------------------------------------
# Token registry + composer
# ----------------------------------------------------------------------------


_TOKEN_BUILDERS: Dict[str, Callable] = {
    "RELIABILITY": _reliability_panel,
    "PINBALL_BY_ALPHA": _pinball_by_alpha_panel,
    "INTERVAL_BAND": _interval_band_panel,
    "WIDTH_DIST": _width_dist_panel,
    "PIT_HIST": _pit_hist_panel,
}

ALLOWED_QUANTILE_PANEL_TOKENS = frozenset(_TOKEN_BUILDERS)


def compose_quantile_figure(
    y_true,
    preds_NK,
    alphas: Sequence[float],
    *,
    panels_template: str = "RELIABILITY PINBALL_BY_ALPHA INTERVAL_BAND WIDTH_DIST PIT_HIST",
    suptitle: str = "",
    max_cols: int = 2,
    cell_width: float = 6.0,
    cell_height: float = 4.0,
) -> FigureSpec:
    """Build a quantile-regression quality FigureSpec from a panel template.

    Inputs:
    - ``y_true``: 1-D ndarray of length N
    - ``preds_NK``: 2-D ndarray of shape (N, K) where K = len(alphas)
    - ``alphas``: ascending tuple of K floats in (0, 1)
    """
    y = np.asarray(y_true)
    P = np.asarray(preds_NK)
    if P.ndim != 2:
        raise ValueError(
            f"compose_quantile_figure requires 2-D preds_NK; "
            f"got shape {P.shape}"
        )
    if P.shape[0] != y.shape[0]:
        raise ValueError(
            f"preds_NK.shape[0]={P.shape[0]} != len(y_true)={y.shape[0]}"
        )
    if P.shape[1] != len(alphas):
        raise ValueError(
            f"preds_NK.shape[1]={P.shape[1]} != len(alphas)={len(alphas)}"
        )

    tokens = parse_panel_template(panels_template)
    unknown = [t for t in tokens if t not in _TOKEN_BUILDERS]
    if unknown:
        raise ValueError(
            f"Unknown quantile panel tokens {unknown}. "
            f"Allowed: {sorted(ALLOWED_QUANTILE_PANEL_TOKENS)}"
        )
    panels: List[PanelSpec] = [
        _TOKEN_BUILDERS[tok](y, P, alphas) for tok in tokens
    ]
    grid = pack_panels(panels, max_cols=max_cols)
    n_rows = len(grid)
    n_cols = max_cols if grid else 0
    return FigureSpec(
        suptitle=suptitle,
        panels=grid,
        figsize=figsize_for_grid(
            n_rows, n_cols, cell_width=cell_width, cell_height=cell_height,
        ),
    )


__all__ = [
    "ALLOWED_QUANTILE_PANEL_TOKENS",
    "compose_quantile_figure",
]
