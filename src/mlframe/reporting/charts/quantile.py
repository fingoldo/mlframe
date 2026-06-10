"""Quantile-regression quality-visualisation panels.

Each panel builder takes ``(y_true, preds_NK, alphas)`` and returns
one ``PanelSpec``. Inputs are 1-D ``y`` of length N + 2-D ``preds``
of shape ``(N, K)`` where ``K = len(alphas)``.

Token catalogue (6 standard panels):
- ``RELIABILITY``    -- empirical coverage vs nominal alpha (diagonal
                        = perfect calibration). The most important
                        single calibration view.
- ``COVERAGE``       -- empirical vs nominal INTERVAL coverage per
                        symmetric quantile pair, with a 95% Wilson CI
                        band and the identity diagonal; mean interval
                        width per level folded into the legend.
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
    AnnotationPanelSpec, FigureSpec, HistogramPanelSpec, LinePanelSpec, PanelSpec,
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
    from mlframe.metrics.quantile import _fast_coverage  # noqa: F401 (import for jit warmup)

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
    from mlframe.metrics.quantile import pinball_loss_per_alpha
    losses_dict = pinball_loss_per_alpha(y_true, preds_NK, alphas)
    a_arr = np.asarray(alphas, dtype=np.float64)
    # Index losses by column position: pinball_loss_per_alpha keys by float(alpha), so a
    # float-key lookup is fragile to representation drift; the dict is ordered by alpha column.
    losses = np.array(list(losses_dict.values()), dtype=np.float64)
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
    """Per-row prediction band: median line + filled lo..hi band, y_true as markers.

    Sorts the plotted rows by predicted median so the band reads left-to-right in
    score-rank order. On large datasets a RANDOM subsample of ``cap`` rows is taken
    (uniform, fixed seed) -- this is a representative draw, not the first-``cap`` rows.
    y_true is drawn as markers (not a connected line; an order-statistic line through
    randomly-sampled points is meaningless) so the operator sees how often the actual
    value leaves the band.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    P = np.asarray(preds_NK, dtype=np.float64)
    a_arr = list(alphas)
    K = len(a_arr)

    # Pick lo/median/hi columns. Lo = first alpha; hi = last; median = closest to 0.5.
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
            P_sorted[:, median_col],
            y_sorted,
        ]),
        series_labels=tuple([
            f"q_alpha={a_arr[median_col]:g} (median)",
            "y_true",
        ]),
        title=f"Prediction interval (random n={len(x_axis)}, sorted by median)",
        xlabel="Sample (sorted by predicted median)",
        ylabel="y / quantile",
        line_styles=("-", "markers"),
        colors=("darkblue", "darkorange"),
        band=(P_sorted[:, lo_col], P_sorted[:, hi_col]),
        band_color="steelblue",
        band_label=f"[q_{a_arr[lo_col]:g}, q_{a_arr[hi_col]:g}]",
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
    a_hi = f"{float(alphas[-1]):g}"
    a_lo = f"{float(alphas[0]):g}"
    return HistogramPanelSpec(
        values=widths,
        bins=bins,
        title=(
            f"Width(q_{a_hi} - q_{a_lo}) "
            f"(mean={widths.mean():.3f}, max={widths.max():.3f})"
        ),
        xlabel=f"Interval width (q_{a_hi} - q_{a_lo})",
        ylabel="Density",
        density=True,
    )


def _symmetric_interval_pairs(alphas) -> List[tuple]:
    """Symmetric (lo_col, hi_col, nominal_coverage) triples from an ascending alpha grid.

    Pairs alpha_j with alpha_{K-1-j} so the interval straddles the median; nominal coverage
    is alpha_hi - alpha_lo. The exact-median column (if K is odd) is skipped (zero-width).
    """
    a = np.asarray(alphas, dtype=np.float64)
    K = a.shape[0]
    pairs: List[tuple] = []
    for j in range(K // 2):
        lo, hi = j, K - 1 - j
        if hi <= lo:
            continue
        pairs.append((lo, hi, float(a[hi] - a[lo])))
    return pairs


def _wilson_ci(p_hat: float, n: int, z: float = 1.959963984540054):
    """Wilson score interval for a binomial proportion (95% by default, z=1.96).

    Robust at the extremes (coverage near 0 or 1) where the Wald interval leaves [0, 1].
    """
    if n <= 0:
        return p_hat, p_hat
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def _coverage_panel(y_true, preds_NK, alphas) -> PanelSpec:
    """Empirical vs nominal interval coverage per symmetric quantile pair (R-5).

    For each nominal level (alpha_hi - alpha_lo) the empirical coverage is the fraction of
    rows inside [q_lo, q_hi]; a 95% Wilson band shows sampling uncertainty and the identity
    diagonal marks perfect calibration. A well-calibrated model lands on the diagonal; an
    overconfident (too-narrow) model sits BELOW it. Mean interval width per level is folded
    into each point's legend label (the read-only LinePanelSpec has no secondary axis).

    Perf: ~0.125s @2M / 9 alphas (cProfile); cost is the per-pair full-n boolean coverage
    reduction (already vectorised numpy) -- no actionable speedup at the panel scale.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    P = np.asarray(preds_NK, dtype=np.float64)
    n = y.shape[0]
    pairs = _symmetric_interval_pairs(alphas)
    if not pairs:
        return AnnotationPanelSpec(
            text="Coverage skipped: needs >= 2 alphas straddling the median",
            title="Interval coverage (empirical vs nominal)",
        )
    nominal = np.array([p[2] for p in pairs], dtype=np.float64)
    empirical = np.empty(len(pairs), dtype=np.float64)
    ci_lo = np.empty(len(pairs), dtype=np.float64)
    ci_hi = np.empty(len(pairs), dtype=np.float64)
    widths = np.empty(len(pairs), dtype=np.float64)
    for i, (lo, hi, _nom) in enumerate(pairs):
        q_lo = P[:, lo]
        q_hi = P[:, hi]
        inside = (y >= q_lo) & (y <= q_hi)
        cov = float(inside.mean()) if n else 0.0
        empirical[i] = cov
        lo_ci, hi_ci = _wilson_ci(cov, n)
        ci_lo[i] = lo_ci
        ci_hi[i] = hi_ci
        widths[i] = float(np.mean(q_hi - q_lo))
    order = np.argsort(nominal)
    nominal = nominal[order]
    empirical = empirical[order]
    ci_lo = ci_lo[order]
    ci_hi = ci_hi[order]
    widths = widths[order]
    diag = nominal.copy()
    width_note = "; ".join(f"nom {nm:g}: width {w:.3g}" for nm, w in zip(nominal, widths))
    return LinePanelSpec(
        x=nominal,
        y=tuple([diag, empirical]),
        series_labels=("perfect", f"empirical ({width_note})"),
        title="Interval coverage (empirical vs nominal)",
        xlabel="Nominal coverage (alpha_hi - alpha_lo)",
        ylabel="Empirical coverage",
        line_styles=(":", "lines+markers"),
        colors=("green", "steelblue"),
        band=(ci_lo, ci_hi),
        band_color="steelblue",
        band_label="95% Wilson CI",
    )


def _pit_hist_panel(y_true, preds_NK, alphas) -> PanelSpec:
    """PIT histogram. Uniform => calibrated.

    Requires K >= 3 alphas for interpolation; with K < 3 we emit an honest annotation
    placeholder rather than a fake [0.0] histogram (which read as a degenerate calibrated PIT).
    """
    from mlframe.metrics.quantile import pit_values

    if len(alphas) < 3:
        return AnnotationPanelSpec(
            text="PIT skipped: requires K >= 3 alphas to interpolate",
            title="PIT histogram",
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
    "COVERAGE": _coverage_panel,
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
