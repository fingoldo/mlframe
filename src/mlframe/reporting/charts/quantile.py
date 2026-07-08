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

Reliability extension (R-6):
- ``QUANTILE_RELIABILITY`` -- per tau, the isotonic-recalibrated observed
                        coverage E[1(y<=q_pred)|q_pred] vs the nominal tau
                        (CORP-style reliability for quantiles). Calibrated
                        models track the horizontal tau line; a curve that
                        sits above/below tau is over/under-predicting at
                        that level.
- ``PINBALL_DECOMP`` -- CORP additive decomposition of pinball loss per tau
                        (miscalibration - discrimination + uncertainty) when
                        ``model-diagnostics`` is importable; otherwise plain
                        mean pinball per tau as a bar (the line-plot variant
                        is ``PINBALL_BY_ALPHA``).
- ``QUANTILE_CROSSING`` -- per adjacent (tau_lo<tau_hi) pair, the fraction of
                        rows where q_tau_lo > q_tau_hi (a monotonicity
                        violation). Independent quantile heads cross silently;
                        any non-zero bar is a correctness problem.

Probabilistic-forecast communication plot:
- ``FAN_CHART``      -- y over time/index with nested shaded quantile bands
                        (darkest at the median, fading outward), the standard
                        way to read a probabilistic forecast: the median path
                        plus how the predictive interval widens with the
                        horizon. Envelope-downsampled per time bucket (per-bucket
                        per-quantile aggregate) so a 1e6-row forecast never plots
                        n raw points.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

from mlframe.reporting.charts._layout import (
    figsize_for_grid, pack_panels, parse_panel_template,
)
from mlframe.reporting.spec import (
    AnnotationPanelSpec, BarPanelSpec, FigureSpec, HistogramPanelSpec,
    LinePanelSpec, PanelSpec,
)


def _model_diagnostics_decompose():
    """Return ``model_diagnostics.scoring.(decompose, PinballLoss)`` or ``None``.

    DEPEND path for the CORP pinball decomposition; the import is gated so a
    host without the wheel (or a future 3.x break) falls back to plain pinball.
    """
    try:
        from model_diagnostics.scoring import PinballLoss, decompose
        return decompose, PinballLoss
    except Exception:
        return None


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
    from mlframe.reporting.charts._sampling import prebin_histogram

    P = np.asarray(preds_NK, dtype=np.float64)
    # Use the column-wise span (max - min across all alpha columns) rather than last-minus-first: under quantile
    # crossing the alphas are not monotone, so P[:, -1] - P[:, 0] can go negative and report nonsense widths.
    if P.ndim == 2 and P.shape[1] > 1:
        widths = np.nanmax(P, axis=1) - np.nanmin(P, axis=1)
    else:
        widths = np.abs(P[:, -1] - P[:, 0])
    # Degenerate: all rows have the same width (e.g. linear quantile
    # regressor that just adds a constant offset per alpha). numpy
    # raises ``Too many bins for data range`` when bins>=2 and
    # max==min. Clamp bins to a safe value -- numpy still emits a 1-bin
    # histogram which renders as a single bar (faithful representation).
    n_unique = int(np.unique(widths).size)
    bins = min(30, max(1, n_unique))
    a_hi = f"{float(alphas[-1]):g}"
    a_lo = f"{float(alphas[0]):g}"
    mean_w = float(np.nanmean(widths)) if widths.size and np.isfinite(widths).any() else 0.0
    max_w = float(np.nanmax(widths)) if widths.size and np.isfinite(widths).any() else 0.0
    heights, centers, width = prebin_histogram(widths, bins, True)
    return HistogramPanelSpec(
        values=heights if centers is not None else widths,
        bins=bins,
        bin_centers=centers,
        bin_width=width,
        title=(f"Width(q_{a_hi} - q_{a_lo}) " f"(mean={mean_w:.3f}, max={max_w:.3f})"),
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
    overconfident (too-narrow) model sits BELOW it. Mean interval width per level rides on a
    secondary y-axis (a different unit than the [0,1] coverage), so sharpness reads directly.

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
    return LinePanelSpec(
        x=nominal,
        y=tuple([diag, empirical, widths]),
        series_labels=("perfect", "empirical", "mean interval width"),
        title="Interval coverage (empirical vs nominal)",
        xlabel="Nominal coverage (alpha_hi - alpha_lo)",
        ylabel="Empirical coverage",
        line_styles=(":", "lines+markers", "lines+markers"),
        colors=("green", "steelblue", "darkorange"),
        band=(ci_lo, ci_hi),
        band_color="steelblue",
        band_label="95% Wilson CI",
        secondary_y=(False, False, True),
        secondary_ylabel="Mean interval width",
    )


def _pit_hist_panel(y_true, preds_NK, alphas) -> PanelSpec:
    """PIT histogram. Uniform => calibrated.

    Requires K >= 3 alphas for interpolation; with K < 3 we emit an honest annotation
    placeholder rather than a fake [0.0] histogram (which read as a degenerate calibrated PIT).
    """
    from mlframe.metrics.quantile import pit_values
    from mlframe.reporting.charts._sampling import prebin_histogram

    if len(alphas) < 3:
        return AnnotationPanelSpec(
            text="PIT skipped: requires K >= 3 alphas to interpolate",
            title="PIT histogram",
        )
    pit = pit_values(y_true, preds_NK, alphas)
    mean = float(pit.mean()) if pit.size else 0.0
    heights, centers, width = prebin_histogram(pit, 20, True)
    return HistogramPanelSpec(
        values=heights if centers is not None else pit,
        bins=20,
        bin_centers=centers,
        bin_width=width,
        title=(f"PIT histogram (uniform = calibrated; " f"mean={mean:.3f})"),
        xlabel="PIT value",
        ylabel="Density",
        density=True,
    )


# ----------------------------------------------------------------------------
# Reliability extension (R-6)
# ----------------------------------------------------------------------------

# Cap on rows fed to the per-tau isotonic fit: O(n log n) sort dominates, and a
# 100k uniform subsample reproduces the recalibrated curve within plotting noise
# while keeping the panel sub-second at n>=1e6 (see _benchmarks/profile harness).
_ISOTONIC_FIT_CAP = 100_000
_RELIABILITY_GRID = 25

# The model-diagnostics CORP decompose is far heavier per row than the bare isotonic fit
# (~6 s/tau at 100k vs ~0.13 s for the reliability curve), so it gets a tighter cap. A 10k
# uniform draw reproduces the miscalibration / discrimination / uncertainty terms within ~1%
# of the full-n decomposition (verified vs n=5e5) while staying sub-second per tau.
_CORP_FIT_CAP = 10_000


def _isotonic_recalibrated_coverage(q_pred, indicator, grid):
    """Isotonic E[indicator | q_pred] evaluated on ``grid`` (clipped out-of-range).

    ``indicator`` is the 0/1 event ``y <= q_pred``; the isotonic fit gives the
    monotone-in-q_pred recalibrated coverage. For a calibrated tau-quantile this
    is ~tau across the whole predicted range.
    """
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(q_pred, indicator)
    return iso.predict(grid)


def _quantile_reliability_panel(y_true, preds_NK, alphas) -> PanelSpec:
    """Isotonic-recalibrated observed coverage vs predicted quantile, per tau (R-6).

    For each tau_k the recalibrated curve E[1(y<=q_k)|q_k] should sit on the
    horizontal tau_k line; sustained deviation is miscalibration at that level.
    Predictions that are constant within a tau (degenerate single-valued q_k)
    give a single point -- still plotted, just uninformative about shape.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    P = np.asarray(preds_NK, dtype=np.float64)
    a_arr = np.asarray(alphas, dtype=np.float64)
    K = a_arr.shape[0]
    n = y.shape[0]

    # Isotonic + np.quantile both need >=1 finite (y, q) row; empty or all-NaN inputs index OOB / raise in sklearn.
    row_ok = np.isfinite(y) & np.all(np.isfinite(P), axis=1) if n else np.zeros(0, dtype=bool)
    if int(row_ok.sum()) < 2:
        return AnnotationPanelSpec(
            text="Quantile reliability skipped: needs >= 2 finite (y, q) rows",
            title="Quantile reliability (isotonic-recalibrated coverage vs tau)",
        )
    y = y[row_ok]
    P = P[row_ok]
    n = y.shape[0]

    if n > _ISOTONIC_FIT_CAP:
        rng = np.random.default_rng(0)
        sub = rng.choice(n, size=_ISOTONIC_FIT_CAP, replace=False)
        y = y[sub]
        P = P[sub]

    from mlframe.reporting.colors import line_color

    quantile_grid = np.linspace(0.02, 0.98, _RELIABILITY_GRID)
    curves: List[np.ndarray] = []
    nominal: List[np.ndarray] = []
    labels: List[str] = []
    styles: List[str] = []
    colors: List[str] = []
    for k in range(K):
        q_k = P[:, k]
        indicator = (y <= q_k).astype(np.float64)
        grid = np.quantile(q_k, quantile_grid)
        recall = _isotonic_recalibrated_coverage(q_k, indicator, grid)
        curves.append(recall)
        nominal.append(np.full(_RELIABILITY_GRID, float(a_arr[k])))
        labels.append(f"tau={a_arr[k]:g} (obs)")
        styles.append("lines+markers")
        colors.append(line_color(k))
    # The per-tau nominal lines are reference levels sharing each obs curve's color; labelling all K floods the
    # legend (~2K entries cover the plot at K>=7). Label one as the dotted-reference key and blank the rest.
    for k in range(K):
        labels.append("nominal tau (dotted)" if k == 0 else "")
        styles.append(":")
        colors.append(line_color(k))
    x_axis = np.linspace(0.0, 1.0, _RELIABILITY_GRID)
    return LinePanelSpec(
        x=x_axis,
        y=tuple(curves + nominal),
        series_labels=tuple(labels),
        title="Quantile reliability (isotonic-recalibrated coverage vs tau)",
        xlabel="Predicted-quantile rank (low -> high)",
        ylabel="Recalibrated P(y <= q_tau)",
        line_styles=tuple(styles),
        colors=tuple(colors),
        legend_outside=True,
    )


def _pinball_decomp_panel(y_true, preds_NK, alphas) -> PanelSpec:
    """CORP pinball-loss decomposition per tau, or plain mean pinball if unavailable (R-6).

    With ``model-diagnostics`` present this draws the additive
    ``miscalibration - discrimination + uncertainty`` bars per tau (a high
    miscalibration share is the actionable signal). Without it, falls back to a
    single mean-pinball bar per tau (complementing the line-plot PINBALL_BY_ALPHA).
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    P = np.asarray(preds_NK, dtype=np.float64)
    a_arr = np.asarray(alphas, dtype=np.float64)
    K = a_arr.shape[0]
    cats = tuple(f"{a_arr[k]:g}" for k in range(K))

    # Both the plain pinball and the CORP isotonic fit need finite (y, q) rows; non-finite raises in sklearn / NaN-poisons.
    row_ok = np.isfinite(y) & np.all(np.isfinite(P), axis=1) if y.size else np.zeros(0, dtype=bool)
    y = y[row_ok]
    P = P[row_ok]
    n = y.shape[0]

    def _plain_pinball_bar() -> PanelSpec:
        if n == 0:
            return AnnotationPanelSpec(
                text="Pinball decomposition skipped: no finite (y, q) rows",
                title="Pinball by tau",
            )
        from mlframe.metrics.quantile import pinball_loss
        losses = np.array([pinball_loss(y, P[:, k], float(a_arr[k])) for k in range(K)], dtype=np.float64)
        return BarPanelSpec(
            categories=cats,
            values=losses,
            title=f"Mean pinball by tau (mean={losses.mean():.4f})",
            xlabel="tau",
            ylabel="Pinball loss",
            colors=("crimson",),
        )

    md = _model_diagnostics_decompose()
    # CORP decompose is an isotonic recalibration fit needing >=2 rows; on degenerate input it raises (out-of-range
    # recalibration) -- fall back to the plain pinball bar rather than crash the whole figure.
    if md is None or n < 2:
        return _plain_pinball_bar()

    decompose, PinballLoss = md
    # The CORP decompose is an isotonic recalibration fit with a heavy constant (~6 s/tau at 100k);
    # uncapped it costs ~243 s at n=1e6. Subsample to _CORP_FIT_CAP -- the decomposition is a
    # diagnostic estimate, faithful within ~1% on a uniform draw, and this keeps it sub-second/tau.
    if n > _CORP_FIT_CAP:
        rng = np.random.default_rng(0)
        sub = rng.choice(n, size=_CORP_FIT_CAP, replace=False)
        y = y[sub]
        P = P[sub]
    miscal = np.empty(K, dtype=np.float64)
    discr = np.empty(K, dtype=np.float64)
    uncert = np.empty(K, dtype=np.float64)
    try:
        for k in range(K):
            df = decompose(y, P[:, k], scoring_function=PinballLoss(level=float(a_arr[k])))
            miscal[k] = float(df["miscalibration"][0])
            discr[k] = float(df["discrimination"][0])
            uncert[k] = float(df["uncertainty"][0])
    except (ValueError, IndexError):
        # Degenerate predictions push isotonic recalibration outside the scoring range; plain pinball still informs.
        return _plain_pinball_bar()
    return BarPanelSpec(
        categories=cats,
        values=(miscal, discr, uncert),
        series_labels=("miscalibration", "discrimination", "uncertainty"),
        title="Pinball CORP decomposition by tau (score = miscal - discr + uncert)",
        xlabel="tau",
        ylabel="Loss component",
        colors=("crimson", "seagreen", "slategray"),
    )


def _quantile_crossing_panel(y_true, preds_NK, alphas) -> PanelSpec:
    """Adjacent-pair quantile-crossing violation rate (R-6).

    For each adjacent pair (tau_j < tau_{j+1}) the bar is the fraction of rows
    where q_{tau_j} > q_{tau_{j+1}} -- a monotonicity violation that independent
    quantile heads produce silently. The title surfaces the single worst pair's
    raw row count so a tiny-but-nonzero rate is not lost to rounding.
    """
    P = np.asarray(preds_NK, dtype=np.float64)
    a_arr = np.asarray(alphas, dtype=np.float64)
    K = a_arr.shape[0]
    n = P.shape[0]
    if K < 2:
        return AnnotationPanelSpec(
            text="Quantile crossing skipped: needs >= 2 alphas",
            title="Quantile crossing (adjacent-pair violation rate)",
        )
    cats: List[str] = []
    rates = np.empty(K - 1, dtype=np.float64)
    counts = np.empty(K - 1, dtype=np.int64)
    for j in range(K - 1):
        viol = P[:, j] > P[:, j + 1]
        c = int(viol.sum())
        counts[j] = c
        rates[j] = (c / n) if n else 0.0
        cats.append(f"{a_arr[j]:g}>{a_arr[j + 1]:g}")
    worst = int(np.argmax(counts))
    title = f"Quantile crossing rate (worst {cats[worst]}: " f"{counts[worst]} rows, {rates[worst]:.3%})"
    return BarPanelSpec(
        categories=tuple(cats),
        values=rates,
        title=title,
        xlabel="Adjacent tau pair",
        ylabel="Fraction of rows with q_lo > q_hi",
        colors=("indianred",),
        xtick_rotation=30.0,
    )


# ----------------------------------------------------------------------------
# Fan chart (probabilistic-forecast communication plot)
# ----------------------------------------------------------------------------

# Number of time buckets the x-axis is collapsed to before plotting. A fan chart communicates the
# SHAPE of the widening interval, not per-sample detail; 400 buckets reads as a smooth fan while
# bounding the plotted vertices regardless of n (a 1e6-row forecast collapses to <=400 per band edge).
_FAN_TIME_BUCKETS: int = 400


def _fan_chart_panel(y_true, preds_NK, alphas) -> PanelSpec:
    """Nested shaded quantile bands over time/index -- the probabilistic-forecast communication plot.

    The x-axis (sample index, taken as the forecast horizon) is collapsed into ``_FAN_TIME_BUCKETS``
    equal-population buckets; within each bucket every quantile column is averaged, so the fan is built
    from at most ``buckets`` vertices per band edge -- never the raw n points. Symmetric alpha pairs are
    drawn as nested bands (widest/lightest = outermost interval, narrowing and darkening toward the
    median line) so the operator reads the predictive interval widening with the horizon at a glance.
    """
    P = np.asarray(preds_NK, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64).ravel()
    a_arr = np.asarray(alphas, dtype=np.float64)
    K = a_arr.shape[0]
    n = P.shape[0]
    if K < 2 or n < 2:
        return AnnotationPanelSpec(
            text="Fan chart skipped: needs >= 2 alphas and >= 2 rows",
            title="Fan chart (nested predictive-interval bands)",
        )

    median_col = int(min(range(K), key=lambda j: abs(a_arr[j] - 0.5)))
    n_buckets = min(_FAN_TIME_BUCKETS, n)
    # Equal-population buckets over the index: bincount-based per-bucket mean (one O(n) pass per column),
    # never a per-row python loop. ``which`` assigns each row to its horizon bucket.
    which = np.minimum((np.arange(n) * n_buckets) // n, n_buckets - 1).astype(np.int64)
    counts = np.bincount(which, minlength=n_buckets).astype(np.float64)
    counts_safe = np.where(counts > 0, counts, 1.0)
    x_axis = np.bincount(which, weights=np.arange(n, dtype=np.float64), minlength=n_buckets) / counts_safe

    q_bucketed = np.empty((n_buckets, K), dtype=np.float64)
    for k in range(K):
        q_bucketed[:, k] = np.bincount(which, weights=P[:, k], minlength=n_buckets) / counts_safe
    y_bucketed = np.bincount(which, weights=y, minlength=n_buckets) / counts_safe

    # LinePanelSpec carries a single ``band`` (lower, upper); nested bands need one band per symmetric
    # pair. We draw the OUTERMOST pair as the spec band (widest envelope) and overlay the inner quantile
    # edges as faint lines so the nesting reads -- darkest at the median, lightening outward.
    pairs = _symmetric_interval_pairs(alphas)
    median_path = q_bucketed[:, median_col]
    series: List[np.ndarray] = [median_path, y_bucketed]
    labels: List[str] = [f"median (tau={a_arr[median_col]:g})", "y_true (bucket mean)"]
    styles: List[str] = ["-", "markers"]
    series_colors: List[str] = ["darkblue", "black"]
    # Inner pair edges as faint lines (skip the outermost, which becomes the filled band). Darker shade
    # for pairs closer to the median communicates the nesting on backends without per-band fills.
    inner = pairs[:-1] if len(pairs) >= 2 else []
    for depth, (lo, hi, _nom) in enumerate(inner):
        # Darker gray for pairs nearer the median (smaller depth) communicates the band nesting on
        # backends that draw only the single outermost filled band. Hex grayscale so both matplotlib
        # and plotly accept it (plotly rejects matplotlib's bare "0.25" fractional-gray string).
        level = int(round(min(0.7, 0.25 + 0.12 * depth) * 255))
        gray = f"#{level:02x}{level:02x}{level:02x}"
        for c_idx in (lo, hi):
            series.append(q_bucketed[:, c_idx])
            labels.append(f"tau={a_arr[c_idx]:g}")
            styles.append(":")
            series_colors.append(gray)

    lo_col, hi_col, nom = pairs[-1]
    return LinePanelSpec(
        x=x_axis,
        y=tuple(series),
        series_labels=tuple(labels),
        title=f"Fan chart (nested predictive intervals; {n_buckets} horizon buckets / {n:,} rows)",
        xlabel="Forecast horizon (sample index)",
        ylabel="y / quantile",
        line_styles=tuple(styles),
        colors=tuple(series_colors),
        band=(q_bucketed[:, lo_col], q_bucketed[:, hi_col]),
        band_color="steelblue",
        band_label=f"[q_{a_arr[lo_col]:g}, q_{a_arr[hi_col]:g}] ({nom:.0%})",
        x_is_time=False,
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
    "QUANTILE_RELIABILITY": _quantile_reliability_panel,
    "PINBALL_DECOMP": _pinball_decomp_panel,
    "QUANTILE_CROSSING": _quantile_crossing_panel,
    "FAN_CHART": _fan_chart_panel,
}

ALLOWED_QUANTILE_PANEL_TOKENS = frozenset(_TOKEN_BUILDERS)


DEFAULT_QUANTILE_PANELS: str = (
    "RELIABILITY COVERAGE PINBALL_BY_ALPHA INTERVAL_BAND WIDTH_DIST PIT_HIST " "QUANTILE_RELIABILITY PINBALL_DECOMP QUANTILE_CROSSING"
)


def compose_quantile_figure(
    y_true,
    preds_NK,
    alphas: Sequence[float],
    *,
    panels_template: str = DEFAULT_QUANTILE_PANELS,
    suptitle: str = "",
    max_cols: int = 2,
    cell_width: float = 7.5,
    cell_height: float = 4.8,
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
        raise ValueError(f"compose_quantile_figure requires 2-D preds_NK; " f"got shape {P.shape}")
    if P.shape[0] != y.shape[0]:
        raise ValueError(f"preds_NK.shape[0]={P.shape[0]} != len(y_true)={y.shape[0]}")
    if P.shape[1] != len(alphas):
        raise ValueError(f"preds_NK.shape[1]={P.shape[1]} != len(alphas)={len(alphas)}")

    tokens = parse_panel_template(panels_template)
    unknown = [t for t in tokens if t not in _TOKEN_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown quantile panel tokens {unknown}. " f"Allowed: {sorted(ALLOWED_QUANTILE_PANEL_TOKENS)}")
    panels: List[PanelSpec] = [_TOKEN_BUILDERS[tok](y, P, alphas) for tok in tokens]
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
    "DEFAULT_QUANTILE_PANELS",
    "compose_quantile_figure",
]
