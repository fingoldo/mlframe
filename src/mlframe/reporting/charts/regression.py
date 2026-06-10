"""Regression-report quality-visualisation panels + composer.

Mirrors the multiclass / quantile / multilabel composer pattern: each
token names a panel builder, ``compose_regression_figure`` parses a
space-separated template and packs the resulting panels into a grid.

Token catalogue:
- ``SCATTER``        -- predictions vs true values with the perfect-fit
                        diagonal. Above ``hexbin_threshold`` points the
                        cloud is drawn as a log-density 2-D histogram
                        (HeatmapPanelSpec) so a 2M-row scatter stays
                        readable; below it, a raw scatter with an
                        extremes-preserving subsample (so the MaxError
                        point quoted in the title is actually plotted).
- ``RESID_HIST``     -- residual histogram + fitted-Normal overlay; the
                        noise-distribution hypothesis + suggested loss
                        ride in the title.
- ``RESID_VS_PRED``  -- residuals vs predicted with a running-median +
                        IQR band overlay. A funnel (band widening with
                        y_hat) is the visual signature of
                        heteroscedasticity; a sloped median band flags
                        prediction-dependent bias.
- ``ERR_BY_DECILE``  -- target binned into deciles; grouped bars of mean
                        |residual| and mean signed residual per decile.
                        Exposes the GBM extreme-compression pathology
                        (top-decile under-prediction shows as a large
                        negative signed-residual bar).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from mlframe.reporting.charts._layout import (
    figsize_for_grid, pack_panels, parse_panel_template,
)
from mlframe.reporting.charts._sampling import subsample_preserving_extremes
from mlframe.reporting.spec import (
    BarPanelSpec, FigureSpec, HeatmapPanelSpec, LinePanelSpec, PanelSpec,
    ScatterPanelSpec,
)

# Above this many finite points the pred-vs-actual cloud is drawn as a log-density 2-D histogram instead of a raw scatter
# (a 2M-point Scattergl/SVG cloud is both slow and a solid unreadable blob; density binning preserves structure).
DEFAULT_HEXBIN_THRESHOLD: int = 50_000
# Raw-scatter subsample cap below the hexbin threshold. 5000 (was 500) keeps the cloud dense enough to read structure
# while the extremes-preserving draw guarantees the MaxError / range-endpoint points stay on the chart.
DEFAULT_REGRESSION_SCATTER_SAMPLE: int = 5_000
DEFAULT_DENSITY_BINS: int = 80


def _finite_pair(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def _worst_k_into_finite(y_true, y_pred, worst_k_indices) -> Optional[np.ndarray]:
    """Remap worst-K positions from the ORIGINAL arrays onto the finite-filtered index space.

    The panel's x/y are ``_finite_pair`` outputs (non-finite rows dropped); the worst-K indices the integrator
    supplies index the original arrays. Map each original position to its rank among the finite rows (dropped rows
    contribute no highlight). Returns positions into the finite arrays, or None when no usable worst-K survives.
    """
    if worst_k_indices is None:
        return None
    wk = np.asarray(worst_k_indices, dtype=np.int64).ravel()
    if wk.size == 0:
        return None
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    finite = np.isfinite(yt) & np.isfinite(yp)
    finite_pos = np.cumsum(finite) - 1  # rank among finite rows for each original position
    wk = wk[(wk >= 0) & (wk < finite.size)]
    wk = wk[finite[wk]]
    if wk.size == 0:
        return None
    return finite_pos[wk].astype(np.int64)


def _remap_through_order(order: np.ndarray, wk_finite: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    """Map finite-index worst-K positions through an ``argsort`` permutation (panel sorted by ``order``)."""
    if wk_finite is None or wk_finite.size == 0:
        return None
    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n, dtype=np.int64)
    return inverse[wk_finite]


def _append_missing_worst_k(s_pred, s_true, yp_finite, yt_finite, wk_finite):
    """Ensure every worst-K row is present in the (subsampled) panel arrays, appending those the subsample dropped.

    The extremes-preserving subsample keeps the largest |resid| rows, but a worst-K set ranked on loss (classification)
    or a custom score may not be a strict subset, so append any missing rows and return highlight positions into the
    final panel arrays.
    """
    if wk_finite is None or wk_finite.size == 0:
        return s_pred, s_true, None
    wk_pred = yp_finite[wk_finite]
    wk_true = yt_finite[wk_finite]
    base = len(s_pred)
    s_pred = np.concatenate([s_pred, wk_pred])
    s_true = np.concatenate([s_true, wk_true])
    highlight = np.arange(base, base + wk_finite.size, dtype=np.int64)
    return s_pred, s_true, highlight


def _scatter_panel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "",
    sample_size: int = DEFAULT_REGRESSION_SCATTER_SAMPLE,
    hexbin_threshold: int = DEFAULT_HEXBIN_THRESHOLD,
    density_bins: int = DEFAULT_DENSITY_BINS,
    seed: int = 42,
    worst_k_indices: Optional[np.ndarray] = None,
    trend_line: Optional[str] = "theil-sen",
) -> PanelSpec:
    """Predictions-vs-true panel.

    Above ``hexbin_threshold`` points: a log-density 2-D histogram (HeatmapPanelSpec) — a hexbin/hist2d analogue that
    stays readable at millions of rows; the robust trend line is fit on the full cloud and drawn beside y=x. Below it:
    a raw scatter with an extremes-preserving subsample so the headline MaxError point (and the axis-anchoring range
    endpoints) are always drawn, with the worst-K residual rows highlighted red. The y=x diagonal is always present.

    ``worst_k_indices`` are positions into the ORIGINAL (pre-finite-filter) ``y_pred``/``y_true``; they are remapped
    onto the finite-filtered scatter's own index space (the renderer resolves them against the panel's full x/y, which
    are the finite arrays). On the hexbin path individual points are not drawn, so the highlight is skipped there.
    ``trend_line`` overlays a robust (Theil-Sen / Huber) fit beside the y=x diagonal so a systematic slope bias is
    visible even when the cloud hugs the diagonal.
    """
    yt, yp = _finite_pair(y_true, y_pred)
    n = yt.size
    showing = min(sample_size, n)
    # Map original-array worst-K positions onto the finite-filtered index space the panel x/y live in.
    wk_finite = _worst_k_into_finite(y_true, y_pred, worst_k_indices)

    if n > hexbin_threshold:
        lo = float(min(yp.min(), yt.min())) if n else 0.0
        hi = float(max(yp.max(), yt.max())) if n else 1.0
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, density_bins + 1)
        counts, _, _ = np.histogram2d(yp, yt, bins=[edges, edges])
        # log1p so a few dense bins don't wash out the long tail; transpose so matrix[row=y_true, col=y_pred] reads bottom-up.
        density = np.log1p(counts.T)
        centers = (edges[:-1] + edges[1:]) / 2.0
        col_labels = tuple(f"{c:.3g}" for c in centers)
        row_labels = tuple(f"{c:.3g}" for c in centers)
        ht = title or f"Predictions vs true (log-density, {n:_} pts)"
        return HeatmapPanelSpec(
            matrix=density,
            row_labels=row_labels,
            col_labels=col_labels,
            title=ht + f"\n(density binned {density_bins}x{density_bins}; y=x is the main diagonal)",
            xlabel="Predictions",
            ylabel="True values",
            colorbar_label="log(1 + count)",
            trend_line=trend_line,
            trend_xy=(yp, yt) if trend_line is not None else None,
        )

    if n > sample_size:
        resid = yt - yp
        idx = subsample_preserving_extremes(yp, yt, sample_size=sample_size, extreme_values=resid, rng=seed)
        s_pred, s_true = yp[idx], yt[idx]
        showing_note = f"(showing {showing:,} / {n:,} sampled)"
        scatter_title = f"{title}\n{showing_note}" if title else showing_note
        # The subsample may not contain every worst-K row; the renderer resolves highlight_indices against the FULL
        # panel x/y, so pass the panel's own (subsampled) data plus the worst-K rows guaranteed present via extremes.
        s_pred, s_true, wk_panel = _append_missing_worst_k(s_pred, s_true, yp, yt, wk_finite)
    else:
        order = np.argsort(yp)
        s_pred, s_true = yp[order], yt[order]
        scatter_title = title
        # Sorted by yp, so remap worst-K finite positions through the argsort inverse.
        wk_panel = _remap_through_order(order, wk_finite, n)
    return ScatterPanelSpec(
        x=s_pred,
        y=s_true,
        title=scatter_title,
        xlabel="Predictions",
        ylabel="True values",
        perfect_fit_line=True,
        point_color="steelblue",
        point_alpha=0.3,
        point_size=10.0,
        highlight_indices=wk_panel,
        highlight_color="red",
        trend_line=trend_line,
    )


def _resid_hist_panel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    audit: Any = None,
    sample_size: int = DEFAULT_REGRESSION_SCATTER_SAMPLE,
    seed: int = 42,
) -> PanelSpec:
    """Residual histogram + fitted-Normal overlay. Hypothesis + suggested loss ride in the title."""
    yt, yp = _finite_pair(y_true, y_pred)
    resid = yt - yp
    if resid.size > sample_size:
        rng = np.random.default_rng(seed)
        resid = resid[rng.choice(resid.size, size=sample_size, replace=False)]
    n_bins = max(20, min(80, int(math.sqrt(resid.size)) if resid.size > 0 else 20))
    if audit is not None:
        suggested = (audit.suggested_loss.split("(")[0].strip()
                     if getattr(audit, "suggested_loss", None) else "")
        hyp_line = f"hypothesis: {audit.hypothesis}"
        if suggested:
            hyp_line += f" (suggested: {suggested})"
        title = (f"Residuals (skew={audit.skew:+.2f}, excess_kurt={audit.excess_kurt:+.2f})"
                 + ("\n" + hyp_line if hyp_line else ""))
        overlay = (audit.mean, audit.std) if audit.std > 0 else None
    else:
        title = "Residuals"
        overlay = None
    from mlframe.reporting.spec import HistogramPanelSpec
    return HistogramPanelSpec(
        values=resid,
        bins=n_bins,
        title=title,
        xlabel="Residual (y_true - y_pred)",
        ylabel="Density",
        density=True,
        overlay_normal=overlay,
    )


def _resid_vs_pred_panel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    audit: Any = None,
    n_pred_bins: int = 20,
) -> LinePanelSpec:
    """Residuals vs predicted with a running-median + IQR band.

    Rather than a raw point cloud, the residual structure is summarised as a per-prediction-bin running median (the
    line) plus a shaded inter-quartile band (q25..q75). A funnel (band widening with y_hat) is heteroscedasticity; a
    sloped median line flags prediction-dependent bias. Robust to the extreme-error points kept by the scatter.
    """
    yt, yp = _finite_pair(y_true, y_pred)
    resid = yt - yp
    n = resid.size
    if n == 0:
        return LinePanelSpec(x=np.array([0.0]), y=np.array([0.0]),
                             title="Residuals vs predicted (no finite data)",
                             xlabel="Predicted (y_hat)", ylabel="Residual")

    lo, hi = float(yp.min()), float(yp.max())
    if hi <= lo:
        # Degenerate: all predictions identical -> a single bin centered on the constant prediction.
        centers = np.array([lo])
        med = np.array([float(np.median(resid))])
        q25 = np.array([float(np.percentile(resid, 25))])
        q75 = np.array([float(np.percentile(resid, 75))])
    else:
        n_bins = min(n_pred_bins, max(2, n // 10))
        edges = np.linspace(lo, hi, n_bins + 1)
        which = np.clip(np.digitize(yp, edges[1:-1]), 0, n_bins - 1)
        centers_l: List[float] = []
        med_l: List[float] = []
        q25_l: List[float] = []
        q75_l: List[float] = []
        for b in range(n_bins):
            sel = resid[which == b]
            if sel.size == 0:
                continue
            centers_l.append((edges[b] + edges[b + 1]) / 2.0)
            # One np.percentile([25,50,75]) does a single partition per bin vs three separate calls; the per-bin
            # boolean-mask group + a single partition beat sorting all n once (a global lexsort over n=2M measured
            # ~4x SLOWER end-to-end: 764ms -> 3147ms, since lexsort fully sorts 2M vs ~20 partial-sorts of ~100k).
            q25_b, med_b, q75_b = np.percentile(sel, [25.0, 50.0, 75.0])
            med_l.append(float(med_b))
            q25_l.append(float(q25_b))
            q75_l.append(float(q75_b))
        centers = np.asarray(centers_l)
        med = np.asarray(med_l)
        q25 = np.asarray(q25_l)
        q75 = np.asarray(q75_l)

    het_marker = ""
    if audit is not None and math.isfinite(getattr(audit, "hetero_spearman", float("nan"))):
        het_marker = ("(!) heteroscedastic" if audit.hetero_significant else "homoscedastic")
        het_marker = f" ({het_marker}; spearman(|resid|,y_hat)={audit.hetero_spearman:+.3f})"
    # Zero-reference line: a flat residual=0 series so the operator sees deviation of the running median from 0.
    zero = np.zeros_like(centers)
    return LinePanelSpec(
        x=centers,
        y=(med, zero),
        series_labels=("running median residual", "zero"),
        title=f"Residuals vs predicted{het_marker}",
        xlabel="Predicted (y_hat)",
        ylabel="Residual (y_true - y_pred)",
        line_styles=("lines+markers", "--"),
        colors=("steelblue", "green"),
        band=(q25, q75),
        band_color="steelblue",
        band_label="IQR (q25-q75)",
    )


def _err_by_decile_panel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_deciles: int = 10,
) -> BarPanelSpec:
    """Per-target-decile error breakdown: mean |residual| + mean signed residual.

    y_true is binned into ``n_deciles`` equal-frequency buckets; each bucket gets two bars — mean absolute residual
    (magnitude) and mean signed residual (bias direction). The signed bar exposes the GBM extreme-compression
    pathology: trees under-predict the top target decile, so its signed residual (y_true - y_pred) is large positive.
    """
    yt, yp = _finite_pair(y_true, y_pred)
    resid = yt - yp
    n = yt.size
    if n == 0:
        return BarPanelSpec(
            categories=("D1",), values=np.array([0.0]),
            title="Error by target decile (no finite data)",
            xlabel="Target decile (low -> high)", ylabel="Residual",
        )
    # Equal-frequency deciles via quantile cut-points + searchsorted: a full argsort over n=2M is the chart's single
    # biggest cost (~0.4s); np.quantile does only a k-way partial sort, then searchsorted is O(n). Ties land in one
    # bucket consistently (acceptable -- ranks split ties arbitrarily anyway), so decile populations stay ~equal.
    k = min(n_deciles, n)
    cuts = np.quantile(yt, np.linspace(0.0, 1.0, k + 1)[1:-1])
    which = np.searchsorted(cuts, yt, side="right")
    which = np.minimum(which, k - 1).astype(np.int64)
    # Vectorized per-bucket means via weighted bincount (one O(n) pass each) instead of k boolean-mask scans.
    counts = np.bincount(which, minlength=k).astype(np.float64)
    counts_safe = np.where(counts > 0, counts, 1.0)
    mean_signed = np.bincount(which, weights=resid, minlength=k) / counts_safe
    mean_abs = np.bincount(which, weights=np.abs(resid), minlength=k) / counts_safe
    cats = tuple(f"D{b + 1}" for b in range(k))
    return BarPanelSpec(
        categories=cats,
        values=(mean_abs, mean_signed),
        series_labels=("mean |resid|", "mean signed resid (y_true - y_pred)"),
        title="Error by target decile (signed > 0 in top decile => under-prediction / compression)",
        xlabel="Target decile (low -> high)",
        ylabel="Residual",
        colors=("steelblue", "darkorange"),
    )


_TOKEN_BUILDERS: Dict[str, Callable] = {
    "SCATTER": _scatter_panel,
    "RESID_HIST": _resid_hist_panel,
    "RESID_VS_PRED": _resid_vs_pred_panel,
    "ERR_BY_DECILE": _err_by_decile_panel,
}

ALLOWED_REGRESSION_PANEL_TOKENS = frozenset(_TOKEN_BUILDERS)

DEFAULT_REGRESSION_PANELS = "SCATTER RESID_HIST RESID_VS_PRED ERR_BY_DECILE"


def compose_regression_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    audit: Any = None,
    panels_template: str = DEFAULT_REGRESSION_PANELS,
    suptitle: str = "",
    metrics_str: str = "",
    sample_size: int = DEFAULT_REGRESSION_SCATTER_SAMPLE,
    hexbin_threshold: int = DEFAULT_HEXBIN_THRESHOLD,
    seed: int = 42,
    worst_k_indices: Optional[np.ndarray] = None,
    trend_line: Optional[str] = "theil-sen",
    max_cols: int = 2,
    cell_width: float = 6.0,
    cell_height: float = 4.0,
    figsize: Optional[Tuple[float, float]] = None,
) -> FigureSpec:
    """Build a regression-quality FigureSpec from a panel template.

    ``audit`` is a duck-typed ResidualAudit (used by RESID_HIST + RESID_VS_PRED). ``metrics_str`` (MAE/RMSE/MaxError/R2
    + optional Spearman) becomes the SCATTER panel title. ``worst_k_indices`` (original-array positions of the worst
    residual rows, from the error-analysis pass) highlights those points red on the pred-vs-actual scatter. ``trend_line``
    overlays a robust fit (Theil-Sen / Huber, None to disable) beside y=x. The default template restores the
    residuals-vs-predicted panel dropped in 2026-05 and adds the per-decile error breakdown.
    """
    tokens = parse_panel_template(panels_template)
    unknown = [t for t in tokens if t not in _TOKEN_BUILDERS]
    if unknown:
        raise ValueError(
            f"Unknown regression panel tokens {unknown}. "
            f"Allowed: {sorted(ALLOWED_REGRESSION_PANEL_TOKENS)}"
        )

    # Fold the Spearman/heteroscedasticity line into the scatter title so the diagnostic stays visible on the headline panel.
    scatter_title = metrics_str
    if audit is not None and math.isfinite(getattr(audit, "hetero_spearman", float("nan"))):
        het = "(!) het" if audit.hetero_significant else "hom"
        line = f"Spearman(|resid|,preds) = {audit.hetero_spearman:+.3f} ({het})"
        scatter_title = f"{metrics_str}\n{line}".strip("\n")

    panels: List[PanelSpec] = []
    for tok in tokens:
        if tok == "SCATTER":
            panels.append(_scatter_panel(
                y_true, y_pred, title=scatter_title,
                sample_size=sample_size, hexbin_threshold=hexbin_threshold, seed=seed,
                worst_k_indices=worst_k_indices, trend_line=trend_line,
            ))
        elif tok == "RESID_HIST":
            panels.append(_resid_hist_panel(
                y_true, y_pred, audit=audit, sample_size=sample_size, seed=seed,
            ))
        elif tok == "RESID_VS_PRED":
            panels.append(_resid_vs_pred_panel(y_true, y_pred, audit=audit))
        elif tok == "ERR_BY_DECILE":
            panels.append(_err_by_decile_panel(y_true, y_pred))

    grid = pack_panels(panels, max_cols=max_cols)
    n_rows = len(grid)
    n_cols = max_cols if grid else 0
    fig_size = figsize if figsize is not None else figsize_for_grid(
        n_rows, n_cols, cell_width=cell_width, cell_height=cell_height,
    )
    return FigureSpec(
        suptitle=suptitle,
        panels=grid,
        figsize=fig_size,
        suptitle_fontsize=11,
    )


def build_regression_panel_spec(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    audit: Any,
    header_str: str = "",
    metrics_str: str = "",
    figsize: Tuple[float, float] = (16.0, 5.0),
    plot_sample_size: int = DEFAULT_REGRESSION_SCATTER_SAMPLE,
    seed: int = 42,
    panels_template: str = DEFAULT_REGRESSION_PANELS,
    worst_k_indices: Optional[np.ndarray] = None,
    trend_line: Optional[str] = "theil-sen",
) -> FigureSpec:
    """Thin adapter preserving the legacy 2-panel call signature.

    Delegates to ``compose_regression_figure``. The legacy callers passed ``figsize`` for a single-row layout; we keep
    honouring an explicit ``figsize`` but let the composer pack >2 panels into a grid when the (now default) template
    asks for more. ``audit`` stays duck-typed (ResidualAudit). ``worst_k_indices`` / ``trend_line`` forward the worst-K
    red overlay + robust fit to the SCATTER panel.
    """
    # Honour the legacy single-row figsize only when the template is the legacy 2-panel set; otherwise let the grid size itself.
    legacy_two = parse_panel_template(panels_template) == ["SCATTER", "RESID_HIST"]
    return compose_regression_figure(
        y_true, y_pred,
        audit=audit,
        panels_template=panels_template,
        suptitle=header_str,
        metrics_str=metrics_str,
        sample_size=plot_sample_size,
        seed=seed,
        worst_k_indices=worst_k_indices,
        trend_line=trend_line,
        figsize=figsize if legacy_two else None,
    )


__all__ = [
    "ALLOWED_REGRESSION_PANEL_TOKENS",
    "DEFAULT_REGRESSION_PANELS",
    "DEFAULT_HEXBIN_THRESHOLD",
    "DEFAULT_REGRESSION_SCATTER_SAMPLE",
    "compose_regression_figure",
    "build_regression_panel_spec",
]
