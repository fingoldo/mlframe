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
- ``WORM``           -- de-trended normal QQ of the residuals: the QQ
                        ordinate minus the y=x identity, plotted against the
                        theoretical normal quantile, with a pointwise CI band.
                        Subtracting the identity flattens the dominant linear
                        trend so small/medium departures (heavy tails, skew)
                        that are invisible on a raw QQ become large vertical
                        excursions; points leaving the band are significant
                        non-normality. Order-statistic-decimated to <=2000
                        plotted points, tails always kept.
- ``RESID_ACF``      -- residual autocorrelation by lag with Bartlett white-
                        noise +-1.96/sqrt(n) bounds (drawn as hlines). A lag-1
                        bar above the bound means the residuals carry serial
                        structure the model missed (mis-specified dynamics /
                        omitted lagged feature). ACF via FFT, lag- and
                        series-tail-capped so it stays bounded at n>=1e6.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np

from mlframe.reporting.charts._layout import (
    figsize_for_grid, pack_panels, parse_panel_template,
)
from mlframe.reporting.charts._acf import MAX_ACF_LAGS, acf_fft, significance_band
from mlframe.reporting.charts._sampling import subsample_preserving_extremes
from mlframe.reporting.spec import (
    AnnotationPanelSpec, BarPanelSpec, FigureSpec, HeatmapPanelSpec, LinePanelSpec,
    PanelSpec, ScatterPanelSpec,
)

# Above this many finite points the pred-vs-actual cloud is drawn as a log-density 2-D histogram instead of a raw scatter
# (a 2M-point Scattergl/SVG cloud is both slow and a solid unreadable blob; density binning preserves structure).
DEFAULT_HEXBIN_THRESHOLD: int = 50_000
# Raw-scatter subsample cap below the hexbin threshold. 5000 (was 500) keeps the cloud dense enough to read structure
# while the extremes-preserving draw guarantees the MaxError / range-endpoint points stay on the chart.
DEFAULT_REGRESSION_SCATTER_SAMPLE: int = 5_000
DEFAULT_DENSITY_BINS: int = 80


def _finite_pair(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten ``y_true``/``y_pred`` to float64 and drop any row where either value is non-finite, keeping the pair aligned."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def _uniform_bin_index(v: np.ndarray, edges: np.ndarray, nbins: int) -> np.ndarray:
    """Bin index of ``v`` into uniform ``edges`` (length ``nbins+1``), == ``searchsorted(edges, v, 'right')-1`` clamped.

    Uniform edges let the index come from one arithmetic scale+floor (O(n), no binary search) instead of searchsorted's
    O(n log nbins) the numpy histogram path pays. FP rounding can leave a value 1 bin off near an edge, so a single
    vectorized compare against its bin's two edges corrects it -- making the result bit-identical to searchsorted even
    for ULP-nudged on-edge values. Caller masks out-of-range values; on/inside-range values match bit-for-bit.
    """
    lo = edges[0]
    hi = edges[-1]
    idx = ((v - lo) * (nbins / (hi - lo))).astype(np.intp)
    np.clip(idx, 0, nbins - 1, out=idx)
    over = (v < edges[idx]) & (idx > 0)
    idx[over] -= 1
    under = (v >= edges[idx + 1]) & (idx < nbins - 1)
    idx[under] += 1
    return np.asarray(idx)


def _hist2d_uniform(xp: np.ndarray, yp: np.ndarray, edges: np.ndarray, nbins: int) -> np.ndarray:
    """Bit-identical drop-in for ``np.histogram2d(xp, yp, bins=[edges, edges])[0]`` on UNIFORM edges, ~1.7x faster at 1e7.

    Replaces histogramdd's per-axis searchsorted with the arithmetic ``_uniform_bin_index`` + a single weighted bincount.
    """
    inb = (xp >= edges[0]) & (xp <= edges[-1]) & (yp >= edges[0]) & (yp <= edges[-1])
    ix = _uniform_bin_index(xp, edges, nbins)
    iy = _uniform_bin_index(yp, edges, nbins)
    flat = (ix * nbins + iy)[inb]
    return np.bincount(flat, minlength=nbins * nbins).reshape(nbins, nbins).astype(np.float64)


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
    return np.asarray(finite_pos[wk].astype(np.int64))


def _remap_through_order(order: np.ndarray, wk_finite: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    """Map finite-index worst-K positions through an ``argsort`` permutation (panel sorted by ``order``)."""
    if wk_finite is None or wk_finite.size == 0:
        return None
    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n, dtype=np.int64)
    return np.asarray(inverse[wk_finite])


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
    trend_line: Optional[Literal["theil-sen", "huber"]] = "theil-sen",
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
        # Uniform edges -> arithmetic binning (bit-identical to np.histogram2d, ~1.7x faster at 1e7); binning is this panel's dominant O(n) cost.
        counts = _hist2d_uniform(yp, yt, edges, density_bins)
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
        suggested = audit.suggested_loss.split("(")[0].strip() if getattr(audit, "suggested_loss", None) else ""
        hyp_line = f"hypothesis: {audit.hypothesis}"
        if suggested:
            hyp_line += f" (suggested: {suggested})"
        title = f"Residuals (skew={audit.skew:+.2f}, excess_kurt={audit.excess_kurt:+.2f})" + ("\n" + hyp_line if hyp_line else "")
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
        return LinePanelSpec(
            x=np.array([0.0]), y=np.array([0.0]), title="Residuals vs predicted (no finite data)", xlabel="Predicted (y_hat)", ylabel="Residual"
        )

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
        # Uniform edges -> arithmetic bin index (bit-identical to clip(digitize(...)), ~2.8x faster at 1e7); the digitize searchsorted was this panel's top line.
        which = _uniform_bin_index(yp, edges, n_bins)
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
        het_marker = "(!) heteroscedastic" if audit.hetero_significant else "homoscedastic"
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


# Plotted-point cap for the de-trended QQ. The tails are the diagnostic payload, so the decimation keeps
# every point in the extreme heads/tails and thins only the dense central body to hit the cap.
_WORM_PLOT_CAP: int = 2000
# How many extreme order statistics at EACH tail are kept verbatim (never thinned) on the worm plot.
_WORM_TAIL_KEEP: int = 100


def _decimate_keep_tails(n: int, cap: int, tail_keep: int) -> np.ndarray:
    """Sorted index subset of ``0..n-1`` of length <= ``cap`` that keeps the first/last ``tail_keep``.

    The dense central body is uniformly strided down to the remaining budget; the heads and tails (where
    QQ departures live) are retained verbatim. Returns all indices when ``n <= cap``.
    """
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    tail_keep = min(tail_keep, n // 2)
    head = np.arange(tail_keep, dtype=np.int64)
    tail = np.arange(n - tail_keep, n, dtype=np.int64)
    mid_budget = max(1, cap - 2 * tail_keep)
    mid = np.linspace(tail_keep, n - tail_keep - 1, mid_budget).astype(np.int64)
    return np.unique(np.concatenate([head, mid, tail]))


def _worm_panel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    audit: Any = None,
) -> PanelSpec:
    """De-trended normal QQ (worm plot) of the residuals with a pointwise CI band.

    Sorts the residuals, standardises them, and compares each order statistic to its theoretical normal
    quantile. The plotted ordinate is the DETRENDED QQ -- ``sample_quantile - theoretical_quantile`` --
    so the dominant y=x slope is removed and only the departure remains: a flat worm near zero is normal,
    upward tails are heavy/over-dispersed tails, an S-shape is skew. The CI band is the order-statistic
    pointwise normal-theory band ``+- z * sqrt(p(1-p)/n) / phi(theoretical_quantile)`` (the asymptotic SE
    of the p-th sample quantile, mapped through the standardisation); points outside it are significant.
    """
    yt, yp = _finite_pair(y_true, y_pred)
    resid = yt - yp
    n = resid.size
    if n < 3:
        return AnnotationPanelSpec(
            text="Worm plot skipped: needs >= 3 finite residuals",
            title="Worm plot (de-trended normal QQ)",
        )
    from scipy.stats import norm

    mu = float(resid.mean())
    sd = float(resid.std(ddof=1))
    if sd <= 0.0:
        return AnnotationPanelSpec(
            text="Worm plot skipped: residuals are constant (zero variance)",
            title="Worm plot (de-trended normal QQ)",
        )
    order_stats = np.sort(resid)
    keep = _decimate_keep_tails(n, _WORM_PLOT_CAP, _WORM_TAIL_KEEP)
    # Decimate FIRST, then evaluate norm.ppf only on the <=2000 kept plotting positions: the per-row ppf
    # over the full n is the worm panel's biggest cost and the caller plots only the kept points anyway.
    z_sample = (order_stats[keep] - mu) / sd
    # Plotting positions (Blom): (i - 3/8) / (n + 1/4); robust at the tails vs i/(n+1).
    p_k = (keep.astype(np.float64) + 1.0 - 0.375) / (n + 0.25)
    zt = norm.ppf(p_k)
    detrended = z_sample - zt
    # Asymptotic SE of the p-th sample quantile (standardised scale): sqrt(p(1-p)/n)/phi(z_theo).
    phi = norm.pdf(zt)
    phi = np.where(phi > 1e-12, phi, 1e-12)
    se = np.sqrt(p_k * (1.0 - p_k) / n) / phi
    z95 = 1.959963984540054
    ci = z95 * se
    zero = np.zeros_like(zt)
    # Auto-interpretation so the panel is self-explanatory: the worm tests whether the model's RESIDUALS
    # are Gaussian. Flat-on-zero = normal (the usual prediction-interval / RMSE assumptions hold);
    # both tails bending UP-and-down away from zero = heavy tails (a few errors much larger than Gaussian
    # -> RMSE understates worst-case, prediction intervals too narrow); a consistent up- or down-tilt =
    # skew (systematic over/under-prediction in one tail).
    _frac_out = float(np.mean(np.abs(detrended) > ci)) if ci.size else 0.0
    _rt = float(np.median(detrended[zt >= 1.0])) if np.any(zt >= 1.0) else 0.0
    _lt = float(np.median(detrended[zt <= -1.0])) if np.any(zt <= -1.0) else 0.0
    if _frac_out < 0.05:
        _shape = "residuals ~ normal (interval/RMSE assumptions hold)"
    elif _rt > 0 and _lt < 0:
        _shape = "HEAVY TAILS -- a few errors far larger than Gaussian (RMSE understates worst-case)"
    elif _rt > 0 and _lt > 0:
        _shape = "RIGHT-SKEW residuals (under-prediction tail)"
    elif _rt < 0 and _lt < 0:
        _shape = "LEFT-SKEW residuals (over-prediction tail)"
    else:
        _shape = "light tails"
    return LinePanelSpec(
        x=zt,
        y=(detrended, zero),
        series_labels=("de-trended QQ (sample - theoretical)", "normal (zero)"),
        title=(f"Worm plot -- are residuals Gaussian? {_shape}\n" f"({_frac_out:.0%} of points outside 95% CI; n={n:,}, plotted {zt.size:,})"),
        xlabel="Theoretical normal quantile",
        ylabel="Sample quantile - theoretical (standardised)",
        line_styles=("lines+markers", "--"),
        colors=("steelblue", "green"),
        band=(zero - ci, zero + ci),
        band_color="steelblue",
        band_label="95% pointwise CI",
    )


def _resid_acf_panel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    nlags: int = MAX_ACF_LAGS,
) -> PanelSpec:
    """Residual autocorrelation by lag with Bartlett white-noise +-bounds (drawn as hlines).

    Residuals of a correctly-specified model are white noise; a lag-1 (or seasonal-lag) ACF bar poking
    above the +-1.96/sqrt(n) band means the model left serial structure on the table (mis-specified
    dynamics, an omitted lagged feature, autocorrelated errors that bias the standard errors). ACF is the
    FFT autocovariance (O(n log n)), with the series tail-capped and the lag count capped so the panel
    stays bounded at n>=1e6.
    """
    yt, yp = _finite_pair(y_true, y_pred)
    resid = yt - yp
    if resid.size < 3:
        return AnnotationPanelSpec(
            text="Residual ACF skipped: needs >= 3 finite residuals",
            title="Residual autocorrelation (Bartlett band)",
        )
    acf_lags, n_used = acf_fft(resid, nlags=nlags)
    if acf_lags.size == 0:
        return AnnotationPanelSpec(
            text="Residual ACF skipped: residuals are constant (zero variance)",
            title="Residual autocorrelation (Bartlett band)",
        )
    band = significance_band(n_used)
    lags = np.arange(1, acf_lags.size + 1)
    cats = tuple(str(int(lo)) for lo in lags)
    sig = int(np.sum(np.abs(acf_lags) > band))
    return BarPanelSpec(
        categories=cats,
        values=acf_lags.astype(np.float64),
        title=(f"Residual ACF (n={n_used:,}; {sig} of {acf_lags.size} lags beyond " f"+-{band:.3f} Bartlett band => serial structure)"),
        xlabel="Lag",
        ylabel="Autocorrelation",
        colors=("steelblue",),
        hline=(band, "red", f"+-1.96/sqrt(n) = {band:.3f}"),
    )


_TOKEN_BUILDERS: Dict[str, Callable] = {
    "SCATTER": _scatter_panel,
    "RESID_HIST": _resid_hist_panel,
    "RESID_VS_PRED": _resid_vs_pred_panel,
    "ERR_BY_DECILE": _err_by_decile_panel,
    "WORM": _worm_panel,
    "RESID_ACF": _resid_acf_panel,
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
    trend_line: Optional[Literal["theil-sen", "huber"]] = "theil-sen",
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
        raise ValueError(f"Unknown regression panel tokens {unknown}. " f"Allowed: {sorted(ALLOWED_REGRESSION_PANEL_TOKENS)}")

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
        elif tok == "WORM":
            panels.append(_worm_panel(y_true, y_pred, audit=audit))
        elif tok == "RESID_ACF":
            panels.append(_resid_acf_panel(y_true, y_pred))

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
    trend_line: Optional[Literal["theil-sen", "huber"]] = "theil-sen",
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
