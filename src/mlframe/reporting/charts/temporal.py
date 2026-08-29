"""Target-temporal-audit chart spec builder.

Renders the full temporal-audit diagnostic via the spec vocabulary:
- the kept-bins target-rate line,
- dropped/sparse bins as a faded marker series (so the operator sees
  what was filtered, not a silent gap),
- per-segment mean as a horizontal step overlay,
- Pelt change-points as vertical reference lines,
- ``x_is_time`` so renderers format the time axis (rotate/auto-fmt).

All series share the full (kept + dropped) timeline; the kept line and
the dropped markers carry NaN at the other's positions so each draws only
where it has data (matplotlib + plotly both skip NaN).

Target serial-structure panels (token-based composer):
- ``TARGET_ACF``  -- target autocorrelation by lag with Bartlett +-bounds.
                     A decaying ACF reveals trend / momentum the model can
                     exploit via a lagged feature.
- ``TARGET_PACF`` -- target partial autocorrelation by lag with the same
                     bounds. The PACF cuts off at the AR order, so a single
                     lag-1 spike that then drops to ~0 says "an AR(1) lag is
                     all the serial information there is".
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np


from ._captions import caption_for_tokens

from mlframe.reporting.charts._acf import (
    MAX_ACF_LAGS, acf_fft, lag_tick_labels, pacf_levinson, significance_band,
)
from mlframe.reporting.charts._layout import (
    figsize_for_grid, pack_panels, parse_panel_template,
)
from mlframe.reporting.spec import (
    AnnotationPanelSpec, BarPanelSpec, FigureSpec, LinePanelSpec, PanelSpec,
)


def _wilson_bounds(rates: np.ndarray, counts: np.ndarray, z: float = 1.959963984540054):
    """Per-bin 95% Wilson interval for a binomial rate, NaN where the bin holds no rows or has no rate."""
    n = np.where(counts > 0, counts, np.nan)
    denom = 1.0 + (z * z) / n
    centre = (rates + (z * z) / (2.0 * n)) / denom
    half = (z * np.sqrt(rates * (1.0 - rates) / n + (z * z) / (4.0 * n * n))) / denom
    return np.clip(centre - half, 0.0, 1.0), np.clip(centre + half, 0.0, 1.0)


def _median_gap(x_axis: np.ndarray) -> Any:
    """Median inter-point gap of a sorted x-axis. Returns a timedelta for datetime x, a float otherwise.

    Used to size the change-point span half-width; falls back to a unit gap on a single-point axis.
    """
    if len(x_axis) < 2:
        arr = np.asarray(x_axis)
        return np.timedelta64(1, "D") if np.issubdtype(arr.dtype, np.datetime64) else 1.0
    diffs = np.diff(np.asarray(x_axis))
    return np.median(diffs)


def build_temporal_audit_spec(
    audit_result: Any,  # TemporalAuditResult instance
    *,
    figsize: Tuple[float, float] = (12.0, 4.5),
) -> FigureSpec:
    """Build a temporal-audit FigureSpec surfacing all audit structure.

    Reads from ``audit_result``:
    - ``bins``: List[TimeBin] with ``bin_start`` + ``target_rate`` + ``kept``
    - ``segments``: per-segment dicts with ``start_idx`` / ``end_idx``
      (into the kept-bin list) + ``mean_rate``
    - ``change_point_indices``: indices into the kept-bin list
    - ``target_name``, ``granularity``, ``target_type``, ``timestamp_col``
    """
    bins = list(getattr(audit_result, "bins", []) or [])
    kept = [b for b in bins if getattr(b, "kept", True)]
    dropped = [b for b in bins if not getattr(b, "kept", True)]

    target_name = getattr(audit_result, "target_name", "target")
    granularity = getattr(audit_result, "granularity", "")
    target_type_str = str(getattr(audit_result, "target_type", ""))
    timestamp_col = getattr(audit_result, "timestamp_col", "Time")
    segments = list(getattr(audit_result, "segments", []) or [])
    change_points = list(getattr(audit_result, "change_point_indices", []) or [])

    if not kept and not dropped:
        # Degenerate audit -- emit a 1-point flat line so renderers don't crash.
        line = LinePanelSpec(
            x=np.array([0.0]), y=np.array([0.0]),
            title=f"Target rate over time: {target_name} (no bins)",
            xlabel="Time", ylabel="mean(y)", line_styles=("-",),
        )
        return FigureSpec(
            suptitle="", panels=((line,),), figsize=figsize,
            caption="No time bin survived the audit's minimum-support filter, so there is no timeline to read.",
        )

    # Full timeline (kept + dropped), sorted by bin_start so the x-axis is monotone.
    all_bins = sorted(bins, key=lambda b: b.bin_start)
    _starts = [b.bin_start for b in all_bins]
    # Coerce timestamp-like bin starts to datetime64[ns] so renderers format the time axis and span arithmetic
    # (cx +- half_w via timedelta) works; numeric / non-datetime starts pass through unchanged.
    if _starts and hasattr(_starts[0], "isoformat"):
        import pandas as _pd
        x_axis = _pd.DatetimeIndex(_starts).to_numpy()
    else:
        x_axis = np.asarray(_starts)
    order_of = {i: k for k, i in enumerate(np.argsort([b.bin_start for b in bins], kind="stable"))}
    pos = {i: order_of[i] for i in range(len(bins))}
    kept_pos = [pos[i] for i, b in enumerate(bins) if getattr(b, "kept", True)]
    dropped_pos = [pos[i] for i, b in enumerate(bins) if not getattr(b, "kept", True)]

    kept_y = np.full(len(all_bins), np.nan, dtype=np.float64)
    for b, k in zip(kept, kept_pos):
        kept_y[k] = float(b.target_rate)

    dropped_y = np.full(len(all_bins), np.nan, dtype=np.float64)
    for b, k in zip(dropped, dropped_pos):
        dropped_y[k] = float(b.target_rate)

    # Per-segment mean as a step series over the full timeline. Each kept bin inherits its segment's mean_rate; the
    # value holds flat across the segment and steps at boundaries -- the visual "is the rate stable within a regime".
    seg_step = np.full(len(all_bins), np.nan, dtype=np.float64)
    for s in segments:
        start_idx = int(s.get("start_idx", 0))
        end_idx = int(s.get("end_idx", 0))  # exclusive into kept[]
        mean_rate = float(s.get("mean_rate", np.nan))
        for ki in range(start_idx, min(end_idx, len(kept))):
            seg_step[kept_pos[ki]] = mean_rate

    # Change-points (indices into kept[]) -> thin shaded vertical spans at the corresponding timeline positions.
    # vspans (add_vrect / axvspan) rather than vlines: plotly's add_vline does annotation-position arithmetic on the
    # x value, which raises on a datetime axis (Timestamp has no integer add); add_vrect handles datetime on both
    # backends. The span half-width is a small fraction of the median inter-bin gap so it reads as a crisp boundary.
    half_w = _median_gap(x_axis) * 0.15
    vspans: List[Tuple[Any, Any, str, float]] = []
    for ci in change_points:
        if 0 <= ci < len(kept):
            cx = x_axis[kept_pos[ci]]
            vspans.append((cx - half_w, cx + half_w, "red", 0.5))

    ylabel = "P(y=1)" if "binary" in target_type_str.lower() else "mean(y)"
    n_seg = len(segments)
    title = (
        f"Target rate over time: {target_name} "
        f"({target_type_str}, {granularity}-binned; {n_seg} segment(s), "
        f"{len(change_points)} change-point(s), {len(dropped)} sparse bin(s))"
    )

    # Wilson band per kept bin: a binary rate from 12 rows swings across most of [0,1] on sampling alone, so the
    # band is what separates a real regime change from a thin bin. Only defined for a binary target (a mean of a
    # continuous target has no binomial interval), and drawn from each bin's OWN n rather than a pooled count.
    counts = np.array([float(getattr(b, "n_obs", 0) or 0) for b in all_bins], dtype=np.float64)
    band = None
    if "binary" in target_type_str.lower():
        with np.errstate(invalid="ignore", divide="ignore"):
            _lo, _hi = _wilson_bounds(kept_y, counts)
        band = (_lo, _hi)
    hover = tuple(
        f"{lbl}: rate={rate:.4g} over {int(c):,} rows" if np.isfinite(rate) else f"{lbl}: no rate ({int(c):,} rows)"
        for lbl, rate, c in zip([str(getattr(b, "bin_label", "")) for b in all_bins], kept_y, counts)
    )

    line = LinePanelSpec(
        x=x_axis,
        y=(kept_y, seg_step, dropped_y),
        band=band,
        band_color="steelblue",
        band_label="95% Wilson interval on each bin's own n",
        hovertext=hover,
        series_labels=(target_name, "segment mean", f"sparse (filtered, n={len(dropped)})"),
        title=title,
        xlabel=f"{timestamp_col} ({granularity})",
        ylabel=ylabel,
        line_styles=("lines+markers", "-", "markers"),
        colors=("steelblue", "orange", "gray"),
        vspans=tuple(vspans) or None,
        x_is_time=True,
    )

    return FigureSpec(
        suptitle="",
        panels=((line,),),
        figsize=figsize,
        caption=(
            "How to read: each point is one time bin's mean target, so a drift here means the relationship the model was fitted on is not the one it will be scored against. Shaded spans are the detected segments and the vertical marks their change points; a bin's own row count decides how much of its movement is real, which is why sparse bins are drawn separately rather than silently merged."
        ),
    )


# ----------------------------------------------------------------------------
# Target serial-structure panels (ACF / PACF)
# ----------------------------------------------------------------------------


def _target_acf_panel(y: np.ndarray, *, nlags: int = MAX_ACF_LAGS) -> PanelSpec:
    """Target autocorrelation by lag with Bartlett white-noise +-bounds (hline).

    A decaying ACF means the target carries momentum / trend a lagged feature could exploit; a flat ACF
    inside the band is serially-independent (a lagged feature would add nothing). FFT autocovariance with
    the series tail-capped and the lag count capped so it stays bounded at n>=1e6.
    """
    acf_lags, n_used = acf_fft(y, nlags=nlags)
    if acf_lags.size == 0:
        return AnnotationPanelSpec(
            text="Target ACF skipped: needs >= 2 finite points with non-zero variance",
            title="Target autocorrelation (Bartlett band)",
        )
    band = significance_band(n_used)
    sig = int(np.sum(np.abs(acf_lags) > band))
    return BarPanelSpec(
        categories=lag_tick_labels(int(acf_lags.size)),
        values=acf_lags.astype(np.float64),
        title=f"Target ACF (n={n_used:,}; {sig} of {acf_lags.size} lags beyond +-{band:.3f})",
        xlabel="Lag",
        ylabel="Autocorrelation",
        colors=("steelblue",),
        hline=(band, "red", f"+-1.96/sqrt(n) = {band:.3f}"),
    )


def _target_pacf_panel(y: np.ndarray, *, nlags: int = MAX_ACF_LAGS) -> PanelSpec:
    """Target partial autocorrelation by lag with Bartlett +-bounds (hline).

    The PACF isolates the direct lag-k effect with the intervening lags partialled out; it cuts off at
    the AR order, so a single lag-1 spike that drops to ~0 says the target is AR(1) (one lagged feature
    captures the serial information). Durbin-Levinson over the capped ACF -- O(nlags^2), never over n.
    """
    pacf_lags, n_used = pacf_levinson(y, nlags=nlags)
    if pacf_lags.size == 0:
        return AnnotationPanelSpec(
            text="Target PACF skipped: needs >= 2 finite points with non-zero variance",
            title="Target partial autocorrelation (Bartlett band)",
        )
    band = significance_band(n_used)
    sig = int(np.sum(np.abs(pacf_lags) > band))
    return BarPanelSpec(
        categories=lag_tick_labels(int(pacf_lags.size)),
        values=pacf_lags.astype(np.float64),
        title=f"Target PACF (n={n_used:,}; {sig} of {pacf_lags.size} lags beyond +-{band:.3f})",
        xlabel="Lag",
        ylabel="Partial autocorrelation",
        colors=("seagreen",),
        hline=(band, "red", f"+-1.96/sqrt(n) = {band:.3f}"),
    )


_TOKEN_BUILDERS: Dict[str, Callable] = {
    "TARGET_ACF": _target_acf_panel,
    "TARGET_PACF": _target_pacf_panel,
}

ALLOWED_TEMPORAL_PANEL_TOKENS = frozenset(_TOKEN_BUILDERS)

# One sentence per token, joined for the tokens ACTUALLY rendered (see ``_captions.caption_for_tokens``). The
# figure-level caption used to describe the DEFAULT template, so a caller asking for a narrower mix read about
# panels that were not on their figure.
_TOKEN_CAPTIONS: Dict[str, str] = {
    "TARGET_ACF": (
        "The target ACF asks whether the target predicts its own future. Row order is read as time, and non-finite rows stay as gaps rather than being closed up, so lag k means k steps of that grid."
    ),
    "TARGET_PACF": (
        "The PACF strips out the correlation already explained by shorter lags, which is what names the ORDER of the dependence rather than just its presence."
    ),
}


DEFAULT_TEMPORAL_TARGET_PANELS = "TARGET_ACF TARGET_PACF"


def compose_target_acf_figure(
    y: np.ndarray,
    *,
    panels_template: str = DEFAULT_TEMPORAL_TARGET_PANELS,
    nlags: int = MAX_ACF_LAGS,
    suptitle: str = "",
    max_cols: int = 2,
    cell_width: float = 6.0,
    cell_height: float = 4.0,
) -> FigureSpec:
    """Build a target ACF/PACF FigureSpec from a panel template.

    ``y`` is the 1-D target series in temporal order (the audit's per-bin rate, or the raw target). The
    ACF/PACF are computed on the (tail-capped, mean-centred) series; the white-noise band uses the
    post-cap length so it matches the bars.
    """
    y_arr = np.asarray(y).ravel()
    tokens = parse_panel_template(panels_template)
    unknown = [t for t in tokens if t not in _TOKEN_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown temporal panel tokens {unknown}. " f"Allowed: {sorted(ALLOWED_TEMPORAL_PANEL_TOKENS)}")
    panels: List[PanelSpec] = [_TOKEN_BUILDERS[tok](y_arr, nlags=nlags) for tok in tokens]
    grid = pack_panels(panels, max_cols=max_cols)
    n_rows = len(grid)
    n_cols = max_cols if grid else 0
    return FigureSpec(
        suptitle=suptitle,
        panels=grid,
        figsize=figsize_for_grid(n_rows, n_cols, cell_width=cell_width, cell_height=cell_height),
        caption=caption_for_tokens(
            "How to read: these panels look for structure in the target over time; a bar outside the shaded band is correlation the white-noise null does not explain.",
            tokens,
            _TOKEN_CAPTIONS,
        ),
    )


__all__ = [
    "build_temporal_audit_spec",
    "ALLOWED_TEMPORAL_PANEL_TOKENS",
    "DEFAULT_TEMPORAL_TARGET_PANELS",
    "compose_target_acf_figure",
]
