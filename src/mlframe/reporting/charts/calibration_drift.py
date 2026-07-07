"""Calibration drift over time: reliability degradation across rolling time windows.

PSI / covariate drift answers "did the FEATURE distribution move". This answers the
complementary probability-space question: "did the model's calibration decay over time".
A model can keep stable inputs yet drift in P(y=1|score) as the world changes underneath it
(label shift, concept drift), and that shows up as a rising per-window miscalibration (ECE)
even when covariate PSI is flat.

The diagnostic slices ``(y_true, y_score)`` into ``n_windows`` consecutive time windows
(equal-population by default so every window has enough samples to bin), computes a scalar
ECE per window via the same njit binning the calibration report uses, and renders ECE-over-time
as a ``LinePanelSpec`` (``x_is_time`` when the timestamps are datetimes). Optional small-multiple
reliability curves overlay each window's curve so the operator sees WHERE the curve bent.

Efficiency: one ``argsort`` on the timestamps, then ``np.searchsorted`` on the sorted axis to find
the window boundaries (no per-window mask over the full array). Each window's ECE comes from the
existing O(n) njit ``compute_ece_and_brier_decomposition`` -- aggregate-first, no per-row Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

from mlframe.metrics.calibration import compute_ece_debiased
from mlframe.metrics.calibration import fast_calibration_binning
from mlframe.reporting.spec import FigureSpec, LinePanelSpec

# Below this per-window count the ECE estimate is too noisy to trust; the window's ECE is set to NaN
# (the renderer skips it) rather than reporting a spuriously-large miscalibration off a handful of rows.
MIN_WINDOW_SAMPLES: int = 30


@dataclass(frozen=True)
class CalibrationDriftResult:
    """Per-window calibration-drift summary.

    ``window_ece`` is the scalar ECE per window (NaN for under-populated windows); ``window_centers``
    is the representative timestamp of each window (midpoint of its time span, or the index midpoint
    for non-temporal axes). ``reliability_curves`` carries the per-window (mean_pred, obs_freq) pairs
    for the small-multiple overlay; empty when ``collect_curves=False``.
    """

    window_ece: np.ndarray
    window_centers: np.ndarray
    window_counts: np.ndarray
    window_edges: np.ndarray
    base_rate: float
    reliability_curves: Tuple[Tuple[np.ndarray, np.ndarray], ...] = field(default_factory=tuple)
    n_windows: int = 0
    n_bins: int = 0

    @property
    def ece_trend(self) -> float:
        """Late-minus-early ECE: mean ECE of the last third of windows minus the first third.

        Positive => calibration degraded over time (the headline the biz_value test pins). Computed
        over finite windows only so under-populated NaN windows do not poison the trend.
        """
        finite = np.isfinite(self.window_ece)
        if finite.sum() < 2:
            return float("nan")
        vals = self.window_ece[finite]
        k = max(1, len(vals) // 3)
        return float(np.mean(vals[-k:]) - np.mean(vals[:k]))


def _window_edges_by_population(n: int, n_windows: int) -> np.ndarray:
    """Equal-population window boundaries as positions into the sorted axis (len = n_windows + 1).

    Equal-population (not equal-time-span) so every window holds ~n/n_windows samples and bins reliably
    even when the timestamps clump (bursty logging, business-hours skew). ``np.linspace`` of positions
    rounded to int; deduped so a tiny n with many windows does not manufacture empty windows.
    """
    raw = np.linspace(0, n, n_windows + 1)
    return np.unique(np.round(raw).astype(np.int64))


def calibration_drift(
    y_true: np.ndarray,
    y_score: np.ndarray,
    timestamps: np.ndarray,
    *,
    n_windows: int = 10,
    n_bins: int = 10,
    collect_curves: bool = True,
) -> CalibrationDriftResult:
    """Compute per-window calibration drift (ECE over rolling time windows).

    Parameters
    ----------
    y_true : binary {0,1} array.
    y_score : predicted positive-class probability, same length.
    timestamps : per-sample time key (datetime64 / numeric / pandas DatetimeIndex). Used only to ORDER
        and SLICE the samples; the values become the x-axis of the ECE-over-time line when datetime.
    n_windows : number of consecutive equal-population time windows (default 10).
    n_bins : reliability bins per window for the ECE estimate (default 10).
    collect_curves : also return each window's reliability curve for the small-multiple overlay.

    Returns
    -------
    CalibrationDriftResult

    Efficiency: a single ``argsort`` orders the three arrays; window boundaries are positions into the
    sorted axis (``_window_edges_by_population``), so each window is a contiguous slice -- no per-window
    boolean mask over the whole array. Per-window ECE reuses the O(n) njit calibration kernel.
    """
    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    ts = _as_sortable_timestamps(timestamps)
    n = yt.shape[0]
    if not (ys.shape[0] == n == ts.shape[0]):
        raise ValueError(f"calibration_drift: y_true ({n}), y_score ({ys.shape[0]}), timestamps ({ts.shape[0]}) " "must have equal length")
    if n_windows < 1:
        raise ValueError(f"n_windows must be >= 1, got {n_windows}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")

    base_rate = float(np.mean(yt)) if n else 0.0

    if n == 0:
        empty_f = np.empty(0, dtype=np.float64)
        return CalibrationDriftResult(
            window_ece=empty_f, window_centers=empty_f, window_counts=np.empty(0, dtype=np.int64),
            window_edges=np.empty(0, dtype=np.int64), base_rate=base_rate,
            reliability_curves=(), n_windows=0, n_bins=n_bins,
        )

    # One stable sort orders all three arrays by time; gather views (contiguous slices below avoid copies per window).
    order = np.argsort(ts, kind="stable")
    ts_sorted = ts[order]
    yt_sorted = yt[order].astype(np.int64, copy=False)
    ys_sorted = ys[order]

    edges = _window_edges_by_population(n, n_windows)
    n_eff = len(edges) - 1

    eces = np.full(n_eff, np.nan, dtype=np.float64)
    centers = np.empty(n_eff, dtype=ts_sorted.dtype)
    counts = np.zeros(n_eff, dtype=np.int64)
    curves: List[Tuple[np.ndarray, np.ndarray]] = []

    for w in range(n_eff):
        lo, hi = int(edges[w]), int(edges[w + 1])
        cnt = hi - lo
        counts[w] = cnt
        # Window center = time midpoint of its samples (first/last in the sorted slice); robust for datetime + numeric.
        centers[w] = ts_sorted[lo + cnt // 2] if cnt > 0 else ts_sorted[min(lo, n - 1)]
        if cnt < MIN_WINDOW_SAMPLES:
            if collect_curves:
                curves.append((np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)))
            continue
        yt_w = yt_sorted[lo:hi]
        ys_w = ys_sorted[lo:hi]
        # Debiased ECE subtracts each window's per-bin Bernoulli noise floor: the plug-in ECE on small unequal
        # windows is inflated by ~1/sqrt(n_w), so a no-drift stream with varying window sizes produced a spurious
        # ece_trend that was purely a sample-size artifact. The bias-corrected estimator collapses that floor to ~0.
        ece = compute_ece_debiased(yt_w.astype(np.float64), ys_w, n_bins)
        eces[w] = float(ece)
        if collect_curves:
            fp, ftr, _hits = fast_calibration_binning(yt_w, ys_w, nbins=n_bins)
            curves.append((fp, ftr))

    return CalibrationDriftResult(
        window_ece=eces,
        window_centers=centers,
        window_counts=counts,
        window_edges=edges,
        base_rate=base_rate,
        reliability_curves=tuple(curves),
        n_windows=n_eff,
        n_bins=n_bins,
    )


def _as_sortable_timestamps(timestamps: Any) -> np.ndarray:
    """Coerce timestamps to a 1-D sortable ndarray, preserving datetime64 where possible.

    pandas DatetimeIndex / Series -> datetime64[ns] ndarray (so the line gets ``x_is_time``); python
    datetime objects -> datetime64[ns]; everything else -> ``np.asarray`` (numeric epoch / ordinal).
    """
    if hasattr(timestamps, "to_numpy"):
        arr = timestamps.to_numpy()
    else:
        arr = np.asarray(timestamps)
    arr = np.ravel(arr)
    if arr.dtype == object and arr.size and hasattr(arr.flat[0], "isoformat"):
        import pandas as _pd
        arr = _pd.DatetimeIndex(arr).to_numpy()
    return np.asarray(arr)


def build_calibration_drift_spec(
    result: CalibrationDriftResult,
    *,
    title: str = "Calibration drift over time (ECE per window)",
    show_reliability_curves: bool = True,
    max_curve_panels: int = 6,
    figsize: Tuple[float, float] = (12.0, 4.5),
) -> FigureSpec:
    """Build the calibration-drift FigureSpec from a ``CalibrationDriftResult``.

    Top panel: ECE-over-time line (``x_is_time`` when the window centers are datetimes), with a band
    at the irreducible noise floor and a vertical marker on the worst (highest-ECE) window so the eye
    lands on the worst regime. Optional second panel: small-multiple reliability curves (one series per
    window, decimated to ``max_curve_panels`` evenly-spaced windows) sharing the perfect diagonal.
    """
    x = result.window_centers
    ece = result.window_ece
    x_is_time = np.issubdtype(np.asarray(x).dtype, np.datetime64)

    finite = np.isfinite(ece)
    vlines: Optional[Tuple[Tuple[Any, str, str], ...]] = None
    if finite.any():
        worst_local = int(np.nanargmax(np.where(finite, ece, -np.inf)))
        vlines = ((x[worst_local], "red", f"worst window (ECE={ece[worst_local]:.3f})"),)

    ece_line = LinePanelSpec(
        x=np.asarray(x),
        y=ece,
        series_labels=("ECE",),
        title=f"{title} (trend {result.ece_trend:+.3f})",
        xlabel="time" if x_is_time else "window",
        ylabel="ECE (miscalibration)",
        line_styles=("lines+markers",),
        colors=("crimson",),
        vlines=vlines,
        x_is_time=bool(x_is_time),
    )

    if not (show_reliability_curves and result.reliability_curves):
        return FigureSpec(suptitle="", panels=((ece_line,),), figsize=figsize)

    rel_panel = _reliability_small_multiple(result, max_curve_panels=max_curve_panels)
    if rel_panel is None:
        return FigureSpec(suptitle="", panels=((ece_line,),), figsize=figsize)
    # constrained_layout reserves space between the rows so the top panel's rotated date x-tick labels do not
    # collide with the bottom panel's title; extra height gives both panels room once that gap is reserved.
    return FigureSpec(
        suptitle="",
        panels=((ece_line,), (rel_panel,)),
        figsize=(figsize[0], figsize[1] * 2.0),
        row_height_ratios=(1.0, 1.2),
        constrained_layout=True,
    )


def _reliability_small_multiple(
    result: CalibrationDriftResult,
    *,
    max_curve_panels: int,
) -> Optional[LinePanelSpec]:
    """Overlay per-window reliability curves (decimated to <= ``max_curve_panels`` windows) + perfect diagonal.

    Each kept window contributes one ``lines+markers`` series on a shared [0,1] x-grid; the perfect
    diagonal is the first (dotted) series. Returns None when no window had a usable curve.
    """
    curves = result.reliability_curves
    usable = [i for i, (fp, _ftr) in enumerate(curves) if fp.size > 0]
    if not usable:
        return None
    if len(usable) > max_curve_panels:
        pick = np.linspace(0, len(usable) - 1, max_curve_panels).round().astype(int)
        usable = [usable[i] for i in np.unique(pick)]

    diag = np.linspace(0.0, 1.0, 11)
    series_x: List[np.ndarray] = [diag]
    series_y: List[np.ndarray] = [diag]
    labels: List[str] = ["perfect"]
    styles: List[str] = [":"]
    for i in usable:
        fp, ftr = curves[i]
        series_x.append(np.asarray(fp, dtype=np.float64))
        series_y.append(np.asarray(ftr, dtype=np.float64))
        labels.append(f"w{i}")
        styles.append("lines+markers")

    return LinePanelSpec(
        x=tuple(series_x),
        y=tuple(series_y),
        series_labels=tuple(labels),
        title="Reliability curve per window",
        xlabel="predicted probability",
        ylabel="observed frequency",
        line_styles=tuple(styles),
    )


__all__ = [
    "calibration_drift",
    "build_calibration_drift_spec",
    "CalibrationDriftResult",
    "MIN_WINDOW_SAMPLES",
]
