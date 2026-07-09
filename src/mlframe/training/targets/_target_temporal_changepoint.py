"""Change-point detection helpers for ``target_temporal_audit``.

Wave 106 (2026-05-21): split out from ``training/target_temporal_audit.py``
to keep that file below the 1k-line monolith threshold. Behaviour preserved
bit-for-bit; every moved symbol is re-exported from ``target_temporal_audit``
so existing imports continue to work.

Pre-split header (kept for context):

Temporal target audit вЂ” detect P(y) shifts over time, find change points.

A real-world drift incident (a job-board scraping pipeline running in
forward mode) revealed that:

- The target rate isn't constant over time. The historical period was
  positive-only (selection-biased) at ~98%; a several-month window
  had an unbiased ~40% rate; the most recent month is unbiased ~40% again.
- Without a per-bin time-series view of the target, this regime change
  is invisible вЂ” the operator sees the AGGREGATE rate (74%) and
  thinks they have a balanced classification problem when in fact
  they have multiple regimes mixed.

This module ships:

- `audit_target_over_time(...)` вЂ” group-by-time + change-point
  detection, returns a structured `TemporalAuditResult`.
- `plot_target_over_time(...)` вЂ” matplotlib-based time-series plot
  with change points marked, saves to disk (or returns Figure).
- `_pick_granularity(...)` вЂ” auto-pick a time bin width that yields
  30-50 non-empty bins (configurable via env / args).
- `find_change_points_zscore(...)` вЂ” generic 1-D change-point
  detector. Could move to `pyutilz` later (useful for trading too вЂ”
  the user said "Р±СѓРґРµС‚ РїРѕР»РµР·РЅРѕ Рё РґР»СЏ С‚СЂРµР№РґРёРЅРіР° РїРѕР·Р¶Рµ").

Wiring: opt-in via `TrainingBehaviorConfig.target_temporal_audit_*`.
When the config field `target_temporal_audit_column` is set,
`train_mlframe_models_suite` runs the audit per-target right before
training, logs the report, plots to disk under the per-target charts
folder, and stores the structured result on `metadata`.

Public surface:
- audit_target_over_time
- plot_target_over_time
- find_change_points_zscore
- TemporalAuditResult (dataclass)
- DEFAULT_MIN_BIN_FRACTION_FOR_FILTER  вЂ” bin must have в‰Ґ this many obs
  to be plotted / used; default 0.5 of the median bin size.
- DEFAULT_ZSCORE_THRESHOLD вЂ” 3.0 (3 MADs from rolling median)
"""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from .target_temporal_audit import ChangePointMethod

import numpy as np

try:
    import polars as pl  # noqa: F401
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False

# Wave 106 (2026-05-21): the DEFAULT_* constants + Granularity type alias
# live in target_temporal_audit (parent module). That module imports us
# from its BOTTOM (after these constants are defined at its TOP), so by
# the time Python resolves these names the parent is partially loaded
# and the symbols are already bound -- single source of truth, no
# duplication.
from .target_temporal_audit import (
    DEFAULT_ZSCORE_THRESHOLD,
    DEFAULT_PELT_MODEL,
    DEFAULT_PELT_MIN_SEGMENT_SIZE,
    _import_ruptures,
)


def find_change_points_pelt(
    rates: np.ndarray,
    *,
    model: str = DEFAULT_PELT_MODEL,
    penalty: float | None = None,
    min_segment_size: int = DEFAULT_PELT_MIN_SEGMENT_SIZE,
    weights: np.ndarray | None = None,
) -> list[int]:
    """Find change points via PELT (Killick et al. 2012) using
    ``ruptures.Pelt``.

    PELT is the canonical algorithm for offline change-point detection:
    given a cost function and a constant penalty per change point, it
    returns the partition that minimises ``total_cost +
    penalty Г— n_changepoints``. With pruning it runs in O(n) practical
    / O(nВІ) worst-case.

    Penalty auto-tuning
    -------------------
    When ``penalty`` is ``None`` (default), we use a BIC-style
    estimator: ``pen = log(n) Г— max(var(rates), eps)``. For binary-rate
    series this lands on a sensible default вЂ” empirically detected
    all 4 transitions in the user's production drift pattern. Override
    explicitly if the auto-pen leaves you with too many or too few
    points.

    Parameters
    ----------
    rates : ndarray, shape (n,)
        Per-bin target rate.
    model : str
        ruptures cost model. ``"l2"`` (default; mean-shift) is the
        fast and standard choice. Others: ``"l1"`` (median-shift,
        outlier-robust), ``"rbf"`` (kernel, slow but detects general
        distributional changes).
    penalty : float, optional
        Per-change-point penalty (BIC-like). Higher = fewer points.
        ``None`` = auto-tune (see above).
    min_segment_size : int
        Minimum number of bins per segment. Default 2.
    weights : ndarray, optional
        Currently unused by ruptures' built-in costs; we accept the
        kwarg for signature parity with ``find_change_points_zscore``.
        Future: drop tiny-n bins from the input before fitting.

    Returns
    -------
    list of int
        Boundary indices in pairs ``[start_0, end_0_excl, start_1,
        end_1_excl, ...]`` вЂ” same convention as
        ``find_change_points_zscore``. Conversion from ruptures' format
        (which returns ``[end_0, end_1, ..., n]`` excluding 0) to our
        anomaly-pair format is handled internally.
    """
    rates = np.asarray(rates, dtype=np.float64)
    n = len(rates)
    if n < min_segment_size * 2:
        return []

    # Optional weight-based bin filter вЂ” drop very-small-n bins so they
    # don't bias the segmentation. We keep the full series shape and
    # only re-weight by replacing dropped bins with the running median
    # so they neither add a "segment" nor force a split.
    rates_for_fit = rates.copy()
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        wpos = weights[weights > 0]
        if wpos.size > 0:
            wmedian = float(np.median(wpos))
            tiny = weights < max(1.0, 0.1 * wmedian)
            if tiny.any():
                # Replace tiny-bin rates with their nearest non-tiny neighbour.
                non_tiny_idx = np.where(~tiny)[0]
                if non_tiny_idx.size > 0:
                    for i in np.where(tiny)[0]:
                        nearest = non_tiny_idx[np.argmin(np.abs(non_tiny_idx - i))]
                        rates_for_fit[i] = rates[nearest]

    # Auto-tune penalty: BIC-style. The within-segment residual variance
    # is unknown, so we use overall data variance as an upper bound;
    # times log(n). A floor protects against degenerate constant series
    # (var в‰€ 0 в†’ no segmentation).
    if penalty is None:
        var = float(np.var(rates_for_fit))
        penalty = max(0.05, var * math.log(max(n, 2)))

    rpt = _import_ruptures()
    algo = rpt.Pelt(model=model, min_size=min_segment_size, jump=1)
    algo.fit(rates_for_fit.reshape(-1, 1))
    # ruptures returns ascending end-indices ending at n (e.g. [14, 18, 33]
    # for n=33 means segments [0:14], [14:18], [18:33]).
    bkps = algo.predict(pen=float(penalty))

    # Convert to our [start_0, end_0_excl, start_1, end_1_excl, ...]
    # boundary-pairs format. The "anomaly intervals" are the segments
    # whose mean differs from the global median by more than
    # 0.5 Г— MAD-equivalent. Operators want a flat list of segment
    # boundaries вЂ” convert directly.
    if not bkps or len(bkps) <= 1:
        return []
    boundaries: list[int] = []
    prev = 0
    for end in bkps:
        if end > prev:
            boundaries.append(prev)
            boundaries.append(int(end))
            prev = end
    # The very first / last "segments" cover [0:end_0] and [end_{-2}:n]
    # вЂ” these are NOT anomaly intervals, they're the regime-segments.
    # For change-point output we want the *transition points*: the
    # indices where regime switches. So remove the [0, end_0] and
    # [last_start, n] outer pairs to get just the inner transitions.
    # Actually our format expects all segment boundaries; let the
    # caller (`_segments_from_change_points`) interpret them. Return as
    # flat list of inner change points (not the trivial 0 / n).
    inner = [b for b in bkps[:-1]]  # drop the trivial final n
    if not inner:
        return []
    # Build pairs [c0, c1, c1, c2, ...] so segments[0]=[0:c0],
    # segments[1]=[c0:c1], etc.
    pairs: list[int] = []
    for c in inner:
        pairs.append(int(c))
        pairs.append(int(c))
    # Drop the duplicate at the end so we have inner change points
    # exactly (each appears twice = both end of prev seg and start of
    # next seg in our pair format). _segments_from_change_points uses
    # `set([0, n] + boundaries)` to collapse duplicates.
    return pairs


def find_change_points_zscore(
    rates: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    z_threshold: float = DEFAULT_ZSCORE_THRESHOLD,
    min_anomaly_run: int = 1,
    window: int | None = None,
    abs_min_spread: float = 0.05,
) -> list[int]:
    """Find indices where ``rates`` departs from a baseline by more
    than ``z_threshold`` modified-z-score units (or by an absolute
    minimum spread when the baseline MAD is degenerate).

    Two modes:

    * **Global-median baseline** (default, ``window=None``). The
      median of ``rates`` is the baseline; bins are flagged if
      ``|rate - median| > z_threshold Г— MAD`` (with a fall-through
      ``|rate - median| > abs_min_spread`` for the degenerate
      MAD-near-zero case). Right for "dominant baseline + a few
      anomaly regimes" patterns вЂ” the user's selection-bias case
      where ~80% of bins sit at one rate and a contiguous regime
      sits at another.

    * **Local rolling-window baseline** (``window=K``). Centred rolling
      median + MAD over a window of width K. Right when no single
      regime dominates (e.g. multiple roughly-equal-sized regimes).
      Window must be wider than the longest anomaly to detect it
      (median of a centred window inside an anomaly returns the
      anomaly's value). Falls back to the global-median branch for
      bins where the local window is degenerate.

    Modified z-score uses median + MAD (Iglewicz & Hoaglin 1993) so
    a single outlier doesn't inflate the baseline.

    Parameters
    ----------
    rates : ndarray, shape (n,)
        Per-bin target rate.
    weights : ndarray, shape (n,), optional
        Per-bin weight (typically n_obs). When provided, bins with
        n_obs below ``0.1 Г— median(weights)`` are excluded from the
        baseline median (they're too noisy to anchor it).
    z_threshold : float
        Modified-z-score threshold. 3.0 = strong outlier; 2.0 = loose.
    min_anomaly_run : int
        Minimum consecutive flagged bins to count as a change point.
        ``1`` flags single-bin spikes; ``2+`` filters them out.
    window : int, optional
        If given, use centred rolling-window baseline (must be odd; +1
        applied if even). If None (default), use global median.
    abs_min_spread : float
        Absolute baseline-vs-rate spread above which a bin is flagged
        even when the modified z-score is degenerate (MAD в‰€ 0). 0.05
        means "5 percentage points off the baseline" for binary
        targets.

    Returns
    -------
    list of int
        Boundary indices in pairs: ``[start_0, end_0_exclusive,
        start_1, end_1_exclusive, ...]``. For one dip in the middle
        of an otherwise-stable series the result is
        ``[dip_start, dip_end + 1]``.
    """
    rates = np.asarray(rates, dtype=np.float64)
    n = len(rates)
    if n == 0:
        return []

    # Bin-validity mask: drop tiny-n bins from baseline.
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        wpos = weights[weights > 0]
        if wpos.size > 0:
            wmedian = float(np.median(wpos))
            valid_for_baseline = weights >= max(1.0, 0.1 * wmedian)
        else:
            valid_for_baseline = np.ones(n, dtype=bool)
    else:
        valid_for_baseline = np.ones(n, dtype=bool)

    if window is None:
        # ---- Global-median baseline ----
        valid_rates = rates[valid_for_baseline]
        if valid_rates.size < 3:
            return []
        med = float(np.median(valid_rates))
        mad = float(np.median(np.abs(valid_rates - med)))
        if mad < 1e-9:
            flagged = np.abs(rates - med) > abs_min_spread
        else:
            mz = 0.6745 * np.abs(rates - med) / mad
            # Combine z-score + absolute-spread floor (catches narrow
            # but real shifts in series with tiny baseline MAD).
            flagged = (mz > z_threshold) | (np.abs(rates - med) > abs_min_spread * 2)
    else:
        # ---- Local rolling-window baseline ----
        if window % 2 == 0:
            window += 1
        half = window // 2
        flagged = np.zeros(n, dtype=bool)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            window_vals = rates[lo:hi][valid_for_baseline[lo:hi]]
            if window_vals.size < 2:
                continue
            med_i = float(np.median(window_vals))
            mad_i = float(np.median(np.abs(window_vals - med_i)))
            if mad_i < 1e-9:
                flagged[i] = abs(rates[i] - med_i) > abs_min_spread
            else:
                mz_i = 0.6745 * abs(rates[i] - med_i) / mad_i
                flagged[i] = (mz_i > z_threshold) or (abs(rates[i] - med_i) > abs_min_spread * 2)

    # Group consecutive flags into runs; report run boundaries.
    boundaries: list[int] = []
    in_run = False
    run_start = -1
    for i, f in enumerate(flagged):
        if f and not in_run:
            run_start = i
            in_run = True
        elif not f and in_run:
            run_len = i - run_start
            if run_len >= min_anomaly_run:
                boundaries.append(run_start)
                boundaries.append(i)
            in_run = False
    if in_run:
        run_len = n - run_start
        if run_len >= min_anomaly_run:
            boundaries.append(run_start)
            boundaries.append(n)
    return boundaries


def find_change_points(
    rates: np.ndarray,
    *,
    method: ChangePointMethod = "pelt",
    weights: np.ndarray | None = None,
    **method_kwargs: Any,
) -> list[int]:
    """Top-level dispatcher: ``method="pelt"`` (default) or
    ``method="zscore"``.

    Pelt is the principled choice (auto-detects K via penalty, optimal
    given cost+penalty, handles balanced regimes). Z-score is the
    simpler / more interpretable alternative for "dominant baseline +
    anomaly" patterns. See module docstring for trade-offs.

    All ``method_kwargs`` are forwarded to the underlying detector.
    Returns the same ``[start_0, end_0, start_1, end_1, ...]`` boundary
    list shape regardless of method.
    """
    if method == "pelt":
        return find_change_points_pelt(rates, weights=weights, **method_kwargs)
    if method == "zscore":
        return find_change_points_zscore(rates, weights=weights, **method_kwargs)
    raise ValueError(f"Unknown change-point method: {method!r}")


def _segments_from_change_points(
    rates: np.ndarray,
    weights: np.ndarray,
    boundaries: list[int],
    bin_labels: list[str],
) -> list[dict[str, Any]]:
    """Split [0..n) into intervals by the change-point boundaries.

    Boundaries from `find_change_points_zscore` come in pairs
    (anomaly_start, anomaly_end). We add 0 and n to get all segment
    edges, then dedup-sort.
    """
    n = len(rates)
    edges = sorted(set([0, n, *list(boundaries)]))
    segments: list[dict[str, Any]] = []
    for s, e in zip(edges[:-1], edges[1:]):
        if e <= s:
            continue
        seg_rates = rates[s:e]
        seg_weights = weights[s:e]
        total_w = float(seg_weights.sum())
        wmean = float(np.average(seg_rates, weights=seg_weights)) if total_w > 0 else float(seg_rates.mean()) if seg_rates.size else float("nan")
        segments.append(
            {
                "start_idx": int(s),
                "end_idx": int(e),
                "start_label": bin_labels[s] if s < len(bin_labels) else "",
                "end_label": bin_labels[e - 1] if e - 1 < len(bin_labels) else "",
                "n_bins": int(e - s),
                "n_obs": int(seg_weights.sum()),
                "mean_rate": wmean,
            }
        )
    return segments


# -----------------------------------------------------------------------------
# Top-level audit
# -----------------------------------------------------------------------------
