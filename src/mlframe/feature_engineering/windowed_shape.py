"""Rolling-window shape features.

Per-row shape statistics computed over a fixed-K trailing window inside
each group. Distinct from ``numerical.compute_numaggs`` which produces
SCALAR features over a whole vector; here we produce one feature value
per ROW (rolling).

Primitives implemented:

* ``add_rolling_mean_abs_d2_features`` -- mean(|x_{t+1} - 2*x_t + x_{t-1}|)
  per window; bounded acceleration proxy useful for telemetry / sensor
  drift detection.
* ``add_rolling_extrema_count_features`` -- (n_peaks, n_troughs) per
  window; pattern-of-shape signal orthogonal to rolling-std.
* ``add_rolling_integral_above_baseline_features`` -- cumulative
  positive deviation from a baseline (well-median by default); useful
  for "excess signal" accumulation features.

Underpinned by ``grouped.per_group_sliding_window`` so per-group
boundary handling is consistent across the family.
"""

from __future__ import annotations

__all__ = [
    "rolling_mean_abs_d2",
    "rolling_n_peaks",
    "rolling_n_troughs",
    "rolling_extrema_density",
    "rolling_integral_above_baseline",
]

from typing import Optional

import numpy as np

from .grouped import per_group_sliding_window


def rolling_mean_abs_d2(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int = 20,
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Mean absolute second-difference inside a trailing K-window per group.

    For a window ``w`` of length K the statistic is
    ``mean(|w[2:] - 2*w[1:-1] + w[:-2]|)``: a discrete proxy for
    |d^2 x / dt^2| averaged over the window. Sensitive to sharp turns,
    insensitive to linear trends or DC offsets.
    """
    out = np.full(values.size, fill_value, dtype=np.float64)
    for _sort_idx_seg, wins, write_idx in per_group_sliding_window(
        values, group_ids, window_K=window_K,
    ):
        # second-difference of each window: shape (n_windows, K - 2)
        d2 = np.diff(wins, n=2, axis=1)
        out[write_idx] = np.abs(d2).mean(axis=1)
    return out


def _peak_trough_counts(wins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-window counts of strict local maxima and minima.

    Strict definition: position ``i`` is a peak iff
    ``w[i-1] < w[i] > w[i+1]``. End positions (0, K-1) are never
    counted because they cannot be checked symmetrically.
    """
    wmid = wins[:, 1:-1]
    wleft = wins[:, :-2]
    wright = wins[:, 2:]
    n_peaks = ((wmid > wleft) & (wmid > wright)).sum(axis=1)
    n_troughs = ((wmid < wleft) & (wmid < wright)).sum(axis=1)
    return n_peaks.astype(np.float64), n_troughs.astype(np.float64)


def rolling_n_peaks(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int = 20,
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Count of strict local maxima inside a trailing K-window per group."""
    out = np.full(values.size, fill_value, dtype=np.float64)
    for _sort_idx_seg, wins, write_idx in per_group_sliding_window(
        values, group_ids, window_K=window_K,
    ):
        n_peaks, _ = _peak_trough_counts(wins)
        out[write_idx] = n_peaks
    return out


def rolling_n_troughs(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int = 20,
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Count of strict local minima inside a trailing K-window per group."""
    out = np.full(values.size, fill_value, dtype=np.float64)
    for _sort_idx_seg, wins, write_idx in per_group_sliding_window(
        values, group_ids, window_K=window_K,
    ):
        _, n_troughs = _peak_trough_counts(wins)
        out[write_idx] = n_troughs
    return out


def rolling_extrema_density(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int = 20,
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    """``(n_peaks + n_troughs) / (K - 2)`` per group.

    Single composite measuring shape-change rate (orthogonal to
    rolling-std which measures magnitude). High density = wavy /
    oscillatory window; low density = monotonic or flat.
    """
    out = np.full(values.size, fill_value, dtype=np.float64)
    denom = max(1, window_K - 2)
    for _sort_idx_seg, wins, write_idx in per_group_sliding_window(
        values, group_ids, window_K=window_K,
    ):
        n_peaks, n_troughs = _peak_trough_counts(wins)
        out[write_idx] = (n_peaks + n_troughs) / denom
    return out


def rolling_integral_above_baseline(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int = 50,
    *,
    baseline: Optional[np.ndarray] = None,
    baseline_fn: str = "median",
    fill_value: float = np.nan,
) -> np.ndarray:
    """Cumulative excess of ``values`` above a baseline over a K-window.

    For each row, returns ``sum(clip(window - baseline, 0, +inf))``
    over the trailing K-window. The baseline can be:

    * a 1-D array aligned with ``values`` (per-row baseline)
    * derived per-group via ``baseline_fn`` in {"median", "mean", "p25"}
      computed over the WHOLE group (constant per group)

    Useful for "excess signal accumulation" features: how much of the
    last K observations sit above the typical level for this group.
    """
    out = np.full(values.size, fill_value, dtype=np.float64)
    if baseline is not None and baseline_fn != "median":
        # The two are mutually exclusive -- one chosen explicit baseline
        # wins, fn is only used when no array supplied.
        pass
    for sort_idx_seg, wins, write_idx in per_group_sliding_window(
        values, group_ids, window_K=window_K,
    ):
        if baseline is not None:
            # per-row baseline for the SAME group rows; the baseline at
            # the write_idx anchor is used as the constant.
            b = np.asarray(baseline, dtype=np.float64)[write_idx]
        else:
            seg_vals = wins[0].copy() if wins.shape[0] > 0 else np.array([])
            # Re-derive over the WHOLE group (= the values at sort_idx_seg)
            group_full = np.asarray(values, dtype=np.float64)[sort_idx_seg]
            if baseline_fn == "median":
                b_scalar = float(np.nanmedian(group_full)) if group_full.size else 0.0
            elif baseline_fn == "mean":
                b_scalar = float(np.nanmean(group_full)) if group_full.size else 0.0
            elif baseline_fn == "p25":
                b_scalar = float(np.nanpercentile(group_full, 25)) if group_full.size else 0.0
            else:
                raise ValueError(
                    f"baseline_fn={baseline_fn!r} not in "
                    f"('median', 'mean', 'p25')"
                )
            b = b_scalar  # broadcast scalar
        excess = np.clip(wins - (b if np.ndim(b) == 0 else b[:, None]), 0.0, None)
        out[write_idx] = excess.sum(axis=1)
    return out
