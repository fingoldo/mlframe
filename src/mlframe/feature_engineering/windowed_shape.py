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
    "rolling_total_variation",
    "rolling_zero_crossings",
    "rolling_longest_monotone_run",
    "rolling_quantile_spread",
    "rolling_shannon_entropy_binned",
]

from typing import Optional

import numpy as np
from numba import njit, prange

from .grouped import per_group_sliding_window


@njit(cache=True, fastmath=True, parallel=True)
def _longest_monotone_run_kernel(d: np.ndarray, direction_code: int) -> np.ndarray:
    """Longest strictly-monotone sign-run length per window row of a diff matrix.

    ``direction_code``: 0=up, 1=down, 2=any (max of up/down). Returns the
    monotone-subsequence length (sign-run + 1, min 1) per row. Rows are
    independent, so the outer scan runs in parallel. Mirrors the original
    per-row Python loop exactly (verified bit-identical, ~150x faster at 200k
    windows / K=20).
    """
    n_wins, n_d = d.shape
    max_run = np.empty(n_wins, dtype=np.int32)
    for r in prange(n_wins):
        longest_up = 0
        longest_dn = 0
        cur_up = 0
        cur_dn = 0
        for c in range(n_d):
            v = d[r, c]
            if v > 0.0:
                cur_up += 1
                cur_dn = 0
                if cur_up > longest_up:
                    longest_up = cur_up
            elif v < 0.0:
                cur_dn += 1
                cur_up = 0
                if cur_dn > longest_dn:
                    longest_dn = cur_dn
            else:
                cur_up = 0
                cur_dn = 0
        if direction_code == 0:
            run = longest_up
        elif direction_code == 1:
            run = longest_dn
        else:
            run = longest_up if longest_up > longest_dn else longest_dn
        max_run[r] = run + 1 if run > 0 else 1
    return max_run


@njit(cache=True, fastmath=False, parallel=True)
def _quantile_spread_kernel(wins: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    """Per-window ``quantile(q_high) - quantile(q_low)`` over the rows of a sliding-window matrix.

    Replaces ``np.quantile(wins, [q_low, q_high], axis=1)`` (which partitions the whole
    ``(n_windows, K)`` matrix + allocates a copy + interpolates in separate passes) with a single
    prange pass: each window's K values are copied into a thread-local buffer, sorted, and the two
    linear-interpolated quantiles are read off. Bit-identical by construction to numpy ``method='linear'``:
    virtual index ``q*(K-1)``, neighbours floor/ceil, and the exact two-branch ``_lerp``
    (``a+(b-a)*t`` for ``t<0.5`` else ``b-(b-a)*(1-t)``). Assumes finite input (the Python wrapper
    falls back to numpy for any NaN-containing segment, matching numpy's sort-NaN-to-end semantics).
    """
    n_wins, K = wins.shape
    out = np.empty(n_wins, dtype=np.float64)
    inv = K - 1
    for r in prange(n_wins):
        buf = np.empty(K, dtype=np.float64)
        for c in range(K):
            buf[c] = wins[r, c]
        buf.sort()
        vlo = q_low * inv
        vhi = q_high * inv
        # low quantile
        if vlo >= inv:
            qlo = buf[K - 1]
        else:
            p = int(np.floor(vlo))
            t = vlo - p
            a = buf[p]
            b = buf[p + 1]
            d = b - a
            qlo = a + d * t if t < 0.5 else b - d * (1.0 - t)
        # high quantile
        if vhi >= inv:
            qhi = buf[K - 1]
        else:
            p = int(np.floor(vhi))
            t = vhi - p
            a = buf[p]
            b = buf[p + 1]
            d = b - a
            qhi = a + d * t if t < 0.5 else b - d * (1.0 - t)
        out[r] = qhi - qlo
    return out


@njit(cache=True, fastmath=False, parallel=True)
def _shannon_entropy_binned_kernel(wins: np.ndarray, n_bins: int, quantile_strategy: bool) -> np.ndarray:
    """Per-window binned Shannon entropy over the rows of a sliding-window matrix.

    Replaces the per-row Python loop (one ``np.quantile`` + ``np.unique`` + ``np.histogram`` +
    ``np.log`` dispatch per window) with a single prange pass. Bit-identical by construction:
    quantile edges use numpy's linear interpolation, the histogram uses the same half-open
    ``[e[i], e[i+1])`` binning with a closed last bin, and the entropy sum walks the same nonzero
    probabilities. Rows are independent so the outer scan parallelises.
    """
    n_wins, K = wins.shape
    ent = np.full(n_wins, np.nan, dtype=np.float64)
    n_edges = n_bins + 1
    for r in prange(n_wins):
        buf = np.empty(K, dtype=np.float64)
        m = 0
        for c in range(K):
            x = wins[r, c]
            if np.isfinite(x):
                buf[m] = x
                m += 1
        if m < 2:
            continue
        fin = buf[:m]
        fin.sort()
        if quantile_strategy:
            # numpy linear-interpolation quantiles at probs linspace(0,1,n_bins+1) over sorted fin.
            edges = np.empty(n_edges, dtype=np.float64)
            for i in range(n_edges):
                if n_bins == 0:
                    q = 0.0
                else:
                    q = i / n_bins
                pos = q * (m - 1)
                lo_i = int(np.floor(pos))
                if lo_i >= m - 1:
                    edges[i] = fin[m - 1]
                else:
                    frac = pos - lo_i
                    edges[i] = fin[lo_i] + (fin[lo_i + 1] - fin[lo_i]) * frac
            # np.unique: collapse duplicate edges (sorted already).
            uniq = np.empty(n_edges, dtype=np.float64)
            nu = 0
            for i in range(n_edges):
                if nu == 0 or edges[i] != uniq[nu - 1]:
                    uniq[nu] = edges[i]
                    nu += 1
            if nu < 2:
                ent[r] = 0.0
                continue
            counts = np.zeros(nu - 1, dtype=np.int64)
            for j in range(m):
                x = fin[j]
                # half-open [e[k], e[k+1]); last bin closed at the right edge.
                # binary search for the rightmost edge <= x.
                lo = 0
                hi = nu - 1
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if uniq[mid] <= x:
                        lo = mid
                    else:
                        hi = mid - 1
                b = lo
                if b >= nu - 1:
                    b = nu - 2  # closed last bin (x == right edge)
                counts[b] += 1
            total = 0
            for b in range(nu - 1):
                total += counts[b]
            e = 0.0
            for b in range(nu - 1):
                cb = counts[b]
                if cb > 0:
                    p = cb / total
                    e -= p * np.log(p)
            ent[r] = e
        else:
            lo_v = fin[0]
            hi_v = fin[m - 1]
            if hi_v - lo_v < 1e-12:
                ent[r] = 0.0
                continue
            right = hi_v + 1e-12
            width = (right - lo_v) / n_bins
            counts = np.zeros(n_bins, dtype=np.int64)
            for j in range(m):
                x = fin[j]
                b = int((x - lo_v) / width)
                if b < 0:
                    b = 0
                elif b >= n_bins:
                    b = n_bins - 1
                counts[b] += 1
            total = 0
            for b in range(n_bins):
                total += counts[b]
            e = 0.0
            for b in range(n_bins):
                cb = counts[b]
                if cb > 0:
                    p = cb / total
                    e -= p * np.log(p)
            ent[r] = e
    return ent


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


def rolling_total_variation(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int = 20,
    *,
    normalize: bool = False,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Sum of absolute first-differences in trailing K-window per group.

    Path-length / "wiggle intensity" feature: integrates |dx/dt| over
    the window. ``normalize=True`` divides by ``max - min`` of the
    window so the result is a unitless wiggle-ratio (1.0 = monotonic;
    >>1 = noisy / oscillatory). Orthogonal to:
    * rolling-std (measures magnitude, not arc-length)
    * rolling_mean_abs_d2 (measures curvature, not arc-length)
    * rolling_extrema_density (counts, not path-length).

    Use cases: telemetry activity intensity, finance realized variation,
    IMU/gyro roughness.
    """
    out = np.full(values.size, fill_value, dtype=np.float64)
    for _sort_idx_seg, wins, write_idx in per_group_sliding_window(
        values, group_ids, window_K=window_K,
    ):
        tv = np.abs(np.diff(wins, axis=1)).sum(axis=1)
        if normalize:
            wmax = wins.max(axis=1)
            wmin = wins.min(axis=1)
            denom = (wmax - wmin) + 1e-12
            tv = tv / denom
        out[write_idx] = tv
    return out


def rolling_zero_crossings(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int = 20,
    *,
    center: str = "zero",
    fill_value: float = np.nan,
) -> np.ndarray:
    """Count of sign changes of (window - center) per group.

    ``center`` in ``{"zero", "median", "mean"}``. Per-row rolling
    variant of the "level crossings" feature: how many times does the
    signal cross its baseline inside the K-window? Distinct from
    ``rolling_extrema_density`` which counts local maxima / minima
    regardless of level.

    Use cases: residual / detrended-signal regime detection (e.g.
    after detrending via ``ewma_residual``), EEG / ECG oscillation
    rate, mean-reversion frequency.
    """
    if center not in {"zero", "median", "mean"}:
        raise ValueError(f"center={center!r} not in {{'zero','median','mean'}}")
    out = np.full(values.size, fill_value, dtype=np.float64)
    for _sort_idx_seg, wins, write_idx in per_group_sliding_window(
        values, group_ids, window_K=window_K,
    ):
        if center == "median":
            c = np.median(wins, axis=1, keepdims=True)
        elif center == "mean":
            c = wins.mean(axis=1, keepdims=True)
        else:
            c = 0.0
        s = wins - c
        # Sign-flip count: sum of (sign[t] * sign[t-1] < 0) along axis=1.
        s_sign = np.sign(s)
        # Treat zeros as continuing prior sign (no crossing on flat run).
        # That's the conventional definition.
        cross = (s_sign[:, 1:] * s_sign[:, :-1]) < 0
        out[write_idx] = cross.sum(axis=1).astype(np.float64)
    return out


def rolling_longest_monotone_run(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int = 20,
    *,
    direction: str = "any",
    fill_value: float = np.nan,
) -> np.ndarray:
    """Length of longest strictly-monotone subsequence in K-window.

    ``direction`` in ``{"up", "down", "any"}`` (any = max of up/down).
    Captures persistence/streak structure: a 19-up + 1-down ramp and a
    sawtooth with the same slope are indistinguishable by slope+std
    but differ here by ~18 vs ~2. Cheap O(K) per window via
    np.diff sign-run length.

    Use cases: trend-following signal strength (finance), monotonic
    gradient zones (geosteering / sensor degradation curves),
    manufacturing ramp-up detection.
    """
    if direction not in {"up", "down", "any"}:
        raise ValueError(f"direction={direction!r}")
    direction_code = 0 if direction == "up" else (1 if direction == "down" else 2)
    out = np.full(values.size, fill_value, dtype=np.float64)
    for _sort_idx_seg, wins, write_idx in per_group_sliding_window(
        values, group_ids, window_K=window_K,
    ):
        # For each window row, the longest run of consecutive +/- signs over
        # np.diff(wins) gives the longest strictly-monotone subsequence
        # (length = run_of_signs + 1). The per-row scan is njit + parallel.
        d = np.ascontiguousarray(np.diff(wins, axis=1))
        max_run = _longest_monotone_run_kernel(d, direction_code)
        out[write_idx] = max_run.astype(np.float64)
    return out


def rolling_quantile_spread(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int = 20,
    *,
    q_low: float = 0.1,
    q_high: float = 0.9,
    fill_value: float = np.nan,
) -> np.ndarray:
    """``quantile(window, q_high) - quantile(window, q_low)`` per row.

    Generic inter-quantile range. Rolling-IQR is one config
    (``q_low=0.25, q_high=0.75``); ``(0.05, 0.95)`` gives a robust
    range; ``(0.5, 0.95)`` gives upper-tail spread. Robust-std analogue
    that doesn't break down on a single outlier in the window.

    Use cases: outlier-resilient volatility (finance / IoT with spikes),
    server-latency tail dispersion, anomaly detection.
    """
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError(f"need 0 <= q_low < q_high <= 1, got {q_low}, {q_high}")
    out = np.full(values.size, fill_value, dtype=np.float64)
    for sort_idx_seg, wins, write_idx in per_group_sliding_window(
        values, group_ids, window_K=window_K,
    ):
        # njit prange kernel matches numpy method='linear' bit-for-bit on finite windows; a window
        # containing NaN has different sort-to-end semantics, so fall back to numpy for those segments.
        seg = values[sort_idx_seg]
        if np.isfinite(seg).all():
            wins_c = np.ascontiguousarray(wins)
            out[write_idx] = _quantile_spread_kernel(wins_c, q_low, q_high)
        else:
            q = np.quantile(wins, [q_low, q_high], axis=1)
            out[write_idx] = q[1] - q[0]
    return out


def rolling_shannon_entropy_binned(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int = 20,
    *,
    n_bins: int = 8,
    bin_strategy: str = "quantile",
    fill_value: float = np.nan,
) -> np.ndarray:
    """Shannon entropy of the K-window's histogram per group.

    ``bin_strategy``:
    * ``"quantile"`` (default) — quantile-bin the WINDOW itself; output
      varies with shape, not absolute level. Distinguishes unimodal vs
      bimodal vs flat-uniform mass distributions.
    * ``"uniform"`` — equal-width bins on the window's [min, max] range.

    Distinct from ``mlframe.feature_engineering.compute_numaggs`` MI
    (which measures DEPENDENCE between two vectors). Here we measure
    "how concentrated vs spread the values in this window are".

    Use cases: regime/state detection (markets, sensor modes),
    anomaly scoring (entropy shift = mass redistribution),
    facies-style classification.
    """
    if bin_strategy not in {"quantile", "uniform"}:
        raise ValueError(f"bin_strategy={bin_strategy!r}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    out = np.full(values.size, fill_value, dtype=np.float64)
    quantile_strategy = bin_strategy == "quantile"
    for _sort_idx_seg, wins, write_idx in per_group_sliding_window(
        values, group_ids, window_K=window_K,
    ):
        wins_c = np.ascontiguousarray(wins)
        out[write_idx] = _shannon_entropy_binned_kernel(wins_c, n_bins, quantile_strategy)
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
