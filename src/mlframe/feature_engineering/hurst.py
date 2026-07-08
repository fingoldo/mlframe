"""Compute the Hurst Exponent of a 1D array via R/S analysis.

https://en.wikipedia.org/wiki/Hurst_exponent
"""

from __future__ import annotations

__all__ = [
    "compute_hurst_rs",
    "precompute_hurst_exponent",
    "compute_hurst_exponent",
    "rolling_hurst",
    "rolling_dfa_alpha",
    "rolling_higuchi_fd",
    "higuchi_fd",
    "dfa_alpha",
    "multi_scale_hurst",
    "dfa_alpha2_quadratic",
    "multifractal_dfa",
]

import logging

logger = logging.getLogger(__name__)

import numpy as np
from numba import njit, prange

_FASTMATH = False
_ZERO_EPS = 1e-12


@njit(fastmath=_FASTMATH, cache=True)
def compute_hurst_rs(arr: np.ndarray) -> float:  # pragma: no cover
    """Rescaled-range (R/S) statistic for one window.

    Standard deviation uses ddof=1 (sample std) per the Mandelbrot / Hurst (1951) convention
    for R/S analysis: the rescaled range is divided by the unbiased estimator of dispersion.
    """
    mean = np.mean(arr)
    deviations = arr - mean
    Z = np.cumsum(deviations)
    R = np.max(Z) - np.min(Z)
    n = len(arr)
    if n < 2:
        return np.nan
    var = np.sum(deviations * deviations) / (n - 1)
    S = np.sqrt(var)
    if R <= _ZERO_EPS or S <= _ZERO_EPS:
        return np.nan
    return float(R / S)


@njit(fastmath=_FASTMATH, cache=True)
def precompute_hurst_exponent(
    arr: np.ndarray,
    min_window: int = 5,
    max_window: int = -1,
    windows_log_step: float = 0.25,
    take_diffs: bool = False,
):  # pragma: no cover
    """R/S aggregated across a geometric ladder of window sizes.

    ``max_window=-1`` is the "auto" sentinel and resolves to ``L-1``.  ``take_diffs`` defaults to
    ``False`` to match the consumer-facing ``compute_hurst_exponent``; pass ``True`` for raw price
    or random-walk paths where the Hurst signal lives in the increment series.
    """
    if take_diffs:
        arr = arr[1:] - arr[:-1]

    L = len(arr)
    if L < 2 or min_window < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    if max_window <= 0:
        max_window = L - 1
    if max_window <= min_window:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    # Geometric ladder: log10-spaced floats cast to int. With the default windows_log_step=0.25,
    # `min_window` is the first ladder rung exactly. For larger log_steps the first rung can be
    # `int(10**ceil(log10(min_window) / log_step) * log_step)` which may overshoot `min_window`,
    # leaving a small gap at the lower end - this is by design (no R/S estimate below the
    # explicit min_window). Pass `min_window=2` and `windows_log_step <= 0.25` for full coverage.
    raw_sizes = (10.0 ** np.arange(np.log10(min_window), np.log10(max_window), windows_log_step)).astype(np.int64)
    window_sizes = np.unique(raw_sizes)

    n_sizes = len(window_sizes)
    out_sizes = np.empty(n_sizes, dtype=np.int64)
    out_rs = np.empty(n_sizes, dtype=np.float64)
    out_count = 0

    for idx in range(n_sizes):
        w = window_sizes[idx]
        if w < 2 or w > L:
            continue
        n_windows = L // w
        if n_windows == 0:
            continue
        sum_rs = 0.0
        cnt_rs = 0
        for k in range(n_windows):
            start = k * w
            partial_rs = compute_hurst_rs(arr[start : start + w])
            if not np.isnan(partial_rs):
                sum_rs += partial_rs
                cnt_rs += 1
        if cnt_rs > 0:
            out_sizes[out_count] = w
            out_rs[out_count] = sum_rs / cnt_rs
            out_count += 1

    return out_sizes[:out_count], out_rs[:out_count]


def compute_hurst_exponent(
    arr: np.ndarray,
    min_window: int = 5,
    max_window=None,
    windows_log_step: float = 0.25,
    take_diffs: bool = False,
) -> tuple:
    """Hurst exponent + intercept constant for a 1D array.

    Returns ``(H, c)`` such that ``E[R/S(n)] ~= c * n**H``.  Returns ``(nan, nan)`` when the array
    is too short or the R/S ladder is degenerate (no positive points to log-fit).

    ``max_window=None`` resolves to ``len(arr) - 1`` inside the kernel; the sentinel must be an int
    because the kernel is ``@njit`` and cannot accept ``None``.
    """
    if len(arr) < min_window:
        return np.nan, np.nan
    max_window_int = -1 if max_window is None else int(max_window)
    window_sizes, rs = precompute_hurst_exponent(
        arr=arr,
        min_window=min_window,
        max_window=max_window_int,
        windows_log_step=windows_log_step,
        take_diffs=take_diffs,
    )
    if len(rs) < 2:
        return np.nan, np.nan
    rs_arr = np.asarray(rs, dtype=float)
    window_sizes_arr = np.asarray(window_sizes, dtype=float)
    if np.any(rs_arr <= 0) or np.any(window_sizes_arr <= 0):
        return np.nan, np.nan
    # Closed-form 2-parameter OLS for the single-regressor log-log fit ``y = h*xv + c``: the covariance
    # normal equations give the same slope/intercept as ``np.linalg.lstsq`` without an SVD of an (M x 2)
    # design matrix or the vstack/ones-column assembly. ~1.5x faster on the R/S ladder; matches lstsq to <1e-15.
    xv = np.log10(window_sizes_arr)
    yv = np.log10(rs_arr)
    xm = xv.mean()
    dx = xv - xm
    var_x = (dx * dx).sum()
    if var_x <= 0:
        return np.nan, np.nan
    h = (dx * (yv - yv.mean())).sum() / var_x
    c = 10 ** (yv.mean() - h * xm)
    return h, c


# =============================================================================
# Rolling-window variants + sister fractal metrics (DFA, Higuchi).
#
# All three are O(K * N) per group when run as rolling features at
# window size K -- naive numpy would be O(K^2 * N). The numba kernels
# below collapse the inner per-window loop and parallelise across
# windows via ``prange``. Sister to ``compute_hurst_rs`` which already
# lives above; the single-window primitives expose the same surface
# so callers can use them in either streaming or rolling mode.
# =============================================================================


@njit(cache=True, fastmath=True)
def _hurst_rs_single(x: np.ndarray) -> float:
    """R/S statistic on a single window. Same definition as
    ``compute_hurst_rs`` but inlined for the rolling-kernel hot path
    to avoid per-call dispatch overhead.
    """
    n = x.size
    if n < 20:
        return np.nan
    mu = x.mean()
    y = x - mu
    z = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc += y[i]
        z[i] = acc
    rng_v = z.max() - z.min()
    sd = x.std()
    if sd <= _ZERO_EPS or rng_v <= _ZERO_EPS:
        return np.nan
    return float(np.log(rng_v / sd) / np.log(n))


@njit(cache=True, fastmath=True, parallel=True)
def _rolling_hurst_kernel(
    arr: np.ndarray, K: int, out: np.ndarray,
) -> None:
    """Per-row R/S Hurst over a trailing K-window.

    For row i >= K-1, out[i] = log(R/S(arr[i-K+1:i+1])) / log(K). For
    rows i < K-1 the kernel leaves ``out[i]`` at the caller-supplied
    sentinel (typically NaN); see ``rolling_hurst`` below.
    """
    n = arr.size
    for i in prange(K - 1, n):
        out[i] = _hurst_rs_single(arr[i - K + 1 : i + 1])


@njit(cache=True, fastmath=True)
def dfa_alpha(x: np.ndarray) -> float:
    """Detrended Fluctuation Analysis alpha exponent on one window.

    Standard DFA-1 (linear detrend) on log-spaced subwindow sizes.
    Returns the slope of log F(s) vs log s. alpha ~ 0.5 = white noise;
    alpha > 0.5 = persistent (long-memory); alpha < 0.5 = anti-
    persistent (mean-reverting).
    """
    n = x.size
    if n < 50:
        return np.nan
    mu = x.mean()
    y = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc += x[i] - mu
        y[i] = acc
    sizes = np.array([10, 20, 40, 80])
    sizes = sizes[sizes < n // 2]
    if sizes.size < 2:
        return np.nan
    log_s = np.empty(sizes.size)
    log_f = np.empty(sizes.size)
    for k in range(sizes.size):
        s = sizes[k]
        m = n // s
        var_sum = 0.0
        # ``t`` (and its mean ``tm``) depend only on the loop-invariant segment length ``s``; hoist out of the per-segment ``j`` loop.
        t = np.arange(s).astype(np.float64)
        tm = t.mean()
        for j in range(m):
            seg = y[j * s : (j + 1) * s]
            sm = seg.mean()
            num = 0.0
            den = 0.0
            for i in range(s):
                num += (t[i] - tm) * (seg[i] - sm)
                den += (t[i] - tm) ** 2
            slope = num / (den + 1e-12)
            intercept = sm - slope * tm
            resid_sq = 0.0
            for i in range(s):
                fit = intercept + slope * t[i]
                resid_sq += (seg[i] - fit) ** 2
            var_sum += resid_sq / s
        f_s = np.sqrt(var_sum / m)
        log_s[k] = np.log(s)
        log_f[k] = np.log(f_s + 1e-12)
    lm = log_s.mean()
    fm = log_f.mean()
    num = 0.0
    den = 0.0
    for k in range(log_s.size):
        num += (log_s[k] - lm) * (log_f[k] - fm)
        den += (log_s[k] - lm) ** 2
    return num / (den + 1e-12)


@njit(cache=True, fastmath=True, parallel=True)
def _rolling_dfa_kernel(arr: np.ndarray, K: int, out: np.ndarray) -> None:
    n = arr.size
    for i in prange(K - 1, n):
        out[i] = dfa_alpha(arr[i - K + 1 : i + 1])


@njit(cache=True, fastmath=True)
def higuchi_fd(x: np.ndarray, kmax: int = 8) -> float:
    """Higuchi fractal dimension on one window.

    Standard impl: for each k in 1..kmax compute average curve-length
    L_m(k) over offsets m, then fit log L(k) ~ log(1/k). The slope is
    the fractal dimension. kmax=8 is the conventional default for
    biomedical signals; raise for longer windows.
    """
    n = x.size
    if n < kmax * 4:
        return np.nan
    Lk = np.empty(kmax)
    for k in range(1, kmax + 1):
        Lk_avg = 0.0
        for m in range(k):
            i_max = (n - m - 1) // k
            if i_max < 1:
                continue
            L = 0.0
            for i in range(1, i_max + 1):
                L += abs(x[m + i * k] - x[m + (i - 1) * k])
            norm = (n - 1) / (i_max * k * k)
            Lk_avg += L * norm
        Lk[k - 1] = Lk_avg / k
    log_lk = np.log(Lk + 1e-12)
    log_kk = np.log(1.0 / np.arange(1, kmax + 1).astype(np.float64))
    lm = log_kk.mean()
    fm = log_lk.mean()
    num = 0.0
    den = 0.0
    for k in range(kmax):
        num += (log_kk[k] - lm) * (log_lk[k] - fm)
        den += (log_kk[k] - lm) ** 2
    return num / (den + 1e-12)


@njit(cache=True, fastmath=True, parallel=True)
def _rolling_hfd_kernel(
    arr: np.ndarray, K: int, kmax: int, out: np.ndarray,
) -> None:
    n = arr.size
    for i in prange(K - 1, n):
        out[i] = higuchi_fd(arr[i - K + 1 : i + 1], kmax)


def _per_group_rolling(
    values: np.ndarray,
    group_ids: np.ndarray | None,
    window_K: int,
    kernel,
    *kernel_args,
) -> np.ndarray:
    """Shared per-group dispatch for the three rolling kernels above.

    When ``group_ids`` is None, runs the kernel on the entire array
    (single group). When supplied, uses
    ``feature_engineering.grouped.iter_group_segments`` so per-group
    boundary handling is consistent with the rest of the rolling
    module family.
    """
    arr = np.ascontiguousarray(values, dtype=np.float64)
    out = np.full(arr.size, np.nan, dtype=np.float64)
    if group_ids is None:
        # Replace NaN with 0 for the kernel; non-finite handling is
        # caller's job for ML pipelines that need it.
        arr_finite = np.where(np.isfinite(arr), arr, 0.0)
        kernel(arr_finite, *kernel_args, out)
        return out
    from .grouped import iter_group_segments
    sort_idx, starts, ends = iter_group_segments(group_ids)
    arr_sorted = arr[sort_idx]
    for s, e in zip(starts, ends):
        seg = arr_sorted[s:e]
        if seg.size < window_K:
            continue
        seg_finite = np.where(np.isfinite(seg), seg, 0.0).copy()
        seg_out = np.full(seg.size, np.nan, dtype=np.float64)
        kernel(seg_finite, *kernel_args, seg_out)
        out[sort_idx[s:e]] = seg_out
    return out


def rolling_hurst(
    values: np.ndarray,
    *,
    group_ids: np.ndarray | None = None,
    window_K: int = 200,
) -> np.ndarray:
    """Rolling R/S Hurst exponent inside trailing K-windows per group.

    Per-row output: log(R/S(window)) / log(K). Equivalent to a single-
    scale R/S Hurst (no log-log fit across scales; one scale = K).
    For multi-scale estimation, call ``compute_hurst_exponent`` on
    each window's full range; the rolling variant trades scale-
    diversity for speed.

    ``window_K >= 20`` enforced (R/S is unstable on tiny windows).
    """
    if window_K < 20:
        raise ValueError(f"window_K must be >= 20 for stable R/S, got {window_K}")
    return _per_group_rolling(values, group_ids, window_K, _rolling_hurst_kernel, window_K)


def rolling_dfa_alpha(
    values: np.ndarray,
    *,
    group_ids: np.ndarray | None = None,
    window_K: int = 200,
) -> np.ndarray:
    """Rolling DFA-1 alpha exponent inside trailing K-windows per group.

    ``window_K >= 50`` enforced; DFA needs at least a few log-spaced
    subwindow sizes to fit the F(s) vs s slope. ``50`` admits
    [10, 20] subwindow scales (n // 2 = 25 limit).
    """
    if window_K < 50:
        raise ValueError(f"window_K must be >= 50 for DFA, got {window_K}")
    return _per_group_rolling(values, group_ids, window_K, _rolling_dfa_kernel, window_K)


def rolling_higuchi_fd(
    values: np.ndarray,
    *,
    group_ids: np.ndarray | None = None,
    window_K: int = 100,
    kmax: int = 8,
) -> np.ndarray:
    """Rolling Higuchi fractal dimension inside trailing K-windows per group.

    ``window_K >= kmax * 4`` enforced; below that the L(k) fit has
    fewer than 2 valid points and the kernel returns NaN.
    """
    if window_K < kmax * 4:
        raise ValueError(f"window_K must be >= kmax*4 ({kmax * 4}) for Higuchi, got {window_K}")
    return _per_group_rolling(values, group_ids, window_K, _rolling_hfd_kernel, window_K, kmax)


def multi_scale_hurst(
    arr: np.ndarray,
    *,
    breaks: tuple = (20, 100),
    min_window: int = 5,
    windows_log_step: float = 0.25,
) -> tuple:
    """Hurst exponent on disjoint scale ranges (short / mid / long).

    Reuses ``precompute_hurst_exponent`` for the R/S ladder, then fits
    log-log H separately on scale segments ``(min_window, breaks[0])``,
    ``(breaks[0], breaks[1])``, ``(breaks[1], n-1)``. Returns
    ``(H_short, H_mid, H_long, c_short, c_mid, c_long)``; NaN where a
    segment has fewer than 2 ladder rungs.

    Use case: regime detection across multiple time horizons (intraday
    vs daily vs weekly memory in finance; alpha-vs-gamma in EEG).
    Single primitive replaces 3 separate `compute_hurst_exponent` calls
    with disjoint min/max windows.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size < min_window:
        return (np.nan,) * 6
    window_sizes, rs = precompute_hurst_exponent(
        arr=arr, min_window=min_window, max_window=-1,
        windows_log_step=windows_log_step, take_diffs=False,
    )
    if len(rs) < 2:
        return (np.nan,) * 6
    ws = np.asarray(window_sizes, dtype=np.float64)
    rs_arr = np.asarray(rs, dtype=np.float64)
    finite = (rs_arr > 0) & (ws > 0)
    ws = ws[finite]
    rs_arr = rs_arr[finite]
    if ws.size < 2:
        return (np.nan,) * 6

    def _fit(mask: np.ndarray) -> tuple:
        if mask.sum() < 2:
            return (np.nan, np.nan)
        x = np.log10(ws[mask])
        y = np.log10(rs_arr[mask])
        X = np.vstack([x, np.ones(x.size)]).T
        h, c = np.linalg.lstsq(X, y, rcond=None)[0]
        return (float(h), 10.0 ** float(c))

    mask_short = ws < breaks[0]
    mask_mid = (ws >= breaks[0]) & (ws < breaks[1])
    mask_long = ws >= breaks[1]
    h_s, c_s = _fit(mask_short)
    h_m, c_m = _fit(mask_mid)
    h_l, c_l = _fit(mask_long)
    return (h_s, h_m, h_l, c_s, c_m, c_l)


@njit(cache=True, fastmath=True)
def dfa_alpha2_quadratic(x: np.ndarray) -> float:
    """DFA-2 (quadratic detrending) alpha exponent on one window.

    Same algorithm as ``dfa_alpha`` but each sub-segment is detrended
    by a quadratic fit (not linear). DFA-1 misattributes polynomial
    trends as long-range correlation, inflating alpha; DFA-2 removes
    that bias. The ratio DFA-2 / DFA-1 is itself a feature that
    detects whether long-memory is real vs a smooth trend.

    Use case: climate / battery decay / HRV signals with trend +
    curvature where DFA-1 is biased.
    """
    n = x.size
    if n < 50:
        return np.nan
    mu = x.mean()
    y = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc += x[i] - mu
        y[i] = acc
    sizes = np.array([10, 20, 40, 80])
    sizes = sizes[sizes < n // 2]
    if sizes.size < 2:
        return np.nan
    log_s = np.empty(sizes.size)
    log_f = np.empty(sizes.size)
    for k in range(sizes.size):
        s = sizes[k]
        m = n // s
        var_sum = 0.0
        # ``t`` and the normal-equation design moments (S0..S4, the 3x3 matrix and its inverse cofactors) depend only on the loop-invariant segment length ``s``; hoist out of the per-segment ``j`` loop. Only Sy/Sty/St2y and the residuals depend on ``seg``.
        t = np.arange(s).astype(np.float64)
        # Quadratic fit via normal equations.
        S0 = float(s)
        S1 = t.sum()
        S2 = (t * t).sum()
        S3 = (t * t * t).sum()
        S4 = (t * t * t * t).sum()
        # Solve [[S0,S1,S2],[S1,S2,S3],[S2,S3,S4]] [a,b,c] = [Sy,Sty,St2y]
        # via explicit 3x3 inverse (numba can't call linalg.solve robustly).
        M00 = S0; M01 = S1; M02 = S2
        M11 = S2; M12 = S3
        M22 = S4
        det = M00 * (M11 * M22 - M12 * M12) - M01 * (M01 * M22 - M12 * M02) + M02 * (M01 * M12 - M11 * M02)
        if abs(det) < 1e-12:
            # Design matrix degenerate for this scale (depends only on ``s``): every segment would have been skipped, so f_s collapses to 0 -- matches the pre-hoist per-segment ``continue``.
            log_s[k] = np.log(s)
            log_f[k] = np.log(1e-12)
            continue
        # Cofactors / adjugate solve
        inv_det = 1.0 / det
        c00 = (M11 * M22 - M12 * M12) * inv_det
        c01 = -(M01 * M22 - M12 * M02) * inv_det
        c02 = (M01 * M12 - M11 * M02) * inv_det
        c11 = (M00 * M22 - M02 * M02) * inv_det
        c12 = -(M00 * M12 - M01 * M02) * inv_det
        c22 = (M00 * M11 - M01 * M01) * inv_det
        for j in range(m):
            seg = y[j * s : (j + 1) * s]
            Sy = seg.sum()
            Sty = (t * seg).sum()
            St2y = (t * t * seg).sum()
            a = c00 * Sy + c01 * Sty + c02 * St2y
            b = c01 * Sy + c11 * Sty + c12 * St2y
            cq = c02 * Sy + c12 * Sty + c22 * St2y
            # Residuals.
            resid_sq = 0.0
            for i in range(s):
                fit = a + b * t[i] + cq * t[i] * t[i]
                d = seg[i] - fit
                resid_sq += d * d
            var_sum += resid_sq / s
        f_s = np.sqrt(var_sum / m)
        log_s[k] = np.log(s)
        log_f[k] = np.log(f_s + 1e-12)
    lm = log_s.mean()
    fm = log_f.mean()
    num = 0.0
    den = 0.0
    for k in range(log_s.size):
        num += (log_s[k] - lm) * (log_f[k] - fm)
        den += (log_s[k] - lm) ** 2
    return num / (den + 1e-12)


@njit(cache=True, fastmath=True)
def multifractal_dfa(
    x: np.ndarray,
    q_values: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Multifractal DFA: emit H(q) for each q in ``q_values``.

    Standard DFA collapses the H(q) spectrum into one scalar (q=2).
    Multifractal DFA varies q over [-5, 5] to expose the singularity
    spectrum. ``q_values`` typically ``[-5, -3, -1, 2, 3, 5]`` (omit
    q=0 — needs separate log-form handling). Returns shape
    ``(len(q_values),)`` Hurst exponents.

    Derived features (compute outside this kernel):
    * ``mfdfa_width = max(H) - min(H)`` — multifractality strength
    * ``mfdfa_asymmetry = (H_neg_q + H_pos_q - 2*H_q2) / mfdfa_width``

    Use case: finance (vol clustering, regime), turbulence, EEG
    seizure-onset, HRV classification.
    """
    n = x.size
    out = np.full(q_values.size, np.nan)
    if n < 50:
        return out
    mu = x.mean()
    y = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc += x[i] - mu
        y[i] = acc
    scales_valid = scales[scales < n // 2]
    if scales_valid.size < 2:
        return out
    log_s = np.empty(scales_valid.size)
    for k in range(scales_valid.size):
        log_s[k] = np.log(scales_valid[k])
    # F_q(s) per (q, s)
    F_qs = np.full((q_values.size, scales_valid.size), np.nan)
    for ks in range(scales_valid.size):
        s = scales_valid[ks]
        m = n // s
        if m < 1:
            continue
        # Variance per sub-segment (linear-detrended).
        var_per_seg = np.empty(m)
        # ``t`` (and its mean ``tm``) depend only on the loop-invariant segment length ``s``; hoist out of the per-segment ``j`` loop.
        t = np.arange(s).astype(np.float64)
        tm = t.mean()
        for j in range(m):
            seg = y[j * s : (j + 1) * s]
            sm = seg.mean()
            num = 0.0
            den = 0.0
            for i in range(s):
                num += (t[i] - tm) * (seg[i] - sm)
                den += (t[i] - tm) ** 2
            slope = num / (den + 1e-12)
            intercept = sm - slope * tm
            resid_sq = 0.0
            for i in range(s):
                fit = intercept + slope * t[i]
                resid_sq += (seg[i] - fit) ** 2
            var_per_seg[j] = resid_sq / s
        # F_q(s) = ( mean(var^(q/2)) )^(1/q) for q != 0
        for kq in range(q_values.size):
            q = q_values[kq]
            if abs(q) < 1e-6:
                continue
            powered = 0.0
            for j in range(m):
                v = var_per_seg[j]
                if v <= 0:
                    powered += 0.0
                else:
                    powered += v ** (q * 0.5)
            mean_pow = powered / m
            F_qs[kq, ks] = (mean_pow + 1e-12) ** (1.0 / q)
    # Fit H(q) = slope of log F_q(s) vs log s.
    for kq in range(q_values.size):
        row = F_qs[kq]
        finite = 0
        x_sum = 0.0
        y_sum = 0.0
        for ks in range(scales_valid.size):
            if row[ks] > 0 and not np.isnan(row[ks]):
                finite += 1
                x_sum += log_s[ks]
                y_sum += np.log(row[ks])
        if finite < 2:
            continue
        x_mean = x_sum / finite
        y_mean = y_sum / finite
        num = 0.0
        den = 0.0
        for ks in range(scales_valid.size):
            if row[ks] > 0 and not np.isnan(row[ks]):
                lr = log_s[ks]
                lf = np.log(row[ks])
                num += (lr - x_mean) * (lf - y_mean)
                den += (lr - x_mean) ** 2
        if den > 0:
            out[kq] = num / den
    return out
