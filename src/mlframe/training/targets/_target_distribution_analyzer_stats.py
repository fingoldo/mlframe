"""Statistical helper detectors carved out of
``mlframe.training._target_distribution_analyzer``: moment-based stats
(kurtosis, skewness), lag-autocorrelation variants (single / multi-lag
/ per-group Fisher-z aggregated), and a within-group ordering heuristic.

Bound back into the parent's namespace via re-export at the parent's
module bottom so historical
``from mlframe.training._target_distribution_analyzer import _excess_kurtosis``
resolves transparently.
"""
from __future__ import annotations

import math

import numpy as np


def _excess_kurtosis(y: np.ndarray) -> float:
    """Biased (Pearson) excess kurtosis; gaussian baseline = 0.

    z**4 via chained multiplication (z2 = z*z; z2*z2) avoids np.power's
    general-purpose dispatch (~3x faster at n=50k..5M; bench:
    profiling/bench_target_moments_no_power.py). Same antipattern fix
    as iter129 for regression_residual_audit.
    """
    n = y.size
    if n < 4:
        return 0.0
    mu = float(np.mean(y))
    sigma = float(np.std(y))
    if sigma <= 0.0 or not math.isfinite(sigma):
        return 0.0
    z = (y - mu) / sigma
    z2 = z * z
    return float(np.mean(z2 * z2)) - 3.0


def _skewness(y: np.ndarray) -> float:
    """Biased moment-based skewness.

    z**3 via chained multiplication (z*z*z) avoids np.power's
    general-purpose dispatch (~2.5-3.6x faster at n=50k..5M; bench:
    profiling/bench_target_moments_no_power.py).
    """
    n = y.size
    if n < 3:
        return 0.0
    mu = float(np.mean(y))
    sigma = float(np.std(y))
    if sigma <= 0.0 or not math.isfinite(sigma):
        return 0.0
    z = (y - mu) / sigma
    return float(np.mean(z * z * z))


def _lag1_autocorr(y: np.ndarray) -> float:
    """Pearson autocorrelation between y[:-1] and y[1:].

    A naive autocorr would assume y is time-ordered. This function assumes
    rows are in their training order; AR detection is meaningful only when
    the caller knows the rows have a natural sequence. The suite caller
    skips this detector when ``has_time_axis=False``.
    """
    return _lag_autocorr(y, lag=1)


def _lag_autocorr(y: np.ndarray, lag: int = 1) -> float:
    """Pearson autocorrelation at the given lag (lag-1 by default).

    Computed directly as ``(da . db) / sqrt((da . da) * (db . db))`` on the mean-centred slices (three BLAS dot
    products) instead of ``np.corrcoef``, which stacks a 2 x n array and builds a full 2 x 2 covariance matrix to
    return a single off-diagonal value -- 2.9x faster on the lag-scan at n=300k. Numerically equivalent to the corrcoef
    form to ~1 ULP (the (n-1) covariance normalisation cancels in the ratio); the sub-1e-15 reduction-order delta cannot
    move the strong-AR threshold or the reported diagnostic. The zero-variance guard (constant slice -> 0.0) is preserved
    via ``va <= 0`` / ``vb <= 0`` (va == n * Var(a), so it fires exactly when the old ``np.std`` guard did).
    """
    if y.size < (lag + 3) or lag < 1:
        return 0.0
    a, b = y[:-lag], y[lag:]
    da = a - a.mean()
    db = b - b.mean()
    va = float(da @ da)
    vb = float(db @ db)
    if va <= 0.0 or vb <= 0.0:
        return 0.0
    return float((da @ db) / math.sqrt(va * vb))


def _max_abs_lag_autocorr(y: np.ndarray, lags: tuple[int, ...] = (1, 2, 3, 5)) -> tuple[float, int]:
    """Return (max |autocorr|, lag-at-max) across the supplied lags.

    E5.1 (2026-05-21): single-lag detection misses long-memory series whose
    lag-1 cancels to a near-zero value (e.g. heavily smoothed targets, AR(2)
    with negative phi_1, seasonal patterns at lag 5/7/12). Aggregating
    across lags 1/2/3/5 catches the same class of distribution-shape
    pathologies the lag-1 detector was designed for, while keeping the
    statistical cost minimal (4 corrcoef calls on a 1-D array).
    """
    best_corr = 0.0
    best_lag = 0
    for lag in lags:
        corr = _lag_autocorr(y, lag=lag)
        if math.isfinite(corr) and abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = int(lag)
    return best_corr, best_lag


def _lag1_autocorr_grouped(y: np.ndarray, group_ids: np.ndarray, min_group_size: int = 4) -> float:
    """Per-group lag-1 autocorr aggregated across groups via Fisher-z + reverse.

    For data where rows have a natural sequence WITHIN each group but not across
    groups (the classic per-customer-time-series / per-subject-EEG / per-asset-depth
    layout), naive ``_lag1_autocorr`` measures cross-group transitions as if they
    were temporal -- spurious low/zero AR. Instead, compute lag-1 autocorr for
    every group independently and aggregate via Fisher-z transformation (atanh)
    + reverse (tanh).

    Why Fisher-z and NOT size-weighted mean: a size-weighted plain average is
    dominated by ONE huge group when group sizes are skewed (one 100k-row well +
    999 tiny ones -> the AR signal from the big well buries the others). Fisher
    z aggregates per-group correlations as DISTINCT samples, faithfully
    reflecting "AR is present within most groups". A critique-agent
    flagged the original size-weighted form for exactly this skew sensitivity.

    Groups smaller than ``min_group_size`` rows are skipped (with a stamp in the
    returned via global ``_lag1_grouped_skipped_count`` -- not yet wired but
    reserved for future observability). Returns NaN when no qualifying groups
    remain.
    """
    if y.size != group_ids.size or y.size < min_group_size:
        return float("nan")
    z_values: list[float] = []
    uniq = np.unique(group_ids)
    skipped = 0
    for g in uniq:
        mask = group_ids == g
        n_g = int(mask.sum())
        if n_g < min_group_size:
            skipped += 1
            continue
        yg = y[mask]
        ar_g = _lag1_autocorr(yg)
        if not math.isfinite(ar_g):
            continue
        # Fisher-z; cap |ar| to avoid atanh(+/-1) = +/-inf for perfect correlation.
        ar_capped = max(-0.9999, min(0.9999, ar_g))
        z_values.append(math.atanh(ar_capped))
    if not z_values:
        return float("nan")
    z_mean = float(np.mean(z_values))
    return math.tanh(z_mean)


def _check_within_group_ordering(group_ids: np.ndarray, n_check: int = 1024) -> bool:
    """Heuristic ordering check: sample at most ``n_check`` TRULY adjacent group_id
    pairs (i, i+1) spread evenly across the array; if more than 50% are within-group
    transitions, the rows are plausibly sorted by group (rows of the same group are
    contiguous). Returns True for plausibly-ordered data.

    The per-group AR detector assumes rows within a group are in their natural
    sequence (e.g. depth/time-sorted). If rows were shuffled randomly
    AFTER the FTE step, within-group autocorr drops to ~0 even when the
    underlying signal is strongly AR -- a false-negative the operator never
    sees. This check surfaces the assumption violation.

    Bug fixed 2026-07-19: the previous version strided THROUGH the array (comparing
    group_ids[::step] elements to each other) instead of sampling true adjacent pairs.
    For many small groups (e.g. hundreds of wells, each far smaller than group_ids.size //
    n_check), the stride lands in a DIFFERENT group almost every time even when the data
    is perfectly sorted, so it always reported "not ordered" regardless of the true
    same-group fraction of real adjacent rows. Now samples evenly-spaced true (i, i+1)
    pairs instead.
    """
    if group_ids.size < 2:
        return False
    # Step-sample true adjacent-pair indices to keep the check O(n_check) on huge arrays.
    step = max(1, (group_ids.size - 1) // n_check)
    idx = np.arange(0, group_ids.size - 1, step)
    if idx.size < 2:
        return False
    same_group = group_ids[idx] == group_ids[idx + 1]
    return float(np.mean(same_group)) > 0.5
