"""Winkler interval score for composite-target prediction intervals -- honest interval-quality scoring.

A conformal / quantile method emits a central ``(1 - alpha)`` prediction interval ``[lo, hi]`` per row.
Coverage alone is gameable (a trivially wide interval covers everything); width alone ignores misses. The
Winkler interval score (Winkler 1972; Gneiting & Raftery 2007) is the PROPER scoring rule that combines
both -- reward for sharpness, penalty for misses scaled by ``2/alpha`` -- so sharper AND better-calibrated
intervals score lower:

    S(y, lo, hi) = (hi - lo)
                   + (2/alpha) * (lo - y) * [y < lo]
                   + (2/alpha) * (y - hi) * [y > hi]

Lower is better. ``alpha`` is the nominal MISCOVERAGE (e.g. ``alpha = 0.1`` for a 90% interval); the target
coverage is ``1 - alpha``. This module reuses the numba-JITed unweighted kernels in
:mod:`mlframe.metrics.quantile` (``winkler_score`` / ``coverage`` / ``mean_interval_width``) and adds the
composite conveniences the base module lacks: sample WEIGHTS, a PER-ROW score, a PER-GROUP score (per-well /
per-symbol interval quality), and a combined :func:`interval_quality_summary`.

Performance (cProfile, ``_benchmarks/bench_winkler.py``, 1M rows / 500 groups). The mean unweighted score is
~1.7 ms (delegates to the already-optimized ``mlframe.metrics.quantile`` njit kernel); the weighted path is a
single fused ``numba.njit`` pass (:func:`_weighted_winkler_njit`). The per-group score was originally 2.68 s --
``mean_width`` was recomputed PER group by an O(n) ``codes == g`` rescan + masked dot (O(n_groups * n) total);
folding the weighted width into the ONE grouped njit sweep (:func:`_grouped_winkler_njit`; bincount fallback)
alongside the Winkler / coverage / count accumulators cut it to ~24 ms (~114x). ``pd.factorize`` of the group
ids is then the only O(n) non-kernel cost -- no actionable further speedup at this shape.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from mlframe.metrics.quantile import (
    coverage as _base_coverage,
    mean_interval_width,
    winkler_score as _base_winkler,
)

try:
    import pandas as pd

    _HAVE_PANDAS = True
except Exception:  # pragma: no cover
    _HAVE_PANDAS = False

_NJIT_KW = dict(fastmath=False, cache=True, nogil=True)

try:
    import numba

    @numba.njit(**_NJIT_KW)
    def _weighted_winkler_njit(y, lo, hi, w, alpha_miscov):
        n = y.shape[0]
        two_over_a = 2.0 / max(alpha_miscov, 1e-12)
        s = 0.0
        wsum = 0.0
        for i in range(n):
            wi = w[i]
            width = hi[i] - lo[i]
            pen = 0.0
            if y[i] < lo[i]:
                pen = two_over_a * (lo[i] - y[i])
            elif y[i] > hi[i]:
                pen = two_over_a * (y[i] - hi[i])
            s += wi * (width + pen)
            wsum += wi
        return s / wsum if wsum > 0 else 0.0

    @numba.njit(**_NJIT_KW)
    def _weighted_coverage_njit(y, lo, hi, w):
        n = y.shape[0]
        s = 0.0
        wsum = 0.0
        for i in range(n):
            wi = w[i]
            if lo[i] <= y[i] <= hi[i]:
                s += wi
            wsum += wi
        return s / wsum if wsum > 0 else 0.0

    @numba.njit(**_NJIT_KW)
    def _per_row_winkler_njit(y, lo, hi, alpha_miscov):
        n = y.shape[0]
        out = np.empty(n, dtype=np.float64)
        two_over_a = 2.0 / max(alpha_miscov, 1e-12)
        for i in range(n):
            width = hi[i] - lo[i]
            pen = 0.0
            if y[i] < lo[i]:
                pen = two_over_a * (lo[i] - y[i])
            elif y[i] > hi[i]:
                pen = two_over_a * (y[i] - hi[i])
            out[i] = width + pen
        return out

    @numba.njit(cache=True)
    def _grouped_winkler_njit(codes, y, lo, hi, w, alpha_miscov, n_groups):
        """Per-group weighted Winkler + coverage in one sweep. Rows with code < 0 are dropped."""
        two_over_a = 2.0 / max(alpha_miscov, 1e-12)
        S = np.zeros(n_groups)
        Wd = np.zeros(n_groups)
        W = np.zeros(n_groups)
        cov = np.zeros(n_groups)
        cnt = np.zeros(n_groups, dtype=np.int64)
        for i in range(codes.shape[0]):
            g = codes[i]
            if g < 0:
                continue
            wi = w[i]
            width = hi[i] - lo[i]
            pen = 0.0
            if y[i] < lo[i]:
                pen = two_over_a * (lo[i] - y[i])
            elif y[i] > hi[i]:
                pen = two_over_a * (y[i] - hi[i])
            S[g] += wi * (width + pen)
            Wd[g] += wi * width
            W[g] += wi
            if lo[i] <= y[i] <= hi[i]:
                cov[g] += wi
            cnt[g] += 1
        return S, Wd, W, cov, cnt

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False


__all__ = [
    "winkler_interval_score",
    "winkler_score_per_row",
    "winkler_score_per_group",
    "mean_coverage",
    "interval_quality_summary",
]


def _prep(y_true, q_lo, q_hi):
    y = np.ascontiguousarray(np.asarray(y_true, dtype=np.float64).reshape(-1))
    lo = np.ascontiguousarray(np.asarray(q_lo, dtype=np.float64).reshape(-1))
    hi = np.ascontiguousarray(np.asarray(q_hi, dtype=np.float64).reshape(-1))
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError(f"shape mismatch y={y.shape}, q_lo={lo.shape}, q_hi={hi.shape}")
    return y, lo, hi


def _check_alpha(alpha: float) -> float:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha (nominal miscoverage) must be in (0, 1); got {alpha}")
    return float(alpha)


def winkler_interval_score(y_true, q_lo, q_hi, alpha: float, sample_weight: Any = None) -> float:
    """Mean Winkler interval score for a central ``(1 - alpha)`` interval. Lower is better.

    ``alpha`` is the nominal miscoverage (0.1 for a 90% interval). Unweighted calls delegate to the optimized
    :func:`mlframe.metrics.quantile.winkler_score`; weighted calls use the fused weighted kernel here.
    """
    alpha = _check_alpha(alpha)
    if sample_weight is None:
        return _base_winkler(y_true, q_lo, q_hi, alpha)
    y, lo, hi = _prep(y_true, q_lo, q_hi)
    w = np.ascontiguousarray(np.asarray(sample_weight, dtype=np.float64).reshape(-1))
    if w.shape != y.shape:
        raise ValueError(f"sample_weight shape {w.shape} != y shape {y.shape}")
    if y.shape[0] == 0:
        return 0.0
    if _HAVE_NUMBA:
        return float(_weighted_winkler_njit(y, lo, hi, w, alpha))
    width = hi - lo
    pen = np.where(y < lo, lo - y, 0.0) + np.where(y > hi, y - hi, 0.0)
    per = width + (2.0 / alpha) * pen
    wsum = float(w.sum())
    return float(np.dot(per, w) / wsum) if wsum > 0 else 0.0


def winkler_score_per_row(y_true, q_lo, q_hi, alpha: float) -> np.ndarray:
    """Per-row Winkler score (width + one-sided miss penalty). Returns an ``(n,)`` array."""
    alpha = _check_alpha(alpha)
    y, lo, hi = _prep(y_true, q_lo, q_hi)
    if y.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    if _HAVE_NUMBA:
        return _per_row_winkler_njit(y, lo, hi, alpha)
    width = hi - lo
    pen = np.where(y < lo, lo - y, 0.0) + np.where(y > hi, y - hi, 0.0)
    return width + (2.0 / alpha) * pen


def mean_coverage(y_true, q_lo, q_hi, sample_weight: Any = None) -> float:
    """Empirical coverage: (weighted) fraction of ``y`` inside ``[q_lo, q_hi]``, in ``[0, 1]``.

    HOLDOUT CONTRACT: coverage is only honest on rows the interval method did NOT calibrate on -- computed
    in-sample it is optimistically inflated. The caller must pass a holdout split.
    """
    if sample_weight is None:
        return _base_coverage(y_true, q_lo, q_hi)
    y, lo, hi = _prep(y_true, q_lo, q_hi)
    w = np.ascontiguousarray(np.asarray(sample_weight, dtype=np.float64).reshape(-1))
    if w.shape != y.shape:
        raise ValueError(f"sample_weight shape {w.shape} != y shape {y.shape}")
    if y.shape[0] == 0:
        return 0.0
    if _HAVE_NUMBA:
        return float(_weighted_coverage_njit(y, lo, hi, w))
    inside = (y >= lo) & (y <= hi)
    wsum = float(w.sum())
    return float(np.dot(inside.astype(np.float64), w) / wsum) if wsum > 0 else 0.0


def winkler_score_per_group(
    y_true, q_lo, q_hi, alpha: float, group_ids: Any, sample_weight: Any = None,
) -> dict:
    """Per-group Winkler + coverage + width + count. Returns ``{label: {winkler, coverage, mean_width, n}}``.

    Null / NaN group labels are dropped (they cannot be attributed to a group).
    """
    alpha = _check_alpha(alpha)
    y, lo, hi = _prep(y_true, q_lo, q_hi)
    n = y.shape[0]
    if _HAVE_PANDAS:
        codes, uniq = pd.factorize(np.asarray(group_ids), sort=False)
        codes = np.asarray(codes, dtype=np.int64)
        uniq = list(uniq)
    else:
        uniq_arr, codes = np.unique(np.asarray(group_ids), return_inverse=True)
        codes = np.asarray(codes, dtype=np.int64)
        uniq = list(uniq_arr)
    if codes.shape[0] != n:
        raise ValueError(f"group_ids length {codes.shape[0]} != y length {n}")
    n_groups = len(uniq)
    w = np.ones(n, dtype=np.float64) if sample_weight is None else np.ascontiguousarray(np.asarray(sample_weight, dtype=np.float64).reshape(-1))
    if w.shape[0] != n:
        raise ValueError(f"sample_weight length {w.shape[0]} != y length {n}")
    if n == 0 or n_groups == 0:
        return {}
    codes = np.ascontiguousarray(codes)
    if _HAVE_NUMBA:
        S, Wd, W, cov, cnt = _grouped_winkler_njit(codes, y, lo, hi, w, alpha, n_groups)
    else:
        S, Wd, W, cov, cnt = _grouped_winkler_bincount(codes, y, lo, hi, w, alpha, n_groups)
    out: dict[Any, dict] = {}
    for g in range(n_groups):
        if cnt[g] == 0 or W[g] <= 0:
            continue
        label = uniq[g]
        key = label.item() if isinstance(label, np.generic) else label
        out[key] = {
            "winkler": float(S[g] / W[g]),
            "coverage": float(cov[g] / W[g]),
            "mean_width": float(Wd[g] / W[g]),
            "n": int(cnt[g]),
        }
    return out


def _grouped_winkler_bincount(codes, y, lo, hi, w, alpha, n_groups):
    two_over_a = 2.0 / alpha
    width = hi - lo
    pen = np.where(y < lo, lo - y, 0.0) + np.where(y > hi, y - hi, 0.0)
    per = width + two_over_a * pen
    inside = ((y >= lo) & (y <= hi)).astype(np.float64)
    valid = codes >= 0
    cv = codes[valid]
    S = np.bincount(cv, weights=(per * w)[valid], minlength=n_groups)
    Wd = np.bincount(cv, weights=(width * w)[valid], minlength=n_groups)
    W = np.bincount(cv, weights=w[valid], minlength=n_groups)
    cov = np.bincount(cv, weights=(inside * w)[valid], minlength=n_groups)
    cnt = np.bincount(cv, minlength=n_groups).astype(np.int64)
    return S, Wd, W, cov, cnt


def interval_quality_summary(
    y_true, q_lo, q_hi, alpha: float, sample_weight: Any = None,
) -> dict:
    """Combined honest interval-quality block: ``winkler`` (lower better), ``coverage`` vs ``target_coverage``,
    ``mean_width``, and the below/above miss rates. All computed on the SAME rows.
    """
    alpha = _check_alpha(alpha)
    y, lo, hi = _prep(y_true, q_lo, q_hi)
    n = y.shape[0]
    w = np.ones(n, dtype=np.float64) if sample_weight is None else np.ascontiguousarray(np.asarray(sample_weight, dtype=np.float64).reshape(-1))
    if w.shape[0] != n:
        raise ValueError(f"sample_weight length {w.shape[0]} != y length {n}")
    wsum = float(w.sum())
    below = float(np.dot((y < lo).astype(np.float64), w) / wsum) if wsum > 0 else 0.0
    above = float(np.dot((y > hi).astype(np.float64), w) / wsum) if wsum > 0 else 0.0
    return {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "coverage": mean_coverage(y, lo, hi, sample_weight=w if sample_weight is not None else None),
        "winkler": winkler_interval_score(y, lo, hi, alpha, sample_weight=w if sample_weight is not None else None),
        "mean_width": mean_interval_width(lo, hi),
        "miss_rate": below + above,
        "below_rate": below,
        "above_rate": above,
        "n": int(n),
    }
