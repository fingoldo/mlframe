"""Residual rescue: candidate columns the rank-MI selection dropped, judged on the residual of the selected design in the target's own scale.

The greedy MI selection ranks candidates by information about the (quantile-binned) target. A term that is weak in rank space but carries most of the target's variance is invisible to it:
``a**2/b`` next to ``3*log(c)*sin(d)`` with 3% outliers has 83% of the variance (half of it in five rows) and MI 0.07 against 1.85, so the screen never picks it. The error metric of the
downstream models is MAE / RMSE, i.e. the target's own scale, so this stage asks the regression question instead:

* the selected design (raw and engineered columns, each with a free-form main effect of ``main_effect_bins`` levels) is backfitted on the even rows against the winsorised, standardised target
  (``_pair_residual_screen``: clipping keeps the additive structure of the bulk of the rows, a rank transform would squash it);
* every other column of the pool gets the even-row mean residual in its quantile bins (shrunk toward 0 by ``SHRINK`` pseudo-counts), applied to the odd rows; the evidence is the paired per-row
  reduction of the squared residual ``d_i = e_i^2 - (e_i - m_bin(i))^2`` and its t statistic ``mean(d) / (sd(d) / sqrt(n_odd))``;
* the best column is admitted when its t exceeds the one-sided normal quantile of a family-wise ``alpha`` over the pool (Bonferroni) and its relative gain ``mean(d)`` over the odd-row variance of the target is at least
  ``min_relative_gain`` (a practical-effect floor, a constructor parameter of the estimator); it joins the design and the residual is refitted, up to ``max_admit`` rounds, so a column that
  merely repeats an admitted one scores nothing in the next round.

Columns are binned by their own quantiles (no target information), the bin means come from even rows and are scored on odd rows, so a noise column cannot score beyond the family-wise level.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from numba import njit, prange

from ._pair_residual_screen import BACKFIT_SWEEPS, FAMILY_ALPHA, SHRINK, _backfit, _winsorised, main_effect_bins
from ._y_encoding import FEW_CLASSES_MAX

__all__ = ["residual_rescue", "MAX_POOL", "DEFAULT_SCAN_ROWS"]

MAX_POOL = 300  # columns scored per round: the pool is cut to the highest marginal variation of the residual in its bins
DEFAULT_SCAN_ROWS = 50_000
CODE_SAMPLE = 4096  # finite values per column that set its bin edges
_MIN_ROWS = 800


@njit(parallel=True, cache=True)
def _score_columns(Q, resid, even, n_bins, shrink):
    """Per column of ``Q``: the t statistic and the mean of ``d`` (odd-row reduction of the squared residual by the even-row bin means),"""
    p, n = Q.shape
    t = np.zeros(p)
    gain = np.zeros(p)
    for j in prange(p):
        s = np.zeros(n_bins)
        c = np.zeros(n_bins)
        for i in range(n):
            if even[i]:
                s[Q[j, i]] += resid[i]
                c[Q[j, i]] += 1.0
        m1 = 0.0
        m2 = 0.0
        cnt = 0
        for i in range(n):
            if not even[i]:
                b = Q[j, i]
                mc = s[b] / (c[b] + shrink)
                e = resid[i]
                d = e * e - (e - mc) * (e - mc)
                m1 += d
                m2 += d * d
                cnt += 1
        if cnt > 1:
            mean = m1 / cnt
            var = max(m2 / cnt - mean * mean, 0.0)
            t[j] = mean / (np.sqrt(var / cnt) + 1e-300)
            gain[j] = mean
    return t, gain


@njit(parallel=True, cache=True)
def _codes_matrix(M, n_bins, sample, min_finite, few_classes):
    """Equal-frequency bin codes of every row of ``M`` (columns of the data) and a usable mask.

    The edges are the quantiles of a strided sample of ``sample`` finite values (a full sort of every column is the whole cost of this stage otherwise); non-finite values go to bin 0. A column
    is unusable when it has fewer than ``min_finite`` finite values or at most ``few_classes`` distinct values in the sample (constant or nominal-like)."""
    p, n = M.shape
    codes = np.zeros((p, n), dtype=np.int64)
    ok = np.zeros(p, dtype=np.bool_)
    for j in prange(p):
        fin = 0
        for i in range(n):
            if np.isfinite(M[j, i]):
                fin += 1
        if fin < min_finite:
            continue
        step = max(n // sample, 1)
        buf = np.empty((n + step - 1) // step)
        m = 0
        for i in range(0, n, step):
            if np.isfinite(M[j, i]):
                buf[m] = M[j, i]
                m += 1
        if m < 2:
            continue
        s = np.sort(buf[:m])
        distinct = 1
        for k in range(1, m):
            if s[k] != s[k - 1]:
                distinct += 1
        if distinct <= few_classes:
            continue
        edges = np.empty(n_bins - 1)
        for k in range(n_bins - 1):
            edges[k] = s[min(int((k + 1) * m / n_bins), m - 1)]
        for i in range(n):
            v = M[j, i]
            if np.isfinite(v):
                codes[j, i] = np.searchsorted(edges, v, side="right")
        ok[j] = True
    return codes, ok


def _code_columns(cols: "Mapping[str, np.ndarray]", rows: np.ndarray, n_bins: int) -> "tuple[list[str], np.ndarray]":
    """Names and ``(p, len(rows))`` codes of the usable columns of ``cols`` on ``rows``."""
    names = list(cols)
    if not names:
        return [], np.empty((0, len(rows)), dtype=np.int64)
    M = np.ascontiguousarray(np.stack([np.asarray(cols[k])[rows] for k in names]).astype(np.float64, copy=False))
    codes, ok = _codes_matrix(M, n_bins, CODE_SAMPLE, _MIN_ROWS, FEW_CLASSES_MAX)
    keep = np.flatnonzero(ok)
    return [names[k] for k in keep], np.ascontiguousarray(codes[keep])


def _residual(Qm: np.ndarray, yr: np.ndarray, even: np.ndarray, n_main: int) -> np.ndarray:
    """Residual of the additive free-form model of the design columns ``Qm`` (rows are columns) on the standardised target."""
    if Qm.shape[0] == 0:
        return np.asarray(yr - yr[even].mean())
    f, mu = _backfit(Qm, yr, even, n_main, BACKFIT_SWEEPS)
    pred = np.zeros(len(yr))
    for j in range(Qm.shape[0]):
        pred += f[j, Qm[j]]
    return np.asarray(yr - mu - pred)


def residual_rescue(
    design: Mapping[str, np.ndarray],
    pool: Mapping[str, np.ndarray],
    y: np.ndarray,
    *,
    alpha: float = FAMILY_ALPHA,
    min_relative_gain: float = 0.01,
    max_admit: int = 3,
    scan_rows: int = DEFAULT_SCAN_ROWS,
    max_pool: int = MAX_POOL,
) -> "list[tuple[str, float, float]]":
    """Pool columns that reduce the held-out squared residual of the design, as ``(name, t statistic, relative gain)`` in admission order.

    ``design`` and ``pool`` map names to full-length columns; ``y`` is the continuous target. Empty when there are too few rows, no usable pool column or nothing passes."""
    from scipy.stats import norm

    n = len(y)
    if n < _MIN_ROWS or not len(pool) or int(max_admit) <= 0:
        return []
    rows = np.arange(n) if n <= scan_rows else np.linspace(0, n - 1, int(scan_rows)).astype(np.int64)
    yr = _winsorised(np.asarray(y, dtype=np.float64).ravel()[rows])
    even = np.arange(len(rows)) % 2 == 0
    total = float(np.mean((yr[~even] - yr[~even].mean()) ** 2)) or 1.0  # odd-row variance of the target: gains are relative to it, not to the shrinking residual
    n_main = main_effect_bins((len(rows) + 1) // 2)
    _, dcodes = _code_columns(design, rows, n_main)
    design_codes = list(dcodes)
    pool_names, Qp = _code_columns(pool, rows, n_main)
    if not pool_names:
        return []
    out: "list[tuple[str, float, float]]" = []
    alive = np.ones(len(pool_names), dtype=bool)
    for _ in range(int(max_admit)):
        Qd = np.ascontiguousarray(np.stack(design_codes)) if design_codes else np.empty((0, len(rows)), dtype=np.int64)
        resid = _residual(Qd, yr, even, n_main)
        idx = np.flatnonzero(alive)
        if idx.size > max_pool:
            var = np.array([float(np.var(np.bincount(Qp[j], weights=resid, minlength=n_main) / np.maximum(np.bincount(Qp[j], minlength=n_main), 1))) for j in idx])
            idx = idx[np.argsort(-var, kind="stable")[:max_pool]]
        t, gain = _score_columns(np.ascontiguousarray(Qp[idx]), resid, even, n_main, SHRINK)
        z = float(norm.isf(float(alpha) / max(len(pool_names), 1)))
        rel = gain / total
        ok = np.flatnonzero((t > z) & (rel >= float(min_relative_gain)))
        if ok.size == 0:
            break
        k = int(ok[np.argmax(t[ok])])
        j = int(idx[k])
        out.append((pool_names[j], float(t[k]), float(rel[k])))
        alive[j] = False
        design_codes.append(Qp[j])
    return out
