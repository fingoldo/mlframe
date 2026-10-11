"""Residual rescue: candidate columns the rank-MI selection dropped, judged on the residual of the selected design in the target's own scale.

The greedy MI selection ranks candidates by information about the (quantile-binned) target. A term that is weak in rank space but carries most of the target's variance is invisible to it:
``a**2/b`` next to ``3*log(c)*sin(d)`` with 3% outliers has 83% of the variance (half of it in five rows) and MI 0.07 against 1.85, so the screen never picks it. The error metric of the
downstream models is MAE / RMSE, i.e. the target's own scale, so this stage asks the regression question instead:

* the selected design (raw and engineered columns: a linear part first, then a free-form main effect of ``main_effect_bins`` levels each) is fitted on the even rows against the fence-clipped,
  standardised target (``_pair_residual_screen``: clipping keeps the additive structure of the bulk of the rows, a rank transform would squash it);
* every other column of the pool gets the even-row mean residual in its quantile bins (shrunk toward 0 by ``SHRINK`` pseudo-counts), applied to the odd rows; the evidence is the paired per-row
  reduction of the squared residual ``d_i = e_i^2 - (e_i - m_bin(i))^2`` and its t statistic ``mean(d) / (sd(d) / sqrt(n_odd))``;
* the best column is admitted when its t exceeds the one-sided normal quantile of a family-wise ``alpha`` over the pool (Bonferroni) and its relative gain ``mean(d)`` over the odd-row variance of the target is at least
  ``min_relative_gain`` (a practical-effect floor, a constructor parameter of the estimator); it joins the design and the residual is refitted, up to ``max_admit`` rounds, so a column that
  merely repeats an admitted one scores nothing in the next round.

Columns are binned by their own quantiles (no target information), the bin means come from even rows and are scored on odd rows, so a noise column cannot score beyond the family-wise level.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
from numba import njit, prange

from ._pair_residual_screen import BACKFIT_SWEEPS, FAMILY_ALPHA, OUTER_FENCE, SHRINK, _backfit, _winsorised, main_effect_bins
from ._y_encoding import FEW_CLASSES_MAX

__all__ = ["residual_rescue", "design_residual", "linear_oos_gain", "MAX_POOL", "DEFAULT_SCAN_ROWS", "MIN_ROWS"]

MAX_POOL = 300  # columns scored per round: the pool is cut to the highest marginal variation of the residual in its bins
DEFAULT_SCAN_ROWS = 50_000
CODE_SAMPLE = 4096  # finite values per column that set its bin edges
MIN_ROWS = 800  # fewer rows: the held-out evidence is too thin to admit or to prune anything


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
    codes, ok = _codes_matrix(M, n_bins, CODE_SAMPLE, MIN_ROWS, FEW_CLASSES_MAX)
    keep = np.flatnonzero(ok)
    return [names[k] for k in keep], np.ascontiguousarray(codes[keep])


def _standardised(v: np.ndarray) -> Optional[np.ndarray]:
    """``v`` with non-finite values set to the median, clipped to Tukey's far-out fences and standardised; ``None`` for a constant column."""
    v = np.asarray(v, dtype=np.float64)
    finite = np.isfinite(v)
    if not finite.any():
        return None
    v = np.where(finite, v, np.median(v[finite]))
    q1, q3 = np.quantile(v, (0.25, 0.75))
    iqr = q3 - q1
    if iqr > 0:
        v = np.clip(v, q1 - OUTER_FENCE * iqr, q3 + OUTER_FENCE * iqr)
    sd = float(v.std())
    return (v - v.mean()) / sd if sd > 0 else None


def _without_linear_part(lin_cols: "list[np.ndarray]", yr: np.ndarray, even: np.ndarray) -> np.ndarray:
    """``yr`` minus its least-squares fit (even rows) on the standardised design columns and an intercept. The free-form effects below are binned, so a smooth trend inside a bin is left in their
    residual and a column that shares the trend (a raw operand beside an engineered column built from it) scores a spurious gain; fitting the linear part first removes that artefact."""
    L = np.column_stack([*lin_cols, np.ones(len(yr))]) if lin_cols else np.ones((len(yr), 1))
    beta, *_ = np.linalg.lstsq(L[even], yr[even], rcond=None)
    return np.asarray(yr - L @ beta)


def _residual(Qm: np.ndarray, yr: np.ndarray, even: np.ndarray, n_main: int) -> np.ndarray:
    """Residual of the additive free-form model of the design columns ``Qm`` (rows are columns) on the standardised target."""
    if Qm.shape[0] == 0:
        return np.asarray(yr - yr[even].mean())
    f, mu = _backfit(Qm, yr, even, n_main, BACKFIT_SWEEPS)
    pred = np.zeros(len(yr))
    for j in range(Qm.shape[0]):
        pred += f[j, Qm[j]]
    return np.asarray(yr - mu - pred)


def linear_oos_gain(rest: Mapping[str, np.ndarray], candidate: np.ndarray, y: np.ndarray) -> "tuple[float, float]":
    """Held-out gain of ``candidate`` for a LINEAR model on the other columns: ``(t statistic, share of the target variance)``.

    OLS (fence-clipped, standardised columns and an intercept) fitted on the even rows with and without the candidate; the evidence is the paired per-row reduction of the squared error on the odd rows,
    ``d_i = e0_i^2 - e1_i^2``, its t statistic and its mean over the odd-row variance of the target. This is the question a linear downstream model asks of a column, whatever the free-form
    effects of the other columns could absorb: a raw ``a`` beside ``add(exp(a), ...)`` has no free-form gain (the compound is a proxy of ``a``) but a linear model can only use the compound as one
    block, so a private linear term of ``a`` is a real gain there."""
    yr = _winsorised(np.asarray(y, dtype=np.float64).ravel())
    n = len(yr)
    even = np.arange(n) % 2 == 0
    cols = [c for c in (_standardised(v) for v in rest.values()) if c is not None]
    x = _standardised(candidate)
    if x is None:
        return 0.0, 0.0

    def residual(columns):
        """Odd-row residual of the OLS on ``columns`` fitted on the even rows."""
        L = np.column_stack([*columns, np.ones(n)])
        beta, *_ = np.linalg.lstsq(L[even], yr[even], rcond=None)
        return (yr - L @ beta)[~even]

    d = residual(cols) ** 2 - residual([*cols, x]) ** 2
    sd = float(d.std())
    total = float(np.var(yr[~even])) or 1.0
    return float(d.mean() / (sd / np.sqrt(len(d)) + 1e-300)), float(d.mean() / total)


def design_residual(design: Mapping[str, np.ndarray], y: np.ndarray) -> np.ndarray:
    """Residual of the free-form additive model of the ``design`` columns on the winsorised, standardised target, one value per row (the model is backfitted on the even rows, so the odd
    rows are out of sample). The scale is the standardised target's, whatever the scale of ``y``."""
    n = len(y)
    rows = np.arange(n)
    yr = _winsorised(np.asarray(y, dtype=np.float64).ravel())
    even = rows % 2 == 0
    n_main = main_effect_bins((n + 1) // 2)
    _, dcodes = _code_columns(design, rows, n_main)
    lin = [c for c in (_standardised(v) for v in design.values()) if c is not None]
    return _residual(np.ascontiguousarray(dcodes), _without_linear_part(lin, yr, even), even, n_main)


def residual_rescue(
    design: Mapping[str, np.ndarray],
    pool: Mapping[str, np.ndarray],
    y: np.ndarray,
    *,
    alpha: float = FAMILY_ALPHA,
    min_relative_gain: float = 0.001,
    max_admit: int = 8,
    scan_rows: int = DEFAULT_SCAN_ROWS,
    max_pool: int = MAX_POOL,
) -> "list[tuple[str, float, float]]":
    """Pool columns that reduce the held-out squared residual of the design, as ``(name, t statistic, relative gain)`` in admission order.

    ``design`` and ``pool`` map names to full-length columns; ``y`` is the continuous target. Empty when there are too few rows, no usable pool column or nothing passes."""
    from scipy.stats import norm

    n = len(y)
    if n < MIN_ROWS or not len(pool) or int(max_admit) <= 0:
        return []
    rows = np.arange(n) if n <= scan_rows else np.linspace(0, n - 1, int(scan_rows)).astype(np.int64)
    yr = _winsorised(np.asarray(y, dtype=np.float64).ravel()[rows])
    even = np.arange(len(rows)) % 2 == 0
    total = float(np.mean((yr[~even] - yr[~even].mean()) ** 2)) or 1.0  # odd-row variance of the target: gains are relative to it, not to the shrinking residual
    n_main = main_effect_bins((len(rows) + 1) // 2)
    _, dcodes = _code_columns(design, rows, n_main)
    design_codes = list(dcodes)
    lin_cols = [c for c in (_standardised(np.asarray(v)[rows]) for v in design.values()) if c is not None]
    pool_names, Qp = _code_columns(pool, rows, n_main)
    if not pool_names:
        return []
    out: "list[tuple[str, float, float]]" = []
    alive = np.ones(len(pool_names), dtype=bool)
    for _ in range(int(max_admit)):
        Qd = np.ascontiguousarray(np.stack(design_codes)) if design_codes else np.empty((0, len(rows)), dtype=np.int64)
        resid = _residual(Qd, _without_linear_part(lin_cols, yr, even), even, n_main)
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
        added = _standardised(np.asarray(pool[pool_names[j]])[rows])
        if added is not None:
            lin_cols.append(added)
    return out
