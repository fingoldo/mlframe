"""Pair interaction screen on the residual of an additive model.

Screening column pairs by their joint MI with the target (or by a pair-form table against the target) picks the wrong pair on a strongly additive target: two columns with large main effects
carry a lot of joint information with no interaction at all. This screen first fits the additive part and asks only what is left.

* ``f_1(x_1) + ... + f_p(x_p)`` with free-form main effects is backfitted on the even rows against the winsorised, standardised target (``BACKFIT_SWEEPS`` Gauss-Seidel sweeps over all the columns
  together, so a pair is judged against the whole additive model, not against its own two main effects). The main effects use ``main_effect_bins`` levels (``clip(n_even / 250, 15, 60)``): a
  coarser additive model leaves the lack of fit of a steep effect (a sine, a parabola) in the residual and the screen would call it an interaction.
* For each pair, the mean residual in each ``N_BINS x N_BINS`` cell is estimated on the even rows (shrunk toward 0 by ``SHRINK`` pseudo-counts) and applied to the odd rows. The evidence is the
  paired per-row reduction of the squared residual on the odd rows, ``d_i = e_i^2 - (e_i - m_cell(i))^2``; its t statistic ``mean(d) / (sd(d) / sqrt(n_odd))`` is the pair's score (a held-out
  gain over the additive model, so noise cells cannot score).

A pair is reported when its score exceeds the one-sided normal quantile of a family-wise 5% level over all the pairs screened (Bonferroni), strongest first. All the pairs of up to ``MAX_COLS`` columns cost a few tens of milliseconds (one ``n`` pass per pair, in parallel).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from numba import njit, prange

from ._fe_gain_stats import quantile_codes
from ._y_encoding import FEW_CLASSES_MAX

__all__ = ["pair_interaction_screen", "main_effect_bins", "N_BINS", "FAMILY_ALPHA"]

N_BINS = 10
SHRINK = 3.0  # pseudo-counts that pull a cell's mean residual toward 0
BACKFIT_SWEEPS = 4
FAMILY_ALPHA = 0.05  # family-wise error level over all the pairs of one screen (Bonferroni)
WINSOR_Q = (0.01, 0.99)  # quantiles the target is clipped to
MAIN_BINS_PER_ROWS = 250  # even rows per main-effect level
MAIN_BINS_RANGE = (15, 60)
MAX_COLS = 40
DEFAULT_SCAN_ROWS = 50_000
_MIN_ROWS = 800


@njit(cache=True, nogil=True)
def _backfit(Q, y, even, n_bins, sweeps):
    """Gauss-Seidel backfit of ``p`` free-form main effects on the even rows; returns the ``(p, n_bins)`` effect tables and the intercept."""
    p, n = Q.shape
    f = np.zeros((p, n_bins))
    n_even = 0
    mu = 0.0
    for i in range(n):
        if even[i]:
            mu += y[i]
            n_even += 1
    mu /= max(n_even, 1)
    pred = np.zeros(n)
    for _ in range(sweeps):
        for j in range(p):
            s = np.zeros(n_bins)
            c = np.zeros(n_bins)
            for i in range(n):
                if even[i]:
                    b = Q[j, i]
                    s[b] += y[i] - mu - (pred[i] - f[j, b])
                    c[b] += 1.0
            for b in range(n_bins):
                new = s[b] / c[b] if c[b] > 0.0 else 0.0
                f[j, b] = new
            for i in range(n):
                pred[i] = 0.0
            for jj in range(p):
                for i in range(n):
                    pred[i] += f[jj, Q[jj, i]]
    return f, mu


@njit(parallel=True, cache=True)
def _score_pairs(Q, resid, even, pairs, n_bins, shrink):
    """t statistic of the odd-row reduction of the squared residual by the even-row cell means of each pair."""
    P = pairs.shape[0]
    n = Q.shape[1]
    out = np.zeros(P)
    for k in prange(P):
        a = pairs[k, 0]
        b = pairs[k, 1]
        s = np.zeros(n_bins * n_bins)
        c = np.zeros(n_bins * n_bins)
        for i in range(n):
            if even[i]:
                cell = Q[a, i] * n_bins + Q[b, i]
                s[cell] += resid[i]
                c[cell] += 1.0
        m1 = 0.0
        m2 = 0.0
        cnt = 0
        for i in range(n):
            if not even[i]:
                cell = Q[a, i] * n_bins + Q[b, i]
                mc = s[cell] / (c[cell] + shrink)
                e = resid[i]
                d = e * e - (e - mc) * (e - mc)
                m1 += d
                m2 += d * d
                cnt += 1
        if cnt > 1:
            mean = m1 / cnt
            var = max(m2 / cnt - mean * mean, 0.0)
            se = np.sqrt(var / cnt)
            out[k] = mean / se if se > 0.0 else mean * 1e300
    return out


def _winsorised(y: np.ndarray) -> np.ndarray:
    """``y`` clipped to its ``WINSOR_Q`` quantiles and standardised. A rank transform would be the robust choice but it is a nonlinear squashing (the CDF) of an additive target, which makes the
    squashed target interact; clipping keeps the additive structure of the bulk of the rows."""
    lo, hi = np.quantile(y, WINSOR_Q)
    w = np.clip(y, lo, hi)
    sd = float(w.std())
    return np.asarray((w - w.mean()) / sd if sd > 0 else w - w.mean())


def main_effect_bins(n_even: int) -> int:
    """Levels of a main effect in the additive model: one per ``MAIN_BINS_PER_ROWS`` even rows, within ``MAIN_BINS_RANGE``."""
    return int(min(max(n_even // MAIN_BINS_PER_ROWS, MAIN_BINS_RANGE[0]), MAIN_BINS_RANGE[1]))


def pair_interaction_screen(
    X: "pd.DataFrame", y: np.ndarray, cols: Optional[Sequence[str]] = None, *, scan_rows: int = DEFAULT_SCAN_ROWS, alpha: float = FAMILY_ALPHA, max_cols: int = MAX_COLS
) -> "list[tuple[str, str, float]]":
    """Column pairs with a significant interaction beyond the additive model, strongest first, as ``(column a, column b, t statistic)``.

    Columns that are constant or nominal-like (at most ``FEW_CLASSES_MAX`` distinct values) are left out; at most ``max_cols`` columns (the highest marginal variation of the rank target in
    their bins) enter the additive model. Empty when there are too few rows or columns."""
    n = len(X)
    cand = [c for c in (cols if cols is not None else X.columns) if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    if n < _MIN_ROWS or len(cand) < 2 or y is None:
        return []
    rows = np.arange(n) if n <= scan_rows else np.linspace(0, n - 1, int(scan_rows)).astype(np.int64)
    yr = _winsorised(np.asarray(y, dtype=np.float64).ravel()[rows])
    names, codes, main_codes = [], [], []
    for c in cand:
        x = np.asarray(X[c].to_numpy(), dtype=np.float64)[rows]
        finite = np.isfinite(x)
        if finite.sum() < _MIN_ROWS or np.unique(x[finite]).size <= FEW_CLASSES_MAX:
            continue
        names.append(c)
        codes.append(quantile_codes(x, N_BINS))
        main_codes.append(quantile_codes(x, main_effect_bins((len(rows) + 1) // 2)))
    if len(names) < 2:
        return []
    Q = np.ascontiguousarray(np.stack(codes))
    Qm = np.ascontiguousarray(np.stack(main_codes))
    if len(names) > max_cols:
        mean_by_bin = [np.bincount(Q[j], weights=yr, minlength=N_BINS) / np.maximum(np.bincount(Q[j], minlength=N_BINS), 1) for j in range(len(names))]
        keep = np.argsort([-float(np.var(m)) for m in mean_by_bin], kind="stable")[:max_cols]
        Q, Qm, names = np.ascontiguousarray(Q[keep]), np.ascontiguousarray(Qm[keep]), [names[j] for j in keep]
    even = np.arange(len(rows)) % 2 == 0
    n_main = main_effect_bins((len(rows) + 1) // 2)
    f, mu = _backfit(Qm, yr, even, n_main, BACKFIT_SWEEPS)
    pred = np.zeros(len(rows))
    for j in range(Qm.shape[0]):
        pred += f[j, Qm[j]]
    resid = yr - mu - pred
    pairs = np.array([(a, b) for a in range(len(names)) for b in range(a + 1, len(names))], dtype=np.int64)
    t = _score_pairs(Q, resid, even, pairs, N_BINS, SHRINK)
    from scipy.stats import norm

    z = float(norm.isf(float(alpha) / len(pairs)))
    order = np.argsort(-t, kind="stable")
    return [(names[pairs[k, 0]], names[pairs[k, 1]], float(t[k])) for k in order if t[k] > z]
