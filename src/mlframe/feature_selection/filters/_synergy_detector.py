"""Cheap data-dependent synergy detector for the ``redundancy_aggregator='auto'`` gate.

WHY
---
``fleuret.py`` documents that the default Fleuret/CMIM redundancy gate REJECTS synergistic features (an operand useless alone but
informative jointly with an already-selected partner). The synergy-aware JMIM aggregator (Bennasar 2015) recovers them, but
benchmarks show JMIM OVER-SELECTS correlated decoys on additive/main-effect data (precision/parsimony regression). So JMIM must
engage ONLY when the data actually contains synergy. This module is the cheap pre-fit probe that decides.

SIGNATURE OF SYNERGY
--------------------
A pair ``(X, Z)`` is synergistic for target ``y`` when the target depends on ``(X, Z)`` in a way NO additive model of the two
marginal effects reproduces: the log-linear model without the three-way term (``logit P(y|x,z) = f(x) + g(z)`` for a binary target)
fits the ``kx x kz x ky`` count table badly. The lack of fit is the deviance ``G = 2 sum n log(n / m)`` against the maximum-likelihood
no-three-way fit ``m`` (iterative proportional fitting of the three two-way margins). ``(G - df) / (2 n)`` is a bias-corrected
interaction information in nats, ``df = (kx-1)(kz-1)(ky-1)``. A pure XOR / sign-product pair reads about 0.6 nats (most of ``H(y)``).
Two additive drivers behind a logistic or hard-threshold link, correlated decoys of one driver and main-effect-only targets all
read at the chance level, because their joint IS additive in the logit: that is what the earlier ``I(XZ;Y) - I(X;Y) - I(Z;Y)``
excess could not separate (a saturating link makes it a few percent of the joint and its permutation null did not model that).

DECISION (subsample, every pair)
--------------------------------
Rows are subsampled to ``max_rows``, columns quantile-binned, and the deviance is computed for EVERY pair of up to ``max_features``
columns (default 300, 44850 pairs, one numba prange pass over the integer codes; random column sampling is the fallback above that).
No pair is sampled away, so a planted pair is never missed by coverage. A pair counts when BOTH hold:
  * significance: ``G`` exceeds the chi-square quantile of its own ``df`` at the Bonferroni level ``alpha / n_pairs``, so the null
    maximum over all pairs is controlled analytically (no label permutations, no null-scale multiple);
  * effect size: ``(G - df) / (2 n) >= _MIN_SYNERGY_SHARE * H(y)``, which keeps a genuine but tiny interaction (discretisation of a
    smooth additive link, sparse multiclass tables) from firing.
The cut points ``_ALPHA`` and ``_MIN_SYNERGY_SHARE`` are fixed by the calibration table in ``audits/2026-11-09/_TRACKER.md`` (OP11-17).
Bin count per pair is capped so the mean cell holds at least ``_MIN_ROWS_PER_CELL`` rows, which keeps the chi-square approximation valid.
``n_null`` is accepted for API compatibility and unused.
"""

from __future__ import annotations

import logging

import numpy as np
from numba import njit, prange

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

from ._fe_synergy_screen import joint_synergy_mi  # noqa: F401
from ._numba_utils import _pair_mm_mi_njit
from ._fe_usability_signal import _crit_np_dtype


def _pair_mm_mi(code_x: np.ndarray, code_y: np.ndarray, yc: np.ndarray, min_rows_per_cell: float = 5.0) -> float:
    """Fused-njit equivalent of ``joint_synergy_mi(code_x, code_y, yc)``.

    Builds the dense mixed-radix ``(kx*ky) x kt`` joint histogram in one O(n) pass (no
    ``np.unique`` densify, no ``np.add.at`` scatter) and returns the SAME Miller-Madow-
    corrected joint MI. ~9.6x faster per call at the detector's (n=4000, nbins=8) shape;
    output matches ``joint_synergy_mi`` to FP reduction order (~5e-16), so the synergy
    verdict is unchanged (selection-equivalent). Bench:
    ``_benchmarks/bench_synergy_detector_pair_mi_njit.py``."""
    cx = np.ascontiguousarray(np.asarray(code_x).astype(np.int64).ravel())
    cy = np.ascontiguousarray(np.asarray(code_y).astype(np.int64).ravel())
    yt = np.ascontiguousarray(np.asarray(yc).astype(np.int64).ravel())
    n = cx.shape[0]
    if n == 0 or cy.shape[0] != n or yt.shape[0] != n:
        return 0.0
    kx = int(cx.max()) + 1
    ky = int(cy.max()) + 1
    kt = int(yt.max()) + 1
    return float(_pair_mm_mi_njit(cx, cy, yt, kx, ky, kt, float(min_rows_per_cell)))


def _quantize(col: np.ndarray, nbins: int, rng: np.random.Generator) -> np.ndarray:
    """Integer-bin a 1-D column. Low-cardinality columns (<=nbins distinct) are factorised directly so
    XOR/parity bits keep their exact 0/1 codes; continuous columns get quantile bins."""
    col = np.asarray(col, dtype=_crit_np_dtype()).ravel()
    finite = col[np.isfinite(col)]
    uniq = np.unique(finite)
    if uniq.size <= nbins:
        # direct factorisation - preserves discrete bits exactly
        out = np.searchsorted(uniq, col).astype(np.int64)  # every finite value is one of the sorted distinct values: its rank is its code
        out[~np.isfinite(col)] = 0  # NaN / inf are not levels: code 0, as the dict lookup's default gave
        return out
    qs = np.quantile(finite, np.linspace(0, 1, nbins + 1)[1:-1])
    return np.clip(np.searchsorted(qs, col), 0, nbins - 1).astype(np.int64)


#: Family-wise significance level of the max-over-pairs test (Bonferroni across every pair examined).
_ALPHA = 0.01
#: A pair counts only if its bias-corrected interaction information is at least this share of the target entropy H(y). XOR / sign-product
#: reads ~0.9 of H(y); the residual non-additivity of a binned logistic / threshold link reads at most a few percent.
_MIN_SYNERGY_SHARE = 0.10
#: Mean rows per cell of the (kx, kz, ky) table kept by capping the pair bin count; keeps the chi-square tail valid for multiclass targets at small n.
_MIN_ROWS_PER_CELL = 5.0
_IPF_ITERS = 12


@njit(cache=True, nogil=True)
def _two_way_margins(tab, ka, kb, kt):
    """The (a, b), (a, t) and (b, t) margins of a flat ``ka x kb x kt`` count table (index ``(a*kb+b)*kt+t``)."""
    nab = np.zeros((ka, kb))
    nat = np.zeros((ka, kt))
    nbt = np.zeros((kb, kt))
    for a in range(ka):
        for b in range(kb):
            for t in range(kt):
                v = tab[(a * kb + b) * kt + t]
                nab[a, b] += v
                nat[a, t] += v
                nbt[b, t] += v
    return nab, nat, nbt


@njit(cache=True, nogil=True)
def _fit_ab(m, nab, ka, kb, kt):
    """One proportional-fitting pass of the fitted table ``m`` to the (a, b) margin."""
    for a in range(ka):
        for b in range(kb):
            s = 0.0
            for t in range(kt):
                s += m[(a * kb + b) * kt + t]
            if s > 0.0:
                f = nab[a, b] / s
                for t in range(kt):
                    m[(a * kb + b) * kt + t] *= f


@njit(cache=True, nogil=True)
def _fit_at(m, nat, ka, kb, kt):
    """One proportional-fitting pass of ``m`` to the (a, t) margin."""
    for a in range(ka):
        for t in range(kt):
            s = 0.0
            for b in range(kb):
                s += m[(a * kb + b) * kt + t]
            if s > 0.0:
                f = nat[a, t] / s
                for b in range(kb):
                    m[(a * kb + b) * kt + t] *= f


@njit(cache=True, nogil=True)
def _fit_bt(m, nbt, ka, kb, kt):
    """One proportional-fitting pass of ``m`` to the (b, t) margin."""
    for b in range(kb):
        for t in range(kt):
            s = 0.0
            for a in range(ka):
                s += m[(a * kb + b) * kt + t]
            if s > 0.0:
                f = nbt[b, t] / s
                for a in range(ka):
                    m[(a * kb + b) * kt + t] *= f


@njit(cache=True, nogil=True)
def _deviance_sum(tab, m):
    """``2 sum n log(n / m)`` over the occupied cells."""
    g = 0.0
    for c in range(tab.shape[0]):
        if tab[c] > 0.0 and m[c] > 0.0:
            g += tab[c] * np.log(tab[c] / m[c])
    return 2.0 * g


@njit(cache=True, nogil=True)
def _no_three_way_deviance(tab, ka, kb, kt, iters):
    """Deviance ``2 sum n log(n/m)`` of a ``ka x kb x kt`` count table (flat, index ``(a*kb+b)*kt+t``) against the maximum-likelihood
    model with the three two-way margins fixed and no three-way term, fitted by iterative proportional fitting."""
    m = np.ones_like(tab)
    nab, nat, nbt = _two_way_margins(tab, ka, kb, kt)
    for _ in range(iters):
        _fit_ab(m, nab, ka, kb, kt)
        _fit_at(m, nat, ka, kb, kt)
        _fit_bt(m, nbt, ka, kb, kt)
    return _deviance_sum(tab, m)


@njit(cache=True, nogil=True, parallel=True)
def _pair_deviances(codes, levels, yc, kt, pair_a, pair_b, iters):
    """No-three-way-interaction deviance for each ``(pair_a[q], pair_b[q])`` column pair; ``codes`` is ``(p, n)`` int8 (feature-major),
    ``levels[j]`` the number of codes of column ``j``, ``yc`` the int target codes in ``[0, kt)``."""
    n = codes.shape[1]
    out = np.zeros(pair_a.shape[0])
    for q in prange(pair_a.shape[0]):
        a = pair_a[q]
        b = pair_b[q]
        ka = levels[a]
        kb = levels[b]
        if ka < 2 or kb < 2:
            continue
        tab = np.zeros(ka * kb * kt)
        ca = codes[a]
        cb = codes[b]
        for r in range(n):
            tab[(ca[r] * kb + cb[r]) * kt + yc[r]] += 1.0
        out[q] = _no_three_way_deviance(tab, ka, kb, kt, iters)
    return out


def _all_pairs(p: int) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays of every unordered column pair ``(i < j)`` of ``p`` columns."""
    ia, ib = np.triu_indices(p, k=1)
    return ia.astype(np.int64), ib.astype(np.int64)


def _critical_deviance(df: np.ndarray, alpha: float) -> np.ndarray:
    """Chi-square upper-tail critical value of each ``df`` at level ``alpha`` (one scipy call per distinct ``df``)."""
    from scipy.stats import chi2

    crit = np.zeros(df.shape[0])
    for d in np.unique(df):
        if d > 0:
            crit[df == d] = chi2.isf(alpha, d)
    return crit


def detect_synergy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_rows: int = 4000,
    max_features: int = 300,
    max_pairs: int = 50000,
    n_null: int = 3,
    nbins: int = 8,
    random_seed: int = 0,
) -> tuple[bool, dict]:
    """Cheap pre-fit probe: is ``(X, y)`` synergistic enough to warrant the JMIM aggregator?

    Returns ``(is_synergistic, info)``. Every pair of up to ``max_features`` columns (``max_pairs`` caps the pair count; the default covers all
    pairs of 300 columns) is scored by the no-three-way-interaction deviance on a ``max_rows`` row subsample, see the module docstring. ``info``
    keeps the keys earlier callers read: ``real_excess`` (best pair's bias-corrected interaction information in nats), ``threshold`` (the larger
    of the pair's chi-square critical effect at the Bonferroni level and the entropy-share floor), ``null_excess`` (the chi-square critical
    effect of that pair, the null maximum), ``n_pairs``, ``n_features``; plus ``best_pair`` (column indices in ``X``), ``target_entropy`` and
    ``effect_floor``. ``n_null`` is accepted for API compatibility and unused: the null is analytic.

    Caveats: columns beyond ``max_features`` are dropped by random sampling, so a planted pair is found with probability ``(max_features/p)^2``
    there; the tables use equal-frequency bins, so a synergy living inside one bin is invisible."""
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] < 2 or X.shape[0] < 50:
        return False, {"reason": "too_small"}
    rng = np.random.default_rng(int(random_seed))
    n, p = X.shape

    if n > max_rows:
        ridx = rng.choice(n, size=max_rows, replace=False)
        Xs, ys = X[ridx], np.asarray(y)[ridx]
    else:
        Xs, ys = X, np.asarray(y)

    cols = np.arange(p)
    if p > max_features:
        cols = np.sort(rng.choice(p, size=max_features, replace=False))
    pp = cols.shape[0]

    yc = _quantize(ys, nbins, rng)
    if np.unique(yc).size < 2:
        return False, {"reason": "degenerate_target"}
    kt = int(yc.max()) + 1
    ns = Xs.shape[0]
    ycount = np.bincount(yc, minlength=kt).astype(np.float64)
    py = ycount[ycount > 0] / ns
    h_y = float(-(py * np.log(py)).sum())

    # Cap the per-pair bin count so a (kb x kb x kt) table averages >= _MIN_ROWS_PER_CELL rows per cell.
    pair_bins = int(max(3, min(nbins, np.floor(np.sqrt(ns / (_MIN_ROWS_PER_CELL * kt))))))
    codes = np.empty((pp, ns), dtype=np.int8)
    levels = np.empty(pp, dtype=np.int64)
    for k, j in enumerate(cols.tolist()):
        c = _quantize(Xs[:, j], pair_bins, rng)
        codes[k] = c
        levels[k] = int(c.max()) + 1
    yc = np.ascontiguousarray(yc, dtype=np.int64)

    pa, pb = _all_pairs(pp)
    if pa.shape[0] > max_pairs:
        sel = np.sort(rng.choice(pa.shape[0], size=int(max_pairs), replace=False))
        pa, pb = pa[sel], pb[sel]
    n_pairs = int(pa.shape[0])
    g = _pair_deviances(codes, levels, yc, kt, pa, pb, _IPF_ITERS)
    df = ((levels[pa] - 1) * (levels[pb] - 1) * (kt - 1)).astype(np.float64)
    valid = df > 0
    if not valid.any():
        return False, {"reason": "degenerate_features"}

    crit = _critical_deviance(df, _ALPHA / n_pairs)
    floor = _MIN_SYNERGY_SHARE * h_y
    eff = (g - df) / (2.0 * ns)
    thr = np.maximum(floor, (crit - df) / (2.0 * ns))
    margin = np.where(valid, eff - thr, -np.inf)
    k = int(np.argmax(margin))
    is_syn = bool(margin[k] > 0.0)
    return is_syn, {
        "real_excess": float(eff[k]),
        "null_excess": float((crit[k] - df[k]) / (2.0 * ns)),
        "threshold": float(thr[k]),
        "effect_floor": float(floor),
        "target_entropy": h_y,
        "best_pair": (int(cols[pa[k]]), int(cols[pb[k]])),
        "n_pairs": n_pairs,
        "n_features": int(pp),
    }


__all__ = ["detect_synergy"]
