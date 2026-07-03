"""Tail-concentrated LINEAR-usability signal shared by the MRMR FE gates.

Under heavy operand outliers a genuine ratio (``a**2/b``) becomes TAIL-CONCENTRATED: in the clean bulk its
rank association with y collapses (Spearman ~0) so EVERY rank-MI / CMI statistic under-credits it, yet it
carries strong LINEAR usability (``|corr(continuous y)|`` ~0.99, outlier-inflated -- which is exactly right
for a tail signal) that a spurious high-rank-MI form does not (~0.37). The rank gates cannot, BY
CONSTRUCTION, separate a tail-concentrated-true (low rank-MI, high ``|corr|``) from a spurious (high rank-MI,
low ``|corr|``) pair. These leaf helpers (numpy only -> importable from both ``_mrmr_fe_step`` and
``_feature_engineering_pairs`` with no cycle) compute the RAW-operand ``|corr|`` distinguisher and the
tail-concentration predicate the three FE gates (prospective-pair admission, per-pair winner selection,
engineered-MI prevalence) consult so the credit is CONSISTENT across all three.

Everything is best-effort: the callers wrap use in try/except so any failure leaves the strict rank-MI
decision untouched (canonical fixtures never loosened on error).
"""
from __future__ import annotations

import os

import numpy as np


def usability_operand_continuous(self, X, cols, var_idx):
    """RAW CONTINUOUS values (outliers INTACT) for a cols-space operand index, or None if unresolvable.

    Usability is scored on the raw operands, NOT the binned codes: binning clips the outlier tail into the
    top code, destroying the very magnitude that carries a tail-concentrated ratio's signal (why a binned-code
    OLS proxy cannot recover it). A raw feature resolves by name from ``X`` (the full-n input frame / array,
    row-aligned with the binned matrix and the continuous y); an ENGINEERED operand (no raw position) returns
    None, so usability credit is claimed only for raw-operand pairs."""
    try:
        if cols is None or X is None:
            return None
        _i = int(var_idx)
        if not (0 <= _i < len(cols)):
            return None
        _nm = cols[_i]
        if hasattr(X, "columns"):
            if _nm in getattr(X, "columns", []):
                return np.asarray(X[_nm], dtype=np.float64).ravel()
            return None
        _names = list(getattr(self, "feature_names_in_", []) or [])
        if _nm in _names:
            return np.asarray(X[:, _names.index(_nm)], dtype=np.float64).ravel()
        return None
    except Exception:
        return None


# Row cap for the usability |corr| estimate. Pearson |corr| is a population statistic a large deterministic
# subsample estimates to ~1e-3, while every consumer compares it against WIDE-margin thresholds (min_corr
# 0.6; the tail-concentration gap is ~0.99 vs ~0.06). A STRIDED subsample preserves the outlier proportion
# (hence the outlier-inflated |corr| the tail-concentration signal depends on), so the keep/reject decisions
# stay selection-equivalent while the per-call cost drops ~n/cap (abs_pearson runs ~8 full-n passes and is
# called a few hundred times per fit). Env-tunable; 0 -> full-n. 250k is already sub-1e-3 for the corr.
try:
    _ABS_PEARSON_MAX_ROWS = int(os.environ.get("MLFRAME_USABILITY_CORR_MAX_ROWS", "250000"))
except (ValueError, TypeError):
    _ABS_PEARSON_MAX_ROWS = 250000


def abs_pearson(y, v):
    """``|Pearson corr|`` of ``y`` vs ``v`` over jointly-FINITE rows; 0.0 when <2 valid rows or either is
    constant / non-finite. Outlier-inflated by construction -- which is exactly the tail-concentrated signal
    the coarse rank-MI under-credits. Above ``_ABS_PEARSON_MAX_ROWS`` a deterministic strided subsample is
    used (outlier-proportion preserving, selection-equivalent -- see the constant note)."""
    y = np.asarray(y)
    v = np.asarray(v)
    if _ABS_PEARSON_MAX_ROWS > 0 and y.shape[0] > _ABS_PEARSON_MAX_ROWS:
        _stride = int(y.shape[0] // _ABS_PEARSON_MAX_ROWS)
        if _stride > 1:
            y = y[::_stride]
            v = v[::_stride]
    m = np.isfinite(y) & np.isfinite(v)
    if int(m.sum()) < 2:
        return 0.0
    yy = y[m]
    vv = v[m]
    ys = float(yy.std())
    vs = float(vv.std())
    if ys <= 0.0 or vs <= 0.0:
        return 0.0
    c = float(np.mean((yy - yy.mean()) * (vv - vv.mean())) / (ys * vs))
    return abs(c) if np.isfinite(c) else 0.0


def usability_form_corrs(y, x0, x1):
    """Return (best PAIR-form ``|corr(y)|``, best SINGLE-operand ``|corr(y)|``) over a small scale/sign-robust
    bivariate dictionary of the RAW operands. Pair forms: the two ratio orderings, the two squared-numerator
    ratios, and the product -- the tail-concentrated ratio ``a**2/b`` lands here (``|corr|`` ~0.986 vs y).
    Single forms: each operand and its square. Ratios use a tiny denominator floor so a near-zero-divisor row
    becomes NaN and is dropped by the finite-mask in ``abs_pearson`` (no spurious inf). The PAIR-vs-SINGLE
    margin is the PAIRNESS discriminator: a GENUINE interaction (the true denominator IMPROVES corr over the
    numerator alone) beats its best single operand, while a cross-mix / noise pair where one operand dominates
    (dividing by an unrelated operand only ADDS noise) does not -- so cross pairs and a pure-noise operand are
    rejected without ever measuring them against a canonical baseline."""
    _eps = 1e-12
    _y = np.asarray(y, dtype=np.float64).ravel()
    _x0 = np.asarray(x0, dtype=np.float64).ravel()
    _x1 = np.asarray(x1, dtype=np.float64).ravel()

    def _safe_div(n, d):
        dd = np.where(np.abs(d) < _eps, np.nan, d)
        return n / dd

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        _pair_forms = [
            _safe_div(_x0, _x1), _safe_div(_x1, _x0),
            _safe_div(_x0 * _x0, _x1), _safe_div(_x1 * _x1, _x0),
            _x0 * _x1,
        ]
        _single_forms = [_x0, _x1, _x0 * _x0, _x1 * _x1]
        _cp = max((abs_pearson(_y, f) for f in _pair_forms), default=0.0)
        _cs = max((abs_pearson(_y, f) for f in _single_forms), default=0.0)
    return _cp, _cs


def pair_is_tail_concentrated(y, x0, x1, *, min_corr, pairness_margin):
    """True when the RAW pair (``x0``,``x1``) is tail-concentrated w.r.t. continuous ``y``: its best bivariate
    form is strongly linearly usable (``|corr|`` >= ``min_corr``) AND beats the best single-operand form by
    ``pairness_margin`` (genuine pairness). The rank-vs-linear DISAGREEMENT is supplied by the CALL SITE (this
    predicate is consulted only where the rank-MI path has already rejected / under-ranked the pair), so this
    returns the LINEAR half of the detector. FALSE for pure-noise pairs (best-form ``|corr|`` ~0.02-0.2), for a
    pair whose signal is a lone dominant operand (fails the margin), and for the 4 passing F2 profiles /
    canonical fixtures (there ``a**2/b`` is BOTH the rank-MI and usability leader, so the rank path admits it
    and this predicate is never reached). Best-effort: any error -> False (strict rank-MI decision stands)."""
    try:
        _cp, _cs = usability_form_corrs(y, x0, x1)
        return bool(_cp >= float(min_corr) and _cp >= float(pairness_margin) * float(_cs))
    except Exception:
        return False


def _rank_transform(v):
    """Ordinal ranks of ``v`` (argsort-of-argsort). Ties are broken by position (stable) -- adequate for a
    monotone-association proxy; a Pearson corr over these ranks is a Spearman-style rank correlation."""
    order = np.argsort(v, kind="stable")
    ranks = np.empty(order.shape[0], dtype=np.float64)
    ranks[order] = np.arange(order.shape[0], dtype=np.float64)
    return ranks


def pair_is_tail_concentrated_rankaware(y, x0, x1, *, min_corr, pairness_margin, max_rank_frac=0.7):
    """RANK-AWARE tail-concentration predicate for the FIRST-SWEEP prevalence relaxation pre-scan.

    Fires only when the pair is (1) linearly usable + genuinely pairwise (``pair_is_tail_concentrated``) AND
    (2) its linear-best bivariate form's RANK (Spearman-style) association with y COLLAPSES relative to its
    linear ``|corr|`` -- ``rank_corr <= max_rank_frac * linear_corr``. That second leg is the tail-
    concentration SIGNATURE: under outliers ``a**2/b`` tracks y LINEARLY (outlier-inflated |corr| ~0.99) while
    its RANK association is near-zero (binned-MI ~0.06). On the 4 BALANCED F2 profiles + canonical fixtures a
    genuinely usable ratio tracks y in BOTH rank and linear (they AGREE), so the rank leg FAILS and this
    returns False -> the pre-scan does NOT relax the prevalence bar there (byte-identical). Best-effort:
    any error -> False."""
    try:
        _y = np.asarray(y, dtype=np.float64).ravel()
        _x0 = np.asarray(x0, dtype=np.float64).ravel()
        _x1 = np.asarray(x1, dtype=np.float64).ravel()
        _cp, _cs = usability_form_corrs(_y, _x0, _x1)
        if not (_cp >= float(min_corr) and _cp >= float(pairness_margin) * float(_cs)):
            return False
        _eps = 1e-12

        def _safe_div(n, d):
            dd = np.where(np.abs(d) < _eps, np.nan, d)
            return n / dd

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            _forms = [
                _safe_div(_x0, _x1), _safe_div(_x1, _x0),
                _safe_div(_x0 * _x0, _x1), _safe_div(_x1 * _x1, _x0),
                _x0 * _x1,
            ]
        _best_form, _best_lin = None, -1.0
        for _f in _forms:
            _a = abs_pearson(_y, _f)
            if _a > _best_lin:
                _best_lin, _best_form = _a, _f
        if _best_form is None:
            return False
        _m = np.isfinite(_best_form) & np.isfinite(_y)
        if int(_m.sum()) < 3:
            return False
        _rank_corr = abs_pearson(_rank_transform(_y[_m]), _rank_transform(np.asarray(_best_form)[_m]))
        return bool(_rank_corr <= float(max_rank_frac) * _best_lin)
    except Exception:
        return False


def tail_concentration_form_override(perf, usability, *, min_corr, pairness_margin, mi_band, best_single_corr=0.0):
    """Return the FORM (a ``perf`` key) to PROMOTE when a per-pair form set is tail-concentrated, else None.

    ``perf`` is ``{form_config: rank_mi}``; ``usability`` is ``{form_config: |corr(continuous y)|}``. Override
    fires only on a rank-vs-linear DISAGREEMENT a plain tie-band cannot bridge: the rank-MI leader is NOT the
    ``|corr|`` leader, the rank-MI gap between them EXCEEDS the Miller-Madow tie band (so it is a REAL rank-MI
    lead, not the binning noise the existing usability tie-break already resolves), the ``|corr|``-leader form
    is strongly usable (``|corr|`` >= ``min_corr``), and it beats ``best_single_corr`` by ``pairness_margin``.
    Under those conditions the coarse rank-MI is provably the wrong arbiter (the true signal is
    tail-concentrated) so the ``|corr|``-leader form is promoted. Returns None -- leaving the rank-MI winner
    untouched -- whenever ANY condition fails, so canonical fixtures + the 4 passing profiles (rank leader ==
    ``|corr|`` leader, or the gap is within the band) are byte-identical."""
    try:
        if not perf or not usability:
            return None
        _mi_leader = max(perf.items(), key=lambda kv: float(kv[1]))[0]
        _corr_leader, _corr_val = max(
            ((k, float(usability.get(k, 0.0))) for k in perf.keys()), key=lambda kv: kv[1]
        )
        if _corr_leader == _mi_leader:
            return None
        _band = float(mi_band) if (mi_band and mi_band > 0.0) else 0.0
        if (float(perf[_mi_leader]) - float(perf.get(_corr_leader, 0.0))) <= _band:
            return None
        if _corr_val < float(min_corr):
            return None
        if _corr_val < float(pairness_margin) * float(best_single_corr):
            return None
        return _corr_leader
    except Exception:
        return None
