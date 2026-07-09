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

import numba
import numpy as np


@numba.njit(cache=True, fastmath={"reassoc", "contract", "arcp", "afn", "nsz"})
def _abs_pearson_njit(y, v):
    """One-pass ``|Pearson corr|`` over jointly-finite rows: accumulate sums/sum-sq/cross in a single walk, no
    isfinite-mask + boolean-index copies + two np.std + a mean temp. FP-equivalent to the numpy form to ~1e-15
    (selection-safe for the wide usability |corr| thresholds). 0.0 on <2 valid rows or a (near-)constant side.

    Accepts f32 OR f64 arrays: each value is promoted to float64 BEFORE the squares/products, so the arithmetic (and the
    result) is bit-identical whatever the input dtype -- letting the caller pass an f32 array WITHOUT a full-length f64
    copy, while the sum-of-squares still runs in f64 (no catastrophic cancellation on a large-mean column).

    Perf (2026-07): BRANCHLESS + REASSOC-fastmath accumulation -> 2.1-2.5x over the plain-branch fastmath=False form at
    n=600..30000 (bench ``_benchmarks/bench_abs_pearson_fastmath.py``). The per-row ``if isfinite: accumulate`` control
    flow blocked SIMD vectorisation; replacing it with a select (``av = a if finite else 0.0``; a non-finite row
    contributes 0 to every sum and 0 to ``n``) removes the branch so the reduction vectorises, and the ``reassoc``
    fastmath flag lets LLVM tree-reduce the sums. The NaN-drop semantics are PRESERVED EXACTLY (a NaN/inf operand
    zeroes its own row's contribution and is excluded from ``n`` -- verified vs the numpy masked-pearson reference on
    NaN-injected data), because ``nnan``/``ninf`` are DELIBERATELY EXCLUDED from the fastmath set: a full
    ``fastmath=True`` here would let LLVM assume no-NaN and drop the ``isfinite`` test, silently admitting the NaN rows
    the ratio forms produce and CORRUPTING the |corr| (a selection-BREAKING ~1e-2 error, not the ~1e-16 reassoc ULP
    delta this safe set gives). Result diverges from the old order by <=~1e-16 (single ULP), selection-equivalent under
    the wide usability thresholds (min_corr 0.6; tail-concentration gap ~0.99 vs ~0.06)."""
    n = 0
    sa = 0.0; sv = 0.0; saa = 0.0; svv = 0.0; sav = 0.0
    for i in range(y.shape[0]):
        a = np.float64(y[i]); b = np.float64(v[i])
        finite = np.isfinite(a) and np.isfinite(b)
        # Branchless: a non-finite row contributes 0 to every accumulator and 0 to ``n`` (exact row-drop, SIMD-friendly).
        av = a if finite else 0.0
        bv = b if finite else 0.0
        n += finite
        sa += av; sv += bv; saa += av * av; svv += bv * bv; sav += av * bv
    if n < 2:
        return 0.0
    inv = 1.0 / n
    va = saa - sa * sa * inv
    vv2 = svv - sv * sv * inv
    if va <= 0.0 or vv2 <= 0.0:
        return 0.0
    den = (va * vv2) ** 0.5
    if den <= 0.0:
        return 0.0
    c = (sav - sa * sv * inv) / den
    if not np.isfinite(c):
        return 0.0
    return -c if c < 0.0 else c


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
                return np.asarray(X[_nm], dtype=_crit_np_dtype()).ravel()
            return None
        # feature_names_in_ is an ndarray; "or []" would test truthiness and raise on a multi-element array.
        _names = list(getattr(self, "feature_names_in_", []))
        if _nm in _names:
            return np.asarray(X[:, _names.index(_nm)], dtype=_crit_np_dtype()).ravel()
        return None
    except Exception:
        return None


# Row cap for the usability |corr| estimate. Pearson |corr| is a population statistic a deterministic subsample
# estimates to ~1e-3 (30k -> ~3e-3), while every consumer compares it against WIDE-margin thresholds (min_corr
# 0.6; the tail-concentration gap is ~0.99 vs ~0.06). A STRIDED subsample preserves the outlier proportion
# (hence the outlier-inflated |corr| the tail-concentration signal depends on), so the keep/reject decisions
# stay selection-equivalent while the per-call cost drops ~n/cap. This is HOT: profiling a 100k FE fit showed
# pair_is_tail_concentrated_rankaware -> usability_form_corrs at ~85k calls x ~9 corr passes = ~15% of the run,
# and at 100k the old 250k cap did NOT fire (full-n passes). 30k matches the FE-decision subsample
# (UNIFIED_FE_SUBSAMPLE_N) -- the usability corr needs no more rows than the decision it feeds; measured 3.46x
# on the corr pass at 100k (8x at 1M) with the canonical FE selection pins unchanged. Env-tunable; 0 -> full-n.
try:
    _ABS_PEARSON_MAX_ROWS = int(os.environ.get("MLFRAME_USABILITY_CORR_MAX_ROWS", "30000"))
except (ValueError, TypeError):
    _ABS_PEARSON_MAX_ROWS = 30000


def _crit_np_dtype():
    """np.float32 when the precision-critical relaxation is on (``MLFRAME_CRIT_DTYPE_RELAXED``, DEFAULT ON),
    else np.float64 -- the HOST/numpy mirror of ``_fe_gpu_batch._devices.crit_float_dtype`` (read directly here
    to keep this a numpy-only leaf module with no cupy import). The usability |corr| feeds WIDE-margin
    threshold decisions (min_corr 0.6; the tail-concentration gap is ~0.99 vs ~0.06), so operand casts follow
    the SAME f32 relaxation the discretiser-input / fourier-detrend / prewarp hotspots already use; set the env
    to 0 to force strict f64 everywhere (matching the rest of MRMR)."""
    if os.environ.get("MLFRAME_CRIT_DTYPE_RELAXED", "1").strip().lower() in ("0", "false", "off", "no"):
        return np.float64
    return np.float32


def _corr_stride(n_rows: int) -> int:
    """Row stride for the usability-|corr| subsample (1 = no subsample). See ``_ABS_PEARSON_MAX_ROWS``."""
    if _ABS_PEARSON_MAX_ROWS > 0 and n_rows > _ABS_PEARSON_MAX_ROWS:
        return max(1, int(n_rows // _ABS_PEARSON_MAX_ROWS))
    return 1


def _subsample_for_corr(*arrs):
    """Strided-subsample 1-D arrays TOGETHER (same rows) to the |corr| row cap. Applied once at the top of
    the usability helpers so the ratio-form materialisation AND the ranks AND the corr all run on the
    subsample -- not just the final corr. Idempotent: arrays already at/below the cap are returned as-is
    (so a caller that already subsampled -> ``usability_form_corrs`` does NOT subsample twice)."""
    a0 = arrs[0]
    st = _corr_stride(a0.shape[0])
    if st <= 1:
        return arrs
    return tuple(a[::st] for a in arrs)


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
    # One-pass njit finite-masked |corr| (no mask + boolean-index copies + two np.std + a mean temp); ~13x over the numpy
    # form. The kernel promotes each value to f64 internally, so pass the arrays in their NATIVE dtype: when both are f32
    # (the common _crit_np_dtype path) this skips the full-length f64 copy the old code always paid -- bit-identical
    # result, half the operand memory, and no wasted O(n) copy. Only upcast (copy) when the two dtypes differ.
    _dt = np.float32 if (y.dtype == np.float32 and v.dtype == np.float32) else np.float64
    return float(_abs_pearson_njit(np.ascontiguousarray(y, _dt), np.ascontiguousarray(v, _dt)))


def usability_form_corrs(y, x0, x1, *, return_best_pair_form=False):
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
    _y = np.asarray(y, dtype=_crit_np_dtype()).ravel()
    _x0 = np.asarray(x0, dtype=_crit_np_dtype()).ravel()
    _x1 = np.asarray(x1, dtype=_crit_np_dtype()).ravel()
    _y, _x0, _x1 = _subsample_for_corr(_y, _x0, _x1)  # forms + corr run on the subsample (not just abs_pearson)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # Reuse the squares (single AND pair numerators) and the floored denominators (each used by two ratio
        # forms) instead of recomputing them per form -- bit-identical, just fewer elementwise passes per call.
        _x0sq = _x0 * _x0
        _x1sq = _x1 * _x1
        _x0f = np.where(np.abs(_x0) < _eps, np.nan, _x0)
        _x1f = np.where(np.abs(_x1) < _eps, np.nan, _x1)
        _pair_forms = [
            _x0 / _x1f, _x1 / _x0f,
            _x0sq / _x1f, _x1sq / _x0f,
            _x0 * _x1,
        ]
        _single_forms = [_x0, _x1, _x0sq, _x1sq]
        _cs = max((abs_pearson(_y, f) for f in _single_forms), default=0.0)
        if return_best_pair_form:
            # Track WHICH pair form is the |corr| leader so the tail-concentration caller reuses it instead of
            # rebuilding + re-abs_pearson-ing the identical 5 forms. First-on-tie via strict '>' matches the caller's
            # own prior loop, so _cp and the chosen form are bit-identical to the recompute path.
            _best_form, _best_cp = None, -1.0
            for _f in _pair_forms:
                _a = abs_pearson(_y, _f)
                if _a > _best_cp:
                    _best_cp, _best_form = _a, _f
            _cp = _best_cp if _best_form is not None else 0.0
            return _cp, _cs, _best_form
        _cp = max((abs_pearson(_y, f) for f in _pair_forms), default=0.0)
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
        _y = np.asarray(y, dtype=_crit_np_dtype()).ravel()
        _x0 = np.asarray(x0, dtype=_crit_np_dtype()).ravel()
        _x1 = np.asarray(x1, dtype=_crit_np_dtype()).ravel()
        _y, _x0, _x1 = _subsample_for_corr(_y, _x0, _x1)  # subsample once; the ranks + forms below reuse it
        # Reuse the |corr|-leader pair form from usability_form_corrs instead of rebuilding + re-abs_pearson-ing the
        # identical 5 forms (this predicate is ~15% of a wide FE fit at ~85k calls; the recompute was pure waste).
        _cp, _cs, _best_form = usability_form_corrs(_y, _x0, _x1, return_best_pair_form=True)
        if not (_cp >= float(min_corr) and _cp >= float(pairness_margin) * float(_cs)):
            return False
        _best_lin = _cp
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
        _corr_leader, _corr_val = max(((k, float(usability.get(k, 0.0))) for k in perf.keys()), key=lambda kv: kv[1])
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
