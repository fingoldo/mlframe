"""Regression: pair_is_tail_concentrated_rankaware reuses the |corr|-leader pair form from usability_form_corrs
instead of rebuilding + re-abs_pearson-ing the identical 5 forms. The predicate is ~15% of a wide FE fit (~85k
calls), and the recompute was pure waste. This pins that the dedup is BIT-IDENTICAL to the pre-dedup recompute
across tail-concentrated / clean-product / noise / ratio pairs (the FE bar is selection-equivalence)."""

import numpy as np

from mlframe.feature_selection.filters._fe_usability_signal import (
    pair_is_tail_concentrated_rankaware as NEW,
    usability_form_corrs,
    abs_pearson,
    _subsample_for_corr,
    _rank_transform,
    _crit_np_dtype,
)


def _reference(y, x0, x1, *, min_corr, pairness_margin, max_rank_frac=0.7):
    """Pre-dedup logic: rebuild the 5 forms + re-abs_pearson to find the best form (what the caller used to do)."""
    try:
        _y = np.asarray(y, dtype=_crit_np_dtype()).ravel()
        _x0 = np.asarray(x0, dtype=_crit_np_dtype()).ravel()
        _x1 = np.asarray(x1, dtype=_crit_np_dtype()).ravel()
        _y, _x0, _x1 = _subsample_for_corr(_y, _x0, _x1)
        _cp, _cs = usability_form_corrs(_y, _x0, _x1)
        if not (_cp >= float(min_corr) and _cp >= float(pairness_margin) * float(_cs)):
            return False

        def _sd(n, d):
            """Safe divide n/d, mapping near-zero denominators to NaN instead of inf."""
            return n / np.where(np.abs(d) < 1e-12, np.nan, d)

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            forms = [_sd(_x0, _x1), _sd(_x1, _x0), _sd(_x0 * _x0, _x1), _sd(_x1 * _x1, _x0), _x0 * _x1]
        best, best_lin = None, -1.0
        for f in forms:
            a = abs_pearson(_y, f)
            if a > best_lin:
                best_lin, best = a, f
        if best is None:
            return False
        m = np.isfinite(best) & np.isfinite(_y)
        if int(m.sum()) < 3:
            return False
        rc = abs_pearson(_rank_transform(_y[m]), _rank_transform(np.asarray(best)[m]))
        return bool(rc <= float(max_rank_frac) * best_lin)
    except Exception:
        return False


def test_tail_concentration_dedup_is_bit_identical():
    """Tail concentration dedup is bit identical."""
    rng = np.random.default_rng(0)
    kw = dict(min_corr=0.6, pairness_margin=1.05)
    mism = 0
    for i in range(240):
        n = int(rng.integers(200, 3000))
        x0 = rng.standard_normal(n).astype(np.float32)
        x1 = (rng.standard_normal(n) + 0.1).astype(np.float32)
        kind = i % 4
        if kind == 0:
            y = (x0 * x0 / np.where(np.abs(x1) < 1e-3, np.nan, x1)).astype(np.float32)
            y[rng.integers(0, n, 5)] *= 50
        elif kind == 1:
            y = (x0 * x1).astype(np.float32)
        elif kind == 2:
            y = rng.standard_normal(n).astype(np.float32)
        else:
            y = (x0 / np.where(np.abs(x1) < 1e-3, np.nan, x1)).astype(np.float32)
        y = np.nan_to_num(y)
        if NEW(y, x0, x1, **kw) != _reference(y, x0, x1, **kw):
            mism += 1
    assert mism == 0, f"dedup diverged from the recompute reference on {mism}/240 pairs"


def test_usability_form_corrs_best_form_matches_scalar_leader():
    """The 3-tuple (return_best_pair_form) path returns the SAME _cp/_cs as the 2-tuple path, plus a non-None best form."""
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal(1500).astype(np.float32)
    x1 = (rng.standard_normal(1500) + 0.2).astype(np.float32)
    y = (x0 * x0 / np.where(np.abs(x1) < 1e-3, np.nan, x1)).astype(np.float32)
    cp2, cs2 = usability_form_corrs(y, x0, x1)
    cp3, cs3, best = usability_form_corrs(y, x0, x1, return_best_pair_form=True)
    assert cp3 == cp2 and cs3 == cs2, "returning the best form must not change the scalar corrs"
    assert best is not None and best.ndim == 1
