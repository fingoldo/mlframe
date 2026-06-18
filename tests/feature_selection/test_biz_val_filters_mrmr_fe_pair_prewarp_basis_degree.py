"""biz_value: ``fe_pair_prewarp_basis`` (choice) and ``fe_pair_prewarp_max_degree`` (lever) for the per-operand
learned pre-warp on the unary/binary pair path.

``test_biz_value_mrmr_pair_prewarp.py`` pins the prewarp's recovery + uplift gate at the DEFAULT basis/degree
(chebyshev / 4). It does not exercise the two choice knobs that select the warp's polynomial family + capacity, both
consumed in ``_mrmr_fe_step/_step_core.py`` (``_prewarp_basis`` / ``_prewarp_max_degree``). This file pins them:

  A. BASIS choice: the prewarp recovers the non-monotone separable inner ``(a**3-2a)*(b**2-b)`` under the non-default
     ``hermite`` AND ``legendre`` families (each spans the same degree<=d space), so a regression in a basis builder/eval
     is caught by the FAILED recovery -- measured |corr| ~1.0, pinned >=0.85.
  B. DEGREE lever: ``fe_pair_prewarp_max_degree`` controls the warp's representational capacity. A degree-2 warp cannot
     represent the cubic operand ``a**3-2a``; a degree>=3 warp can. The degree>=4 fit strictly beats degree=2 -- the
     DELTA is the knob's measurable win (degree 2 |corr| ~0.68, degree 4 ~1.0).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


N = 4000
RAW = {"a", "b", "c", "e"}
_LEAN = dict(dcd_enable=False, build_friend_graph=False, cluster_aggregate_enable=False)


def _make_poly(seed: int = 202, n: int = N):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2.5, 2.5, n)
    b = rng.uniform(-2.5, 2.5, n)
    c = rng.normal(0, 1, n)
    e = rng.normal(0, 1, n)
    true = (a**3 - 2 * a) * (b**2 - b)
    y = true + rng.normal(0, 0.05 * np.std(true), n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "e": e}), pd.Series(y, name="y"), true


def _prewarp_mrmr(basis="chebyshev", max_degree=4):
    return MRMR(verbose=0, n_jobs=1, random_seed=0,
                fe_smart_polynom_iters=0, fe_hybrid_orth_enable=False,
                fe_pair_prewarp_enable=True, fe_pair_prewarp_basis=basis,
                fe_pair_prewarp_max_degree=max_degree, **_LEAN)


def _best_engineered_corr(fs, df, true):
    names = list(fs.get_feature_names_out())
    eng = [nm for nm in names if nm not in RAW]
    if not eng:
        return 0.0
    Xt = np.asarray(fs.transform(df))
    best = 0.0
    for i, nm in enumerate(names):
        if nm not in eng:
            continue
        col = Xt[:, i]
        if not np.isfinite(col).all() or float(np.std(col)) < 1e-12:
            continue
        best = max(best, abs(float(np.corrcoef(col, true)[0, 1])))
    return best


def _fit(make):
    MRMR.clear_fit_cache()
    df, y, true = _make_poly()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = make().fit(df, y)
    return fs, df, true


@pytest.mark.parametrize("basis", ["hermite", "legendre"])
def test_prewarp_basis_choice_recovers_non_monotone_inner(basis):
    fs, df, true = _fit(lambda: _prewarp_mrmr(basis=basis, max_degree=4))
    corr = _best_engineered_corr(fs, df, true)
    assert corr >= 0.85, f"prewarp basis={basis} failed to recover F-POLY inner: |corr|={corr:.3f}"


def test_prewarp_max_degree_is_a_lever_high_degree_beats_low():
    fs_hi, df, true = _fit(lambda: _prewarp_mrmr(basis="chebyshev", max_degree=4))
    fs_lo, _, _ = _fit(lambda: _prewarp_mrmr(basis="chebyshev", max_degree=2))
    corr_hi = _best_engineered_corr(fs_hi, df, true)
    corr_lo = _best_engineered_corr(fs_lo, df, true)
    # degree 4 recovers the cubic operand; degree 2 cannot -> strict DELTA.
    assert corr_hi >= 0.85, f"degree-4 prewarp should recover F-POLY: |corr|={corr_hi:.3f}"
    assert corr_hi >= corr_lo + 0.15, (
        f"max_degree is not a lever: deg4 |corr|={corr_hi:.3f} vs deg2 |corr|={corr_lo:.3f}"
    )
