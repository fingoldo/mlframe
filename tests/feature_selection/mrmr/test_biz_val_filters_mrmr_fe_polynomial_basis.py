"""biz_value: ``fe_polynomial_basis`` CHOICE-VALUE coverage for the smart_polynom orthogonal-poly FE path.

``fe_polynomial_basis`` selects which orthogonal-polynomial family the smart_polynom pair optimiser fits when it
reconstructs a separable non-monotone inner ``y = P(a) * Q(b)``. The four wired values (see ``hermite_fe._BASIS_FUNCS``:
chebyshev / hermite / legendre / laguerre) span the same degree<=d polynomial space, so each should reconstruct a
low-degree separable inner that the elementary unary/binary search structurally cannot represent.

The existing pre-distortion suite parametrises only ``["chebyshev", "hermite"]``; this file pins the remaining wired
choices ``legendre`` and ``laguerre`` (laguerre is the documented basis for skewed-positive inputs) so a regression that
breaks a basis's build/eval/warm-start is caught by the FAILED recovery, not just by an interface check. Each basis is
asserted to recover a separable cubic*quadratic inner AND to strictly beat the elementary unary/binary search (which
cannot represent the non-monotone inner) on the same fixture -- the DELTA is the win, per the biz_value contract.
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


def _orth_smart_polynom(basis):
    return MRMR(verbose=0, n_jobs=1, random_seed=0,
                fe_smart_polynom_iters=20, fe_smart_polynom_optimization_steps=400,
                fe_polynomial_basis=basis, fe_optimizer="cma_batch",
                fe_hybrid_orth_enable=False, fe_auto_escalation_enable=False, **_LEAN)


def _unb():
    return MRMR(verbose=0, n_jobs=1, random_seed=0,
                fe_smart_polynom_iters=0, fe_hybrid_orth_enable=False,
                fe_pair_prewarp_enable=False, fe_auto_escalation_enable=False, **_LEAN)


def _best_engineered_corr(fs, df, true):
    names = list(fs.get_feature_names_out())
    eng = [nm for nm in names if nm not in RAW]
    if not eng:
        return None, 0.0
    Xt = np.asarray(fs.transform(df))
    best = (None, 0.0)
    for i, nm in enumerate(names):
        if nm not in eng:
            continue
        col = Xt[:, i]
        if not np.isfinite(col).all() or float(np.std(col)) < 1e-12:
            continue
        r = abs(float(np.corrcoef(col, true)[0, 1]))
        if r > best[1]:
            best = (nm, r)
    return best


def _fit(make_mrmr, df, y):
    MRMR.clear_fit_cache()
    fs = make_mrmr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs.fit(df, y)
    return fs


@pytest.mark.parametrize("basis", ["legendre", "laguerre"])
def test_basis_choice_recovers_separable_polynomial_inner(basis):
    """Each wired orthogonal-poly basis reconstructs the separable cubic*quadratic inner the elementary search cannot.

    Measured chebyshev/hermite recover at |corr| ~0.97 on this F-POLY fixture; pin >=0.80 with wide margin so a basis
    build/eval/warm-start regression trips while seed noise does not.
    """
    df, y, true = _make_poly()
    fs = _fit(lambda: _orth_smart_polynom(basis), df, y)
    name, corr = _best_engineered_corr(fs, df, true)
    assert name is not None, f"basis={basis} engineered nothing on separable-poly inner"
    assert corr >= 0.80, f"basis={basis} best engineered |corr|={corr:.3f} < 0.80 ({name})"


@pytest.mark.parametrize("basis", ["legendre", "laguerre"])
def test_basis_choice_beats_elementary_unary_binary(basis):
    """The orthogonal-poly basis strictly beats the elementary unary/binary search, which structurally cannot represent
    the non-monotone inner ``P(a)*Q(b)`` (no library unary equals ``a**3-2a``). The DELTA is the basis's measurable win."""
    df, y, true = _make_poly()
    fs_orth = _fit(lambda: _orth_smart_polynom(basis), df, y)
    fs_unb = _fit(_unb, df, y)
    _, corr_orth = _best_engineered_corr(fs_orth, df, true)
    _, corr_unb = _best_engineered_corr(fs_unb, df, true)
    assert corr_orth >= corr_unb + 0.20, (
        f"basis={basis} did not beat elementary search: orth |corr|={corr_orth:.3f} vs unb |corr|={corr_unb:.3f}"
    )
