"""Biz-value speed+quality test for the MRMR ``fe_fast_search`` master toggle (2026-06-14).

Asserts, on the two canonical interaction synthetics:
  y1 = a**2/b + f/5 + log(c)*sin(d)            (CASE1)
  y2 = 0.2*a**2/b + f/5 + log(c*2)*sin(d/3)    (CASE2, warped + down-weighted (a,b))

  1. QUALITY: with the fast path ON (default) BOTH interactions are still recovered -- an
     engineered feature jointly covering (a,b) AND one covering (c,d) (a fused composite or a
     conditional-gate composite over (c,d) both count) -- and the Ridge-holdout MAE does NOT
     regress materially vs the exhaustive (fe_fast_search=False) selection.
  2. SPEED: the fast path is materially faster than the exhaustive path.

Tractable n (20_000) so the test is CI-runnable; the production target is n=100_000 (< 60s each,
from ~130s / ~100s warm). Marked ``slow`` -- it fits MRMR four times.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR


def _make_case(case: int, n: int = 20_000):
    rng = np.random.default_rng(0)
    a, b, c, d, e, f = (rng.random(n) for _ in range(6))
    if case == 1:
        y = a**2 / b + f / 5 + np.log(c) * np.sin(d)
    else:
        y = 0.2 * a**2 / b + f / 5 + np.log(c * 2) * np.sin(d / 3)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    return df, pd.Series(y, name="y")


def _ridge_holdout_mae(model, df, y):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error

    Xt = model.transform(df)
    Xt = Xt.values if hasattr(Xt, "values") else np.asarray(Xt)
    Xt = np.asarray(Xt, dtype=float)
    yv = np.asarray(y, dtype=float)
    msk = np.isfinite(Xt).all(1) & np.isfinite(yv)
    Xt, yv = Xt[msk], yv[msk]
    ntr = int(len(yv) * 0.7)
    sc = StandardScaler().fit(Xt[:ntr])
    reg = Ridge(alpha=1.0).fit(sc.transform(Xt[:ntr]), yv[:ntr])
    return float(mean_absolute_error(yv[ntr:], reg.predict(sc.transform(Xt[ntr:]))))


def _covers(names, *operands):
    """A selected name 'covers' an interaction when its engineered form names BOTH operands
    inside one feature (a fused composite, a gate composite, or a binary op of the two)."""
    for nm in names:
        s = str(nm)
        if all((op + ")") in s or (op + ",") in s or ("," + op) in s or ("(" + op) in s or ("_" + op + "_") in s for op in operands):
            return True
    return False


def _recovers_signal(case, names):
    ab = _covers(names, "a", "b")
    cd = _covers(names, "c", "d")
    return ab, cd


@pytest.mark.slow
@pytest.mark.parametrize("case", [1, 2])
def test_fast_search_recovers_signal_and_is_faster(case):
    df, y = _make_case(case)

    # Exhaustive reference (fast OFF) -- establishes the MAE bar + timing baseline.
    t0 = time.time()
    m_ref = MRMR(verbose=0, random_seed=0, fe_fast_search=False).fit(df.copy(), y.copy())
    t_ref = time.time() - t0
    list(m_ref.get_feature_names_out())
    mae_ref = _ridge_holdout_mae(m_ref, df, y)

    # Fast path (default ON).
    t0 = time.time()
    m_fast = MRMR(verbose=0, random_seed=0, fe_fast_search=True).fit(df.copy(), y.copy())
    t_fast = time.time() - t0
    names_fast = list(m_fast.get_feature_names_out())
    mae_fast = _ridge_holdout_mae(m_fast, df, y)

    # QUALITY 1: both interactions recovered on the fast path.
    ab, cd = _recovers_signal(case, names_fast)
    assert ab, f"CASE{case} fast path lost the (a,b) interaction; selected={names_fast}"
    assert cd, f"CASE{case} fast path lost the (c,d) interaction; selected={names_fast}"

    # QUALITY 1b -- CLEANLINESS (CASE1): the fast path must NOT re-introduce the over-materialised junk it
    # used to (spurious cross-group gate_mask / cross-signal sub(sqr(a),invcbrt(c)) / rint composites). On
    # equal-coefficient CASE1 the clean recovery is exactly div(sqr(a),..)+mul(log(c),sin(d)); assert no
    # gate_mask / rint column survives and no cross-group (a&c / a&d / b&c / b&d) binary node. CASE2's
    # warped (c,d) carrier IS a gate composite, so cleanliness is pinned on CASE1 only.
    if case == 1:
        assert not any("gate_mask" in str(nm) for nm in names_fast), f"CASE1 fast path kept a spurious gate_mask column: {names_fast}"
        assert not any("rint" in str(nm) for nm in names_fast), f"CASE1 fast path kept a spurious rint composite: {names_fast}"
        _cross = [nm for nm in names_fast if (_covers([nm], "a", "c") or _covers([nm], "a", "d") or _covers([nm], "b", "c") or _covers([nm], "b", "d"))]
        assert not _cross, f"CASE1 fast path kept a cross-group cross-signal artefact: {_cross}"

    # QUALITY 2: holdout MAE within 10% of the exhaustive selection (the user's tolerance).
    assert mae_fast <= mae_ref * 1.10 + 1e-9, f"CASE{case} fast-path MAE {mae_fast:.5f} regressed >10% vs reference {mae_ref:.5f}"

    # SPEED: fast path is materially faster than exhaustive (>=20% wall reduction). The production
    # target is < 60s at n=100k; at this tractable n we only assert the relative win to avoid
    # hardware-coupling the absolute threshold.
    assert t_fast < t_ref * 0.80, f"CASE{case} fast path not materially faster: fast={t_fast:.1f}s ref={t_ref:.1f}s"


@pytest.mark.slow
def test_fast_search_toggle_restores_knobs():
    """fe_fast_search must NOT permanently mutate the constructor-arg knobs (clone/pickle/repeat-fit
    stability) -- the fast profile is applied per-fit and restored in ``finally``."""
    df, y = _make_case(1, n=4000)
    m = MRMR(verbose=0, random_seed=0, fe_fast_search=True)
    before = (m.fe_max_steps, m.fe_pair_prewarp_enable, m.fe_stability_vote_enable, m.fe_escalation_underdelivery_enable, m.fe_check_pairs_subsample_n)
    m.fit(df, y)
    after = (m.fe_max_steps, m.fe_pair_prewarp_enable, m.fe_stability_vote_enable, m.fe_escalation_underdelivery_enable, m.fe_check_pairs_subsample_n)
    assert before == after, f"fast-search profile leaked into constructor knobs: {before} -> {after}"
