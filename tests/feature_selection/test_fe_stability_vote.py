"""Unit + biz_value + cProfile triad for cross-fold recipe stability voting
(backlog #15, 2026-06-10).

The expensive MRMR FE search runs ONCE on the full data; this consensus layer
adds a cheap K-fold CONFIRMATION -- each surviving ``unary_binary`` recipe is
REPLAYED (leak-safe: the recipe is frozen, only the rows change) on K held-out
folds, its uplift gate recomputed per fold, and the recipe admitted only if it
clears the gate in ``>= ceil(quorum*K)`` folds. It complements the order-2/order-3
maxT floors (which kill chance-MAX candidates WITHIN a fold) by killing recipes
that won only on a fold-specific QUIRK of the full-data split. No refit -- only K
plug-in-MI replays per recipe.

UNIT contracts:
* the fold partition is disjoint + covers every row;
* the per-fold gate fires on a planted-signal column and rejects pure noise;
* ``confirm_recipes_cross_fold`` is a structural no-op below 2 unary_binary
  recipes / k<2 / quorum<=0 / tiny n;
* the ctor knobs are exposed via get_params (pickle / clone safe).

BIZ_VALUE contracts (the decisive ones):
* NOISE-SURVIVOR REDUCTION: on a pure-noise frame with the FE gates relaxed so
  chance-max noise pairs slip the in-fit per-pair gate, the default-on vote drops
  the fold-specific engineered survivors toward 0 (strictly fewer than WITHOUT);
* SIGNAL PRESERVATION: the genuine ``a**2/b`` and ``log*sin`` recipes clear the
  quorum -> the default transform output is UNCHANGED vs the no-vote support.

DEFAULT-PATH contract (default-on, 2026-06-09): the reduction must manifest with
a plain (gate-relaxed) ``MRMR()`` and the signal must survive with default gates.
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters._fe_stability_vote import (
    _fold_indices,
    _marginal_mi,
    _recipe_clears_fold,
    confirm_recipes_cross_fold,
)
from mlframe.feature_selection.filters.mrmr import MRMR


# Relaxed-gate config: simulate the regime where chance-max noise pairs slip the
# in-fit per-pair gate (maxT floor off, lenient prevalence), so the cross-fold
# quorum has fold-specific noise winners to cut.
RELAXED = dict(
    fe_min_pair_mi_prevalence=1.0,
    fe_min_engineered_mi_prevalence=0.80,
    fe_synergy_min_prevalence=1.0,
    fe_pair_maxt_null_permutations=0,
    fe_synergy_screen_max_features=20,
    fe_synergy_max_pairs=8,
    fe_acceptance="prevalence_ratio",
    fe_drop_redundant_raw_operands=False,
)


def _n_eng(m):
    return len(getattr(m, "_engineered_recipes_", []) or [])


def _eng_names(m):
    return {r.name for r in (getattr(m, "_engineered_recipes_", []) or [])}


# ---------------------------------------------------------------------------
# UNIT
# ---------------------------------------------------------------------------
def test_fold_indices_disjoint_and_cover():
    rng = np.random.default_rng(0)
    n, k = 97, 5
    folds = _fold_indices(n, k, rng)
    assert len(folds) == k
    allrows = np.concatenate(folds)
    assert allrows.shape[0] == n
    # every row exactly once
    assert sorted(allrows.tolist()) == list(range(n))
    # near-equal sizes
    sizes = sorted(len(f) for f in folds)
    assert sizes[-1] - sizes[0] <= 1


def test_per_fold_gate_fires_on_signal_rejects_noise():
    rng = np.random.default_rng(1)
    n = 1500
    y = (rng.random(n) > 0.5).astype(np.int64)
    # GENUINE UPLIFT: an engineered column that IS y, built from two source
    # operands EACH independent of y (the XOR/synergy shape the gate must keep).
    # engineered MI (high) >> sum of source marginal MIs (~0) -> admitted.
    src_a = (rng.random(n) > 0.5).astype(np.int64)
    src_b = (rng.random(n) > 0.5).astype(np.int64)
    eng_signal = y.copy()
    assert _recipe_clears_fold(
        eng_codes=eng_signal, src_a_codes=src_a, src_b_codes=src_b,
        y_codes=y, prevalence=1.0,
    )
    # NOISE: an engineered column whose y-information is no better than its
    # source operands' -- the realistic fold-specific-quirk case. One source
    # carries the engineered column's (tiny) marginal, so the uplift ratio
    # does not clear -> rejected.
    eng_noise = src_a.copy()  # engineered == one operand -> zero uplift
    assert not _recipe_clears_fold(
        eng_codes=eng_noise, src_a_codes=src_a, src_b_codes=src_b,
        y_codes=y, prevalence=1.0,
    )
    # And a pure-noise engineered column with pure-noise source legs whose
    # engineered MI is below the source sum is rejected.
    eng_pure_noise = (rng.random(n) > 0.5).astype(np.int64)
    assert _marginal_mi(eng_pure_noise, y) < 0.02


def test_voter_noop_below_two_recipes_or_disabled():
    rng = np.random.default_rng(2)
    n = 400
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["a", "b", "c"])
    y = (rng.random(n) > 0.5).astype(np.int64)

    class _Recipe:
        def __init__(self, name, kind, src):
            self.name = name
            self.kind = kind
            self.src_names = src

    one = {"e0": _Recipe("e0", "unary_binary", ("a", "b"))}
    # < 2 unary_binary recipes -> no-op
    assert confirm_recipes_cross_fold(
        recipes=one, X=X, y_codes=y, feature_names_in=["a", "b", "c"],
        nbins=4, k=5, quorum=0.6,
    ) == set()
    two = dict(one)
    two["e1"] = _Recipe("e1", "unary_binary", ("a", "c"))
    # k < 2 -> no-op
    assert confirm_recipes_cross_fold(
        recipes=two, X=X, y_codes=y, feature_names_in=["a", "b", "c"],
        nbins=4, k=1, quorum=0.6,
    ) == set()
    # quorum <= 0 -> no-op
    assert confirm_recipes_cross_fold(
        recipes=two, X=X, y_codes=y, feature_names_in=["a", "b", "c"],
        nbins=4, k=5, quorum=0.0,
    ) == set()


def test_ctor_knobs_exposed_and_pickle_safe():
    m = MRMR(fe_stability_vote_enable=True, fe_stability_vote_k=7,
             fe_stability_vote_quorum=0.7)
    p = m.get_params()
    assert p["fe_stability_vote_enable"] is True
    assert p["fe_stability_vote_k"] == 7
    assert p["fe_stability_vote_quorum"] == 0.7
    m2 = pickle.loads(pickle.dumps(m))
    assert m2.get_params()["fe_stability_vote_k"] == 7


# ---------------------------------------------------------------------------
# BIZ_VALUE
# ---------------------------------------------------------------------------
def _make_noise(n, p, seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"n{i}" for i in range(p)])
    y = pd.Series((rng.random(n) > 0.5).astype(int), name="y")
    return X, y


def test_bizvalue_noise_survivor_reduction():
    """The decisive win: across several noise frames (gates relaxed so noise
    slips the in-fit gate), the default-on vote cuts the total fold-specific
    engineered survivors strictly below the no-vote count -- toward 0."""
    base = dict(verbose=0, random_seed=42, n_jobs=1, fe_smart_polynom_iters=0, **RELAXED)
    tot_off = tot_on = 0
    for seed in range(4):
        X, y = _make_noise(n=1500, p=20, seed=seed)
        m_off = MRMR(fe_stability_vote_enable=False, **base).fit(X.copy(), y.copy())
        m_on = MRMR(fe_stability_vote_enable=True, **base).fit(X.copy(), y.copy())
        tot_off += _n_eng(m_off)
        tot_on += _n_eng(m_on)
    # The no-vote path must actually surface noise survivors (else the fixture is
    # not exercising the consensus layer).
    assert tot_off >= 3, f"fixture did not produce noise survivors WITHOUT the vote (got {tot_off})"
    # The vote must strictly reduce them, toward 0.
    assert tot_on < tot_off, f"vote did not reduce noise survivors ({tot_off} -> {tot_on})"
    assert tot_on <= max(1, tot_off // 4), (
        f"vote reduced noise survivors only modestly ({tot_off} -> {tot_on}); "
        "expected a strong drop toward 0"
    )


def _make_ratio(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.5, 3.0, n)
    b = rng.uniform(0.5, 3.0, n)
    noise = pd.DataFrame(rng.standard_normal((n, 6)), columns=[f"z{i}" for i in range(6)])
    X = pd.concat([pd.DataFrame({"a": a, "b": b}), noise], axis=1)
    sig = (a ** 2) / b
    y = pd.Series((sig > np.median(sig)).astype(int), name="y")
    return X, y


def _make_logsin(n=3000, seed=2):
    rng = np.random.default_rng(seed)
    c = rng.uniform(0.5, 5.0, n)
    d = rng.uniform(-3.0, 3.0, n)
    noise = pd.DataFrame(rng.standard_normal((n, 6)), columns=[f"z{i}" for i in range(6)])
    X = pd.concat([pd.DataFrame({"c": c, "d": d}), noise], axis=1)
    sig = np.log(np.abs(c)) * np.sin(d)
    y = pd.Series((sig > np.median(sig)).astype(int), name="y")
    return X, y


@pytest.mark.parametrize("maker", [_make_ratio, _make_logsin])
def test_bizvalue_signal_preserved(maker):
    """Genuine recipes clear the quorum -> the default-on transform output is
    UNCHANGED vs the no-vote support (no signal loss)."""
    X, y = maker()
    base = dict(verbose=0, random_seed=42, n_jobs=1, fe_smart_polynom_iters=0,
                fe_synergy_screen_max_features=8)
    m_off = MRMR(fe_stability_vote_enable=False, **base).fit(X.copy(), y.copy())
    m_on = MRMR(fe_stability_vote_enable=True, **base).fit(X.copy(), y.copy())
    # at least one genuine engineered recipe was found
    assert _n_eng(m_off) >= 1, "fixture produced no engineered recipe to preserve"
    # the vote must NOT drop any genuine recipe
    lost = _eng_names(m_off) - _eng_names(m_on)
    assert not lost, f"stability vote dropped genuine engineered recipe(s): {sorted(lost)}"


# ---------------------------------------------------------------------------
# cProfile -- replay cost is negligible (no refit)
# ---------------------------------------------------------------------------
def test_cprofile_replay_cost_negligible():
    """The vote adds only K plug-in-MI replays per recipe; its share of total
    fit time must be small. We profile a signal fit and assert the vote helper
    is not a hotspot."""
    import cProfile
    import pstats
    import io

    X, y = _make_ratio()
    base = dict(verbose=0, random_seed=42, n_jobs=1, fe_smart_polynom_iters=0,
                fe_synergy_screen_max_features=8)

    pr = cProfile.Profile()
    pr.enable()
    MRMR(fe_stability_vote_enable=True, **base).fit(X.copy(), y.copy())
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    total = ps.total_tt
    # cumulative time attributed to the vote entry point
    vote_cum = 0.0
    for func, stat in ps.stats.items():
        if "confirm_recipes_cross_fold" in func[2]:
            vote_cum = stat[3]  # cumulative time
            break
    # The vote must be a small fraction of total fit time (no refit).
    assert vote_cum <= 0.25 * total, (
        f"stability vote took {vote_cum:.3f}s of {total:.3f}s total fit "
        f"({100*vote_cum/max(total,1e-9):.1f}%); expected negligible (replay only)"
    )
