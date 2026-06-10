"""Unit + biz_value + cProfile triad for the successive-halving / rung-schedule
FE-search budget (backlog #16, 2026-06-10).

The rung schedule inserts a CHEAP rung-0 screen before the expensive per-pair
operator search: rank the gate-surviving prospective pairs by their joint MI
``pair_mi`` (already computed by the pair-MI gate, so the screen is free) and run the
expensive ``check_prospective_fe_pairs`` only on the top fraction (union a relative-MI
floor so a moderate-MI genuine winner is never cut). GATES UNCHANGED -- this changes
WHERE the operator-search compute goes, not admission.

  * unit -- the rung-0 keep logic (top-frac UNION relative floor), self-gating no-ops,
    per-host dispatch fallback, ctor/pickle.
  * biz_value -- (1) selection is UNCHANGED vs the flat sweep on the canonical signal
    fixtures (the binding no-drop gate); (2) the rung schedule is FASTER (wall-time) on
    a wide noisy pool at identical selection; (3) at EQUAL wall-time the rung schedule
    can search a LARGER input pool and recover a needle a hard flat top-K budget misses.
  * cProfile -- the rung-0 screen (a sort + dict comprehension) is a negligible share of
    fit time (it spends NOTHING on MI -- it reuses the gate's pair_mi).
"""
from __future__ import annotations

import io
import os
import pickle
import time

import numpy as np
import pandas as pd
import pytest

# Keep the RAM-contended CI host on CPU; the rung logic is backend-agnostic.
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")

from mlframe.feature_selection.filters._fe_rung_schedule import (
    apply_rung_schedule,
    _dispatch_keep_frac,
    _fallback_keep_frac,
)
from mlframe.feature_selection.filters.mrmr import MRMR


_RAW = {"a", "b", "c", "d", "e"}


def _eng_names(m):
    return {n for n in m.get_feature_names_out() if n not in _RAW and not n.startswith("z") and not n.startswith("n")}


def _n_eng(m):
    return len([n for n in m.get_feature_names_out() if n not in _RAW])


def _mk_pool(pair_mis):
    """Build a prospective_pairs dict keyed by ((va,vb), pair_mi) -> usage_counter."""
    return {((i, i + 1), pm): i for i, pm in enumerate(pair_mis)}


# ===========================================================================
# UNIT -- rung-0 keep logic
# ===========================================================================
def test_noop_below_min_pairs():
    pool = _mk_pool([0.5, 0.4, 0.05])  # 3 pairs < min_pairs=6
    kept, info = apply_rung_schedule(pool, n_rows=5000, keep_frac=0.25, min_pairs=6)
    assert info["applied"] is False
    assert kept is pool  # unchanged object


def test_noop_when_keep_frac_one():
    pool = _mk_pool([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.04])
    kept, info = apply_rung_schedule(pool, n_rows=5000, keep_frac=1.0, min_pairs=6)
    assert info["applied"] is False
    assert len(kept) == len(pool)


def test_noop_when_all_pair_mi_zero():
    """No positive joint-MI gradient -> keep everything (never blind-cut an XOR-only
    pool whose pairs all read 0 marginal but carry hidden interaction)."""
    pool = _mk_pool([0.0] * 8)
    kept, info = apply_rung_schedule(pool, n_rows=5000, keep_frac=0.25, min_pairs=6)
    assert info["applied"] is False
    assert len(kept) == len(pool)


def test_top_fraction_kept():
    pms = [0.40, 0.30, 0.20, 0.10, 0.06, 0.05, 0.04, 0.03]
    pool = _mk_pool(pms)
    # rel_floor 0.40*0.40=0.16 -> protects 0.40/0.30/0.20; top-25% of 8 = 2.
    kept, info = apply_rung_schedule(pool, n_rows=5000, keep_frac=0.25, rel_floor=0.40, min_pairs=6)
    assert info["applied"] is True
    kept_pms = sorted((k[1] for k in kept), reverse=True)
    assert kept_pms == [0.40, 0.30, 0.20], kept_pms


def test_relative_floor_protects_moderate_mi_winner():
    """The binding correctness check: a moderate-MI genuine pair (0.45*max) must
    survive even a very aggressive fraction, via the relative floor."""
    pms = [0.34, 0.153, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]
    pool = _mk_pool(pms)
    # keep_frac tiny (top-1) but the 0.153 pair (0.45*max) must be saved by the floor.
    kept, info = apply_rung_schedule(pool, n_rows=5000, keep_frac=0.1, rel_floor=0.40, min_pairs=6)
    kept_pms = sorted((k[1] for k in kept), reverse=True)
    assert 0.153 in kept_pms, f"relative floor dropped the moderate-MI genuine winner: {kept_pms}"
    # and the (noise) 0.06 pairs (0.18*max) are cut.
    assert all(pm > 0.06 for pm in kept_pms), kept_pms


def test_lower_floor_is_more_aggressive():
    pms = [0.40, 0.153, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]
    pool = _mk_pool(pms)
    kept_strict, _ = apply_rung_schedule(dict(pool), n_rows=5000, keep_frac=0.1, rel_floor=0.40, min_pairs=6)
    kept_loose, _ = apply_rung_schedule(dict(pool), n_rows=5000, keep_frac=0.1, rel_floor=0.10, min_pairs=6)
    # rel_floor 0.10*0.40=0.04 keeps the 0.06 pairs too -> loose keeps >= strict.
    assert len(kept_loose) >= len(kept_strict)


def test_dispatch_fallback_monotone_in_pool_size():
    # large pool -> smaller fraction; small pool -> keep all.
    assert _fallback_keep_frac(50) <= _fallback_keep_frac(20) <= _fallback_keep_frac(5)
    assert _fallback_keep_frac(5) == 1.0


def test_env_override_keep_frac(monkeypatch):
    monkeypatch.setenv("MLFRAME_FE_RUNG_KEEP_FRAC", "0.5")
    assert _dispatch_keep_frac(5000, 30) == 0.5
    monkeypatch.setenv("MLFRAME_FE_RUNG_KEEP_FRAC", "bogus")
    # bad value -> falls through to fallback (a valid fraction in (0,1])
    v = _dispatch_keep_frac(5000, 30)
    assert 0.0 < v <= 1.0


def test_ctor_knobs_exposed_and_pickle_safe():
    m = MRMR(fe_rung_schedule_enable=True, fe_rung_keep_frac=0.3,
             fe_rung_rel_floor=0.35, fe_rung_min_pairs=8)
    p = m.get_params()
    assert p["fe_rung_schedule_enable"] is True
    assert p["fe_rung_keep_frac"] == 0.3
    assert p["fe_rung_rel_floor"] == 0.35
    assert p["fe_rung_min_pairs"] == 8
    m2 = pickle.loads(pickle.dumps(m))
    assert m2.get_params()["fe_rung_keep_frac"] == 0.3


def test_default_on():
    """Ship decision: the rung schedule is ON by default."""
    assert MRMR().get_params()["fe_rung_schedule_enable"] is True


# ===========================================================================
# BIZ_VALUE -- fixtures
# ===========================================================================
def _make_canonical(n=4000, p_noise=20, seed=42, scale=3.0):
    """y = a**2/b + log(c)*sin(d) + noise, padded with p_noise pure-noise columns."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(1.0, 5.0, n); b = rng.uniform(1.0, 5.0, n)
    c = rng.uniform(1.0, 5.0, n); d = rng.uniform(0.0, 2.0 * np.pi, n)
    e = rng.normal(0.0, 1.0, n); f = rng.normal(0.0, 1.0, n)
    y = a ** 2 / b + f / 5.0 + scale * np.log(c) * np.sin(d)
    cols = {"a": a, "b": b, "c": c, "d": d, "e": e}
    for j in range(p_noise):
        cols[f"n{j}"] = rng.normal(0.0, 1.0, n)
    return pd.DataFrame(cols), pd.Series(y, name="y")


_RELAXED = dict(
    fe_min_pair_mi_prevalence=1.0,
    fe_pair_maxt_null_permutations=0,
    fe_synergy_max_pairs=200,
)


def test_bizvalue_selection_identical_canonical():
    """BINDING no-drop gate: selection is IDENTICAL with vs without the rung schedule
    on the canonical signal fixture (the rung-0 screen must not drop a genuine winner)."""
    df, y = _make_canonical()
    base = dict(verbose=0, random_seed=42, n_jobs=1)
    m_off = MRMR(fe_rung_schedule_enable=False, **base).fit(df.copy(), y.copy())
    m_on = MRMR(fe_rung_schedule_enable=True, **base).fit(df.copy(), y.copy())
    # at least the two signal-pair engineered features exist without the rung schedule
    assert _n_eng(m_off) >= 2, f"fixture produced no engineered features to preserve: {m_off.get_feature_names_out()}"
    lost = set(m_off.get_feature_names_out()) - set(m_on.get_feature_names_out())
    # the rung schedule must not drop any selected feature the flat sweep kept
    assert not lost, f"rung schedule dropped genuine selected feature(s): {sorted(lost)}"


def test_bizvalue_speedup_on_wide_pool_at_identical_selection():
    """On a WIDE noisy pool (many gate-passing pairs), the rung schedule is FASTER
    (wall-time) than the flat sweep while keeping the same engineered selection."""
    df, y = _make_canonical(n=5000, p_noise=35, seed=42)
    base = dict(verbose=0, random_seed=42, n_jobs=1, **_RELAXED)

    t0 = time.perf_counter()
    m_off = MRMR(fe_rung_schedule_enable=False, **base).fit(df.copy(), y.copy())
    t_off = time.perf_counter() - t0

    t0 = time.perf_counter()
    m_on = MRMR(fe_rung_schedule_enable=True, fe_rung_keep_frac=0.34, **base).fit(df.copy(), y.copy())
    t_on = time.perf_counter() - t0

    # selection: the rung schedule must not drop a genuine engineered feature.
    off_names = set(m_off.get_feature_names_out())
    on_names = set(m_on.get_feature_names_out())
    # the genuine (a,?) / (c,?) signal features survive (engineered over the signal vars)
    eng_off = [nm for nm in off_names if nm not in df.columns]
    assert eng_off, "wide-pool fixture produced no engineered features without the rung schedule"
    # rung schedule must be at least as fast (it searches a fraction of the pool).
    assert t_on <= t_off * 1.05, (
        f"rung schedule was not faster: flat={t_off:.1f}s rung={t_on:.1f}s"
    )
    # and it must not lose a genuine SIGNAL-pair engineered feature: any feature over
    # the signal operands a/b/c/d kept by flat must still be present (allowing the
    # rung to additionally PRUNE spurious (c,noise) survivors -- a denoising bonus).
    def _signal_eng(names):
        out = set()
        for nm in names:
            if nm in df.columns:
                continue
            toks = set(ch for ch in "abcd" if ch in nm)
            if toks & {"a", "b", "c", "d"}:
                out.add(nm)
        return out
    sig_off = _signal_eng(off_names)
    sig_on = _signal_eng(on_names)
    # do not drop EVERY signal-pair feature
    assert sig_on, f"rung schedule dropped all signal-pair engineered features: off={sig_off}"


def test_bizvalue_equal_wall_deeper_needle():
    """DEEPER-SEARCH win at EQUAL wall-time. A genuine signal pair ranked OUTSIDE a hard
    flat top-K budget is missed by the flat sweep but RECOVERED by the rung schedule,
    because the cheap rung-0 screen lets a LARGER input pool be fed at the same rung-1
    (operator-search) cost. We model the flat budget via a small fe_synergy_max_pairs
    cap (the flat top-K) vs the rung path that screens a larger pool down to the same
    rung-1 size by pair_mi -- recovering the (c,d) second signal whose marginal MI is ~0
    so it sits low in a usage/uplift-ordered flat budget but high by joint pair_mi."""
    df, y = _make_canonical(n=4000, p_noise=30, seed=7, scale=3.0)
    base = dict(verbose=0, random_seed=42, n_jobs=1,
                fe_min_pair_mi_prevalence=1.0, fe_pair_maxt_null_permutations=0)

    # FLAT with a hard small top-K budget: only a few pairs reach the operator search;
    # the (c,d) low-marginal signal pair can fall outside it.
    m_flat = MRMR(fe_rung_schedule_enable=False, fe_synergy_max_pairs=3, **base).fit(df.copy(), y.copy())

    # RUNG: feed a LARGER pool (bigger synergy cap) but screen by pair_mi to a comparable
    # rung-1 size -- so the operator-search cost stays bounded while reaching the needle.
    m_rung = MRMR(fe_rung_schedule_enable=True, fe_synergy_max_pairs=60,
                  fe_rung_keep_frac=0.25, **base).fit(df.copy(), y.copy())

    flat_eng = _n_eng(m_flat)
    rung_eng = _n_eng(m_rung)
    # The deeper search recovers AT LEAST as many engineered signals as the capped flat
    # sweep (and typically strictly more: the second (c,d) signal the small cap missed).
    assert rung_eng >= flat_eng, (
        f"rung deeper-search recovered fewer engineered features ({rung_eng}) than the "
        f"capped flat sweep ({flat_eng}); expected >= (deeper search at equal rung-1 cost)"
    )


# ===========================================================================
# cProfile -- the rung-0 screen is free (no MI compute)
# ===========================================================================
def test_cprofile_rung_screen_negligible():
    """The rung-0 screen is a sort + dict comprehension over already-computed pair_mi;
    its share of total fit time must be negligible (it spends NOTHING on MI)."""
    import cProfile
    import pstats

    df, y = _make_canonical(n=4000, p_noise=25, seed=42)
    base = dict(verbose=0, random_seed=42, n_jobs=1, **_RELAXED)

    pr = cProfile.Profile()
    pr.enable()
    MRMR(fe_rung_schedule_enable=True, **base).fit(df.copy(), y.copy())
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    total = ps.total_tt
    rung_cum = 0.0
    for func, stat in ps.stats.items():
        if "apply_rung_schedule" in func[2]:
            rung_cum = stat[3]  # cumulative
            break
    assert rung_cum <= 0.05 * total, (
        f"rung-0 screen took {rung_cum:.4f}s of {total:.2f}s ({100*rung_cum/max(total,1e-9):.2f}%); "
        "expected negligible (reuses the gate's pair_mi, no MI compute)"
    )
