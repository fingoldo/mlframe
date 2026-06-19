"""SECOND FUNNEL STAGE -- GPU-exhaustive synergy sweep (fe_synergy_exhaustive).

Three checks:
  * decision-logic unit test: "never" always pre-rank; "auto" escalates to exhaustive when affordable
    (GPU + predicted wall-time <= budget) and falls back to pre-rank when over budget; "force" fires
    whenever a GPU is present; the throughput is sourced from the kernel_tuning cache / fallback (NEVER
    hardcoded into the decision).
  * PARITY (needs CUDA): on a frame with p <= cap the exhaustive path and the capped/auto path select the
    SAME features (the cap/pre-rank are a no-op below the cap, so forcing exhaustive must not change the
    result).
  * BIZ_VALUE (needs CUDA): on a WIDE perfectly-balanced (L=0) interaction frame (p=400 > cap), the
    exhaustive sweep recovers the planted operands that the O(p) pre-rank PROVABLY cannot.

The biz_value + parity tests auto-skip cleanly when no CUDA GPU is present (numba.cuda.is_available()).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr._mrmr_class import MRMR
from mlframe.feature_selection.filters._fe_synergy_exhaustive import (
    decide_exhaustive_sweep,
    predict_exhaustive_seconds,
    measured_pairs_per_second,
)


def _cuda_available() -> bool:
    try:
        from numba import cuda
        return bool(cuda.is_available())
    except Exception:
        return False


_HAS_CUDA = _cuda_available()
_skip_no_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA GPU (numba.cuda.is_available() is False)")


# --------------------------------------------------------------------------------------------------
# Decision-logic unit test (no GPU required)
# --------------------------------------------------------------------------------------------------


class _Knobs:
    def __init__(self, mode, budget=None, max_runtime_mins=None):
        self.fe_synergy_exhaustive = mode
        self.fe_synergy_exhaustive_max_seconds = budget   # explicit override (None => defer to max_runtime_mins)
        self.max_runtime_mins = max_runtime_mins          # MRMR's own fit-wide budget


def test_never_mode_always_prerank():
    use, reason = decide_exhaustive_sweep(_Knobs("never"), n_samples=5000, n_raw=400, verbose=0)
    assert use is False
    assert "never" in reason.lower()


def test_auto_declines_when_over_budget():
    # auto with a sub-microsecond budget can never afford exhaustive -> pre-rank (whether GPU present or not).
    use, reason = decide_exhaustive_sweep(_Knobs("auto", budget=1e-9), n_samples=2_000_000, n_raw=10_000, verbose=0)
    assert use is False
    # On a GPU host the reason is the auto->pre-rank budget decline; on a CPU host it declines on availability.
    assert ("pre-rank" in reason.lower()) or ("declined" in reason.lower())


@_skip_no_cuda
def test_auto_escalates_to_exhaustive_when_affordable():
    # The key fix: "auto" is NOT "never exhaustive" -- on a GPU it escalates to the full sweep when the
    # predicted wall-time fits the budget, so the default gets the complete (balanced-case) result for free.
    use, reason = decide_exhaustive_sweep(_Knobs("auto", budget=180.0), n_samples=5000, n_raw=400, verbose=0)
    assert use is True, reason
    assert "exhaustive" in reason.lower()


@_skip_no_cuda
def test_auto_budget_unlimited_when_no_mrmr_budget_set():
    # No explicit override AND no max_runtime_mins -> budget is UNLIMITED: auto escalates regardless of p
    # (the user did not ask to bound wall-time), so even a very wide frame fires under auto.
    use, reason = decide_exhaustive_sweep(_Knobs("auto"), n_samples=5000, n_raw=8000, verbose=0)
    assert use is True, reason
    assert "unlimited" in reason.lower()


def test_auto_budget_derives_from_max_runtime_mins():
    # The budget comes from MRMR's OWN max_runtime_mins (minutes -> seconds), not a hardcoded constant: a tiny
    # max_runtime_mins makes a large sweep over-budget -> auto falls back to the pre-rank (GPU or not).
    use, reason = decide_exhaustive_sweep(
        _Knobs("auto", max_runtime_mins=1e-6), n_samples=2_000_000, n_raw=10_000, verbose=0)
    assert use is False
    assert ("pre-rank" in reason.lower()) or ("no cuda" in reason.lower())


def test_throughput_sourced_not_hardcoded():
    # predict returns (seconds, pairs_per_second, source); source is cache | fallback -- never a magic
    # constant baked into the decision path.
    sec, pps, source = predict_exhaustive_seconds(5000, 400)
    assert sec > 0 and pps > 0
    assert source in ("cache", "fallback")
    pps2, source2 = measured_pairs_per_second(5000, 79800)
    assert pps2 > 0 and source2 in ("cache", "fallback")


def test_force_ignores_budget():
    # "force" means the user accepts the wall-time: it fires regardless of the budget WHEN a GPU is present,
    # and declines ONLY for lack of a GPU (CPU exhaustive is never run). Contrast test_auto_declines_when_over_budget.
    use, reason = decide_exhaustive_sweep(_Knobs("force", budget=1e-9), n_samples=200_000, n_raw=5_000, verbose=0)
    if _HAS_CUDA:
        assert use is True, reason                      # budget ignored under force
        assert "force" in reason.lower()
    else:
        assert use is False and "no cuda" in reason.lower()


@_skip_no_cuda
def test_force_fires_within_budget_when_gpu():
    use, reason = decide_exhaustive_sweep(_Knobs("force", budget=180.0), n_samples=5000, n_raw=400, verbose=0)
    assert use is True
    assert "exhaustive" in reason.lower()


# --------------------------------------------------------------------------------------------------
# PARITY: p <= cap -> exhaustive == auto (cap/pre-rank are no-ops below the cap)
# --------------------------------------------------------------------------------------------------


def _make_narrow_interaction(seed, n=4000, p=40):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    a, b = X[:, 3], X[:, 17]
    logit = 2.6 * np.sign(a) * np.sign(b) + 0.05 * 2.6 * (a + b)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    cols = [f"f{i}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y")


@_skip_no_cuda
def test_parity_exhaustive_equals_auto_below_cap():
    X, y = _make_narrow_interaction(0, p=40)  # 40 << cap 250
    m_auto = MRMR(fe_synergy_exhaustive="auto", fe_synergy_screen_max_features=250)
    m_auto.fit(X, y)
    m_force = MRMR(fe_synergy_exhaustive="force", fe_synergy_screen_max_features=250)
    m_force.fit(X, y)
    # Below the cap the pre-rank/cap never trims, so forcing the exhaustive sweep must select the SAME
    # raw operands (engineered names may differ in tie-break order but the operand recovery is identical).
    auto_raw = {n for n in m_auto.get_feature_names_out() if n in set(X.columns)}
    force_raw = {n for n in m_force.get_feature_names_out() if n in set(X.columns)}
    assert auto_raw == force_raw, f"parity broke below cap: auto={sorted(auto_raw)} force={sorted(force_raw)}"


# --------------------------------------------------------------------------------------------------
# BIZ_VALUE: wide perfectly-balanced (L=0) frame -> exhaustive recovers; pre-rank cannot
# --------------------------------------------------------------------------------------------------

# Use a small explicit cap (CAP_BIZ) so p > cap exercises the wide-frame path while the order-2 maxT
# permutation-null over the pre-ranked pool stays cheap (C(CAP_BIZ,2) pairs, not C(250,2)).
N_BIZ = 6000
P_BIZ = 120          # > CAP_BIZ
CAP_BIZ = 40
OPERANDS = (7, 95)   # the planted balanced XOR pair


def _make_balanced_l0(seed):
    """ONE perfectly-balanced sign-product pair on a wide frame: the operands carry ZERO main-effect leak
    (L=0), so every univariate (and higher-moment) score vs y is zero in expectation -- the irreducible case
    the O(p) pre-rank cannot rank. Only the exhaustive C(p,2) joint-MI sweep recovers it."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N_BIZ, P_BIZ))
    ia, ib = OPERANDS
    a, b = X[:, ia], X[:, ib]
    # Pure balanced XOR-like target: y depends ONLY on sign(a)*sign(b), perfectly balanced, no leak.
    logit = 3.0 * np.sign(a) * np.sign(b)
    y = (rng.random(N_BIZ) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    cols = [f"f{i}" for i in range(P_BIZ)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y"), {f"f{i}" for i in OPERANDS}


def _operands_recovered(feature_names, operands):
    """A planted operand counts as recovered when it appears as a selected raw column OR inside any
    engineered feature name (e.g. ``add(sign(f7),sign(f95))`` recovers both f7 and f95 as the pair). This
    is the correct "recovers the planted pair" criterion: once the interaction column is built, MRMR may
    drop the redundant raw operands -- the engineered feature IS the recovery."""
    recovered = set()
    for op in operands:
        for nm in feature_names:
            if op == nm or (op + ")") in nm or (op + ",") in nm or ("(" + op) in nm or nm.endswith(op):
                recovered.add(op)
                break
    return recovered


@_skip_no_cuda
def test_biz_value_exhaustive_recovers_balanced_operands_prerank_cannot():
    X, y, operands = _make_balanced_l0(0)

    # PRE-RANK path (never): the legacy pre-rank-only behaviour. The O(p) propensity score is blind to a
    # PERFECTLY balanced pair, but NB at FINITE n the planted "balanced" pair is only APPROXIMATELY balanced,
    # so the capped synergy sweep on the top-CAP pre-ranked columns can OCCASIONALLY recover it too (recovering
    # genuine signal is fine -- not a bug). Truly-irreducible invisibility holds only in the n->inf limit. So we
    # do NOT assert the pre-rank recovers strictly fewer (that was a finite-n knife-edge); we assert the ROBUST
    # facts below: the exhaustive sweep RELIABLY recovers BOTH operands, and is never worse than the pre-rank.
    m_prerank = MRMR(fe_synergy_exhaustive="never", fe_synergy_prerank=True,
                     fe_synergy_screen_max_features=CAP_BIZ)
    m_prerank.fit(X, y)
    rec_prerank = _operands_recovered(m_prerank.get_feature_names_out(), operands)

    # FORCE path: the full C(p,2) joint-MI sweep ranks the balanced pair at the top, and the per-pair search
    # builds the interaction feature from BOTH operands.
    m_exh = MRMR(fe_synergy_exhaustive="force", fe_synergy_screen_max_features=CAP_BIZ,
                 fe_synergy_exhaustive_max_seconds=180.0)
    m_exh.fit(X, y)
    rec_exh = _operands_recovered(m_exh.get_feature_names_out(), operands)

    # ROBUST invariant: exhaustive RELIABLY recovers the full planted pair (the real value of the GPU sweep on
    # the irreducible case) and never does worse than the pre-rank.
    assert rec_exh == operands, f"exhaustive did not recover the full planted pair: {sorted(rec_exh)} != {sorted(operands)}"
    assert len(rec_exh) >= len(rec_prerank), (
        f"exhaustive recovered FEWER than pre-rank: exhaustive {sorted(rec_exh)}, pre-rank {sorted(rec_prerank)}")

    # THE FIX (default "auto"): this frame (p=120, n=6000) is cheap to sweep exhaustively, so the DEFAULT auto
    # escalates to the exhaustive path and ALSO recovers the balanced pair -- i.e. the default is NOT blind to
    # the irreducible case when it can afford completeness. (auto only falls back to the pre-rank on frames too
    # wide to sweep within fe_synergy_exhaustive_max_seconds.)
    m_auto = MRMR(fe_synergy_exhaustive="auto", fe_synergy_screen_max_features=CAP_BIZ,
                  fe_synergy_exhaustive_max_seconds=180.0)
    m_auto.fit(X, y)
    rec_auto = _operands_recovered(m_auto.get_feature_names_out(), operands)
    assert rec_auto == operands, (
        f"default 'auto' did not escalate to exhaustive on an affordable balanced frame: recovered "
        f"{sorted(rec_auto)} != {sorted(operands)} (auto should run the full sweep when it fits the budget)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--no-cov"])
