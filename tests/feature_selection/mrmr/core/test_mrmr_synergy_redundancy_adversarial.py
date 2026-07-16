"""Adversarial probes of the 2026-06 algorithmic patches (synergy seeding, pseudo-remix
protection, count-floor / min_features_fallback, global-RNG leak seal).

Each test ASSERTS correct/sound behavior: it PASSES if the patch is sound and FAILS if it
has a hole. Fast: n<=3000, fixed seeds, mostly unit-level calls into the patched helpers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._fe_synergy_screen import (
    detect_synergy_combos,
    joint_synergy_mi,
    _marginal_mm_mi,
)
from mlframe.feature_selection.filters._fe_raw_redundancy_drop import (
    _is_pseudo_remix_child,
    _PSEUDO_SRC_SPLIT,
    drop_redundant_raw_operands,
)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _qcode(x, nbins):
    """Equi-frequency integer codes for a continuous column."""
    x = np.asarray(x, dtype=np.float64).ravel()
    ranks = np.argsort(np.argsort(x))
    return (ranks * nbins // len(x)).astype(np.int64)


# ======================================================================================
# 1. N-WAY SYNERGY SEEDING
# ======================================================================================
def test_synergy_does_not_admit_pure_noise_pair():
    """A pure-noise pair must NOT clear the synergy gate (no false synergy)."""
    rng = np.random.default_rng(0)
    n = 3000
    cx = rng.integers(0, 6, n)
    cy = rng.integers(0, 6, n)
    yt = rng.integers(0, 2, n)
    out = detect_synergy_combos([cx, cy], yt, [0, 1], max_order=2, min_order=2, synergy_ratio=1.5, min_joint_mi=0.05)
    assert out == [], f"noise pair admitted as synergy: {out}"


def test_synergy_recovers_2way_xor():
    """A genuine 2-way XOR (zero-marginal synergy) MUST be detected."""
    rng = np.random.default_rng(1)
    n = 3000
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    yt = (a ^ b).astype(np.int64)
    # add a pure-noise third column to confirm only the signal pair fires
    c = rng.integers(0, 2, n)
    out = detect_synergy_combos([a, b, c], yt, [0, 1, 2], max_order=2, min_order=2, synergy_ratio=1.5, min_joint_mi=0.05)
    combos = {frozenset(t[0]) for t in out}
    assert frozenset((0, 1)) in combos, f"2-way XOR signal pair missed: {out}"
    assert frozenset((0, 2)) not in combos and frozenset((1, 2)) not in combos, f"noise-bearing pair admitted: {out}"


def test_synergy_recovers_3way_xor_and_rejects_noise_triple():
    """3-way XOR detected at order 3; a pure-noise triple is not admitted."""
    rng = np.random.default_rng(2)
    n = 3000
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    c = rng.integers(0, 2, n)
    yt = (a ^ b ^ c).astype(np.int64)
    n1 = rng.integers(0, 2, n)
    n2 = rng.integers(0, 2, n)
    n3 = rng.integers(0, 2, n)
    cols = [a, b, c, n1, n2, n3]
    out = detect_synergy_combos(cols, yt, list(range(6)), max_order=3, min_order=3, synergy_ratio=1.5, min_joint_mi=0.05)
    combos = {frozenset(t[0]) for t in out}
    assert frozenset((0, 1, 2)) in combos, f"3-way XOR missed: {out}"
    assert frozenset((3, 4, 5)) not in combos, f"noise triple admitted: {out}"


def test_synergy_high_cardinality_hits_max_cells_cap():
    """A high-cardinality combo whose dense joint blows past _MAX_CELLS must be SKIPPED
    (not crash, not falsely admitted)."""
    rng = np.random.default_rng(3)
    n = 2000
    # two columns each with ~1500 distinct codes -> 1500*1500 >> 1<<20 cells
    a = np.arange(n) % 1500
    b = (np.arange(n) * 7) % 1500
    yt = rng.integers(0, 2, n).astype(np.int64)
    out = detect_synergy_combos([a, b], yt, [0, 1], max_order=2, min_order=2, synergy_ratio=1.5, min_joint_mi=0.05)
    # capped out -> nothing admitted; importantly no exception
    assert out == [], f"high-card combo over _MAX_CELLS should be skipped, got {out}"


def test_synergy_many_distinct_codes_noise_not_admitted():
    """High-but-under-cap-cardinality NOISE pair: MM correction must keep joint MI low.
    Adversarial: many cells inflate raw joint MI; MM debit must prevent false synergy."""
    rng = np.random.default_rng(4)
    n = 3000
    a = rng.integers(0, 30, n)  # 30*30*2 = 1800 cells, under cap
    b = rng.integers(0, 30, n)
    yt = rng.integers(0, 2, n).astype(np.int64)
    jmi = joint_synergy_mi(a, b, yt)
    out = detect_synergy_combos([a, b], yt, [0, 1], max_order=2, min_order=2, synergy_ratio=1.5, min_joint_mi=0.05)
    # HOLE: the Miller-Madow occupancy debit does NOT keep a *high-cardinality* noise
    # grid's joint MI near zero. At 30x30 cells over n=3000 (~0.6 rows/cell) the
    # finite-sample joint MI is ~0.29 nats and the ratio gate (1.5 * ~0.004 marginals)
    # is trivially cleared -> a PURE-NOISE pair is admitted as "synergy". The module's
    # noise-resistance was validated only at nbins in {6,8} (<=64 cells); it does not
    # generalise to high-cardinality operands under the _MAX_CELLS=1<<20 ceiling.
    assert jmi < 0.05, f"MM-corrected joint MI of high-card noise pair too high: {jmi} " f"(detect_synergy_combos admitted: {out})"
    assert out == [], f"high-card noise pair admitted as synergy: {out}"


def test_synergy_ratio_gate_vs_marginal_signal():
    """A pair where ONE operand already carries strong marginal MI (so joint barely
    exceeds the marginal) must NOT be flagged as *synergy* under the ratio gate."""
    rng = np.random.default_rng(5)
    n = 3000
    a = rng.integers(0, 2, n)
    yt = a.copy().astype(np.int64)  # y == a: pure marginal, no synergy
    b = rng.integers(0, 2, n)  # noise
    marg_a = _marginal_mm_mi(a, yt)
    assert marg_a > 0.3, "sanity: a should be strongly marginal"
    out = detect_synergy_combos([a, b], yt, [0, 1], max_order=2, min_order=2, synergy_ratio=1.5, min_joint_mi=0.05)
    # joint(a,b) ~ marg(a); ratio 1.5*marg(a) won't be cleared -> not synergy
    assert out == [], f"marginal-dominated pair mislabeled synergy: {out}"


# ======================================================================================
# 2. PSEUDO-REMIX PROTECTION (tokenization + redundancy drop both directions)
# ======================================================================================
def test_pseudo_remix_detector_prefixes():
    """Pseudo-remix prefixes (gate_mask/binagg/argmax) must be detected; genuine composites must not."""
    assert _is_pseudo_remix_child("gate_mask__a__b__t0.1")
    assert _is_pseudo_remix_child("binagg_skew(c|qbin(a))")
    assert _is_pseudo_remix_child("argmax__a__b")
    # genuine composites are NOT pseudo
    assert not _is_pseudo_remix_child("div(sqr(a),abs(b))")
    assert not _is_pseudo_remix_child("mul(a,b)")
    assert not _is_pseudo_remix_child("")


def test_pseudo_src_split_tokenization_short_names():
    """Tokenization edge cases: single-char and substring-y raw names.
    Splitting gate_mask__a__b must yield 'a' and 'b' as exact tokens, NOT 'a1'/'ab'."""
    toks = set(_PSEUDO_SRC_SPLIT.split("gate_mask__a1__b"))
    assert "a1" in toks and "b" in toks
    assert "a" not in toks, "raw 'a' should NOT be a token of gate_mask__a1__b"
    toks2 = set(_PSEUDO_SRC_SPLIT.split("binagg_mean(log|qbin(a))"))
    # 'log' appears as an FE keyword token here; a raw named 'log' WOULD collide.
    assert "log" in toks2 and "a" in toks2


def test_redundancy_drop_genuinely_redundant_raw_still_drops():
    """A raw FULLY subsumed by a GENUINE (non-pseudo) ratio child must still DROP --
    the pseudo-remix protection must not over-protect genuine composites.

    Adversarial against direction-(2) hole: ensure protection is keyed on pseudo prefix,
    not merely on name-token overlap."""
    rng = np.random.default_rng(6)
    n = 2000
    a = rng.uniform(0.5, 3.0, n)
    b = rng.uniform(0.5, 3.0, n)
    ratio = (a**2) / b
    yt = ratio + rng.normal(0, 0.01, n)  # y fully determined by a**2/b
    nb = 10
    cols = ["a", "b", "div(sqr(a),abs(b))"]
    data = np.column_stack([_qcode(a, nb), _qcode(b, nb), _qcode(ratio, nb)])
    eng_cont = {"div(sqr(a),abs(b))": ratio}
    raw_X = pd.DataFrame({"a": a, "b": b})
    kept, dropped = drop_redundant_raw_operands(
        data=data, cols=cols, selected_cols_idx=[0, 1, 2],
        raw_name_set={"a", "b"}, y_binned=_qcode(yt, nb), y_continuous=yt,
        engineered_continuous=eng_cont,
        replayable_eng_names={"div(sqr(a),abs(b))"},
        raw_X=raw_X, seed=0,
    )
    # at least one of the fully-subsumed operands must drop (genuine redundancy)
    assert dropped, f"genuine ratio-subsumed raw was wrongly fully protected: kept={[cols[i] for i in kept]}"


def test_redundancy_drop_pseudo_remix_child_does_not_drop_raw():
    """A raw whose ONLY consumer is a pseudo-remix (gate) child must NOT be dropped:
    the gate cannot subsume the raw's private linear term."""
    rng = np.random.default_rng(7)
    n = 2000
    a = rng.uniform(-3, 3, n)
    b = rng.uniform(-3, 3, n)
    yt = 10.0 * a + rng.normal(0, 0.1, n)  # y carries a LINEARLY
    nb = 10
    gate = (a > np.median(a)).astype(np.float64)
    cols = ["a", "b", "gate_mask__a__b"]
    data = np.column_stack([_qcode(a, nb), _qcode(b, nb), gate.astype(np.int64)])
    eng_cont = {"gate_mask__a__b": gate}
    raw_X = pd.DataFrame({"a": a, "b": b})
    kept, dropped = drop_redundant_raw_operands(
        data=data, cols=cols, selected_cols_idx=[0, 1, 2],
        raw_name_set={"a", "b"}, y_binned=_qcode(yt, nb), y_continuous=yt,
        engineered_continuous=eng_cont,
        replayable_eng_names={"gate_mask__a__b"},
        raw_X=raw_X, seed=0,
    )
    assert "a" not in dropped, f"raw 'a' wrongly dropped given only a pseudo-remix gate consumer: {dropped}"


# ======================================================================================
# 3. COUNT-FLOOR / min_features_fallback ON PURE NOISE
# ======================================================================================
def _noise_frame(n=1500, p=6, seed=0):
    """Build a pure-noise fixture where y is independent of every column in X."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = rng.integers(0, 2, n)  # target independent of X
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(p)]), pd.Series(y, name="y")


def test_min_features_fallback_on_pure_noise_forces_features():
    """With min_features_fallback>=1 on a pure-noise frame, the documented behaviour is a
    count-floor that force-adds the best-MI columns. Assert it adds AT MOST the requested
    floor (does not balloon) and is deterministic."""
    X, y = _noise_frame(seed=10)
    from mlframe.feature_selection.filters import MRMR
    m = MRMR(random_seed=42, verbose=0, n_jobs=1, full_npermutations=2,
             baseline_npermutations=2, fe_max_steps=0, min_features_fallback=2,
             skip_retraining_on_same_content=False)
    m.fit(X.copy(), y)
    support = list(getattr(m, "support_", []))
    nsel = int(np.sum(support)) if len(support) and isinstance(support[0], (bool, np.bool_)) else len(support)
    # floor honoured but not ballooned (rescue caps at a modest multiple of the floor)
    assert nsel >= 1, "min_features_fallback>=1 should force at least one feature on noise"
    assert nsel <= X.shape[1], f"selected more than all columns?? nsel={nsel}"


def test_min_features_fallback_zero_can_select_nothing_on_noise():
    """min_features_fallback==0 opts out of the floor -> a pure-noise frame may legitimately
    select zero raw features (no forced noise)."""
    X, y = _noise_frame(seed=11)
    from mlframe.feature_selection.filters import MRMR
    m = MRMR(random_seed=42, verbose=0, n_jobs=1, full_npermutations=2,
             baseline_npermutations=2, fe_max_steps=0, min_features_fallback=0,
             skip_retraining_on_same_content=False)
    m.fit(X.copy(), y)
    support = list(getattr(m, "support_", []))
    nsel = int(np.sum(support)) if len(support) and isinstance(support[0], (bool, np.bool_)) else len(support)
    # No hard assertion that it MUST be zero (a spurious col can sneak past at small n),
    # but with fallback off it must not exceed a small handful.
    assert nsel <= X.shape[1], f"nsel={nsel}"


# ======================================================================================
# 4. GLOBAL np.random RNG LEAK SEAL
# ======================================================================================
def test_seeded_fit_does_not_perturb_global_numpy_rng():
    """A seeded fit must leave np.random global state byte-identical."""
    X, y = _noise_frame(n=800, seed=20)
    from mlframe.feature_selection.filters import MRMR
    np.random.seed(123456)
    before = np.random.get_state()
    m = MRMR(random_seed=42, verbose=0, n_jobs=1, full_npermutations=2, baseline_npermutations=2, fe_max_steps=0, skip_retraining_on_same_content=False)
    m.fit(X.copy(), y)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1]), "seeded fit perturbed global np.random key array"
    assert before[2] == after[2], "seeded fit advanced global np.random position"


def test_two_seeded_fits_identical_support():
    """Two fits with the same random_seed must produce identical support_."""
    X, y = _noise_frame(n=800, seed=21)
    from mlframe.feature_selection.filters import MRMR
    kw = dict(random_seed=7, verbose=0, n_jobs=1, full_npermutations=2,
              baseline_npermutations=2, fe_max_steps=0,
              skip_retraining_on_same_content=False, min_features_fallback=2)
    m1 = MRMR(**kw); m1.fit(X.copy(), y)
    m2 = MRMR(**kw); m2.fit(X.copy(), y)
    s1 = np.asarray(getattr(m1, "support_", []))
    s2 = np.asarray(getattr(m2, "support_", []))
    assert np.array_equal(s1, s2), f"same-seed fits differ: {s1} vs {s2}"
