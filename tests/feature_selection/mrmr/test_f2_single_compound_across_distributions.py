"""GOLDEN GOAL: the F2 target y = a**2/b + f/5 + log(c)*sin(d) must be recovered as ONE clean fused
compound add(div(sqr(a),b), mul(log(c),sin(d))) (or an algebraically-equivalent neg/neg/abs form) NO MATTER
the input distribution -- uniform, [1,5]-scaled, heavy-tailed, per-feature-mixed, outlier-contaminated.

This pins the distribution-robustness goal: selection must not depend on whether the operands are scaled
[0.1,1.1] vs [1,5] vs lognormal/gamma/etc. The known failure mode (signal-scale imbalance: when a**2/b
dominates Var(y), the weak log(c)*sin(d) half falls below the relevance/prevalence gate and the compound
fragments) is what the residual-aware FE step exists to fix.

Reuses the shared multi-distribution operand generator (tests/feature_selection/_synthetic_distributions.py).
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR
from tests.feature_selection._synthetic_distributions import sample_operands

_ID = re.compile(r"[a-zA-Z_]\w*")
_DOMAINS = {"a": "any", "b": "divisor", "c": "positive", "d": "any", "e": "any"}
# f is the irreducible-noise term -- NOT a feature.
# GOAL SPEC: ``uniform`` is a HARD regression guard (recovered today). The imbalanced / dirty profiles are
# xfail(strict=True) -- the distribution-robustness GOAL (one clean compound regardless of input
# distribution). strict=True makes the test FAIL the moment a fix makes any of them pass, forcing the
# xfail to be removed -- so this self-polices and is NOT a "deferred bug" (it is a known-unimplemented goal).
#
# ROOT CAUSE (corrected 2026-06-22 by two empirical residual-FE investigations -- the ORIGINAL "weak half
# falls below the prevalence gate, fix via residual retarget" premise was REFUTED by per-profile diagnostics):
# variance imbalance breaks the goal in TWO distinct ways, and residual-retarget (targeting MRMR on
# r = y - E_hat[y|selected]) is the WRONG tool for both -- the ridge residual is a/b-shaped, so retarget
# re-chases the dominant half. uniform passes because Var(a**2/b) ~= Var(c*d) there, so both halves are
# constructed AND the nested-parent pair search fuses them into one add(...).
#   * FUSION-BLOCKED (heavy_tailed): BOTH halves ARE constructed as engineered features (the weak
#     mul(log(c),sin(d)) AND an a/b-group half div(neg(b),a__p2sin1)) but never combined. FIXED 2026-06-24
#     by C2 additive-fusion (_fe_additive_fusion.propose_additive_fusions): two surviving engineered
#     halves with DISJOINT raw-token sets whose add(...) MI clears the stronger half's marginal-perm floor
#     are fused via the existing unary_binary + nested_parent_a/b recipe (binary_name="add"; byte-exact
#     replay, no new recipe kind); the fused compound wins re-selection and the now-subsumed fragments
#     (engineered + their raw operands) are dropped. heavy_tailed now recovers the single compound.
#   * FORM-SELECTION-BLOCKED (mixed): FIXED 2026-06-24. The ORIGINAL "a**2/b is NEVER constructed" diagnosis
#     was REFUTED by per-pair diagnostics -- the (a,b) half IS built; the blocker was that the WRONG functional
#     FORM won the per-pair winner selection. Under the per-feature-mixed marginals (a~beta_u, b~gamma) the
#     additive summary ``add(log(a),invsqrt(b))`` scored binned MI 0.1180 vs the EXACT ratio ``div(sqr(a),b)``
#     at 0.1167 -- a 0.0013 BINNING-NOISE gap -- yet its linear usability is far worse (|corr(y)| 0.25 vs 0.46).
#     ``_select_single_best`` broke that epsilon-MI tie on bit-exact ``>``, crowning the additive form; C2 then
#     fused THAT weak form into a poor compound while the correct ratio form (re-derived at step 2 from the still-
#     present raw a/b) survived as a fragment, and the c/d half reappeared too -> 2 full compounds + 2 fragments.
#     The fix snaps the PRIMARY MI key in ``_select_single_best`` to a chance-fluctuation tie band (3% of the pool
#     max, the binned-MI plug-in noise scale ``(k-1)(k_y-1)/2n``) so STATISTICALLY-tied forms are resolved by the
#     EXISTING linear-usability tie-break -- the same "prefer the linearly-usable member of an MI-equivalence
#     class" rule, extended from bit-equal to statistically-equal MI. The ratio half now wins, C2 fuses the same
#     clean halves as uniform, and mixed recovers the single compound add(mul(log(c),sin(d)),div(sqr(a),abs(b))).
#   * DOMINANT-CAPTURE-BLOCKED (scaled_1_5): FIXED 2026-06-24. The blocker was NOT a corrupted a/b capture --
#     under [1,5]-scaling the (c,d) half is an ASYMMETRIC synergy pair (only the bootstrap operand ``c`` is
#     unselected; ``c``'s MARGINAL MI ~= 0 because E[sin(d)] ~= 0, yet log(c) carries real CONDITIONAL signal
#     jointly with d). Its joint-MI ratio (~1.13) cleared the regular 1.05 prevalence bar but FAILED the strict
#     1.5 synergy bar, so the clean ``mul(log(c),sin(d))`` half was never built and C2 had nothing to fuse. The
#     4-part fix lets the clean c/d half survive so C2 fuses it: (a) relax the synergy prevalence bar to the
#     regular 1.05 for an ASYMMETRIC pair whose SINGLE bootstrap operand has ~0 marginal MI (pool-scale-relative
#     gate -- canonical's symmetric/non-~0 noise pairs keep the strict 1.5 bar; _step_pairs_rank.py); (b) admit a
#     fusion on a scale/sign-invariant 2-half OLS multiple-R separability path ALONGSIDE the binned-MI margin
#     (variance imbalance makes binned-MI under-credit the weak half; _fe_additive_fusion.py); (c) run C2 BEFORE
#     the cross-fold stability vote so the weak c/d half (which alone fails the per-fold quorum) is fused before
#     it can be dropped, and the FUSED compound then faces the vote (_step_score.py); (d) preserve each subsumed
#     fragment's recipe in a side-store so a step-2 re-derivation of a composite nesting the fragment stays
#     REPLAYABLE (else it is recorded recipe-less and dropped, collapsing the support back to raw a/b/d;
#     _step_score.py). scaled_1_5 now recovers the single fused compound add(abs(div(sqr(a),b)),mul(log(c),sin(d))).
#   * TAIL-CONCENTRATED-AB-BLOCKED (with_outliers): the PRIOR diagnosis ("a/b corrupted, log(c) DROPPED, no clean
#     c/d half") was REFUTED by per-pair diagnostics 2026-06-24. The c/d half IS built cleanly -- ``mul(log(c),sin(d))``
#     wins its pair at binned MI ~1.34 and is selected; log(c) is NOT dropped. The blocker is entirely the a/b half,
#     and it is NOT "corruption" but a genuine, IRREDUCIBLE rank-MI property of the contaminated data: the 3%/15-IQR
#     outliers injected into operands a and b make ``a**2/b`` a TAIL-CONCENTRATED signal -- in the clean BULK (95% of
#     rows) Var(a**2/b) collapses to ~0.0014 vs Var(log(c)sin(d)) ~0.62 (a ~440x imbalance), and Spearman(a**2/b, y)
#     in the bulk is ~0.05 (essentially zero RANK association). a**2/b tracks y only in the 5% outlier tail. So the
#     rank-based binned MI of EVERY a/b form sits in a tight noise band 0.04-0.07: the true ratio ``div(sqr(a),b)``
#     reads MI 0.0626 while a spurious noise form ``min(reciproc(a),sign(b))`` reads 0.0730 -- a REAL, reproducible
#     0.010 lead (bootstrap P(spurious>true)=0.99), NOT binning noise. No statistically-honest tie band covers it
#     (Miller-Madow bias 0.004, 95th-pct chance-MI floor 0.005, 99th-pct 0.006 are all << 0.010), so the spurious
#     form legitimately wins the per-pair winner-selection AND the (a,b) pair is then dropped upstream by BOTH the
#     prospective-pair ranking (its plain pair-MI 0.094 loses to the d-pairs at ~1.2) AND the engineered-MI
#     prevalence gate -- all of which are calibrated on RANK MI. The two forms DISAGREE overwhelmingly on LINEAR
#     usability (|corr(continuous y)| 0.986 true vs 0.371 spurious -- corr is outlier-inflated), so a usability-aware
#     winner-selection CAN crown ``div(sqr(a),b)``, but that alone does NOT recover the compound: the ratio half (MI
#     0.0626) cannot clear the rank-MI-calibrated engineered-MI / joint-prevalence gate against a pair whose plain
#     pair-MI is 0.094, and the (a,b) pair is dropped from the prospective list entirely in the relaxed retry pass.
#     A safe fix needs the prospective-pair ranking AND the engineered-MI prevalence gate AND the per-pair winner
#     selection to ALL credit tail-concentrated LINEAR-usable (OLS-R / |corr|) signal the coarse rank-MI under-credits
#     -- a multi-gate change that risks flipping selection on the canonical fixtures (where rank-MI is the honest
#     arbiter) and the 4 passing profiles (where div(sqr(a),b) is both the rank-MI AND the usability leader, so no
#     usability override is needed). Attempted 2026-06-24 (usability-dominance leader admission in _pairs_emit +
#     usability_override in _select_single_best): it correctly flips the (a,b) WINNER to div(sqr(a),b) but the pair
#     is still killed upstream by the prospective-ranking + prevalence gates, so with_outliers stays unrecovered;
#     reverted (the winner-selection-only change adds canonical risk without achieving the goal). Stays xfail pending
#     a unified tail-concentrated-signal credit across the upstream rank-MI gates.
_XFAIL_CAPTURE = (
    "GOAL: a/b half is TAIL-CONCENTRATED under outliers (rank-MI ~0 in bulk, signal only in the 5% tail) -- "
    "the true div(sqr(a),b) ratio loses the rank-MI race to a spurious form by a REAL 0.010 (not noise) and "
    "the (a,b) pair is dropped by the rank-MI-calibrated prospective-ranking + prevalence gates. ROOT CAUSE "
    "(2026-06-25 accuracy experiment, score_prospective_pairs in _step_pairs_rank.py:171-220): the admission "
    "gate has ONLY rank-MI/CMI signals (prevalence = pair_mi>sum*bar; maxt = pair_mi>=floor; the perm-CMI "
    "path itself requires _passes_maxt). The DISTINGUISHING signal -- linear |corr| (true 0.986 vs spurious "
    "0.371) -- is unavailable at this gate (discrete codes only; the |corr| re-test lives downstream in the "
    "escalation, BEHIND the maxt clearance the tail-concentrated true form fails). rank-MI/CMI cannot, by "
    "construction, separate a tail-concentrated-true (low rank-MI, high |corr|) from a spurious (high rank-MI, "
    "low |corr|) pair. FIX needs a CROSS-CUTTING change: plumb continuous y + form materialisation (or a "
    "precomputed |corr|/OLS-R signal) INTO score_prospective_pairs and add a usability-gated admission under a "
    "tail-concentration detector -- not a local gate tweak; touches the primary pair-ranking path so it "
    "requires its own canonical biz-value gate (reject if canonical regresses)."
)
_PROFILES = [
    "uniform",
    "scaled_1_5",  # FIXED 2026-06-24 by the 4-part asymmetric-synergy / OLS-separability / pre-vote-fusion / subsumed-recipe-preservation fix.
    "heavy_tailed",  # FIXED 2026-06-24 by C2 additive-fusion (both engineered halves built but unfused).
    "mixed",  # FIXED 2026-06-24 by the binned-MI tie-band form-selection fix (see ROOT CAUSE note below).
    pytest.param("with_outliers", marks=pytest.mark.xfail(strict=True, reason=_XFAIL_CAPTURE)),
]


def _make(profile: str, n: int, seed: int):
    if profile == "scaled_1_5":
        rng = np.random.default_rng(seed)
        ops = {k: rng.uniform(1.0, 5.0, n) for k in ("a", "b", "c")}
        ops["d"] = rng.uniform(0.0, 2 * np.pi, n)
        ops["e"] = rng.uniform(1.0, 5.0, n)
        f = rng.uniform(1.0, 5.0, n)
    else:
        ops = sample_operands(seed, n, _DOMAINS, profile=profile)
        f = sample_operands(seed + 991, n, {"f": "any"}, profile=profile)["f"]
    import pandas as pd
    df = pd.DataFrame({k: ops[k].astype(np.float64) for k in ("a", "b", "c", "d", "e")})
    y = ops["a"] ** 2 / ops["b"] + f / 5.0 + np.log(np.abs(ops["c"]) + 1e-9) * np.sin(ops["d"])
    return df, y


def _cols(nm):
    return set(_ID.findall(nm)) & {"a", "b", "c", "d", "e"}


def _classify(names):
    full, frag_ab, frag_cd = [], [], []
    for nm in names:
        cs = _cols(nm)
        has_ab, has_cd = bool(cs & {"a", "b"}), bool(cs & {"c", "d"})
        if has_ab and has_cd:
            full.append(nm)
        elif has_ab:
            frag_ab.append(nm)
        elif has_cd:
            frag_cd.append(nm)
    return full, frag_ab, frag_cd


@pytest.mark.parametrize("profile", _PROFILES)
def test_f2_one_compound_under_distribution(profile):
    df, y = _make(profile, n=10_000, seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = MRMR(full_npermutations=10, baseline_npermutations=20, fe_max_steps=2,
                  fe_min_pair_mi_prevalence=1.05, verbose=0, n_jobs=1).fit(df, y)
    names = [str(s) for s in fs.get_feature_names_out()]
    full, frag_ab, frag_cd = _classify(names)
    assert all("e" not in _cols(nm) for nm in names), f"[{profile}] noise 'e' referenced: {names}"
    assert len(full) >= 1, f"[{profile}] no feature fuses both a/b and c/d halves: {names}"
    assert not frag_cd, f"[{profile}] redundant c/d-only fragment(s) alongside the compound: {frag_cd} :: {names}"
    assert not frag_ab, f"[{profile}] redundant a/b-only fragment(s) alongside the compound: {frag_ab} :: {names}"
    assert len(full) == 1, f"[{profile}] expected exactly ONE fused compound, got {len(full)}: {full}"
    # the c/d half must keep the log(c) factor (not degrade to bare sin(d))
    assert "c" in _cols(full[0]), f"[{profile}] compound dropped the log(c) factor: {full[0]}"
