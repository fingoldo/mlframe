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
#   * CONSTRUCTION-BLOCKED (mixed): only the c/d half mul(log(c),sin(d)) is built as an engineered feature;
#     the a/b half a**2/b is NEVER constructed (it stays as raw a + raw b), so there is NO second engineered
#     half for C2 to fuse with. This is an upstream CONSTRUCTION miss, not a fusion miss -- C2 cannot help
#     without first building the a/b half (fusing the c/d half with raw a would drop the **2/b and leave
#     raw b as a fragment). Stays xfail pending the a/b-half construction fix.
#   * DOMINANT-CAPTURE-BLOCKED (scaled_1_5, with_outliers): the dominant a/b half is captured corrupted and
#     log(c) is DROPPED (-> cbrt(d)/sin(d)); there is no clean c/d half to fuse. Blocker lives upstream in the
#     pair-search leader/tie-break + linear-usability guard (_step_core.py ~486-498), a research-grade FE-quality
#     fix that must not regress the canonical test_biz_value_mrmr_fe_canonical (same signal, strong config,
#     currently recovered perfectly). Must precede C2; C2 alone gets at most 2/4.
# (Two no-ship iterations are documented in the work plan; the strict xfails stay until a fix lands all 4.)
_XFAIL_CONSTRUCT = "GOAL: a/b half a**2/b NEVER constructed as an engineered feature (stays raw a+b) -- no second engineered half for C2 to fuse; needs upstream a/b-half construction fix"
_XFAIL_CAPTURE = "GOAL: dominant a/b half corrupted under variance imbalance (log(c) dropped) -- needs upstream dominant-capture fix before C2"
_PROFILES = [
    "uniform",
    pytest.param("scaled_1_5", marks=pytest.mark.xfail(strict=True, reason=_XFAIL_CAPTURE)),
    "heavy_tailed",  # FIXED 2026-06-24 by C2 additive-fusion (both engineered halves built but unfused).
    pytest.param("mixed", marks=pytest.mark.xfail(strict=True, reason=_XFAIL_CONSTRUCT)),
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
