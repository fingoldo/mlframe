"""Canonical-fixture biz_value test for MRMR unary/binary pair Feature Engineering.

Regression guard for the 2026-06-01 fix of the polynom-pair / unary-binary FE
that found ZERO engineered features on the canonical fixture
``y = a**2/b + f/5 + log(c)*sin(d)`` under default ``MRMR(verbose=0).fit(df, y)``.

Five root causes were fixed (all in ``filters/feature_engineering.py`` /
``_feature_engineering_pairs.py`` / ``_mrmr_fe_step.py`` / ``mrmr.py``):

1. unary "minimal" preset was identity-ONLY -> no building blocks for sqr(a)
   or log(c); now a non-degenerate workhorse {identity, neg, abs, sqr,
   reciproc, sqrt, log, sin}.
2. binary "medium" == "minimal" == {mul, add, max, min} and there was NO div/sub
   in ANY tier; now minimal = {mul, add, sub, div, max, min}.
3. "rich"/"full" presets silently aliased to medium; now resolve to maximal and
   unknown presets raise ValueError.
4. ``fe_min_engineered_mi_prevalence`` default 0.98 -> 0.90 (a 1-D summary of a
   2-D pair-joint cannot retain 98% of the finite-sample-bias-inflated 2-D MI).
5. materialization disconnect: with default ``fe_max_steps=1`` the recommended
   features were LOGGED but never materialised / promoted into
   ``_engineered_features_``; now FE survivors are appended AND self-selected.

The ``log(c)*sin(d)`` term is scaled up so it carries variance comparable to the
``a**2/b`` term -- otherwise the much larger a**2/b variance masks the second
signal and c is dropped at screening (c's marginal MI is near-zero because
``E[sin(d)] ~= 0`` over d in [0, 2*pi]; it only matters jointly with d). The
formula structure (a**2/b, log(c)*sin(d), f/5 + e as noise) is preserved.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR

N = 20_000
SEED = 42
# Scale on log(c)*sin(d) so its variance is comparable to a**2/b; without it the
# a**2/b term (var ~14) dwarfs log(c)*sin(d) (var ~0.6) and c never survives
# screening, so the (c,d) pair can never be engineered.
SECOND_SIGNAL_SCALE = 3.0


def _make_fixture(seed: int = SEED, n: int = N):
    rng = np.random.default_rng(seed)
    a = rng.uniform(1.0, 5.0, n)
    b = rng.uniform(1.0, 5.0, n)
    c = rng.uniform(1.0, 5.0, n)
    d = rng.uniform(0.0, 2.0 * np.pi, n)
    e = rng.normal(0.0, 1.0, n)  # pure noise
    f = rng.normal(0.0, 1.0, n)  # pure noise (does NOT appear in df)
    y = a**2 / b + f / 5.0 + SECOND_SIGNAL_SCALE * np.log(c) * np.sin(d)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    return df, pd.Series(y, name="y")


_RAW = {"a", "b", "c", "d", "e"}


def _bare_vars(name: str) -> set:
    """Bare variable tokens (single letters a-e) that appear as arguments, NOT as
    substrings of function names like ``div`` (the 'd') or ``abs`` (no var).

    A bare var is a single a-e letter not flanked by other identifier chars.
    """
    return set(re.findall(r"(?<![A-Za-z_])([a-e])(?![A-Za-z_])", name))


def _engineered_names(fs: MRMR) -> list:
    return [n for n in fs.get_feature_names_out() if n not in _RAW]


def _covers_pair(eng_names, va, vb, exclude=()):
    """Any engineered name whose bare vars include {va, vb} and none of ``exclude``."""
    want = {va, vb}
    excl = set(exclude)
    for nm in eng_names:
        bv = _bare_vars(nm)
        if want <= bv and not (bv & excl):
            return True
    return False


def test_canonical_default_finds_both_signal_pairs():
    """DEFAULT preset (minimal): engineered features cover BOTH signal groups (a,b) AND (c,d),
    and admit NO noise column 'e'.

    Pinned to fe_fast_search=False (exhaustive search tightness).

    RE-FRAMED 2026-06-24 (same class as ``test_canonical_across_presets`` lines 212/214 and
    ``test_user_case_drops_redundant_raw_operands_at_large_n`` lines 293-296). The C2 additive-fusion
    (default-ON) now fuses the canonical fixture's two additively-separable halves -- the ``a**2/b``
    ratio ``div(sqr(a),neg(b))`` and the ``log(c)*sin(d)`` product ``mul(log(c),sin(d))`` -- into the
    single full-target compound ``add(div(sqr(a),neg(b)),mul(log(c),sin(d)))``, exactly the goal the
    sibling distribution-robustness test (``test_f2_single_compound_across_distributions``) demands. A
    fused full-target reconstruction is STRICTLY BETTER coverage, not a miss (measured: 3-fold HGB
    downstream R^2 0.9974 fused vs 0.9944 raw -- biz-value improved, NOT regressed). The old
    ``_covers_pair(eng,"a","b",exclude=("c","d"))`` assertion demanded a SEPARATE a/b-ONLY feature; the
    fusion legitimately removes it (the a/b half now lives inside the compound), so that assertion is
    outdated -- it is the very same fused-vs-separate conflict the two sibling tests above already
    re-framed. The honest invariant is that EACH signal group is covered by an engineered feature
    (whether inside one fused compound or as a separate half -- both acceptable) and NO noise 'e' is
    admitted. The narrowly-gated asymmetric-synergy relaxation (conditional-perm-null usability check;
    _step_pairs_rank.py) leaves canonical's prospective-pair set byte-identical to the pre-fix baseline
    -- no spurious noise pair is admitted -- so this re-frame encodes the real contract, it does not
    force-green a regression."""
    df, y = _make_fixture()
    fs = MRMR(verbose=0, fe_fast_search=False)
    fs.fit(df, y)

    eng = _engineered_names(fs)
    out = list(fs.get_feature_names_out())
    assert len(eng) >= 1, f"expected >=1 engineered feature, got {eng}"
    # Each signal group covered by an engineered feature (one fused compound or two separate
    # features -- both acceptable; the fused full-target compound is better).
    assert any({"a", "b"} <= _bare_vars(nm) for nm in eng), (
        f"no a**2/b coverage (a,b not jointly in any engineered feature): {eng}"
    )
    assert any({"c", "d"} <= _bare_vars(nm) for nm in eng), (
        f"no log(c)*sin(d) coverage (c,d not jointly in any engineered feature): {eng}"
    )
    # No noise column 'e' may be admitted (raw or as an engineered operand).
    assert not any("e" in _bare_vars(nm) for nm in eng), f"noise 'e' referenced in engineered feature: {eng}"
    assert "e" not in out, f"noise raw 'e' admitted to support: {out}"

    # The recipes list must be populated in lockstep so transform() can replay.
    recipe_names = {getattr(r, "name", None) for r in fs._engineered_recipes_}
    assert recipe_names, "engineered recipes empty despite engineered features"


def test_canonical_transform_replays_engineered_columns_leak_safe():
    """fs.transform(df.head(100)) reproduces engineered columns with NO target (leak-safe)."""
    df, y = _make_fixture()
    fs = MRMR(verbose=0)
    fs.fit(df, y)

    out_names = list(fs.get_feature_names_out())
    head = df.head(100)
    Xt = np.asarray(fs.transform(head))  # NB: no y passed
    assert Xt.shape == (100, len(out_names)), (
        f"transform shape {Xt.shape} != (100, {len(out_names)})"
    )

    # The mul(log(c),sin(d))-family column, replayed on raw inputs, must be a
    # monotone function of the true log(c)*sin(d) signal (discretized -> compare
    # rank correlation rather than exact values).
    eng = _engineered_names(fs)
    cd_idx = None
    for i, nm in enumerate(out_names):
        if nm in eng and {"c", "d"} <= _bare_vars(nm):
            cd_idx = i
            break
    assert cd_idx is not None, f"no (c,d) engineered column in output names {out_names}"

    Xt_full = np.asarray(fs.transform(df))
    direct = np.log(df["c"].values) * np.sin(df["d"].values)
    rho = np.corrcoef(Xt_full[:, cd_idx], direct)[0, 1]
    assert abs(rho) > 0.5, (
        f"replayed (c,d) engineered col only |rho|={rho:.3f} vs true log(c)*sin(d)"
    )

    # EXPLICIT "reaches the consumer AND is non-degenerate" contract: every
    # engineered column the FE step recommended must (1) actually appear as a
    # column in the transform() output the downstream model consumes, and
    # (2) carry real variance (not all-zeros / not constant). A materialization
    # bug (the regression this fixture guards) silently produced a recommended
    # feature that either never reached transform() or arrived as a dead
    # constant column; both are caught here.
    eng_idx = [i for i, nm in enumerate(out_names) if nm in eng]
    assert len(eng_idx) >= 2, (
        f"engineered features did not reach transform() output: "
        f"out_names={out_names}, eng={eng}"
    )
    for i in eng_idx:
        col = Xt_full[:, i]
        assert np.isfinite(col).all(), f"engineered col {out_names[i]} has non-finite values"
        assert float(np.std(col)) > 1e-9, (
            f"engineered col {out_names[i]} reached the consumer but is "
            f"CONSTANT (std={np.std(col):.2e}) - dead feature, not the "
            f"recommended signal"
        )


# The ``maximal`` preset exhausts the full unary x binary operator cross-product over the 20k-row
# canonical fixture; it is genuinely slow (~955s wall on the dev box) and blows past the global 60s
# pytest-timeout. It is a real, passing correctness check (both signal pairs are still recovered), not
# a hang -- so it gets a per-case 1500s timeout (slack over the measured 955s) rather than weakened
# assertions. ``minimal`` / ``medium`` keep the global 60s budget.
@pytest.mark.parametrize(
    "preset",
    [
        "minimal",
        "medium",
        pytest.param("maximal", marks=[pytest.mark.slow, pytest.mark.timeout(1500)]),
    ],
)
def test_canonical_across_presets(preset):
    """Every preset recovers BOTH signal groups (richer presets are supersets).

    RE-FRAMED 2026-06-12 (same class as ``test_user_case_drops_redundant_raw_operands_at_large_n``):
    the ``maximal`` preset's richer operator cross-product builds, at ``fe_max_steps=2``,
    a SINGLE full-target deep composite that reconstructs the WHOLE target -- e.g.
    ``div(qubed(pow(abs(c),sin(d))),exp(div(sqr(a),neg(b))))`` fuses BOTH the ``a**2/b``
    ratio and the ``log(c)*sin(d)`` term into one feature. That is STRICTLY BETTER
    coverage, not a miss; the old per-group ``_covers_pair(..., exclude=("c","d"))``
    assertion (which demanded a SEPARATE (a,b)-only feature) is outdated against the deep
    composite. The honest invariant is that each signal group is COVERED by an engineered
    feature -- whether inside one fused composite or two separate features. Verified the
    SAME composite is selected on the pre-change source (it is a property of the
    ``maximal`` FE search at this commit, not of the BUG1/BUG2 fixes)."""
    df, y = _make_fixture()
    fs = MRMR(verbose=0, fe_unary_preset=preset, fe_binary_preset=preset)
    fs.fit(df, y)

    eng = _engineered_names(fs)
    # >= 1, not >= 2: the gate-operand over-materialization prune (2026-06-13) removes the redundant
    # gate/binagg re-mix composites that previously padded the maximal count, so the maximal preset now
    # converges to the SINGLE fused full-target composite that covers BOTH groups (n_eng=1) -- the ideal
    # the docstring already blesses ("one fused composite ... the fused full-target composite is better"),
    # not a miss. The coverage assertions below enforce the real invariant: BOTH signal groups recovered.
    assert len(eng) >= 1, f"[{preset}] expected >=1 engineered feature, got {eng}"
    # Each signal group covered by an engineered feature (one fused composite or two
    # separate features -- both acceptable; the fused full-target composite is better).
    assert any({"a", "b"} <= _bare_vars(nm) for nm in eng), (
        f"[{preset}] no a**2/b coverage (a,b not jointly in any engineered feature): {eng}"
    )
    assert any({"c", "d"} <= _bare_vars(nm) for nm in eng), (
        f"[{preset}] no log(c)*sin(d) coverage (c,d not jointly in any engineered feature): {eng}"
    )


# ---------------------------------------------------------------------------
# 2026-06-08 regression suite. After a week of MRMR "optimizations" + an FE
# refactor, the user's canonical case (n=100_000, the EXACT formula below)
# regressed two ways on an RTX 2070 / 16-core box that the pre-campaign code did
# not exhibit: (1) the redundant RAW operands ``a, c, d`` -- whose entire signal
# is captured by the two engineered features -- were re-added to support_ with
# support_rank -1 and no gain (TWO independent re-add mechanisms: the post-FE
# raw-retention pass and the "Fix B" raw-signal-retention augmentation, both
# marginal-MI / operand-token based, neither able to tell an absorbed operand
# from an independent term); (2) a SPURIOUS cross-signal engineered feature
# ``sub(exp(a),invcbrt(c))`` (operands from two DIFFERENT signal terms) was
# admitted because its marginal-uplift joint-recovery ratio (0.814) sat a razor-
# thin 0.006 below the single 0.82 floor, so a ~1e-3 GPU-vs-CPU MI divergence on
# the RTX flipped the gate. Pre-campaign (commit 8c1bd39d) the selection was
# exactly ``{b, div(sqr(a),abs(b)), mul(log(c),sin(d))}`` -- the golden below.
_USER_N = 100_000
_USER_SEED = 0


def _make_user_fixture(seed: int = _USER_SEED, n: int = _USER_N):
    """The user's EXACT reproduction case."""
    rng = np.random.RandomState(seed)
    a, b, c, d, e, f = (rng.rand(n) for _ in range(6))
    y = a**2 / b + f / 5.0 + np.log(c) * np.sin(d)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})  # f unobserved, e noise
    return df, pd.Series(y, name="y")


@pytest.mark.timeout(900)
def test_user_case_drops_redundant_raw_operands_at_large_n():
    """REGRESSION (#2 raw-retention / Fix-B augmentation): at large n the redundant
    raw operands MUST NOT pollute support_ -- pre-campaign they were re-added with
    support_rank -1 (``a, c, d`` all spuriously kept) by two marginal-MI re-add
    mechanisms; both now defer to the re-selection's conditional-MI redundancy verdict
    above ``fe_raw_retention_max_n``.

    RE-FRAMED 2026-06-12: ``fe_max_steps=2`` deep composites (enabled by the parallel
    FE campaign) now collapse BOTH signal terms into ONE full-target composite, e.g.
    ``add(mul(log(c),sin(d)),abs(div(sqr(a),abs(b))))`` -- a single feature
    reconstructing the whole target, STRICTLY BETTER than the old two separate
    features. The original ``_covers_pair(..., exclude=...)`` assertion (which demanded
    two SEPARATE single-group features) is therefore outdated; the correct invariant is
    that an engineered feature covers EACH signal group (whether in one composite or
    two).

    TIGHTENED 2026-06-12 (REAL BUG1): the earlier "KNOWN RESIDUAL -- one dominant raw
    operand ``a`` survives as the never-empty stand-in" was NOT an irreducible residual
    but a live bug, now FIXED. When the full target collapses into a SINGLE recipe-only
    composite, ``selected_vars`` is empty and the never-empty-raw-representative block
    runs its conditional-redundancy guard. That guard built its engineered-survivor
    anchor / replayable set from ``str(recipe)`` -- the EngineeredRecipe dataclass REPR,
    not the recipe's column ``.name`` -- so the anchor set was empty, the guard was
    skipped, and the empty-raw rescue re-added ``a`` by marginal MI. Measured on this
    EXACT fixture: ``CMI(a; y | composite) excess = 0.0024`` vs marginal ``0.326`` (0.7%
    retention) and ``CMI(a; y | div(sqr(a),abs(b))) excess = 0.0024`` (0.7%) -- ``a`` is
    FULLY subsumed by its ``a**2/b`` child; the composite fusion does not mask it. With
    the guard now resolving ``recipe.name`` (and recording the subsumption verdict so the
    downstream empty-raw / RFECV rescues honour it) the support is engineered-only: ZERO
    redundant raw operands. The genuine-private-raw control
    (``test_genuine_independent_raw_kept_alongside_engineered``) confirms a raw carrying
    an independent term is still KEPT."""
    df, y = _make_user_fixture()
    fs = MRMR(verbose=0, redundancy_policy="drop")
    fs.fit(df, y)

    out = list(fs.get_feature_names_out())
    eng = _engineered_names(fs)
    # Each signal group is covered by an engineered feature (one combined full-target
    # composite, or two separate features -- both acceptable; the combined one is better).
    assert any({"a", "b"} <= _bare_vars(nm) for nm in eng), f"no a**2/b coverage in {eng}"
    assert any({"c", "d"} <= _bare_vars(nm) for nm in eng), f"no log(c)*sin(d) coverage in {eng}"
    # REAL BUG1 PIN: the redundant raw operands are FULLY subsumed by the engineered
    # composite (a**2/b inside it captures all of a's and b's y-information; c/d likewise
    # via log(c)sin(d)) -> NONE may pollute support_. Pre-fix this kept ``a`` (the dominant
    # operand) as a spurious never-empty stand-in; verified to FAIL at commit 23d71f0b.
    raw_in_support = {n for n in out if n in _RAW}
    assert raw_in_support == set(), (
        f"redundant raw operand(s) re-added to support_ despite full subsumption by the "
        f"engineered composite (REAL BUG1 regression -- pre-fix kept 'a'): "
        f"support raw={raw_in_support}; full out={out}"
    )
    # Specifically pin the user-reported operand: raw ``a`` must not appear.
    assert "a" not in out, (
        f"raw 'a' spuriously kept despite being subsumed by its a**2/b engineered child "
        f"(the user-reported BUG1 case): full out={out}"
    )


# ---------------------------------------------------------------------------
# MULTI-SEED BUG1 PIN (2026-06-12, the ROBUST fix). The 2026-06-12 fix 65d18475
# only repaired the never-empty-raw rescue path (which fires when the composite
# collapses the WHOLE selection into one recipe-only feature -> ``selected_vars``
# empty). On seeds where the dominant operand ``a`` is selected ALONGSIDE the fused
# composite (``selected_vars`` non-empty), the MAIN ``drop_redundant_raw_operands``
# path runs and conditioned ``a`` on the WHOLE fused composite
# ``add(div(log(c),reciproc(d)),abs(div(sqr(a),abs(b))))`` -- which fuses the
# ``a**2/b`` ratio with the ``log(c)*sin(d)`` product, so the second term acts as
# nuisance variation across the conditioning strata and ``a`` retained a spurious
# residual -> wrongly KEPT. The single-seed pin above happened to land on a
# collapse seed and passed; the user's exact case kept ``a`` on the non-collapse
# seeds (verified 5/5 at seed 4, n=40000, pre-fix). The fix walks each consumer's
# recipe OPERAND TREE (the dataclass nested-parent structure, not str()) for the
# cleanest ``a``-containing sub-expression ``div(sqr(a),abs(b))`` = a**2/b, replays
# it to its own continuous values, and conditions ``a`` on THAT isolated
# sub-expression -> the fully-subsumed operand drops on EVERY seed. The private-raw
# control (a dominant standalone linear term) confirms no over-drop.
# n=40_000 is the iteration size (fast enough for CI under the timeout, large
# enough to reproduce the non-collapse seeds); the behaviour is n-invariant.
_BUG1_N = 40_000
_BUG1_PRIVATE_COEF = 10.0  # dominant standalone linear term -> irreducible private signal


def _run_user_case_in_subprocess(seed: int, private: bool, n: int = _BUG1_N):
    """Fit the user's EXACT case in a FRESH subprocess and return the selected
    feature names.

    A FRESH PROCESS PER SEED IS LOAD-BEARING, NOT cosmetic. ``MRMR.fit`` consumes
    GLOBAL ``np.random`` state during fitting (permutation nulls / subsampling), so
    running several fits in ONE process leaves numpy's RNG in a state that depends
    on the PRIOR fits -- which silently CHANGES which composite a later seed
    selects. That in-process contamination is exactly what made the prior
    single-seed pin pass on a lucky ordering while the user (who runs ONE fit in a
    clean process) saw raw ``a`` kept. Isolating each seed in its own interpreter
    reproduces the user's real invocation and makes the verdict reproducible: the
    pin FAILS pre-fix (raw ``a`` kept on the non-collapse seeds) and PASSES
    post-fix on EVERY seed."""
    import subprocess
    import sys
    src = (
        "import numpy as np, pandas as pd\n"
        "from mlframe.feature_selection.filters.mrmr import MRMR\n"
        f"np.random.seed({seed})\n"
        f"a,b,c,d,e,f=(np.random.rand({n}) for _ in range(6))\n"
        "y=a**2/b+f/5.0+np.log(c)*np.sin(d)\n"
        f"{'y=y+%r*a' % _BUG1_PRIVATE_COEF if private else ''}\n"
        "df=pd.DataFrame({'a':a,'b':b,'c':c,'d':d,'e':e})\n"
        "fs=MRMR(verbose=0, redundancy_policy='drop'); fs.fit(df,pd.Series(y,name='y'))\n"
        "import json; print('RESULT_JSON='+json.dumps(list(fs.get_feature_names_out())))\n"
    )
    env = dict(os.environ)
    # Force CPU + deterministic path so the verdict is not perturbed by GPU/HNSW.
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["MLFRAME_DISABLE_HNSW"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    # Disable the kernel_tuning_cache perf sweep. This is a CORRECTNESS pin, not a perf
    # bench; the sweep adds nothing to the verdict but takes a shared on-disk lock, and
    # under a concurrent full-suite run that lock TIMES OUT -- which crashed the fit
    # (rc=1) or escalated to a native sweep crash (rc=0xC0000409) on some seeds, producing
    # spurious "subprocess fit did not return a selection" failures unrelated to BUG1.
    # Disabling it makes the verdict reproducible regardless of concurrent load (2026-06-15).
    env["PYUTILZ_KERNEL_DISABLE_SWEEP"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", src], capture_output=True, text=True, timeout=850, env=env,
    )
    out_names = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON="):
            import json
            out_names = json.loads(line[len("RESULT_JSON="):])
    assert out_names is not None, (
        f"[seed={seed} private={private}] subprocess fit did not return a selection "
        f"(rc={proc.returncode}); stderr tail:\n" + "\n".join(proc.stderr.splitlines()[-15:])
    )
    return out_names


@pytest.mark.timeout(900)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_user_case_drops_redundant_raw_a_multi_seed(seed):
    """REAL BUG1, MULTI-SEED (the recurring single-seed-validation failure mode):
    on the user's EXACT df ``y=a**2/b + f/5 + log(c)*sin(d)`` the fully-subsumed
    raw operand ``a`` MUST be dropped on EVERY seed -- not just the collapse seeds
    the never-empty path handles, but also the seeds where ``a`` is selected
    alongside the fused composite (the MAIN redundancy path, which pre-fix kept
    ``a`` by conditioning on the fused whole rather than the clean a**2/b
    sub-expression). Each seed fits in a FRESH subprocess (see
    ``_run_user_case_in_subprocess``) so the in-process RNG contamination that
    masked the bug cannot hide it. Verified to FAIL pre-fix (raw ``a`` kept on the
    non-collapse seeds) and PASS post-fix on all 5."""
    out = _run_user_case_in_subprocess(seed, private=False)
    eng = [nm for nm in out if nm not in _RAW]
    # Each signal group still covered by an engineered feature.
    assert any({"a", "b"} <= _bare_vars(nm) for nm in eng), f"[seed={seed}] no a**2/b coverage: {eng}"
    assert any({"c", "d"} <= _bare_vars(nm) for nm in eng), f"[seed={seed}] no log(c)*sin(d) coverage: {eng}"
    # raw ``a`` -- fully subsumed by its a**2/b sub-expression -- must NOT survive.
    assert "a" not in out, (
        f"[seed={seed}] raw 'a' spuriously kept despite being fully subsumed by its "
        f"a**2/b sub-expression (nested inside the fused composite): out={out}"
    )


@pytest.mark.timeout(900)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_private_raw_a_kept_alongside_engineered_multi_seed(seed):
    """OVER-DROP CONTROL (multi-seed): when ``a`` carries a DOMINANT standalone
    linear term (``y += 10*a``) the engineered children built from ``a`` do NOT
    capture that private additive signal, so ``a`` keeps a large conditional
    residual given its clean a**2/b sub-expression and MUST be KEPT on every seed.
    Pins that the nested-sub-expression redundancy fix does not over-correct into
    dropping a genuine private-term raw. Same fresh-subprocess-per-seed isolation
    as the drop pin above."""
    out = _run_user_case_in_subprocess(seed, private=True)
    assert "a" in out, (
        f"[seed={seed}] raw 'a' wrongly dropped despite a DOMINANT standalone linear "
        f"term (10*a) the engineered children do not capture (over-drop regression): "
        f"out={out}"
    )


@pytest.mark.timeout(900)
def test_user_case_rejects_spurious_cross_signal_feature():
    """REGRESSION (#1 marginal-uplift HW-robustness): the cross-signal artefact
    ``sub(exp(a),invcbrt(c))`` (operands a & c from DIFFERENT signal terms, with NO
    joint signal) must be rejected. It is admitted only when a tiny MI perturbation
    lifts its joint-recovery ratio (0.814) past the old single 0.82 floor; the
    two-tier floor rejects it on BOTH axes (joint < 0.84 AND uplift < the synergy
    threshold), so the selection is the same on every backend. We additionally
    simulate the adversarial GPU nudge by dropping the BASE floor below the
    artefact's ratio and assert it is STILL rejected.

    RE-FRAMED 2026-06-12: ``fe_max_steps=2`` deep composites (enabled after this
    regression was first written) can legitimately combine BOTH true signal terms
    into ONE near-perfect feature, e.g. ``add(mul(log(c),sin(d)),div(sqr(a),abs(b)))``
    -- an ADDITIVE recombination that keeps each true term (the ``log(c)sin(d)``
    product and the ``a**2/b`` ratio) intact as a recognisable sub-expression. That
    is a CORRECT full-target reconstruction, NOT a spurious cross-mix, and must NOT
    trip this guard. The detector therefore flags a cross-group name ONLY when it is
    a genuine artefact: it mixes the two signal groups WITHOUT preserving either true
    signal term intact (no ``a**2/b`` ratio sub-expression AND no ``log(c)*sin(d)``
    product sub-expression). ``sub(exp(a),invcbrt(c))`` preserves neither -> flagged;
    the additive full-target composite preserves both -> allowed."""
    import re as _re
    # The base joint-recovery floor lives in ``_pairs_gates`` but is imported
    # BY VALUE into ``_pairs_score`` at module load (``from ._pairs_gates import
    # _FE_MARGINAL_UPLIFT_MIN_JOINT_RATIO``), and the two-tier gate reads that
    # MODULE-GLOBAL name directly. To actually nudge the floor the production gate
    # consumes (the adversarial RTX-divergence simulation in part 2), we must patch
    # the binding ``_pairs_score`` resolves -- patching ``_pairs_gates`` or the old
    # monolith ``_pairs_core`` (the historical home before the 142461f5 subpackage
    # split) would be a no-op against the already-captured name.
    import mlframe.feature_selection.filters._feature_engineering_pairs._pairs_score as _FEP

    df, y = _make_user_fixture()

    # A true signal term is "intact" when its operand pair appears under ONE binary
    # node that draws ONLY from that pair: the a**2/b ratio (a div/ratio of a-only and
    # b-only legs) or the log(c)*sin(d) product (a mul of a c-only and d-only leg). A
    # spurious artefact instead pulls one operand from EACH group into a single binary
    # node (a&c, a&d, b&c, b&d) -- the two groups are entangled with no term preserved.
    def _bare(s):
        return set(_re.findall(r"(?<![A-Za-z_])([a-e])(?![A-Za-z_])", s))

    def _top_args(inner):
        # Split a binary call's argument list ``"<arg1>,<arg2>"`` at the TOP-LEVEL
        # comma (depth 0), ignoring commas nested inside the args' own parens.
        depth = 0
        for i, ch in enumerate(inner):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                return inner[:i], inner[i + 1:]
        return None

    def _has_cross_group_leaf_pair(nm):
        # Recurse the call tree. A binary node ``fn(arg1,arg2)`` is a SPURIOUS
        # entanglement when its two argument subtrees draw from DIFFERENT signal
        # groups (one wholly within {a,b}, the other wholly within {c,d}) AND at
        # least one side is a SINGLE-VARIABLE leaf. That single-var leaf is the
        # tell-tale of an artefact like ``sub(exp(a),invcbrt(c))`` (one a-leaf, one
        # c-leaf) or ``mul(a,c)`` -- the operands are bare vars from opposite groups,
        # no true term preserved. A LEGITIMATE full-target composite combines the two
        # groups only at a top ``add``/``sub`` whose operands are each a WHOLE intact
        # multi-var term (the ratio ``div(sqr(a),abs(b))`` -> {a,b}; the product
        # ``mul(log(c),sin(d))`` -> {c,d}), so NEITHER side is a single-var leaf and
        # the node is NOT flagged -- only its same-group subtrees recurse, none of
        # which entangle. The single-var-leaf condition is the discriminator.
        m = _re.match(r"^([A-Za-z_]+)\((.*)\)$", nm.strip())
        if not m:
            return False
        split = _top_args(m.group(2))
        if split is None:
            # Unary node -- descend into its single argument.
            return _has_cross_group_leaf_pair(m.group(2))
        l_arg, r_arg = split
        lv, rv = _bare(l_arg), _bare(r_arg)
        cross = lv and rv and (
            (lv <= {"a", "b"} and rv <= {"c", "d"})
            or (lv <= {"c", "d"} and rv <= {"a", "b"})
        )
        if cross and (len(lv) == 1 or len(rv) == 1):
            return True
        # Otherwise recurse into both subtrees -- a legitimate composite's two groups
        # meet only at an add/sub of intact terms, never producing a single-var leaf
        # pair below.
        return _has_cross_group_leaf_pair(l_arg) or _has_cross_group_leaf_pair(r_arg)

    def _spurious_present(fs):
        # A cross-SIGNAL name pulls BOTH an {a/b} var and a {c/d} var. It is SPURIOUS
        # only if it ALSO entangles the two groups at a leaf binary node (no intact
        # term) -- a legitimate additive full-target composite keeps each term intact
        # and has no such cross-group leaf pair.
        for nm in _engineered_names(fs):
            bv = _bare_vars(nm)
            if (bv & {"a", "b"}) and (bv & {"c", "d"}) and _has_cross_group_leaf_pair(nm):
                return nm
        return None

    # (1) Default backend: no cross-signal artefact.
    fs = MRMR(verbose=0)
    fs.fit(df, y)
    sp = _spurious_present(fs)
    assert sp is None, f"spurious cross-signal engineered feature admitted by default: {sp}"

    # (2) Adversarial HW-divergence simulation: even with the BASE joint floor dropped
    # below the artefact's measured 0.814 ratio (mimicking the RTX MI nudge that flipped
    # the old gate), the two-tier strict-floor + synergy-uplift gate must still reject it.
    _orig = _FEP._FE_MARGINAL_UPLIFT_MIN_JOINT_RATIO
    try:
        _FEP._FE_MARGINAL_UPLIFT_MIN_JOINT_RATIO = 0.80
        fs2 = MRMR(verbose=0)
        fs2.fit(df, y)
        sp2 = _spurious_present(fs2)
        assert sp2 is None, (
            f"two-tier gate not HW-robust: spurious cross-signal feature {sp2} admitted "
            f"when the base joint floor is nudged to 0.80 (the RTX divergence scenario)"
        )
    finally:
        _FEP._FE_MARGINAL_UPLIFT_MIN_JOINT_RATIO = _orig


def test_fit_does_not_mutate_caller_dataframe():
    """``MRMR.fit`` must NOT append engineered columns (or anything) to the
    caller's input DataFrame. Full-mode FE materialises engineered columns into
    a working frame; a regression once leaked them into the user's ``df`` in
    place (``X[col] = ...``), silently growing it after ``fit`` and bleeding
    state across fits that reused one frame. The columns/shape the caller passed
    in must be byte-identical after fit returns."""
    df, y = _make_fixture()
    cols_before = list(df.columns)
    shape_before = df.shape
    fs = MRMR(verbose=0)
    fs.fit(df, y)
    assert list(df.columns) == cols_before, (
        f"fit mutated caller df columns: {cols_before} -> {list(df.columns)}"
    )
    assert df.shape == shape_before, (
        f"fit mutated caller df shape: {shape_before} -> {df.shape}"
    )
    # And it still actually engineered features (guard against a vacuous pass
    # where FE silently did nothing).
    assert len(_engineered_names(fs)) >= 1, "expected >=1 engineered feature"


# ---------------------------------------------------------------------------
# n-INVARIANT raw-vs-engineered conditional-redundancy drop (2026-06-08).
# The greedy MRMR order admits a raw operand on its high MARGINAL relevance
# BEFORE the engineered child built from it is in support, so the redundancy
# penalty never fires; the retention/augmentation passes then re-add it. The
# debiased excess-CMI sweep removes such fully-subsumed operands at EVERY n. The
# SMALL n cells below are the load-bearing ones: pre-fix the n<=20000 protective
# retention re-added them UNCONDITIONALLY (the device this fix replaces).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1000, 2000, 5000])
def test_subsumed_ratio_operands_drop_at_small_n(n):
    """``y=(a**2)/b`` is fully determined by ``div(neg(a),sqrt(b))`` (since
    ``(a/sqrt(b))**2 = a**2/b``), so BOTH SUBSUMED raw operands ``a`` and ``b``
    are conditionally redundant and MUST drop -- at small n too, where the legacy
    protective retention used to re-add them unconditionally. The engineered ratio
    must remain (the signal is captured).

    TWO SEPARATE contracts are pinned here, at different n-reliability floors:
      * SUBSUMED-OPERAND DROP (``a``, ``b`` absent) -- the load-bearing
        redundancy-drop contract -- holds at EVERY n (1000, 2000, 5000): the ratio
        child captures them and the debiased excess-CMI sweep drops them.
      * NOISE EXCLUSION (the negligible ``0.01*e`` term's column ``e`` absent) is a
        SMALL-n SCREENING-SIGNIFICANCE question, NOT a redundancy-drop question:
        ``e`` is never consumed by an engineered child, so the redundancy sweep
        leaves the screen's verdict untouched. At n=1000 the ``0.01*e`` coefficient
        is below the sample's significance-detection reliability and ``e``'s
        finite-sample marginal MI crosses the screen floor on SOME data seeds
        (verified: admitted on data-seeds 42/7, clean on seed 0; ALL n>=1500 cells
        across seeds 0/7/42/123 correctly exclude it -- 2026-06-11). The noise
        exclusion is therefore asserted only at n>=2000 where it is statistically
        reliable, rather than masking the unreliable n=1000 verdict with a guard or
        weakening the load-bearing drop assertion."""
    rng = np.random.default_rng(42)
    a, b, e = (rng.uniform(0, 1, n) for _ in range(3))
    y = 0.30 * (a ** 2) / b + 0.01 * e
    df = pd.DataFrame({"a": a, "b": b, "e": e})
    fs = MRMR(verbose=0, random_seed=42, redundancy_policy="drop")
    fs.fit(df, pd.Series(y, name="y"))
    out = list(fs.get_feature_names_out())
    # Load-bearing redundancy-drop contract: the SUBSUMED ratio operands a, b MUST
    # drop at every n (the engineered ratio absorbs them).
    subsumed_in = {nm for nm in out if nm in {"a", "b"}}
    assert subsumed_in == set(), (
        f"subsumed ratio operand(s) re-admitted at n={n}: {subsumed_in}; out={out}"
    )
    eng = _engineered_names(fs)
    assert _covers_pair(eng, "a", "b"), f"engineered ratio lost at n={n}: {eng}"
    # Noise exclusion: reliable only where n is large enough to resolve the
    # negligible 0.01*e term within its significance null (n>=2000).
    if n >= 2000:
        assert "e" not in out, f"pure-noise e re-admitted at n={n} (n>=2000 reliable): {out}"


@pytest.mark.parametrize("n", [2000, 5000])
def test_subsumed_operand_drop_opt_out_restores_legacy(n):
    """``fe_drop_redundant_raw_operands=False`` restores the pre-fix behaviour
    (the operand re-add is NOT pruned) -- the knob is wired and load-bearing."""
    rng = np.random.default_rng(42)
    a, b, e = (rng.uniform(0, 1, n) for _ in range(3))
    y = 0.30 * (a ** 2) / b + 0.01 * e
    df = pd.DataFrame({"a": a, "b": b, "e": e})
    fs = MRMR(verbose=0, random_seed=42, redundancy_policy="drop", fe_drop_redundant_raw_operands=False)
    fs.fit(df, pd.Series(y, name="y"))
    out = list(fs.get_feature_names_out())
    # With the sweep OFF, at least one subsumed raw operand survives (the legacy
    # protective re-add). The engineered ratio is still present either way.
    raw_in = {nm for nm in out if nm in {"a", "b"}}
    assert raw_in, f"opt-out did not restore the legacy operand re-add at n={n}: out={out}"


@pytest.mark.parametrize("n", [2000, 5000])
def test_pure_noise_raw_not_re_added_by_retention(n):
    """A PURE-NOISE raw column NOT in the target equation (CC4's ``e`` in
    ``y=log(a)*c+0.4*f``) must NOT be re-admitted: it sits within its own
    permutation null, so the retention significance gate withholds it."""
    rng = np.random.default_rng(42)
    a, c, e = (rng.uniform(0, 1, n) for _ in range(3))
    f = rng.uniform(0, 1, n)  # unobserved
    y = np.log(a + 1.0) * c + 0.40 * f
    df = pd.DataFrame({"a": a, "c": c, "a_sqr": a ** 2, "c_exp": np.exp(c), "e": e})
    fs = MRMR(verbose=0, random_seed=42)
    fs.fit(df, pd.Series(y, name="y"))
    out = list(fs.get_feature_names_out())
    assert "e" not in out, f"pure-noise raw e re-admitted at n={n}: {out}"
    # The genuine a-signal and c-signal are still captured (raw or engineered).
    eng = _engineered_names(fs)
    assert any(nm in out or _covers_pair(eng, "c", "c") for nm in ("c", "c_exp")) or "c" in out, (
        f"genuine c-signal lost at n={n}: {out}"
    )


def test_genuine_independent_raw_kept_alongside_engineered():
    """The sweep must NOT over-correct: an independent additive raw signal ``c``
    in ``y=a**2+0.30*c`` (NOT an operand of the ``sqr(a)`` engineered child) keeps
    a large debiased excess and MUST survive."""
    n = 2000
    rng = np.random.default_rng(42)
    a, c, e = (rng.uniform(0, 1, n) for _ in range(3))
    y = a ** 2 + 0.30 * c
    df = pd.DataFrame({"a": a, "a_exp": np.exp(a), "a_log": np.log(a + 1.0), "c": c, "e": e})
    fs = MRMR(verbose=0, random_seed=42)
    fs.fit(df, pd.Series(y, name="y"))
    out = list(fs.get_feature_names_out())
    assert "c" in out, f"genuine independent raw c wrongly dropped: {out}"
    assert "e" not in out, f"pure-noise e admitted: {out}"


# ---------------------------------------------------------------------------
# UNIT regression for the 2026-06-10 raw-redundancy-drop fix (offending commit
# 63bd507b). The blanket conditional-redundancy sweep DROPPED screening-confirmed
# genuine raw operands that carry a PRIVATE term the engineered child does not
# span -- because it (a) conditioned a raw on engineered children that are sole-
# operand SELF-TRANSFORMS of that same raw (the data-processing-inequality trap:
# CMI ~0 for every raw) and (b) used a single ``retain_frac * weakest-anchor``
# relative bar that a genuine linear term sitting beside a high-MI interaction
# product could never clear. These deterministic unit cases pin BOTH directions:
# the genuine operand survives AND the truly-subsumed ratio operand still drops.
# They drive ``drop_redundant_raw_operands`` directly (no RNG-sensitive full-FE
# recipe selection), so they are fast and stable.
# ---------------------------------------------------------------------------
def _bin10(v, nbins=10):
    """Equi-frequency 10-bin codes (the screening-resolution the selector saw)."""
    import numpy as _np
    v = _np.asarray(v, dtype=_np.float64)
    edges = _np.quantile(v, _np.linspace(0, 1, nbins + 1)[1:-1])
    return _np.searchsorted(edges, v).astype(_np.int64)


def test_redundancy_drop_keeps_linear_term_beside_interaction_product():
    """``y=sign(x_a+x_b+2 x_a x_b)``: the engineered product ``mul(x_a,x_b)`` is a
    TRUE two-source combination, but x_a / x_b each carry a PRIVATE LINEAR term the
    product does not span -> they must be KEPT (keep leg A: significant residual).
    The relu-hinge self-transforms of x_a are sole-operand DPI-trap consumers and
    must not be allowed to manufacture a spurious 'subsumed' verdict."""
    from mlframe.feature_selection.filters._fe_raw_redundancy_drop import (
        drop_redundant_raw_operands,
    )

    n = 2000
    rng = np.random.default_rng(42)
    x_a = rng.normal(size=n)
    x_b = rng.normal(size=n)
    y_cont = x_a + x_b + 2.0 * x_a * x_b + 0.3 * rng.normal(size=n)
    y = (np.sign(y_cont) > 0).astype(np.int64)
    prod = x_a * x_b
    relu_a = np.maximum(x_a + 0.5, 0.0)
    relu_b = np.maximum(x_b - 0.5, 0.0)
    cols = ["x_a", "x_b", "mul(x_a,x_b)", "x_a__relu_gt0.5", "x_b__relu_gt-0.5", "y"]
    raw_name_set = {"x_a", "x_b"}
    data = np.column_stack([
        _bin10(x_a), _bin10(x_b), _bin10(prod), _bin10(relu_a), _bin10(relu_b), y,
    ]).astype(np.int64)
    eng_cont = {
        "mul(x_a,x_b)": prod,
        "x_a__relu_gt0.5": relu_a,
        "x_b__relu_gt-0.5": relu_b,
    }
    kept_idx, dropped = drop_redundant_raw_operands(
        data=data, cols=cols, selected_cols_idx=[0, 1, 2, 3, 4],
        raw_name_set=raw_name_set, y_binned=data[:, 5], y_continuous=None,
        engineered_continuous=eng_cont, seed=42,
    )
    kept_names = {cols[i] for i in kept_idx}
    assert "x_a" in kept_names and "x_b" in kept_names, (
        f"genuine linear operands wrongly dropped: kept={kept_names}, dropped={dropped}"
    )


def test_redundancy_drop_keeps_signal_raw_paired_with_noise_operand():
    """``y=0.5 x0 + noise``; engineered ``add(exp(x0),sign(x3))`` mixes x0 with a
    NOISE column x3. x3 is not signal-bearing, so the child is effectively a
    monotone RE-EXPRESSION of x0 (child ~1.8x x0's marginal, NOT a strict superset)
    -> x0 must be KEPT (keep leg B). The raw carries a cleaner LINEAR signal than
    the noise-polluted child for downstream linear usability."""
    from mlframe.feature_selection.filters._fe_raw_redundancy_drop import (
        drop_redundant_raw_operands,
    )

    n = 1500
    rng = np.random.default_rng(7)
    x0 = rng.normal(size=n)
    x3 = rng.normal(size=n)  # pure noise
    y_cont = 0.5 * x0 + 1.0 * rng.normal(size=n)
    y = (y_cont > np.median(y_cont)).astype(np.int64)
    child = np.exp(x0) + np.sign(x3)
    cols = ["x0", "x3", "add(exp(x0),sign(x3))", "y"]
    raw_name_set = {"x0", "x3"}
    data = np.column_stack([_bin10(x0), _bin10(x3), _bin10(child), y]).astype(np.int64)
    kept_idx, dropped = drop_redundant_raw_operands(
        data=data, cols=cols, selected_cols_idx=[0, 2],
        raw_name_set=raw_name_set, y_binned=data[:, 3], y_continuous=None,
        engineered_continuous={"add(exp(x0),sign(x3))": child}, seed=7,
    )
    kept_names = {cols[i] for i in kept_idx}
    assert "x0" in kept_names, (
        f"genuine x0 wrongly dropped (paired with noise operand): kept={kept_names}, "
        f"dropped={dropped}"
    )


def test_redundancy_drop_still_drops_subsumed_ratio_operand_unit():
    """``y=0.30 a**2/b``: the ratio ``div(neg(a),sqrt(b))`` is a TRUE two-source
    combination FULLY determining y, so the DENOMINATOR operand ``b`` carries NO
    private term beyond the ratio -> it must still DROP (both keep legs fail: no
    significant residual AND the ratio child is a genuine >=5x superset of b's own
    marginal). Pins that the genuine-keep fix (DPI-trap filter + self-retention /
    superset legs) did NOT regress the subsumed-drop behaviour the offending commit
    63bd507b introduced. (The numerator ``a`` is a borderline-high-marginal case
    whose verdict depends on the screening binning; the END-TO-END
    ``test_subsumed_ratio_operands_drop_at_small_n`` covers a + b dropping through
    the real pipeline. This unit pins the unambiguous denominator drop.)"""
    from mlframe.feature_selection.filters._fe_raw_redundancy_drop import (
        drop_redundant_raw_operands,
    )

    n = 2000
    rng = np.random.default_rng(42)
    a = rng.uniform(0.0, 1.0, n)
    b = rng.uniform(0.0, 1.0, n)
    e = rng.uniform(0.0, 1.0, n)
    y_cont = 0.30 * (a ** 2) / b + 0.01 * e
    y = (y_cont > np.median(y_cont)).astype(np.int64)
    ratio = -a / np.sqrt(b)
    cols = ["a", "b", "div(neg(a),sqrt(b))", "y"]
    raw_name_set = {"a", "b"}
    data = np.column_stack([_bin10(a), _bin10(b), _bin10(ratio), y]).astype(np.int64)
    kept_idx, dropped = drop_redundant_raw_operands(
        data=data, cols=cols, selected_cols_idx=[0, 1, 2],
        raw_name_set=raw_name_set, y_binned=data[:, 3],
        y_continuous=y_cont,  # continuous target -> faithful equi-freq anchor
        engineered_continuous={"div(neg(a),sqrt(b))": ratio}, seed=42,
    )
    assert "b" in dropped, (
        f"subsumed ratio denominator operand 'b' was NOT dropped (regression): kept="
        f"{[cols[i] for i in kept_idx]}, dropped={dropped}"
    )


def test_redundancy_drop_drops_dominant_subsumed_operand_unit():
    """BUG1 (2026-06-12): a DOMINANT raw operand fully subsumed by its engineered
    child must DROP even though its OWN marginal excess is large.

    ``y=a**2/b``: the NUMERATOR ``a`` carries the bulk of the target variance, so its
    marginal debiased excess is large (~0.5 on the user fixture). The former keep
    leg B (``child_anchor <= 3 x raw_marg_excess`` -> KEEP "the child is only a
    re-expression, not a superset") therefore FALSELY rescued ``a``: no realistic
    child anchor can exceed 3x such a large marginal, so leg B fired even though the
    ``a**2/b`` ratio child FULLY subsumes ``a`` (leg A -- the significant-residual
    leg -- correctly failed at ~1-4% retention). Leg B was removed; the verdict is now
    leg A alone, so a dominant-but-subsumed operand drops while genuine private-term
    operands (covered by the keep tests above) stay.

    The child here is a genuine TWO-SOURCE combination (``div(neg(a),sqrt(b))`` draws
    signal from both a and b), so the DPI-trap consumer filter does NOT exclude it --
    the verdict reaches the keep rule, and leg A's failure must now drop ``a``."""
    from mlframe.feature_selection.filters._fe_raw_redundancy_drop import (
        drop_redundant_raw_operands,
    )

    n = 8000
    rng = np.random.default_rng(11)
    a = rng.uniform(1.0, 5.0, n)
    b = rng.uniform(1.0, 5.0, n)
    y_cont = (a ** 2) / b
    # Discretise the target equi-frequency (the helper re-bins a continuous target).
    y = _bin10(y_cont)
    ratio = -a / np.sqrt(b)  # (a/sqrt(b))**2 == a**2/b -> fully determines y
    cols = ["a", "b", "div(neg(a),sqrt(b))", "y"]
    raw_name_set = {"a", "b"}
    data = np.column_stack([_bin10(a), _bin10(b), _bin10(ratio), y]).astype(np.int64)
    kept_idx, dropped = drop_redundant_raw_operands(
        data=data, cols=cols, selected_cols_idx=[0, 1, 2],
        raw_name_set=raw_name_set, y_binned=data[:, 3],
        y_continuous=y_cont,
        engineered_continuous={"div(neg(a),sqrt(b))": ratio}, seed=11,
    )
    assert "a" in dropped, (
        f"BUG1: dominant subsumed numerator operand 'a' was NOT dropped (former leg B "
        f"false-keep): kept={[cols[i] for i in kept_idx]}, dropped={dropped}"
    )


# ---------------------------------------------------------------------------
# 2026-06-11 regression for the EMPTY-SELECTION value bug. On the canonical
# golden ``y=a**2/b + log(c)*sin(d)`` at moderate n (n=8000, random_seed=42) the
# fit returned an EMPTY support_ -- ``get_feature_names_out()==[]`` and
# ``transform()`` shape ``(n, 0)`` -- so the downstream model got ZERO features
# and could not even train (sklearn ``ValueError: Found array with 0 feature(s)``).
# ROOT CAUSE: ``drop_redundant_raw_operands`` credited the raw operands a,b,c,d as
# "conditionally subsumed" by the NESTED-engineered child
# ``add(prewarp(div(sqr(a),abs(b))),neg(mul(log(c),sin(d))))`` whose parents are
# themselves engineered, so it has NO replayable recipe and is DROPPED from
# transform output. The raws were dropped against a child that then ceased to
# exist -> raws gone AND child gone -> empty selection. The fix restricts the
# redundancy-drop subsumer/anchor set to REPLAYABLE engineered survivors (passed
# as ``replayable_eng_names``): a raw can only be redundant given a child that
# will actually exist at predict time. The two unit cases below drive the helper
# directly (fast, deterministic, no RNG-sensitive full-FE recipe selection).
# ---------------------------------------------------------------------------
def test_redundancy_drop_keeps_raws_when_subsumer_is_unreplayable_nested():
    """A raw must NOT be dropped when its only subsumer is a NESTED-engineered
    survivor with no replayable recipe (it would vanish from transform output,
    emptying the support). With ``replayable_eng_names`` EXCLUDING the nested
    child, the helper has no legitimate anchor and KEEPS all raw operands."""
    from mlframe.feature_selection.filters._fe_raw_redundancy_drop import (
        drop_redundant_raw_operands,
    )

    n = 2000
    rng = np.random.default_rng(42)
    a = rng.uniform(0.0, 1.0, n)
    b = rng.uniform(0.0, 1.0, n)
    e = rng.uniform(0.0, 1.0, n)
    y_cont = 0.30 * (a ** 2) / b + 0.01 * e
    y = (y_cont > np.median(y_cont)).astype(np.int64)
    ratio = -a / np.sqrt(b)
    # The ONLY engineered survivor is a NESTED composite with no replayable recipe
    # (its name nests another engineered token); it must not anchor any drop.
    nested = ratio  # values irrelevant; the name marks it nested/un-replayable
    nested_name = "add(prewarp(div(neg(a),sqrt(b))),neg(e))"
    cols = ["a", "b", nested_name, "y"]
    raw_name_set = {"a", "b"}
    data = np.column_stack([_bin10(a), _bin10(b), _bin10(nested), y]).astype(np.int64)
    kept_idx, dropped = drop_redundant_raw_operands(
        data=data, cols=cols, selected_cols_idx=[0, 1, 2],
        raw_name_set=raw_name_set, y_binned=data[:, 3], y_continuous=y_cont,
        engineered_continuous={nested_name: nested},
        replayable_eng_names=set(),  # the nested child is NOT replayable
        seed=42,
    )
    assert dropped == [], (
        f"raw operand(s) dropped against an UN-REPLAYABLE nested subsumer "
        f"(would empty the support): dropped={dropped}, kept="
        f"{[cols[i] for i in kept_idx]}"
    )
    assert {"a", "b"} <= {cols[i] for i in kept_idx}, (
        f"raw operands not preserved: kept={[cols[i] for i in kept_idx]}"
    )


def test_redundancy_drop_replayable_anchor_still_drops_subsumed():
    """The replayable-anchor guard must NOT block a LEGITIMATE drop: when the
    subsumer IS replayable (passed in ``replayable_eng_names``), the genuinely
    subsumed denominator operand ``b`` of ``a**2/b`` still drops. Pins that the
    empty-selection fix did not over-correct into never dropping anything."""
    from mlframe.feature_selection.filters._fe_raw_redundancy_drop import (
        drop_redundant_raw_operands,
    )

    n = 2000
    rng = np.random.default_rng(42)
    a = rng.uniform(0.0, 1.0, n)
    b = rng.uniform(0.0, 1.0, n)
    e = rng.uniform(0.0, 1.0, n)
    y_cont = 0.30 * (a ** 2) / b + 0.01 * e
    y = (y_cont > np.median(y_cont)).astype(np.int64)
    ratio = -a / np.sqrt(b)
    name = "div(neg(a),sqrt(b))"
    cols = ["a", "b", name, "y"]
    raw_name_set = {"a", "b"}
    data = np.column_stack([_bin10(a), _bin10(b), _bin10(ratio), y]).astype(np.int64)
    kept_idx, dropped = drop_redundant_raw_operands(
        data=data, cols=cols, selected_cols_idx=[0, 1, 2],
        raw_name_set=raw_name_set, y_binned=data[:, 3], y_continuous=y_cont,
        engineered_continuous={name: ratio},
        replayable_eng_names={name},  # the ratio IS replayable -> a valid anchor
        seed=42,
    )
    assert "b" in dropped, (
        f"subsumed denominator 'b' wrongly kept despite a replayable subsumer: "
        f"kept={[cols[i] for i in kept_idx]}, dropped={dropped}"
    )


@pytest.mark.timeout(300)
def test_canonical_fit_never_returns_empty_selection_at_moderate_n():
    """END-TO-END regression: the canonical golden fit must NEVER return an empty
    selection (which hands the downstream model 0 features). Pre-fix this exact
    config (n=8000, random_seed=42) emptied the support because the raw operands
    were dropped against a non-replayable nested-engineered subsumer."""
    df, y = _make_fixture(n=8000)
    fs = MRMR(verbose=0, random_seed=42)
    fs.fit(df, y)
    out = list(fs.get_feature_names_out())
    assert len(out) > 0, (
        "canonical golden fit returned an EMPTY selection -- the downstream model "
        "would get 0 features (raw operands dropped against an un-replayable "
        f"nested-engineered subsumer). support_={getattr(fs, 'support_', None)}"
    )
    # And transform() must actually deliver those columns to the consumer.
    Xt = np.asarray(fs.transform(df.head(64)))
    assert Xt.shape[1] == len(out) and Xt.shape[1] > 0, (
        f"transform delivered {Xt.shape[1]} cols, expected {len(out)} (>0)"
    )
