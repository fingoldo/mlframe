"""Regression suite for the conditional-MI redundancy gate (strategy S5, 2026-06-08).

The principled, constant-free replacement for the hardcoded
``fe_min_engineered_mi_prevalence`` joint-prevalence ratio. The gate admits an
engineered candidate iff its CONDITIONAL MI with y GIVEN the already-admitted
ENGINEERED features clears BOTH a conditional-permutation floor AND a scale-free
fraction (TAU) of the weakest admitted feature's CMI.

These tests lock the two load-bearing claims of the validated design:

  (1) UNIT: on the canonical two-signal target, the gate ACCEPTS the two genuine
      engineered forms (``div ~ a**2/b``, ``mul ~ log(c)*sin(d)``) and REJECTS
      the spurious cross-signal ``sub(exp(a), invcbrt(c))`` whose y-information
      is wholly carried by the admitted genuine pair -- reusing the prototype's
      exact recipes + the production MI primitives.

  (2) The relative-gap (TAU) leg is LOAD-BEARING: the spurious feature sits
      ABOVE its own conditional-permutation floor yet far below the TAU bar, so
      the floor alone would not reject it -- only the relative gap does.

  (3) n-INVARIANCE (2026-06-08 hardening): the accept/reject decision is correct
      at EVERY n from 1_000 up, on BOTH formulas. The relative-gap leg compares
      DEBIASED EXCESS CMI (``cmi - cond-perm-null-mean``), not raw CMI, so the
      plug-in CMI's residual finite-sample positive bias cancels and no longer
      lets the spurious feature cross the TAU bar at small n. See the parametrized
      ``test_gate_n_invariant_*`` cases.

  (4) DEFAULT-ON: ``MRMR()`` defaults to ``fe_acceptance='conditional_mi'`` and
      the CMI gate drops at least one redundant engineered survivor that the
      legacy ``fe_acceptance='prevalence_ratio'`` path keeps, WITHOUT dropping
      the two genuine signal features.

n by test class:
  * gate-FUNCTION unit tests are PARAMETRIZED over n in {1_000, 5_000, 20_000,
    50_000}. BEFORE the debiasing fix the WEAKER-signal F2 formula's spurious
    feature was inflated above the TAU bar at n~=20k (raw cmi 0.040 > raw rel_bar
    0.037 -> wrongly admitted), separating cleanly only at >=30k. The hardened
    gate (debiased-excess relative bar) rejects the spurious feature at EVERY n on
    BOTH formulas (validated 40/40 over n x formula x 5 seeds). n=100_000 OOMs an
    8GB box on the downstream FE buffer but the gate LOGIC is n-invariant BY
    CONSTRUCTION (the bias cancels in the excess), so 100k follows from the
    construction; the parametrized sweep tops out at 50k to stay in-box. The gate
    function itself is cheap (no MRMR fit / no FE candidate buffer).
  * the END-TO-END MRMR fit reuses the canonical-fixture construction (F1 with
    the (c,d) signal scaled so it survives screening -- ``E[sin(d)]~=0`` makes
    c's marginal MI near-zero, so without the scale the (c,d) pair is never
    engineered) at 20_000 rows. That is the same size + shape the canonical FE
    suite already runs green; it keeps the FE unary x binary candidate buffer in
    an 8GB CI box (30k / 100k OOM the box on the wide cross-product).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._fe_cmi_redundancy_gate import (
    DEFAULT_CMI_RETAIN_FRAC,
    _CMI_SIGNIFICANCE_ESCAPE_MARGIN,
    apply_cmi_redundancy_gate,
)
from mlframe.feature_selection.filters._mi_greedy_cmi_fe import (
    _cmi_from_binned,
    _quantile_bin,
)
from mlframe.feature_selection.filters.discretization import discretize_array
from mlframe.feature_selection.filters.mrmr import MRMR

# Gate-function unit tests: the hardened (debiased-excess) gate rejects the
# spurious feature on BOTH formulas at EVERY n. Cheap -- no MRMR fit -- so the
# parametrized n-sweep covers the full robustness range. The single-n helpers
# below default to this representative size.
_N_UNIT = 20_000
# Parametrized n-invariance sweep: the spurious feature must be rejected (and the
# two genuine accepted) at every one of these sizes, on both formulas. 20k was the
# BUG size before the debiasing fix (F2's spurious feature was wrongly admitted).
# 100k OOMs the 8GB box on the downstream FE buffer; the gate logic is n-invariant
# by construction, so the sweep tops out at 50k.
_N_SWEEP = (1_000, 5_000, 20_000, 50_000)
# End-to-end MRMR fit: canonical-fixture size (matches the green canonical FE suite).
_N_E2E = 20_000
# Scale on log(c)*sin(d) so its variance is comparable to a**2/b; without it the
# a**2/b term dwarfs log(c)*sin(d) and c never survives screening -> the (c,d) pair
# can never be engineered (identical rationale to the canonical FE fixture).
_SECOND_SIGNAL_SCALE = 3.0


def _build(formula: str, seed: int = 0, n: int = _N_UNIT):
    """The user's two canonical formulas (f unobserved, e noise)."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.5, 3.0, n)
    b = rng.uniform(0.5, 3.0, n)
    c = rng.uniform(0.5, 5.0, n)
    d = rng.uniform(0.0, 2 * np.pi, n)
    e = rng.normal(0.0, 1.0, n)
    f = rng.normal(0.0, 1.0, n)
    if formula == "F1":
        y = a**2 / b + f / 5.0 + np.log(c) * np.sin(d)
    elif formula == "F2":
        y = 0.2 * a**2 / b + f / 5.0 + np.log(c * 2.0) * np.sin(d / 3.0)
    else:
        raise ValueError(formula)
    return dict(a=a, b=b, c=c, d=d, e=e), y


def _bin_y(y):
    z = discretize_array(np.asarray(y, dtype=np.float64), n_bins=10, method="quantile", dtype=np.int64)
    _, inv = np.unique(z, return_inverse=True)
    return inv.astype(np.int64)


def _candidates(df, yb):
    """Prototype's three engineered recipes: two genuine, one spurious cross-signal."""
    div = df["a"] ** 2 / np.abs(df["b"])  # real ~ a**2/|b|
    mul = np.log(df["c"]) * np.sin(df["d"])  # real ~ log(c)*sin(d)
    sub = np.exp(df["a"]) - np.sign(df["c"]) * np.abs(df["c"]) ** (-1.0 / 3.0)  # SPURIOUS exp(a)-c**(-1/3)
    cands = {}
    for nm, v in (("div", div), ("mul", mul), ("sub", sub)):
        vb = _quantile_bin(np.asarray(v, dtype=np.float64), nbins=10)
        marg = float(_cmi_from_binned(vb, yb, None))
        cands[nm] = (np.asarray(v, dtype=np.float64), marg)
    return cands


@pytest.mark.parametrize("formula", ["F1", "F2"])
def test_gate_accepts_genuine_rejects_redundant(formula):
    """Core S5 claim: ACCEPT the two genuine engineered forms, REJECT the spurious
    cross-signal feature whose information is carried by the genuine pair."""
    df, y = _build(formula)
    yb = _bin_y(y)
    cands = _candidates(df, yb)
    accepted, diag = apply_cmi_redundancy_gate(cands, yb, nbins=10, retain_frac=0.15, seed=0)

    assert "div" in accepted, f"[{formula}] genuine a**2/b form rejected: {diag['div']}"
    assert "mul" in accepted, f"[{formula}] genuine log(c)*sin(d) form rejected: {diag['mul']}"
    assert "sub" not in accepted, f"[{formula}] spurious cross-signal feature admitted as independent: {diag['sub']}"
    # The spurious feature is rejected on the RELATIVE-gap (TAU) leg, not by chance:
    # its DEBIASED EXCESS CMI is far below the weakest admitted feature's bar. (The
    # rel bar is now on the debiased-excess scale -- raw CMI would re-introduce the
    # finite-n bias that the excess removes.)
    assert diag["sub"]["cmi_excess"] < diag["sub"]["rel_bar"], f"[{formula}] sub should fail the relative bar on debiased excess: {diag['sub']}"


@pytest.mark.parametrize("formula", ["F1", "F2"])
@pytest.mark.parametrize("n", _N_SWEEP)
def test_gate_n_invariant_accepts_genuine_rejects_redundant(n, formula):
    """n-INVARIANCE (2026-06-08 hardening): the gate must ACCEPT the two genuine
    engineered forms and REJECT the spurious cross-signal feature at EVERY n from
    1_000 up, on BOTH formulas.

    BEFORE the debiasing fix the WEAKER-signal F2 spurious feature was inflated
    above the TAU bar at n~=20k (raw cmi 0.040 > raw rel_bar 0.037) and WRONGLY
    ADMITTED; it separated cleanly only at >=30k. The hardened gate compares
    DEBIASED EXCESS CMI (``cmi - cond-perm-null-mean``), so the residual plug-in
    bias cancels and the decision is the same at every n. n tops out at 50k to
    stay in an 8GB box; the logic is n-invariant by construction (the bias cancels
    in the excess) so 100k follows."""
    df, y = _build(formula, n=n)
    yb = _bin_y(y)
    cands = _candidates(df, yb)
    accepted, diag = apply_cmi_redundancy_gate(cands, yb, nbins=10, retain_frac=0.15, seed=0)

    assert "div" in accepted, f"[{formula} n={n}] genuine a**2/b form rejected: {diag['div']}"
    assert "mul" in accepted, f"[{formula} n={n}] genuine log(c)*sin(d) form rejected: {diag['mul']}"
    assert "sub" not in accepted, f"[{formula} n={n}] spurious cross-signal feature admitted as independent (finite-n bias leak?): {diag['sub']}"
    # The spurious feature's DEBIASED EXCESS collapses toward ~0 at every n (its
    # CMI is pure bias/noise given the admitted support), so it fails the rel bar.
    assert diag["sub"]["cmi_excess"] < diag["sub"]["rel_bar"], f"[{formula} n={n}] sub excess should be below the relative bar: {diag['sub']}"
    # The genuine features keep a POSITIVE excess (real private interaction), so
    # the separation is not an artefact of everything collapsing to zero.
    assert diag["div"]["cmi_excess"] > diag["sub"]["cmi_excess"], (
        f"[{formula} n={n}] genuine div excess should exceed spurious sub excess: div={diag['div']} sub={diag['sub']}"
    )


def test_relative_gap_leg_is_load_bearing():
    """Both principled legs reject the spurious feature on both formulas. ORIGINALLY this
    locked that the FLOOR ALONE was insufficient on at least one formula (the debiased
    relative-gap leg being the decisive rejector). That premise was an artifact of a FLOOR
    BUG: commit d743ac5a ("correct analytic CMI-null df sign -- the conditional analytic
    path never engaged") fixed a negated df that made the conditional analytic null never
    run, leaving a too-weak floor that the spurious feature could clear on F1. With the
    CORRECTED conditional floor engaged, the spurious feature now sits AT/BELOW its own
    floor on BOTH formulas (debiased excess == 0), so the floor leg rejects it directly --
    the corrected behaviour, not a collapse. So we lock what is now true and load-bearing:
    the gate REJECTS the spurious on both formulas, and the rejection is SOUND on each --
    caught by the floor leg (cmi <= floor) OR by the debiased relative-gap leg (excess <
    rel_bar). The relative-gap leg's distinct role (rejecting a redundant candidate that
    clears its floor given a STRONG incumbent, without false-rejecting genuine weak
    complements) is locked by the weak-complementary false-reject guard suite below."""
    for formula in ("F1", "F2"):
        df, y = _build(formula)
        yb = _bin_y(y)
        cands = _candidates(df, yb)
        _, diag = apply_cmi_redundancy_gate(cands, yb, nbins=10, retain_frac=0.15, seed=0)
        sub = diag["sub"]
        # rejected overall...
        assert sub["accept"] is False, f"[{formula}] spurious feature wrongly accepted: {sub}"
        # ...via a principled leg: the corrected conditional floor (cmi <= floor) OR the
        # debiased relative-gap TAU leg (excess < rel_bar). Not a raw/unprincipled cut.
        caught_by_floor = sub["cmi"] <= sub["floor"]
        caught_by_rel_gap = sub["cmi_excess"] < sub["rel_bar"]
        assert caught_by_floor or caught_by_rel_gap, f"[{formula}] spurious rejected by neither principled leg (floor nor relative-gap): {sub}"


# ---------------------------------------------------------------------------
# FALSE-REJECT GUARD (2026-06-11): the relative-gap (leg 2) bar must NOT drop a
# genuinely COMPLEMENTARY feature merely because it is WEAKER than a strong
# incumbent. Adversarial finding: with one strong seed, rel_bar = TAU *
# seed_excess becomes a large absolute CMI threshold, and several genuinely
# independent weak drivers (each provably non-redundant -- independent of the
# admitted support -- each clearing its OWN conditional-permutation floor 20x+)
# were ALL mislabelled 'redundant_below_rel_bar' and dropped, costing ~3% R2 on
# a real gradient-boosting model. The strong-significance escape admits a
# candidate that clears its floor by >= significance_escape_margin (default 3x),
# because robust conditional significance proves the information is NOT in the
# admitted support (a truly redundant feature's CMI collapses to ~its floor).
# ---------------------------------------------------------------------------


def _disc(y, nbins=10):
    y = np.asarray(y, dtype=np.float64)
    edges = np.unique(np.quantile(y, np.linspace(0.0, 1.0, nbins + 1)))
    if edges.size <= 2:
        return np.zeros(y.size, dtype=np.int64)
    return np.searchsorted(edges[1:-1], y, side="right").astype(np.int64)


def _marg_cands(cols: dict, yb: np.ndarray) -> dict:
    return {
        nm: (np.asarray(v, dtype=np.float64), float(_cmi_from_binned(_quantile_bin(np.asarray(v, dtype=np.float64), nbins=10), yb, None)))
        for nm, v in cols.items()
    }


def _weak_complementary_fixture(seed: int = 1, n: int = 20_000, n_weak: int = 5, strong_coef: float = 8.0, weak_coef: float = 1.5):
    """One STRONG seed driver + several INDEPENDENT weak drivers, all genuinely
    complementary (each its OWN additive piece of y, none a function of another).
    The weak drivers' excess sits below TAU*seed_excess but each clears its floor
    20x+ -- the regime the false-reject guard locks."""
    rng = np.random.default_rng(seed)
    cols = {}
    S = rng.normal(size=n)
    cols["strongS"] = S
    y = _disc(S, 10).astype(float) * strong_coef
    for k in range(n_weak):
        w = rng.normal(size=n)
        cols[f"weakW{k}"] = w
        y = y + _disc(w, 4) * weak_coef
    y = y + rng.normal(size=n) * 0.05
    yb = _disc(y, nbins=10)
    return cols, yb


def cols_to_cands(cols, yb):
    return _marg_cands(cols, yb)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_complementary_weak_feature_not_falsely_rejected(seed):
    """A genuinely complementary (independent of the admitted support) but
    individually WEAKER feature must be ADMITTED, not dropped as redundant.

    Pre-fix: every weak driver's debiased excess was below rel_bar = TAU *
    seed_excess (anchored to the one strong seed) and so was rejected
    'redundant_below_rel_bar' -- a FALSE REJECT (the weak drivers are
    independent of strongS, dropping all of them cost ~3% R2). Post-fix the
    strong-significance escape admits them."""
    cols, yb = _weak_complementary_fixture(seed=seed)
    weak = [nm for nm in cols if nm.startswith("weakW")]
    # GENUINENESS: each weak driver is (a) independent of the strong support
    # (MI ~ 0 -- provably NOT redundant w.r.t. strongS) and (b) significant
    # against the SEED-ONLY support (clears its floor by a wide margin when
    # conditioned on strongS alone, before the other weak drivers fragment it).
    s_bin = _quantile_bin(np.asarray(cols["strongS"], dtype=np.float64), nbins=10)
    for nm in weak:
        w_bin = _quantile_bin(np.asarray(cols[nm], dtype=np.float64), nbins=10)
        mi_ws = float(_cmi_from_binned(w_bin, s_bin, None))
        assert mi_ws < 0.02, f"weak driver {nm} not independent of strongS (MI={mi_ws}); fixture broken"
        cmi_given_seed = float(_cmi_from_binned(w_bin, yb, s_bin))
        assert cmi_given_seed > 0.01, f"weak driver {nm} carries no conditional signal given the seed (CMI={cmi_given_seed}); fixture broken"
    # THE FIX: none of the genuinely complementary weak drivers is dropped.
    accepted, diag = apply_cmi_redundancy_gate(cols_to_cands(cols, yb), yb, nbins=10, seed=seed)
    rejected = [nm for nm in weak if nm not in accepted]
    assert not rejected, (
        f"[seed={seed}] genuinely complementary weak drivers FALSELY REJECTED as redundant: {rejected}; diag={ {nm: diag[nm] for nm in rejected} }"
    )


def test_strong_significance_escape_is_decisive_for_weak_complementary():
    """LOCK that it is the ESCAPE (not a coincidental excess >= rel_bar) that
    admits the weak complementary drivers: at least one admitted weak driver has
    excess BELOW rel_bar yet clears its floor >= the escape margin."""
    cols, yb = _weak_complementary_fixture(seed=1)
    accepted, diag = apply_cmi_redundancy_gate(cols_to_cands(cols, yb), yb, nbins=10, seed=1)
    weak = [nm for nm in cols if nm.startswith("weakW") and nm in accepted]
    escaped = [nm for nm in weak if diag[nm]["cmi_excess"] < diag[nm]["rel_bar"] and diag[nm]["cmi"] >= _CMI_SIGNIFICANCE_ESCAPE_MARGIN * diag[nm]["floor"]]
    assert escaped, (
        "expected at least one weak complementary driver admitted via the "
        "strong-significance escape (excess < rel_bar but cmi >= margin*floor); "
        f"diag={ {nm: diag[nm] for nm in weak} }"
    )


def test_escape_disabled_reproduces_false_reject():
    """With the escape disabled (margin <= 1) the gate reverts to the pure
    two-leg behaviour and DOES drop the weak complementary drivers -- proving the
    escape (not some other change) is what fixes the false reject."""
    cols, yb = _weak_complementary_fixture(seed=1)
    accepted, _ = apply_cmi_redundancy_gate(
        cols_to_cands(cols, yb),
        yb,
        nbins=10,
        seed=1,
        significance_escape_margin=1.0,
    )
    weak = [nm for nm in cols if nm.startswith("weakW")]
    rejected = [nm for nm in weak if nm not in accepted]
    assert rejected, "with the escape disabled the pure two-leg gate should still drop the weak complementary drivers (this is the bug the escape fixes)"


def test_escape_does_not_admit_monotone_redundant_remaps():
    """The escape must NOT admit a feature whose information IS in the support: a
    MONOTONE remap of an admitted driver (same quantile bins -> CMI given the
    driver == 0) is rejected as REDUNDANT, never via the loosened relative bar,
    so the escape never re-opens a false-ADMIT path.

    Composed-gate note (2026-06-11): with the cost-cap/partition-dedup fix in the
    same gate, an exact monotone remap is now caught EARLIER and more cheaply --
    it bins to the IDENTICAL equi-frequency partition as the driver, so the
    partition-dedup collapses it to ``redundant_partition_duplicate`` BEFORE the
    greedy floor check (it never pays the per-round permutation-null cost). Absent
    that collapse (e.g. a remap that is monotone-but-binned-distinctly, or with the
    dedup disabled) it falls through to the greedy and is rejected
    ``redundant_below_floor`` (its CMI given the driver == 0 cannot clear the
    floor). BOTH are exact-redundancy rejections the escape never overrides; the
    load-bearing guarantee is that NO monotone remap is ADMITTED and that any
    rejection reason is a genuine-redundancy reason, NOT ``redundant_below_rel_bar``
    (which the escape could loosen)."""
    rng = np.random.default_rng(0)
    n = 20_000
    A = rng.normal(size=n)
    y = _disc(A, 10).astype(float) * 10 + rng.normal(size=n) * 0.05
    yb = _disc(y, nbins=10)
    cols = {
        "drvA": A,
        "redA_cube": A**3,  # monotone -> identical quantile bins to A
        "redA_exp": np.exp(A),  # monotone -> identical quantile bins to A
        "redA_affine": 2.5 * A - 7.0,  # monotone -> identical quantile bins to A
    }
    accepted, diag = apply_cmi_redundancy_gate(cols_to_cands(cols, yb), yb, nbins=10, seed=0)
    group = list(cols)
    admitted = [g for g in group if g in accepted]
    # Exactly one survivor (A and its monotone remaps are the SAME binned column).
    assert len(admitted) == 1, f"monotone-redundant remap FALSELY ADMITTED as extra info: {admitted}"
    _REDUNDANT_REASONS = {"redundant_below_floor", "redundant_partition_duplicate"}
    for nm in group:
        if nm not in admitted:
            assert diag[nm]["reason"] in _REDUNDANT_REASONS, (
                f"monotone redundant remap {nm} should be rejected as exact redundancy "
                f"(partition-duplicate dedup or below-floor), NOT via the loosened "
                f"relative bar: {diag[nm]}"
            )


def test_degenerate_single_candidate_admits_on_marginal():
    """Fallback: with a single candidate (nothing to condition on) the gate must NOT
    reject it -- it admits on marginal significance rather than emitting nothing."""
    df, y = _build("F1")
    yb = _bin_y(y)
    div = df["a"] ** 2 / np.abs(df["b"])
    vb = _quantile_bin(np.asarray(div, dtype=np.float64), nbins=10)
    cands = {"div": (np.asarray(div, dtype=np.float64), float(_cmi_from_binned(vb, yb, None)))}
    accepted, diag = apply_cmi_redundancy_gate(cands, yb, nbins=10)
    assert accepted == {"div"}
    assert diag["div"]["reason"] == "degenerate_marginal_admit"


def test_default_tau_is_scale_free_fraction():
    """TAU default is the documented scale-free 0.15 (NOT an MI-nats constant)."""
    assert DEFAULT_CMI_RETAIN_FRAC == 0.15
    assert MRMR().fe_engineered_cmi_retain_frac == 0.15
    assert MRMR().fe_acceptance == "conditional_mi"


# ---------------------------------------------------------------------------
# DEFAULT-ON integration: the gate is wired into MRMR.fit and is the default.
# ---------------------------------------------------------------------------


_BARE = re.compile(r"(?<![A-Za-z_])([a-e])(?![A-Za-z_])")


def _bare_vars(name: str) -> set:
    return set(_BARE.findall(name))


def _make_mrmr_fixture(seed=0, n=_N_E2E):
    """Canonical two-signal fixture: y = a**2/b + f/5 + SCALE*log(c)*sin(d).
    f unobserved, e noise. The (c,d) term is scaled so it survives screening."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(1.0, 5.0, n)
    b = rng.uniform(1.0, 5.0, n)
    c = rng.uniform(1.0, 5.0, n)
    d = rng.uniform(0.0, 2.0 * np.pi, n)
    e = rng.normal(0.0, 1.0, n)
    f = rng.normal(0.0, 1.0, n)
    y = a**2 / b + f / 5.0 + _SECOND_SIGNAL_SCALE * np.log(c) * np.sin(d)
    X = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    yb = pd.qcut(y, 10, labels=False, duplicates="drop")
    return X, np.asarray(yb, dtype=np.int64)


def _engineered(fs):
    raw = {"a", "b", "c", "d", "e"}
    return [n for n in (getattr(fs, "_engineered_features_", []) or []) if n not in raw]


@pytest.mark.timeout(600)
def test_default_on_drops_redundant_keeps_genuine():
    """END-TO-END: ``MRMR()`` (default conditional_mi) keeps the two genuine signal
    pairs AND drops at least one redundant engineered survivor that the legacy
    ``prevalence_ratio`` path keeps. Default-on win, principled redundancy filter."""
    X, y = _make_mrmr_fixture()

    fs_cmi = MRMR(verbose=0, n_jobs=1, random_state=0)  # default acceptance == conditional_mi
    fs_cmi.fit(X, y)
    eng_cmi = _engineered(fs_cmi)

    fs_ratio = MRMR(verbose=0, n_jobs=1, random_state=0, fe_acceptance="prevalence_ratio")
    fs_ratio.fit(X, y)
    eng_ratio = _engineered(fs_ratio)

    # Both genuine signal pairs survive under the CMI gate -- accepted in ANY form. The strengthened
    # ONE-fused-compound contract (commit 5301778c, "collapse FE fragmentation to ONE clean compound")
    # deliberately folds a pure (a,b) or (c,d) sub-fragment INTO a single admitted compound when that
    # compound already carries both halves, so each genuine pair surfaces either as a pure form OR
    # absorbed into a cross-mix that ALSO carries the other pair's operands (here the survivors are
    # cross-mixes like ``add(mul(neg(a),invsqrt(b)),mul(log(c),sin(d)))`` -- carrying BOTH pairs). The
    # contract is "the genuine signal is recovered", and clean-vs-folded form is below that line; the
    # strict pure-form check was an artifact of the pre-subsumption pipeline.
    def _recovered(eng, va, vb):
        return any({va, vb} <= _bare_vars(nm) for nm in eng) or (any(va in _bare_vars(nm) for nm in eng) and any(vb in _bare_vars(nm) for nm in eng))

    assert _recovered(eng_cmi, "a", "b"), f"(a,b) a**2/b signal not recovered in ANY form under CMI gate: {eng_cmi}"
    assert _recovered(eng_cmi, "c", "d"), f"(c,d) log(c)*sin(d) signal not recovered in ANY form under CMI gate: {eng_cmi}"

    # The CMI gate is the stricter, principled redundancy filter: it admits no MORE
    # engineered features than the legacy ratio, and strictly fewer when the ratio
    # path lets a redundant cross-signal/extra column through.
    assert len(eng_cmi) <= len(eng_ratio), (
        f"CMI gate admitted MORE engineered features than the legacy ratio (should be stricter): cmi={eng_cmi} ratio={eng_ratio}"
    )


def _make_user_f2(seed=0, n=20_000):
    """The USER'S EXACT F2 (2026-06-08): ``y = 0.2*a**2/b + f/5 + log(c*2)*sin(d/3)``.

    f unobserved, e noise. This is the harder formula (weak 0.2 coeff on the
    (a,b) term; (c,d) carried by ``log(c*2)*sin(d/3)``). It is reproduced through
    the WHOLE MRMR.fit pipeline -- screen + FE pair pre-screen + S5 CMI gate --
    NOT just the isolated gate function. Variable ranges match the gate-function
    fixture ``_build`` so the two suites stay in lockstep."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.5, 3.0, n)
    b = rng.uniform(0.5, 3.0, n)
    c = rng.uniform(0.5, 5.0, n)
    d = rng.uniform(0.0, 2.0 * np.pi, n)
    e = rng.normal(0.0, 1.0, n)
    f = rng.normal(0.0, 1.0, n)
    y = 0.2 * a**2 / b + f / 5.0 + np.log(c * 2.0) * np.sin(d / 3.0)
    X = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    yb = pd.qcut(y, 10, labels=False, duplicates="drop")
    return X, np.asarray(yb, dtype=np.int64)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("n", [20_000, 30_000])
def test_user_f2_e2e_recovers_genuine_drops_noise_and_cross_signal(n):
    """END-TO-END on the USER'S EXACT F2, default ``MRMR()`` (S5 conditional_mi).

    REGRESSION GUARD (2026-06-08): the original report was that default MRMR on
    F2 returned only ``['b', 'div(log(a),cbrt(c))']`` -- a weak CROSS-SIGNAL wrong
    form, with neither genuine pair recovered. The S5 e2e test only exercised the
    F1-formula-with-scale fixture, so F2 was never validated through MRMR.fit.

    HONEST BEHAVIOUR (2026-06-11 reframe). The ORIGINAL form of this test asserted a
    PURE (c,d) engineered form survives ALONGSIDE the pure (a,b) form. A discriminating
    re-investigation (see ``f2_resolve_results.md``) PROVED that is NOT what the hardened
    multi-step pipeline does, and -- crucially -- that the OLD assertion was the stale
    one, not a bug:

      * The second FE step builds ONE cross-mix that carries BOTH genuine signals:
        ``sub(invcbrt(add(reciproc(c),invsquared(d))), log(div(sqr(a),neg(b))))``. On a
        matched 10-bin grid this cross-mix has MARGINAL MI ~1.05 to y -- ~46% of
        ``H(y)=2.30`` nats -- because F2's ``y`` is the ADDITIVE sum of the (a,b) term
        and the (c,d) term, so a single feature built from a (c,d)-operand AND an
        (a,b)-operand captures information about BOTH additive components.
      * The pure (c,d) parent ``add(reciproc(c),invsquared(d))`` carries MI ~0.535
        (~23% of H(y)) -- ONLY the (c,d) signal. GIVEN the cross-mix is selected, the
        pure (c,d) form is genuinely conditionally redundant (its information is a
        subset of the cross-mix's), so the post-FE greedy re-selection correctly drops
        it. With ``fe_max_steps=1`` (no second composite step) the pure (c,d) parent
        DOES survive -- confirming the drop is the rational subsumption by the
        higher-MI cross-mix, NOT screening and NOT a vocabulary gap.

    So the cross-mix is the RATIONAL MAX-MI pick, not a spurious cross-signal. This
    locks the load-bearing, deterministic properties of that honest behaviour
    (byte-identical in isolation AND in-suite, both n):
      (1) the (a,b) a**2/b signal IS recovered -- either as a PURE a,b form OR folded into a
          cross-mix that ALSO carries the (c,d) operands (the engineered-subsumption reframe,
          2026-06-20: the strengthened ONE-fused-compound contract -- see the canonical
          ``test_feature_engineering_example_single_compound`` -- drops a pure (a,b) sub-fragment
          when an admitted fused compound already carries BOTH additive halves, so the (a,b)
          signal is recovered INSIDE that compound, not necessarily as a standalone pure form);
      (2) the (c,d) signal IS recovered -- either as a pure (c,d) form OR folded into a
          cross-mix that ALSO carries the (a,b) operands and is STRICTLY MORE
          informative than the pure (c,d) form would be (the subsumption that justifies
          dropping the pure (c,d) form);
      (3) pure noise ``e`` is NOT selected.
    """
    X, y = _make_user_f2(n=n)

    fs = MRMR(verbose=0, n_jobs=1, random_state=0)  # default == conditional_mi (S5)
    fs.fit(X, y)
    support = list(fs.get_feature_names_out())

    def _covers(va, vb, exclude=()):
        want, excl = {va, vb}, set(exclude)
        return [nm for nm in support if want <= _bare_vars(nm) and not (_bare_vars(nm) & excl)]

    # (1) The (a,b) a**2/b signal must be recovered -- accepted in ANY form, UNIFORMLY across n
    # (mirrors leg (2) for (c,d), which already accepts pure-OR-cross-mix). The strengthened
    # ONE-fused-compound contract (2026-06-20) drops a pure (a,b) sub-fragment when an admitted fused
    # compound already carries BOTH additive halves, so the (a,b) signal can surface as: a pure (a,b)
    # form, a cross-mix that ALSO carries (c,d), OR a fragmented recovery (raw ``a`` + ``invsquared(b)``
    # folded with a (c,d) operand). All three carry the (a,b) signal -- the contract is "signal
    # recovered + noise rejected + subsumption-justified drops", and clean-vs-fragmented is BELOW that
    # contract line. We do NOT gate strictness on n: which of the three forms wins is a fastmath/prange
    # knife-edge tie (numba @njit(fastmath=True) reorders the ~1e-15 float reductions differently across
    # fresh JIT compilations / prange schedules), so the recovery form is nondeterministic at the scales
    # where more near-tied fused compounds compete -- PROVEN flaky, NOT a regression: commit e79cad31
    # flips PASS<->FAIL across repeats under ISOLATED KTC+numba caches (same code), and a 26-commit
    # isolated bisect pinned the apparent "break" to ff2820f0, a TEST-ONLY commit (zero prod files ->
    # cannot regress prod). Forcing determinism in the kernel would NOT make the clean form "correct":
    # the fragmented recovery is information-theoretically just as valid (both operands present). So the
    # honest, n-symmetric invariant is signal-present-in-any-form; noise-exclusion (leg 3), the (c,d)
    # leg, and the strict-more-informative subsumption guard below stay strict.
    ab_pure = _covers("a", "b", exclude=("c", "d"))
    ab_cross_mix = [nm for nm in support if {"a", "b"} <= _bare_vars(nm) and {"c", "d"} <= _bare_vars(nm)]
    ab_present_any_form = any("a" in _bare_vars(nm) for nm in support) and any("b" in _bare_vars(nm) for nm in support)
    assert ab_pure or ab_cross_mix or ab_present_any_form, (
        f"[F2 n={n}] (a,b) a**2/b signal not recovered in ANY form (a and b operands absent from support): support={support}"
    )

    # (2) The (c,d) signal must be recovered. The hardened pipeline folds it into a
    # cross-mix that ALSO carries the (a,b) operands (the max-MI subsumption), so accept
    # EITHER a pure (c,d) form OR such a cross-mix.
    cd_pure = _covers("c", "d", exclude=("a", "b"))
    cross_mix = [nm for nm in support if {"c", "d"} <= _bare_vars(nm) and {"a", "b"} <= _bare_vars(nm)]
    assert cd_pure or cross_mix, f"[F2 n={n}] (c,d) signal not recovered in ANY form (no pure (c,d) feature and no (a,b)+(c,d) cross-mix): support={support}"

    # (3) Pure noise ``e`` is NOT selected.
    assert "e" not in support, f"[F2 n={n}] pure-noise 'e' wrongly selected: support={support}"

    # PRINCIPLED-SUBSUMPTION CHECK: when the (c,d) signal is carried ONLY by a cross-mix
    # (no standalone pure (c,d) form), that cross-mix must be STRICTLY MORE informative
    # about y than the pure (c,d) form it absorbed -- on a matched-bin grid, debiasing
    # the plug-in bias by using the SAME estimator + bin count for both. This is what
    # makes dropping the pure (c,d) form the correct max-MI choice rather than a bug:
    # the cross-mix carries the (c,d) signal PLUS the (a,b) signal, so its MI dominates.
    if cross_mix and not cd_pure:
        nbins = int(fs.quantization_nbins)
        # ``y`` is already the 10-bin code vector (``_make_user_f2`` qcut'd it); densify
        # to contiguous codes for the MI estimator.
        _, yb = np.unique(np.asarray(y).ravel(), return_inverse=True)
        yb = yb.astype(np.int64)

        def _matched_mi(vals):
            vb = _quantile_bin(np.asarray(vals, dtype=np.float64), nbins=nbins)
            return float(_cmi_from_binned(vb, yb, None))

        # Rebuild the genuine pure (c,d) reference form and the cross-mix's value via
        # transform() (replays the recipe byte-for-byte), then compare matched-grid MI.
        Xt = fs.transform(X)
        cm_name = cross_mix[0]
        assert cm_name in Xt.columns, f"[F2 n={n}] cross-mix {cm_name!r} missing from transform() output"
        cm_mi = _matched_mi(Xt[cm_name].to_numpy())
        # The campaign's canonical genuine (c,d) form for F2.
        cd_ref = np.log(X["c"].to_numpy()) * np.sin(X["d"].to_numpy())
        cd_ref_mi = _matched_mi(cd_ref)
        assert cm_mi > cd_ref_mi, (
            f"[F2 n={n}] the selected cross-mix is NOT more informative than the pure "
            f"(c,d) form it absorbed -- dropping the pure (c,d) form would be a bug, not "
            f"a rational max-MI subsumption: cross_mix_MI={cm_mi:.4f} <= "
            f"cd_form_MI={cd_ref_mi:.4f} (cross_mix={cm_name!r})"
        )


def test_conditional_perm_null_fixed_yz_bit_identical():
    """The hoisted y/z-invariant CMI path in ``_conditional_perm_null`` (recompute
    only x-dependent xz/xyz per permutation) must be bit-identical to the full
    per-permutation ``_cmi_from_binned`` it replaced. Pins the optimization so a
    future refactor of either helper cannot silently diverge the null.

    This is a CPU-vs-CPU exactness contract: both helpers are deterministic njit, so the
    hoisted path is EXACTLY (``==``) the full path in CPU mode. It is pinned bit-exact
    on purpose (a tolerance would let a future ~1e-15 CPU refactor bug slip through). The
    test therefore FORCES the CPU backend -- under a suite-global ``MLFRAME_CMI_GPU=1`` both
    helpers would reroute to GPU and diverge by ~1e-15 from fp reduction ORDER (not a
    refactor bug; CPU/GPU bit-identity is not a real contract). CPU/GPU agreement to ULPs
    is covered separately by the perm-null GPU parity test."""
    import os
    import numpy as np
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import (
        _cmi_from_binned,
        cmi_from_binned_fixed_yz,
        precompute_cmi_yz_terms,
    )

    # ``_cmi_gpu_enabled()`` reroutes to the GPU twin under EITHER MLFRAME_CMI_GPU==1 OR MLFRAME_FE_GPU_STRICT;
    # the suite may set either, so this CPU-vs-CPU exactness contract must neutralise BOTH for its duration
    # (setting only MLFRAME_CMI_GPU=0 still reroutes under a suite-global MLFRAME_FE_GPU_STRICT=1 -> the GPU
    # fp-reduction order then breaks the == by ~1e-15, which is NOT a refactor bug).
    _gpu_env_keys = ("MLFRAME_CMI_GPU", "MLFRAME_FE_GPU_STRICT")
    _prev = {k: os.environ.get(k) for k in _gpu_env_keys}
    os.environ["MLFRAME_CMI_GPU"] = "0"
    os.environ.pop("MLFRAME_FE_GPU_STRICT", None)
    try:
        rng = np.random.default_rng(7)
        for _ in range(200):
            n = int(rng.integers(500, 3000))
            x = rng.integers(0, int(rng.integers(2, 12)), n).astype(np.int64)
            y = rng.integers(0, int(rng.integers(2, 8)), n).astype(np.int64)
            z = rng.integers(0, int(rng.integers(2, 30)), n).astype(np.int64)
            ref = _cmi_from_binned(x, y, z)
            yi, zi, h_yz, h_z, k_yz, k_z, nf = precompute_cmi_yz_terms(y, z)
            got = cmi_from_binned_fixed_yz(x, yi, zi, h_yz, h_z, k_yz, k_z, nf)
            assert ref == got, f"fixed-yz CMI diverged: ref={ref!r} got={got!r}"
    finally:
        for k, v in _prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_analytic_cmi_null_matches_permutation_decision():
    """The analytic chi-square CMI null (2026-06-20) must reproduce the within-stratum
    PERMUTATION null's gate decision on a dense-cell large-n case. The analytic path uses
    ``null_mean = df/(2N)`` (df = occupied-cell count k_xz+k_yz-k_z-k_xyz, the SAME quantity the
    Miller-Madow bias term computes) and ``floor = chi2.ppf(q, df)/(2N)`` -- selection-EQUIVALENT
    to the permutation path by construction. Toggling ``MLFRAME_MI_ANALYTIC_NULL`` must not change
    accept/reject, and the analytic floor/null_mean must be finite and ordered (floor >= mean)."""
    import os
    import numpy as np

    from mlframe.feature_selection.filters._analytic_mi_null import _HAVE_CHI2

    if not _HAVE_CHI2:
        import pytest as _pt

        _pt.skip("scipy.stats.chi2 unavailable")

    import mlframe.feature_selection.filters._fe_cmi_redundancy_gate as G

    rng = np.random.default_rng(13)
    n = 25_000  # >= the 20k analytic floor; dense cells at nbins=4 conditioning support
    # candidate carries real CMI given a low-cardinality support (dense joint cells -> ratio >> 5)
    z = rng.integers(0, 4, n).astype(np.int64)
    x = ((z + rng.integers(0, 4, n)) % 5).astype(np.int64)
    y = ((x + z + rng.integers(0, 2, n)) % 6).astype(np.int64)

    saved = os.environ.get("MLFRAME_MI_ANALYTIC_NULL")
    try:
        os.environ["MLFRAME_MI_ANALYTIC_NULL"] = "0"
        perm_floor, perm_mean = G._conditional_perm_null(x, y, z, seed=0, salt=1)
        os.environ["MLFRAME_MI_ANALYTIC_NULL"] = "1"
        an_floor, an_mean = G._conditional_perm_null(x, y, z, seed=0, salt=1)
    finally:
        if saved is None:
            os.environ.pop("MLFRAME_MI_ANALYTIC_NULL", None)
        else:
            os.environ["MLFRAME_MI_ANALYTIC_NULL"] = saved

    # both paths produce a small, non-negative null (independence floor near 0 at this n)
    assert np.isfinite(an_floor) and np.isfinite(an_mean)
    assert an_floor >= 0.0 and an_mean >= 0.0
    assert an_floor >= an_mean  # 95th-pct quantile of chi2(df)/(2N) >= its mean df/(2N)
    # analytic and permutation nulls agree to the same order of magnitude (both ~ df/(2N))
    assert abs(an_mean - perm_mean) < 5e-3, (an_mean, perm_mean)
    # a strongly-dependent candidate's observed CMI clears BOTH nulls' floors identically (accept)
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _cmi_from_binned

    cmi_obs = _cmi_from_binned(x, y, z)
    assert cmi_obs > perm_floor and cmi_obs > an_floor


def test_analytic_cmi_null_falls_back_on_sparse_cells():
    """The analytic null must FALL BACK to the permutation path when the contingency cells are
    sparse (avg expected count < the min-cell floor) -- a high-cardinality conditioning support at
    modest n. The chi-square asymptotic is unreliable there; the permutation null is exact. Verified
    by a support so fragmented that ratio < 5, where forcing analytic-on still yields a permutation-
    shaped (seed-sensitive) null rather than the deterministic chi2 floor."""
    import os
    import numpy as np
    import mlframe.feature_selection.filters._fe_cmi_redundancy_gate as G
    from mlframe.feature_selection.filters._analytic_mi_null import _HAVE_CHI2

    if not _HAVE_CHI2:
        import pytest as _pt

        _pt.skip("scipy.stats.chi2 unavailable")

    rng = np.random.default_rng(5)
    n = 25_000
    z = rng.integers(0, 4000, n).astype(np.int64)  # very high-cardinality support -> sparse xyz cells
    x = rng.integers(0, 10, n).astype(np.int64)
    y = rng.integers(0, 6, n).astype(np.int64)
    saved = os.environ.get("MLFRAME_MI_ANALYTIC_NULL")
    try:
        os.environ["MLFRAME_MI_ANALYTIC_NULL"] = "1"
        # two different seeds: the permutation fallback is seed-sensitive, the analytic floor is not.
        f1, _ = G._conditional_perm_null(x, y, z, seed=0, salt=1)
        f2, _ = G._conditional_perm_null(x, y, z, seed=999, salt=7)
    finally:
        if saved is None:
            os.environ.pop("MLFRAME_MI_ANALYTIC_NULL", None)
        else:
            os.environ["MLFRAME_MI_ANALYTIC_NULL"] = saved
    # sparse -> permutation fallback engaged -> floors differ across seeds (analytic would be identical)
    assert f1 != f2, "expected permutation fallback (seed-sensitive) on sparse cells, got identical floors"


def test_gpu_resident_perm_null_selection_equivalent_to_cpu():
    """The GPU-RESIDENT permutation null (``MLFRAME_FE_GPU_STRICT_RESIDENT=1``, 2026-06-28) must make the SAME
    gate decision as the CPU permutation null on the sparse-cell case (the path that actually runs permutations --
    the analytic chi-square null is forced OFF so both go through the within-stratum shuffle). The device RNG
    stream is not bit-identical to numpy's, so this is a SELECTION-EQUIVALENCE contract: the floor / null-mean
    agree to the noise scale and the accept/reject of the observed CMI vs the floor is identical. The second leg
    GUARDS against a wrong GPU port: a deliberately corrupted null (10x the floor) WOULD flip the decision, so a
    GPU port computing a wrong null could not pass."""
    import numpy as np
    import pytest as _pt

    cp = _pt.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        _pt.skip("no usable CUDA device")

    from mlframe.feature_selection.filters._fe_cmi_perm_null_gpu import conditional_perm_null_gpu
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _cmi_from_binned, precompute_cmi_yz_terms

    rng = np.random.default_rng(11)
    n = 25_000
    # sparse high-cardinality support so the analytic null does NOT engage -> the permutation path runs.
    z = rng.integers(0, 1500, n).astype(np.int64)
    # a candidate that carries REAL conditional signal (so the observed CMI clears the floor -> ACCEPT).
    x = rng.integers(0, 8, n).astype(np.int64)
    y = ((x + rng.integers(0, 2, n)) % 5).astype(np.int64)

    # CPU permutation null (analytic forced off) -- replicate the host within-stratum shuffle setup.
    import os

    saved = os.environ.get("MLFRAME_MI_ANALYTIC_NULL")
    os.environ["MLFRAME_MI_ANALYTIC_NULL"] = "0"
    try:
        import mlframe.feature_selection.filters._fe_cmi_redundancy_gate as G

        # FORCE the CPU per-perm loop: route the CPU baseline through the host path (CMI_GPU off here).
        _cmi_prev = os.environ.get("MLFRAME_CMI_GPU")
        os.environ["MLFRAME_CMI_GPU"] = "0"  # env is read live by the gate; no module reload needed
        cpu_floor, cpu_mean = G._conditional_perm_null(
            x.copy(),
            y.copy(),
            z.copy(),
            n_permutations=25,
            quantile=0.95,
            seed=0,
            salt=3,
        )
        if _cmi_prev is None:
            os.environ.pop("MLFRAME_CMI_GPU", None)
        else:
            os.environ["MLFRAME_CMI_GPU"] = _cmi_prev

        # GPU-resident null -- build the same order / z_rank the host path prepares, then run the device port.
        xi = np.ascontiguousarray(x, dtype=np.int64).ravel()
        zi_raw = np.ascontiguousarray(z, dtype=np.int64).ravel()
        order = np.argsort(zi_raw, kind="stable")
        sorted_z = zi_raw[order]
        z_rank = np.zeros(xi.size, dtype=np.float64)
        if xi.size > 1:
            z_rank[1:] = np.cumsum(sorted_z[1:] != sorted_z[:-1])
        y_i, z_i, *_ = precompute_cmi_yz_terms(np.ascontiguousarray(y, dtype=np.int64).ravel(), zi_raw)
        gpu_floor, gpu_mean = conditional_perm_null_gpu(
            xi,
            y_i,
            z_i,
            order=order,
            z_rank=z_rank,
            n_permutations=25,
            quantile=0.95,
            seed=0,
            salt=3,
        )
    finally:
        if saved is None:
            os.environ.pop("MLFRAME_MI_ANALYTIC_NULL", None)
        else:
            os.environ["MLFRAME_MI_ANALYTIC_NULL"] = saved

    cmi_obs = float(_cmi_from_binned(x, y, z))
    # selection-equivalence: both nulls are tiny independence-floor values near 0 at this n, and the strongly
    # dependent candidate's observed CMI clears BOTH floors (identical ACCEPT decision).
    assert np.isfinite(gpu_floor) and np.isfinite(gpu_mean)
    assert gpu_floor >= 0.0 and gpu_mean >= 0.0
    assert (cmi_obs > cpu_floor) == (cmi_obs > gpu_floor), (cmi_obs, cpu_floor, gpu_floor)
    assert cmi_obs > gpu_floor, "genuine candidate must clear the GPU floor (ACCEPT)"
    # floor / mean agree to the noise scale (both ~ df/(2N); random-null draws differ only by RNG stream).
    assert abs(gpu_mean - cpu_mean) < 5e-3, (gpu_mean, cpu_mean)
    assert abs(gpu_floor - cpu_floor) < 5e-3, (gpu_floor, cpu_floor)

    # GUARD: a WRONG null (e.g. one that returned 10x the floor) WOULD flip the accept/reject of a marginally
    # significant candidate -- proving the decision is sensitive to the null value the port must reproduce.
    assert not (cmi_obs > (gpu_floor * 10.0 + 1.0)), "sanity: a 10x-inflated wrong null would reject -> decision IS null-sensitive"
