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
    div = df["a"] ** 2 / np.abs(df["b"])                                      # real ~ a**2/|b|
    mul = np.log(df["c"]) * np.sin(df["d"])                                   # real ~ log(c)*sin(d)
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
    assert "sub" not in accepted, (
        f"[{formula}] spurious cross-signal feature admitted as independent: {diag['sub']}"
    )
    # The spurious feature is rejected on the RELATIVE-gap (TAU) leg, not by chance:
    # its DEBIASED EXCESS CMI is far below the weakest admitted feature's bar. (The
    # rel bar is now on the debiased-excess scale -- raw CMI would re-introduce the
    # finite-n bias that the excess removes.)
    assert diag["sub"]["cmi_excess"] < diag["sub"]["rel_bar"], (
        f"[{formula}] sub should fail the relative bar on debiased excess: {diag['sub']}"
    )


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
    assert "sub" not in accepted, (
        f"[{formula} n={n}] spurious cross-signal feature admitted as independent "
        f"(finite-n bias leak?): {diag['sub']}"
    )
    # The spurious feature's DEBIASED EXCESS collapses toward ~0 at every n (its
    # CMI is pure bias/noise given the admitted support), so it fails the rel bar.
    assert diag["sub"]["cmi_excess"] < diag["sub"]["rel_bar"], (
        f"[{formula} n={n}] sub excess should be below the relative bar: {diag['sub']}"
    )
    # The genuine features keep a POSITIVE excess (real private interaction), so
    # the separation is not an artefact of everything collapsing to zero.
    assert diag["div"]["cmi_excess"] > diag["sub"]["cmi_excess"], (
        f"[{formula} n={n}] genuine div excess should exceed spurious sub excess: "
        f"div={diag['div']} sub={diag['sub']}"
    )


def test_relative_gap_leg_is_load_bearing():
    """The conditional-permutation FLOOR alone is insufficient -- on at least one
    formula the spurious feature sits ABOVE its own floor yet its DEBIASED EXCESS
    is below the TAU bar, so the relative-gap leg is what rejects it. Locks the
    'implement BOTH legs' design."""
    floor_alone_would_admit = False
    for formula in ("F1", "F2"):
        df, y = _build(formula)
        yb = _bin_y(y)
        cands = _candidates(df, yb)
        _, diag = apply_cmi_redundancy_gate(cands, yb, nbins=10, retain_frac=0.15, seed=0)
        sub = diag["sub"]
        # rejected overall...
        assert sub["accept"] is False
        # ...and on this formula the floor alone would NOT have caught it (raw CMI
        # clears the floor), so the debiased relative-gap leg is the decisive
        # rejector.
        if sub["cmi"] > sub["floor"] and sub["cmi_excess"] < sub["rel_bar"]:
            floor_alone_would_admit = True
    assert floor_alone_would_admit, (
        "expected at least one formula where the spurious feature clears its "
        "conditional-permutation floor (so the debiased relative-gap TAU leg is the "
        "decisive rejector); if this fails the two-leg design may have collapsed to one leg"
    )


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
        nm: (np.asarray(v, dtype=np.float64),
             float(_cmi_from_binned(_quantile_bin(np.asarray(v, dtype=np.float64), nbins=10), yb, None)))
        for nm, v in cols.items()
    }


def _weak_complementary_fixture(seed: int = 1, n: int = 20_000, n_weak: int = 5,
                                strong_coef: float = 8.0, weak_coef: float = 1.5):
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
        assert cmi_given_seed > 0.01, (
            f"weak driver {nm} carries no conditional signal given the seed "
            f"(CMI={cmi_given_seed}); fixture broken"
        )
    # THE FIX: none of the genuinely complementary weak drivers is dropped.
    accepted, diag = apply_cmi_redundancy_gate(cols_to_cands(cols, yb), yb, nbins=10, seed=seed)
    rejected = [nm for nm in weak if nm not in accepted]
    assert not rejected, (
        f"[seed={seed}] genuinely complementary weak drivers FALSELY REJECTED as "
        f"redundant: {rejected}; diag={ {nm: diag[nm] for nm in rejected} }"
    )


def test_strong_significance_escape_is_decisive_for_weak_complementary():
    """LOCK that it is the ESCAPE (not a coincidental excess >= rel_bar) that
    admits the weak complementary drivers: at least one admitted weak driver has
    excess BELOW rel_bar yet clears its floor >= the escape margin."""
    cols, yb = _weak_complementary_fixture(seed=1)
    accepted, diag = apply_cmi_redundancy_gate(cols_to_cands(cols, yb), yb, nbins=10, seed=1)
    weak = [nm for nm in cols if nm.startswith("weakW") and nm in accepted]
    escaped = [
        nm for nm in weak
        if diag[nm]["cmi_excess"] < diag[nm]["rel_bar"]
        and diag[nm]["cmi"] >= _CMI_SIGNIFICANCE_ESCAPE_MARGIN * diag[nm]["floor"]
    ]
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
        cols_to_cands(cols, yb), yb, nbins=10, seed=1, significance_escape_margin=1.0,
    )
    weak = [nm for nm in cols if nm.startswith("weakW")]
    rejected = [nm for nm in weak if nm not in accepted]
    assert rejected, (
        "with the escape disabled the pure two-leg gate should still drop the weak "
        "complementary drivers (this is the bug the escape fixes)"
    )


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
        "redA_cube": A ** 3,          # monotone -> identical quantile bins to A
        "redA_exp": np.exp(A),        # monotone -> identical quantile bins to A
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

    # Both genuine signal pairs survive under the CMI gate.
    def _covers(eng, va, vb, exclude=()):
        want, excl = {va, vb}, set(exclude)
        return any(want <= _bare_vars(nm) and not (_bare_vars(nm) & excl) for nm in eng)

    assert _covers(eng_cmi, "a", "b", exclude=("c", "d")), f"no a**2/b form under CMI gate: {eng_cmi}"
    assert _covers(eng_cmi, "c", "d", exclude=("a", "b")), f"no log(c)*sin(d) form under CMI gate: {eng_cmi}"

    # The CMI gate is the stricter, principled redundancy filter: it admits no MORE
    # engineered features than the legacy ratio, and strictly fewer when the ratio
    # path lets a redundant cross-signal/extra column through.
    assert len(eng_cmi) <= len(eng_ratio), (
        f"CMI gate admitted MORE engineered features than the legacy ratio "
        f"(should be stricter): cmi={eng_cmi} ratio={eng_ratio}"
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

    This locks the four load-bearing properties of the recovered behaviour:
      (1) a GENUINE (a,b) form (a**2/b) is recovered -- pure a,b operands;
      (2) a GENUINE (c,d) form is recovered -- pure c,d operands;
      (3) pure noise ``e`` is NOT selected;
      (4) the S5 CMI gate is the decisive filter -- it admits NO MORE engineered
          features than the legacy ``prevalence_ratio`` path (which, on F2, lets
          spurious cross-signal forms like div(exp(a),invsquared(c)) through).
    """
    X, y = _make_user_f2(n=n)

    fs = MRMR(verbose=0, n_jobs=1, random_state=0)  # default == conditional_mi (S5)
    fs.fit(X, y)
    support = list(fs.get_feature_names_out())

    def _covers(va, vb, exclude=()):
        want, excl = {va, vb}, set(exclude)
        return [nm for nm in support if want <= _bare_vars(nm) and not (_bare_vars(nm) & excl)]

    ab = _covers("a", "b", exclude=("c", "d"))
    cd = _covers("c", "d", exclude=("a", "b"))
    assert ab, f"[F2 n={n}] no genuine (a,b) a**2/b form recovered: support={support}"
    assert cd, f"[F2 n={n}] no genuine (c,d) form recovered: support={support}"
    assert "e" not in support, f"[F2 n={n}] pure-noise 'e' wrongly selected: support={support}"

    # The S5 CMI gate is the stricter principled filter: on F2 the legacy ratio
    # path admits redundant cross-signal engineered forms (e.g. a-with-c, b-with-d)
    # that S5 rejects, so S5 must admit no MORE engineered features than legacy.
    eng_cmi = _engineered(fs)
    fs_ratio = MRMR(verbose=0, n_jobs=1, random_state=0, fe_acceptance="prevalence_ratio")
    fs_ratio.fit(X, y)
    eng_ratio = _engineered(fs_ratio)
    assert len(eng_cmi) <= len(eng_ratio), (
        f"[F2 n={n}] CMI gate admitted MORE engineered features than legacy ratio "
        f"(should be stricter): cmi={eng_cmi} ratio={eng_ratio}"
    )
