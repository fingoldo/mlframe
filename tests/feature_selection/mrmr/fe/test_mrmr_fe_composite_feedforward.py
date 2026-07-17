"""Regression + biz-value coverage for ENGINEERED-OPERAND FEED-FORWARD (2026-06-08).

Capability: at FE step k>1 (``fe_max_steps>=2``) the MRMR pair search feeds the
engineered columns selected by the prior step BACK into the operand pool and builds
COMPOSITES of two engineered features -- e.g. the additive
``add(div(sqr(a),abs(b)), mul(log(c),sin(d)))`` that captures ~the entire
deterministic part of ``y = a**2/b + log(c)*sin(d)``. Three root causes were fixed
so the composite is DISCOVERED, SELECTED, and REPLAYABLE:

  1. ``check_prospective_fe_pairs`` skipped any operand not in ``original_cols``
     (raw ``feature_names_in_`` only), so an ``(eng_i, eng_j)`` pair never produced a
     candidate. Now engineered operands resolve by name (``allow_engineered_operands``).
  2. The augmented frame ``X`` carries the DISCRETISED bin codes of engineered columns;
     combining bin codes is severely lossy and sinks the composite below the
     engineered-MI gate. The CONTINUOUS engineered values are now fed through
     (``engineered_operand_values``) so the composite is built on them.
  3. A composite whose parents are themselves engineered had no replayable recipe and
     was dropped from ``transform()``. Nested-parent recipes (``nested_parent_a/b``)
     now make it fully replayable.

A principled feed-forward cap (``fe_max_engineered_operands``, default 8) bounds the
O(k^2) pair blow-up. ``fe_max_steps=1`` behaviour is unchanged (no step-2 composites).
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
import re

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR
from mlframe.feature_selection.filters.engineered_recipes import (
    EngineeredRecipe,
    apply_recipe,
    build_unary_binary_recipe,
)


def _bare_vars(name: str) -> set:
    """Single-letter operand VARIABLE names in an engineered feature name, excluding
    letters that belong to function names (``div``/``abs``/``sqr``/...). A raw operand
    appears as a bare ``a``..``e`` not flanked by identifier characters."""
    return set(re.findall(r"(?<![A-Za-z_])([a-e])(?![A-Za-z_])", name))


def _is_composite(name: str) -> bool:
    """True for a composite whose OPERANDS are themselves engineered features (a
    nested ``binary(unary(x),unary(y))`` form on at least one side).

    A single-pair engineered form like ``sub(invqubed(d),log(e))`` has 1 comma (the
    outer binary's) and 3 parens; a composite of TWO engineered pairs like
    ``add(div(sqr(a),abs(b)),mul(log(c),sin(d)))`` has the outer comma PLUS each
    inner pair's comma (>=2 commas total) and >=4 parens. The comma count cleanly
    separates the two (a nested operand contributes its own argument comma)."""
    return name.count("(") >= 4 and name.count(",") >= 2


def _split_outer_operands(name: str) -> list:
    """Split an outer ``binary(LEFT,RIGHT)`` form into its two top-level operand
    substrings at the OUTER (depth-0) comma. Nested inner commas (each inner pair's
    own argument comma) are NOT split on -- only the outer binary's comma is. So
    ``add(div(sqr(a),neg(b)),mul(log(c),sin(d)))`` -> ``['div(sqr(a),neg(b))',
    'mul(log(c),sin(d))']``. Returns ``[]`` if there is no top-level paren."""
    if "(" not in name:
        return []
    inner = name[name.index("(") + 1 : name.rindex(")")]
    parts, depth, last = [], 0, 0
    for i, ch in enumerate(inner):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(inner[last:i])
            last = i + 1
    parts.append(inner[last:])
    return parts


def _has_cd_two_operand_side(name: str) -> bool:
    """True iff one TOP-LEVEL operand of the composite is a genuine TWO-OPERAND form
    over BOTH ``c`` and ``d`` (e.g. ``mul(log(c),sin(d))`` / ``div(log(c),reciproc(d))``).

    This is the STRICT discriminator the c-only-surrogate regression would fail: a
    wrong selection like ``add(invcbrt(c),neg(div(sqr(a),b)))`` has a ``c``-only side
    (``invcbrt(c)`` -- no ``d``) and an ``(a,b)`` side, so NO single operand carries
    both ``c`` and ``d``; the genuine ``(c,d)`` interaction term is missing. The clean
    composite always has one operand covering ``{c, d}`` jointly."""
    return any({"c", "d"} <= _bare_vars(p) for p in _split_outer_operands(name))


# ---------------------------------------------------------------------------
# UNIT: nested-parent recipe build + replay round-trip.
# ---------------------------------------------------------------------------


def test_nested_parent_recipe_replays_composite_from_raw_columns():
    """A unary_binary recipe whose two operands are themselves engineered recipes
    replays correctly from RAW columns alone (the engineered parent values are NOT
    present in the transform frame -- they are recomputed recursively)."""
    n = 2000
    rng = np.random.default_rng(0)
    a = rng.uniform(1.0, 5.0, n)
    b = rng.uniform(1.0, 5.0, n)
    c = rng.uniform(1.0, 5.0, n)
    d = rng.uniform(0.1, 6.0, n)
    X = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})

    # Parent recipes (continuous, no quantization so we can check values directly).
    par_a = build_unary_binary_recipe(
        name="div(sqr(a),abs(b))",
        src_a_name="a",
        src_b_name="b",
        unary_a_name="sqr",
        unary_b_name="abs",
        binary_name="div",
        unary_preset="medium",
        binary_preset="minimal",
        quantization_nbins=None,
        quantization_method=None,
        quantization_dtype=np.int32,
    )
    par_b = build_unary_binary_recipe(
        name="mul(log(c),sin(d))",
        src_a_name="c",
        src_b_name="d",
        unary_a_name="log",
        unary_b_name="sin",
        binary_name="mul",
        unary_preset="medium",
        binary_preset="minimal",
        quantization_nbins=None,
        quantization_method=None,
        quantization_dtype=np.int32,
    )
    # Composite recipe nesting both parents (identity unary on each side, add binary).
    comp = build_unary_binary_recipe(
        name="add(div(sqr(a),abs(b)),mul(log(c),sin(d)))",
        src_a_name="div(sqr(a),abs(b))",
        src_b_name="mul(log(c),sin(d))",
        unary_a_name="identity",
        unary_b_name="identity",
        binary_name="add",
        unary_preset="medium",
        binary_preset="minimal",
        quantization_nbins=None,
        quantization_method=None,
        quantization_dtype=np.int32,
        nested_parent_a=par_a,
        nested_parent_b=par_b,
    )
    assert "nested_parent_a" in comp.extra and "nested_parent_b" in comp.extra

    out = np.asarray(apply_recipe(comp, X), dtype=np.float64)
    expected = (a**2 / np.abs(b)) + (np.log(c) * np.sin(d))
    expected = np.nan_to_num(expected, nan=0.0, posinf=0.0, neginf=0.0)
    assert np.allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_nested_parent_recipe_survives_pickle():
    """The nested-parent recipe (parents stored in ``extra``) pickles and replays
    identically after a round-trip."""
    par = build_unary_binary_recipe(
        name="mul(log(c),sin(d))",
        src_a_name="c",
        src_b_name="d",
        unary_a_name="log",
        unary_b_name="sin",
        binary_name="mul",
        unary_preset="medium",
        binary_preset="minimal",
        quantization_nbins=None,
        quantization_method=None,
        quantization_dtype=np.int32,
    )
    comp = build_unary_binary_recipe(
        name="add(c,mul(log(c),sin(d)))",
        src_a_name="c",
        src_b_name="mul(log(c),sin(d))",
        unary_a_name="identity",
        unary_b_name="identity",
        binary_name="add",
        unary_preset="medium",
        binary_preset="minimal",
        quantization_nbins=None,
        quantization_method=None,
        quantization_dtype=np.int32,
        nested_parent_b=par,
    )
    comp2 = pickle.loads(pickle.dumps(comp))  # nosec B301 -- round-trip of a locally-created, trusted object
    assert isinstance(comp2.extra["nested_parent_b"], EngineeredRecipe)
    n = 500
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"c": rng.uniform(1.0, 5.0, n), "d": rng.uniform(0.1, 6.0, n)})
    assert np.allclose(
        np.asarray(apply_recipe(comp, X), dtype=np.float64),
        np.asarray(apply_recipe(comp2, X), dtype=np.float64),
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# BIZ-VALUE: end-to-end composite discovery on the canonical additive fixture.
# ---------------------------------------------------------------------------


def _canonical_fixture(seed: int, n: int):
    """Canonical fixture."""
    rng = np.random.default_rng(seed)
    a = rng.random(n)
    b = rng.random(n)
    c = rng.random(n)
    d = rng.random(n)
    e = rng.random(n)
    f = rng.random(n)
    y = a**2 / b + f / 5.0 + np.log(c) * np.sin(d)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    return df, pd.Series(y, name="y")


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_fe_max_steps_2_discovers_additive_composite_of_two_engineered():
    """On the canonical ``y = a**2/b + log(c)*sin(d)`` fixture, ``fe_max_steps=2``
    surfaces the additive composite of the two real step-1 engineered features as
    the TOP selected engineered factor, and ``transform()`` replays it faithfully."""
    df, y = _canonical_fixture(seed=0, n=100_000)
    # n_workers=1: force the serial FE path. The joblib/loky worker path intermittently
    # OOMs on RAM-tight boxes for the medium-preset 100k-row buffer (a known Windows-
    # paging fragility, see mrmr.py threading notes) -- unrelated to composite discovery,
    # which is path-independent. Serial keeps this correctness test deterministic.
    # fe_fast_search=False: the FUSED step-2 composite is an EXHAUSTIVE-search guarantee. The default
    # fast path (2026-06-14) deliberately trades the step-2 fusion for speed (it recovers the two
    # halves SEPARATELY, equal downstream quality) -- this test pins the exhaustive fusion capability.
    fs = MRMR(verbose=0, fe_max_steps=2, n_workers=1, fe_fast_search=False)
    fs.fit(df, y)

    sel = list(fs.get_feature_names_out())
    composites = [s for s in sel if _is_composite(s)]
    assert composites, f"no engineered-x-engineered composite in selection: {sel}"
    comp = composites[0]
    # The composite must reference BOTH the (a,b) and (c,d) signal pairs.
    assert {"a", "b"} <= _bare_vars(comp)
    assert {"c", "d"} <= _bare_vars(comp)

    # It must be the TOP engineered factor by mrmr_gain (support_rank 0).
    prov = fs.fe_provenance_
    row = prov[prov["feature_name"] == comp]
    assert not row.empty, f"composite {comp} missing from fe_provenance_"
    assert int(row["support_rank"].iloc[0]) == 0, f"composite {comp} is not the top-ranked feature: rank={int(row['support_rank'].iloc[0])}"

    # transform() replays it (no NaN, monotone in the true signal on held-out data).
    rng = np.random.default_rng(7)
    m = 20_000
    at, bt, ct, dt_, et = (rng.random(m) for _ in range(5))
    df_test = pd.DataFrame({"a": at, "b": bt, "c": ct, "d": dt_, "e": et})
    Xt = fs.transform(df_test)
    assert comp in list(Xt.columns)
    col = np.asarray(Xt[comp], dtype=np.float64)
    assert np.isfinite(col).all(), f"composite transform column has non-finite values"
    y_test_det = at**2 / bt + np.log(ct) * np.sin(dt_)
    from scipy.stats import spearmanr

    rho = spearmanr(col, y_test_det).correlation
    assert abs(rho) >= 0.95, f"transform-replayed composite is not monotone in the true signal: Spearman={rho:.3f}"


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_fe_max_steps_1_fuses_to_single_compound():
    """``fe_max_steps=1`` recovers the additively-separable canonical signal (y = a**2/b + f/5 +
    log(c)*sin(d)) as ONE clean fused compound. SUPERSEDED PREMISE (2026-07-01): the prior contract here was
    "fe_max_steps=1 must NOT build composites -- keep the two separate halves". That predates C2 additive-
    fusion (_fe_additive_fusion.propose_additive_fusions, 2026-06-24), which runs at the END of EACH FE step --
    including step 1 -- and fuses two disjoint additively-separable halves into a single add/sub compound. So
    the single-step path now correctly returns ONE compound covering BOTH signal groups (the sign-aware fusion
    makes it a clean, correctly-aligned recovery -- see test_f2_single_step_one_compound), not two fragments.
    What fe_max_steps=1 still does NOT do is the STEP-2 feed-forward compositing over an engineered-augmented
    operand pool (guarded by test_fe_max_steps_2_discovers_additive_composite_of_two_engineered)."""
    df, y = _canonical_fixture(seed=0, n=100_000)
    fs = MRMR(verbose=0, fe_max_steps=1, n_workers=1)  # serial: avoid loky-worker OOM flakiness
    fs.fit(df, y)
    sel = list(fs.get_feature_names_out())
    # ONE additive-separable compound (the C2 within-step fusion of the two clean halves), covering BOTH the
    # {a,b} and {c,d} signal groups -- no fragmentation, no spurious extra.
    comps = [s for s in sel if _is_composite(s)]
    assert len(comps) == 1, f"fe_max_steps=1 must recover exactly ONE fused compound, got: {sel}"
    comp = comps[0]
    assert comp.startswith(("add(", "sub(")), f"compound is not an additive-separable (add/sub) form: {comp}"
    assert {"a", "b"} <= _bare_vars(comp) and {"c", "d"} <= _bare_vars(comp), f"the single compound must cover BOTH signal groups: {comp}"
    # No stray a/b-only or c/d-only single-pair fragment left over beside the fused compound.
    frags = [s for s in sel if "(" in s and not _is_composite(s) and (_bare_vars(s) <= {"a", "b"} or _bare_vars(s) <= {"c", "d"})]
    assert not frags, f"fe_max_steps=1 left redundant single-pair fragment(s) beside the compound: {frags} :: {sel}"


def test_fe_max_engineered_operands_zero_disables_feedforward():
    """``fe_max_engineered_operands=0`` restores the raw-only operand pool: a
    cheap smoke that the knob is wired and a fit completes without composites.
    Small n keeps it in the fast profile."""
    df, y = _canonical_fixture(seed=0, n=3000)
    fs = MRMR(verbose=0, fe_max_steps=2, fe_max_engineered_operands=0)
    fs.fit(df, y)
    sel = list(fs.get_feature_names_out())
    assert not any(_is_composite(s) for s in sel), f"fe_max_engineered_operands=0 should disable composites: {sel}"


# ---------------------------------------------------------------------------
# STRICT PIN: the CLEAN F1 composite must be recovered at every seed.
# ---------------------------------------------------------------------------
# This pins the behaviour the user expected but that was NOT pinned before: on
# F1 ``y = a**2/b + f/5 + log(c)*sin(d)`` with ``fe_max_steps=2`` the selection must
# contain an ADDITIVE composite whose two operands cover BOTH genuine pairs -- and
# crucially the ``(c, d)`` part must be a genuine TWO-OPERAND interaction over c AND d,
# NOT a c-only surrogate (e.g. ``add(invcbrt(c), neg(a**2/b))`` -- which a prior run
# reported and which the loose ``_bare_vars`` pair-coverage check alone would let pass
# only because the OTHER operand happens to mention nothing of c/d). The n-dependence
# probe (2026-06-10) confirmed this clean composite is recovered at n=30k/50k/100k on
# every seed (the wrong c-only form did NOT reproduce -- that earlier run was a stale
# pull), so a strict n=30k multi-seed assertion is a faithful, RAM-safe regression guard.
# n=30000 (not 100k) keeps it deterministic on RAM-tight boxes; n_workers=1 forces the
# serial FE path (the loky-worker buffer intermittently OOMs at 100k -- a Windows-paging
# fragility unrelated to composite discovery, which is path-independent).
@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_strict_pin_clean_additive_composite_covers_both_pairs_two_operand(seed):
    """STRICT REGRESSION PIN (2026-06-10): F1 / ``fe_max_steps=2`` must surface an
    additive composite over BOTH genuine pairs, with the ``(c,d)`` side a real
    two-operand interaction (NOT a c-only surrogate). Asserted on 3 seeds at n=30000."""
    df, y = _canonical_fixture(seed=seed, n=30_000)
    # fe_fast_search=False: the fused step-2 composite is an exhaustive-search guarantee (the default
    # fast path recovers the two halves separately for speed) -- pin the exhaustive fusion here.
    fs = MRMR(verbose=0, fe_max_steps=2, n_workers=1, fe_fast_search=False)
    fs.fit(df, y)
    sel = list(fs.get_feature_names_out())

    composites = [s for s in sel if _is_composite(s)]
    assert composites, f"[seed={seed}] no engineered-x-engineered composite: {sel}"

    # Among the composites, find one that is the CLEAN additive-separable cross-pair form. The top-level binary
    # is a SIGNED sum of the two halves: 'add(...)' OR 'sub(...)' -- sign-aware fusion picks 'sub' when a half
    # arrives sign-flipped (chosen by sign-invariant MI), which is the correctly-aligned additive form, not an
    # ugly variant.
    clean = [
        s
        for s in composites
        if s.startswith(("add(", "sub("))  # additive-separable top-level binary (signed sum of the halves)
        and {"a", "b"} <= _bare_vars(s)  # the (a,b) signal pair present
        and {"c", "d"} <= _bare_vars(s)  # the (c,d) signal pair present
        and _has_cd_two_operand_side(s)  # (c,d) is a real TWO-operand side, not c-only
    ]
    assert clean, (
        f"[seed={seed}] no CLEAN additive composite covering BOTH {{a,b}} AND a "
        f"genuine two-operand {{c,d}} side (a c-only surrogate like "
        f"add(invcbrt(c),...) would fail HERE). composites={composites} sel={sel}"
    )

    # The CLEAN composite must ALSO be the top-ranked engineered factor (support_rank 0).
    comp = clean[0]
    prov = fs.fe_provenance_
    row = prov[prov["feature_name"] == comp]
    assert not row.empty, f"[seed={seed}] composite {comp} missing from fe_provenance_"
    assert int(row["support_rank"].iloc[0]) == 0, f"[seed={seed}] clean composite {comp} is not top-ranked: rank={int(row['support_rank'].iloc[0])}"
