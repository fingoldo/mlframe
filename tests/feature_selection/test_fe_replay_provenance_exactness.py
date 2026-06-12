"""Robust-prewarp recipe PROVENANCE + EXACT replay parity for the unary_binary
FE path (gaps ``gaps_fe_masking-07`` + ``gaps_fe_masking-14``).

Two adjacent gaps in the FE replay surface:

* ``gaps_fe_masking-07`` -- the Huber-IRLS robust-prewarp persists MAD-anchored
  winsor bounds + a ``robust_fit`` flag flat in ``EngineeredRecipe.extra`` (commit
  f98d1d9), but no test built a unary_binary RECIPE through the production builder
  with a robust prewarp side: the extra-key persistence, the frozen-``extra``
  ``MappingProxyType`` contract, the pickle round-trip of the robust keys, and the
  documented "winsor recorded, NOT applied at replay" behaviour on contaminated
  NEW data were all unexercised.
* ``gaps_fe_masking-14`` -- the MRMR-level replay-parity checks for fit-produced
  unary_binary columns compare via CORRELATION (``|rho| > 0.5`` / ``> 0.999``),
  which tolerates a constant offset / scale / wrong-edge bug. ``transform()`` and
  ``apply_recipe()`` execute the SAME recipe code path, so where the comparison
  target is the recipe (not the true math) exact equality should hold and is
  pinned here with ``assert_allclose(rtol=0, atol=0)``.

What this file pins (measured against the real API in
``filters/hermite_fe`` + ``filters/engineered_recipes``):

(a) PROVENANCE: a unary_binary recipe built via ``build_unary_binary_recipe`` with
    a spike-contaminated operand side carries ``prewarp_a_robust_fit`` +
    ``prewarp_a_winsor_lo/hi`` in ``extra``; pickle preserves them; the frozen
    ``extra`` (``MappingProxyType``) rejects mutation with ``TypeError``.

(b) EXACT replay parity: an MRMR fit on ``y = a*b`` produces a unary_binary
    engineered column whose ``transform()`` output is BIT-EQUAL to
    ``apply_recipe(recipe, X)`` (same recipe path -> rtol=0, atol=0). This is the
    exactness ``gaps_fe_masking-14`` argues for where correlation tolerance hides
    nothing. The ``|rho|`` formula check is NOT used here precisely because the
    comparison target is the recipe path itself, not the true math.

(c) CONTAMINATED-NEW-DATA replay: the robust prewarp recipe applied to NEW rows
    carrying spikes at ~20x IQR returns a FINITE column (the ``nan_to_num`` scrub
    + the basis-axis clip bound the polynomial), and the NON-spiked rows are
    BIT-EQUAL to the clean-replay values. The recorded winsor bounds are NOT
    consulted at replay -- mutating them to absurd values does not change the
    output -- pinning the documented "winsor recorded, not applied" contract so a
    future change that DOES apply winsor at replay must consciously update this
    test.
"""

from __future__ import annotations

import dataclasses
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.hermite_fe import (
    _detect_heavy_tail,
    apply_operand_prewarp,
    fit_operand_prewarp,
)
from mlframe.feature_selection.filters.engineered_recipes import (
    apply_recipe,
    build_unary_binary_recipe,
)
from mlframe.feature_selection.filters.mrmr import MRMR

from tests.feature_selection.conftest import is_fast_mode

warnings.filterwarnings("ignore")

# MRMR knobs that strip the heavy auxiliary stages (DCD, friend-graph, cluster
# aggregate) so the e2e fit is fast and the only engineered output is the
# elementary unary/binary recipe whose replay parity is under test.
_LEAN = dict(dcd_enable=False, build_friend_graph=False, cluster_aggregate_enable=False)


def _spike(rng, x, frac: float = 0.02, scale_iqr: float = 20.0):
    """Replace ``frac`` of x with gross outliers at +/- ``scale_iqr`` * IQR from the median.

    Mirrors the ``_spike`` helper in ``test_robust_warp_fit_gate.py`` -- a thin
    contaminating spike far beyond the bulk that trips ``_detect_heavy_tail`` and
    routes ``fit_operand_prewarp`` through the Huber-IRLS robust path."""
    x = np.asarray(x, dtype=np.float64).copy()
    n = x.size
    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    iqr = max(q3 - q1, 1e-9)
    idx = rng.choice(n, max(1, int(n * frac)), replace=False)
    x[idx] = med + rng.choice([-1.0, 1.0], idx.size) * scale_iqr * iqr
    return x


def _build_robust_prewarp_recipe(seed: int = 7, n: int = 3000):
    """Fit a robust (Huber-IRLS) per-operand prewarp on a SPIKE-contaminated
    operand ``a`` and wrap it into a ``unary_binary`` recipe via the production
    builder. Returns ``(recipe, spec_a, a_spiked, b)``.

    The inner target is the non-monotone product ``(a^3-2a)*(b^2-b)`` so the
    chebyshev warp has genuine curvature to fit; contaminating ``a`` forces the
    robust path so the recipe carries the ``robust_fit`` + winsor provenance."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2.5, 2.5, n)
    b = rng.uniform(-2.5, 2.5, n)
    a_sp = _spike(rng, a, frac=0.02, scale_iqr=20.0)
    true = (a ** 3 - 2 * a) * (b ** 2 - b)
    y = true + rng.normal(0, 0.1, n)
    spec_a = fit_operand_prewarp(a_sp, y, basis="chebyshev", max_degree=4)
    assert spec_a is not None, "prewarp fit returned None on the contaminated operand"
    recipe = build_unary_binary_recipe(
        name="mul(prewarp(a),identity(b))",
        src_a_name="a",
        src_b_name="b",
        unary_a_name="prewarp",
        unary_b_name="identity",
        binary_name="mul",
        unary_preset="minimal",
        binary_preset="minimal",
        quantization_nbins=None,
        quantization_method=None,
        quantization_dtype=np.int16,
        prewarp_a=spec_a,
        prewarp_b=None,
    )
    return recipe, spec_a, a_sp, b


def _make_mul(seed: int = 11, n: int = 3000):
    """Multiplicative-synergy fixture ``y = a*b`` -- MRMR reliably engineers a
    single ``unary_binary`` column (a ``div(unary(a),unary(b))`` reconstruction of
    the product) whose replay parity is exact."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    c = rng.normal(0, 1, n)
    e = rng.normal(0, 1, n)
    true = a * b
    y = true + rng.normal(0, 0.05 * float(np.std(true)), n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "e": e}), pd.Series(y, name="y"), true


def _fit_mul(df, y):
    """Fit the lean MRMR that produces exactly one unary_binary engineered column.

    The univariate-basis stage (adaptive Fourier / chirp) is disabled so it does
    not out-compete the pair path; smart_polynom + hybrid-orth + prewarp are off so
    the engineered output is a plain elementary ``unary_binary`` recipe."""
    MRMR.clear_fit_cache()
    fs = MRMR(
        verbose=0,
        n_jobs=1,
        random_seed=0,
        fe_smart_polynom_iters=0,
        fe_hybrid_orth_enable=False,
        fe_univariate_basis_enable=False,
        fe_pair_prewarp_enable=False,
        **_LEAN,
    )
    fs.fit(df, y)
    return fs


# ===========================================================================
# (a) PROVENANCE: robust-prewarp recipe carries the winsor + robust_fit keys,
#     survives pickle, and the frozen extra rejects mutation.
# ===========================================================================


def test_robust_prewarp_spec_routes_through_huber_path():
    """Precondition sanity: the spike-contaminated operand trips the heavy-tail
    gate and ``fit_operand_prewarp`` returns a spec carrying the robust provenance
    (a clean operand would take the byte-identical OLS path with no robust keys)."""
    _recipe, spec_a, a_sp, _b = _build_robust_prewarp_recipe()
    assert _detect_heavy_tail(a_sp), "contaminated operand did not trip the heavy-tail gate"
    assert spec_a.get("robust_fit") is True, "robust prewarp spec missing robust_fit flag"
    assert "winsor_lo" in spec_a and "winsor_hi" in spec_a, "robust spec missing winsor bounds"
    assert float(spec_a["winsor_lo"]) < float(spec_a["winsor_hi"]), "winsor bounds not ordered"


def test_unary_binary_recipe_extra_carries_robust_prewarp_provenance():
    """The production builder copies the robust prewarp provenance FLAT into
    ``recipe.extra``: the ``prewarp_a_robust_fit`` flag + the MAD-anchored
    ``prewarp_a_winsor_lo/hi`` floats (gaps_fe_masking-07)."""
    recipe, spec_a, _a, _b = _build_robust_prewarp_recipe()
    assert recipe.kind == "unary_binary"
    assert recipe.extra.get("prewarp_a_robust_fit") is True, (
        "recipe.extra missing the prewarp_a_robust_fit provenance flag"
    )
    assert "prewarp_a_winsor_lo" in recipe.extra and "prewarp_a_winsor_hi" in recipe.extra, (
        "recipe.extra missing the prewarp_a_winsor_lo/hi provenance bounds"
    )
    # The persisted bounds equal the spec's bounds (recorded, not transformed).
    assert float(recipe.extra["prewarp_a_winsor_lo"]) == float(spec_a["winsor_lo"])
    assert float(recipe.extra["prewarp_a_winsor_hi"]) == float(spec_a["winsor_hi"])
    # The unwarped (identity) b-side carries NO prewarp/robust keys.
    assert not any(k.startswith("prewarp_b") for k in recipe.extra), (
        "identity b-side should not have persisted any prewarp keys"
    )


def test_robust_prewarp_recipe_survives_pickle_roundtrip():
    """Pickle round-trip preserves the robust provenance keys AND recipe equality
    (the ``__getstate__`` mappingproxy->dict + ``__setstate__`` re-freeze chain)."""
    recipe, _spec, _a, _b = _build_robust_prewarp_recipe()
    r2 = pickle.loads(pickle.dumps(recipe))
    assert r2 == recipe, "pickle round-trip recipe is not equal to the original"
    assert r2.extra.get("prewarp_a_robust_fit") is True
    assert float(r2.extra["prewarp_a_winsor_lo"]) == float(recipe.extra["prewarp_a_winsor_lo"])
    assert float(r2.extra["prewarp_a_winsor_hi"]) == float(recipe.extra["prewarp_a_winsor_hi"])
    # The coef array (the bytes that actually drive replay) round-trips array-equal.
    np.testing.assert_array_equal(
        np.asarray(r2.extra["prewarp_a_coef"]), np.asarray(recipe.extra["prewarp_a_coef"])
    )


def test_robust_prewarp_recipe_extra_is_frozen_mappingproxy():
    """The frozen ``extra`` (``MappingProxyType``) rejects in-place mutation of the
    provenance keys with ``TypeError`` -- the iter-49 immutability contract that
    keeps a stored recipe from being silently corrupted post-construction."""
    recipe, _spec, _a, _b = _build_robust_prewarp_recipe()
    with pytest.raises(TypeError):
        recipe.extra["prewarp_a_winsor_lo"] = -999.0
    with pytest.raises(TypeError):
        recipe.extra["a_brand_new_key"] = 1
    # Attribute rebind is also blocked by the frozen dataclass.
    with pytest.raises(dataclasses.FrozenInstanceError):
        recipe.extra = {}  # type: ignore[misc]


# ===========================================================================
# (b) EXACT replay parity: transform() == apply_recipe() bit-for-bit for a
#     fit-produced unary_binary column (gaps_fe_masking-14).
# ===========================================================================


@pytest.mark.slow
def test_unary_binary_transform_equals_apply_recipe_exactly():
    """MRMR fit on ``y = a*b`` engineers a single ``unary_binary`` column. Because
    ``transform()`` and ``apply_recipe()`` execute the SAME recipe code path, the
    held-out transform output must be BIT-EQUAL to ``apply_recipe`` -- no
    correlation tolerance. ``rtol=0, atol=0`` (gaps_fe_masking-14): if this ever
    diverges, that divergence IS the finding and must be root-caused, never
    re-tolerated."""
    df, y, _ = _make_mul(seed=11, n=3000)
    fs = _fit_mul(df, y)
    names = list(fs.get_feature_names_out())
    recipes = {r.name: r for r in fs._engineered_recipes_}
    ub_names = [nm for nm in names if nm in recipes and recipes[nm].kind == "unary_binary"]
    assert ub_names, (
        f"MRMR engineered no unary_binary column on y=a*b (selected {names}); "
        f"the exact-parity contract has nothing to pin"
    )

    df_test, _yt, _tt = _make_mul(seed=511, n=3000)
    Xt = np.asarray(fs.transform(df_test))
    for nm in ub_names:
        i = names.index(nm)
        direct = np.asarray(apply_recipe(recipes[nm], df_test)).reshape(-1)
        np.testing.assert_allclose(
            Xt[:, i], direct, rtol=0, atol=0,
            err_msg=(
                f"transform() vs apply_recipe() for unary_binary '{nm}' diverged; "
                f"they execute the same recipe path so this must be bit-equal"
            ),
        )


def test_unary_binary_exact_parity_holds_on_majority_of_seeds():
    """The bit-exact transform/apply parity is not a single-seed accident: across a
    seed sweep the ``y = a*b`` fit produces a unary_binary recipe whose replay is
    exact on every seed (majority-of-seeds gate; here it must be exact on ALL since
    bit-equality has no legitimate seed variance).

    Serves as the ``MLFRAME_FAST=1`` representative of the exact-parity contract
    (gaps_fe_masking-14): in fast mode it runs a single seed at n=1500 so the core
    ``transform() == apply_recipe()`` path stays exercised when the heavier slow
    test is skipped."""
    seeds = (11, 0, 7) if not is_fast_mode() else (11,)
    n = 3000 if not is_fast_mode() else 1500
    exact_seeds = 0
    produced = 0
    for seed in seeds:
        df, y, _ = _make_mul(seed=seed, n=n)
        fs = _fit_mul(df, y)
        names = list(fs.get_feature_names_out())
        recipes = {r.name: r for r in fs._engineered_recipes_}
        ub_names = [nm for nm in names if nm in recipes and recipes[nm].kind == "unary_binary"]
        if not ub_names:
            continue
        produced += 1
        df_test, _yt, _tt = _make_mul(seed=seed + 500, n=n)
        Xt = np.asarray(fs.transform(df_test))
        all_exact = True
        for nm in ub_names:
            i = names.index(nm)
            direct = np.asarray(apply_recipe(recipes[nm], df_test)).reshape(-1)
            if not np.array_equal(Xt[:, i], direct):
                all_exact = False
        exact_seeds += int(all_exact)
    assert produced >= 1, "no seed produced a unary_binary recipe to test"
    assert exact_seeds == produced, (
        f"transform/apply bit-exact parity held on only {exact_seeds}/{produced} "
        f"seeds; bit-equality on the same recipe path must hold on every seed"
    )


# ===========================================================================
# (c) CONTAMINATED-NEW-DATA replay: finite output + non-spiked rows bit-equal,
#     and the recorded winsor bounds are NOT applied at replay (07).
# ===========================================================================


def test_robust_prewarp_replay_on_spiked_new_data_is_finite():
    """Replaying the robust prewarp recipe on NEW rows carrying ~20x-IQR spikes
    returns a FINITE column: the basis-axis clip bounds the polynomial input and
    the ``nan_to_num`` scrub at ``_recipe_unary_binary.py`` catches any residual
    inf/NaN. A spike must never propagate a non-finite engineered value."""
    recipe, _spec, _a, _b = _build_robust_prewarp_recipe()
    rng = np.random.default_rng(909)
    n = 3000
    a_new = rng.uniform(-2.5, 2.5, n)
    b_new = rng.uniform(-2.5, 2.5, n)
    q1, med, q3 = np.quantile(a_new, [0.25, 0.5, 0.75])
    iqr = max(q3 - q1, 1e-9)
    spike_idx = rng.choice(n, 30, replace=False)
    a_spiked = a_new.copy()
    a_spiked[spike_idx] = med + rng.choice([-1.0, 1.0], spike_idx.size) * 20.0 * iqr
    df_spiked = pd.DataFrame({"a": a_spiked, "b": b_new})
    out = np.asarray(apply_recipe(recipe, df_spiked)).reshape(-1)
    assert np.isfinite(out).all(), "spiked-row replay produced non-finite engineered values"


def test_robust_prewarp_replay_nonspiked_rows_bit_equal_to_clean():
    """The non-spiked rows of a contaminated replay are BIT-EQUAL to the clean
    replay: a transform-time spike on one row must not perturb any other row's
    engineered value (the prewarp is a row-local closed-form on x alone)."""
    recipe, _spec, _a, _b = _build_robust_prewarp_recipe()
    rng = np.random.default_rng(909)
    n = 3000
    a_new = rng.uniform(-2.5, 2.5, n)
    b_new = rng.uniform(-2.5, 2.5, n)
    q1, med, q3 = np.quantile(a_new, [0.25, 0.5, 0.75])
    iqr = max(q3 - q1, 1e-9)
    spike_idx = rng.choice(n, 30, replace=False)
    df_clean = pd.DataFrame({"a": a_new.copy(), "b": b_new.copy()})
    a_spiked = a_new.copy()
    a_spiked[spike_idx] = med + rng.choice([-1.0, 1.0], spike_idx.size) * 20.0 * iqr
    df_spiked = pd.DataFrame({"a": a_spiked, "b": b_new.copy()})

    out_clean = np.asarray(apply_recipe(recipe, df_clean)).reshape(-1)
    out_spiked = np.asarray(apply_recipe(recipe, df_spiked)).reshape(-1)
    mask = np.ones(n, dtype=bool)
    mask[spike_idx] = False
    np.testing.assert_array_equal(
        out_clean[mask], out_spiked[mask],
        err_msg="non-spiked rows diverged between clean and spiked replay",
    )


def test_recorded_winsor_bounds_are_not_applied_at_replay():
    """PIN the documented "winsor recorded, NOT applied at replay" contract: replay
    is closed-form on ``coef`` (``apply_operand_prewarp`` ignores the winsor keys),
    so mutating the persisted ``prewarp_a_winsor_lo/hi`` to absurd values does NOT
    change the apply output. A future change that DOES apply winsor at replay must
    consciously update this test."""
    recipe, _spec, a_sp, b = _build_robust_prewarp_recipe()
    df = pd.DataFrame({"a": a_sp, "b": b})
    out0 = np.asarray(apply_recipe(recipe, df)).reshape(-1)

    mutated_extra = dict(recipe.extra)
    mutated_extra["prewarp_a_winsor_lo"] = -9999.0
    mutated_extra["prewarp_a_winsor_hi"] = 9999.0
    recipe_mut = dataclasses.replace(recipe, extra=mutated_extra)
    out1 = np.asarray(apply_recipe(recipe_mut, df)).reshape(-1)

    np.testing.assert_array_equal(
        out0, out1,
        err_msg=(
            "mutating the recorded winsor bounds changed the replay output; the "
            "winsor bounds are provenance-only and must NOT be applied at replay"
        ),
    )


def test_robust_prewarp_apply_matches_manual_closed_form_exactly():
    """The recipe's apply path equals the manual closed-form ``prewarp(a) * b`` (the
    unwarped identity b-side) BIT-FOR-BIT -- nailing that the recipe replays the
    same closed-form ``apply_operand_prewarp`` the spec encodes, with the
    fit/transform-consistent ``nan_to_num`` scrub on the product."""
    recipe, spec_a, a_sp, b = _build_robust_prewarp_recipe()
    df = pd.DataFrame({"a": a_sp, "b": b})
    out = np.asarray(apply_recipe(recipe, df)).reshape(-1)
    warped_a = apply_operand_prewarp(np.asarray(a_sp, dtype=np.float64), spec_a)
    manual = np.nan_to_num(warped_a * np.asarray(b, dtype=np.float64),
                           nan=0.0, posinf=0.0, neginf=0.0)
    np.testing.assert_array_equal(
        out, manual,
        err_msg="recipe apply diverged from the manual closed-form prewarp(a)*b",
    )
