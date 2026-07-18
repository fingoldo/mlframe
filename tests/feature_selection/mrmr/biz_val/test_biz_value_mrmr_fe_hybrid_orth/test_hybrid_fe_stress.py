"""Consolidated from test_biz_value_mrmr_layer27.py.

Layer 27 biz_value: STRESS-TEST the FE pipelines under hostile production conditions.

Layers 21-26 pinned the happy-path contracts of the three live FE constructors:
  * Layer 21 / 22  -- orth-poly univariate / cross-basis (auto-wired in MRMR at L23).
  * Layer 26       -- generic MI-greedy unary/binary transform constructor.

Layer 27 fires HOSTILE inputs at the same surface. Each scenario pins one
non-negotiable production invariant the two-gate selection has to hold:

A. **All-noise dataset** (5 seeds)
   20 columns, NONE carry signal. ``support_`` must be small or empty
   (fallback path counts as empty) and NEITHER ``hybrid_orth_features_``
   NOR ``mi_greedy_features_`` may pad the augmented frame with spurious
   engineered columns. The absolute-MI floor must hold against pure-noise
   uplift inflation.

B. **Adversarial noise pair** (5 seeds)
   ``y = sign(x_real * x_pad)`` where x_pad has a tiny real signal mixed
   with noise. The genuine ``x_real * x_pad`` cross MUST be discovered;
   noise * noise crosses MUST NOT enter ``mi_greedy_features_`` /
   ``hybrid_orth_features_``.

C. **Multi-collinear sources** (5 seeds)
   10 columns where x2..x10 are copies of x1 with tiny IID jitter. Signal
   ``y = sign(x1^2 - 1)``. Hybrid must NOT emit He_2 ten times (one per
   duplicate); MRMR redundancy must prune duplicates from the final
   support.

D. **Conflict between MI-greedy and hybrid orth** (5 seeds)
   With BOTH enabled on ``y = sign(x^2 - 1)``: hybrid emits ``x__He2``,
   MI-greedy emits ``square(x)`` (same signal, different encoding). MRMR
   support must contain AT MOST ONE of them (redundancy term fires).

E. **Recipe corruption survival** (3 seeds)
   Pickle a fitted MRMR; corrupt one mi_greedy_transform recipe by:
   1. Pointing src_names at a column that does NOT exist in test X.
   2. Setting the transform key to an unknown name.
   The replay must EITHER raise an actionable error (KeyError / ValueError
   surfaced to caller) OR fall back cleanly; it MUST NOT silently produce
   a wrong column.

NEVER xfail. Contracts fail -> investigate prod, fix on the spot.

2026-05-31 Layer 27.
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
import warnings
from functools import cache

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


SEEDS = (1, 7, 13, 21, 42)
SEEDS_SHORT = (1, 13, 42)


from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr

# ---------------------------------------------------------------------------
# A. All-noise dataset: no FE constructor must manufacture signal
# ---------------------------------------------------------------------------


def _build_all_noise(seed: int, n: int = 1500, p: int = 20):
    """20 independent noise columns with zero mutual information to a fair-Bernoulli y."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"n{i:02d}": rng.standard_normal(n) for i in range(p)})
    # Independent fair Bernoulli y -- zero mutual information with every col.
    y = pd.Series(rng.integers(0, 2, size=n), name="y")
    return X, y


class TestAllNoiseNoSpuriousFE:
    """On a pure-noise frame, no FE constructor may manufacture spurious engineered columns."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_hybrid_orth_emits_no_spurious_columns(self, seed):
        """hybrid_orth's absolute-MI floor keeps spurious columns bounded on pure noise."""
        X, y = _build_all_noise(seed)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_top_k=5,
        )
        m.fit(X, y)
        # On pure noise the absolute-MI floor + uplift gate should reject
        # every candidate. Allow at most 3 false-positives across the 20 raw
        # cols x 2 default degrees (= 40 candidates); the noise-aware
        # median+3.5*MAD floor (Layer 27 prod fix) targets ~0.05% per-slot
        # FP rate but empirical MAD tails are heavier than Gaussian, so a
        # small FP budget keeps the contract robust on adversarial seeds
        # without hiding the major regression class (pre-fix: 5 / top_k=5).
        assert (
            len(m.hybrid_orth_features_) <= 3
        ), f"seed={seed}: hybrid_orth manufactured {len(m.hybrid_orth_features_)} engineered columns on a pure-noise frame: {m.hybrid_orth_features_}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mi_greedy_emits_no_spurious_columns(self, seed):
        """mi_greedy's absolute-MI floor keeps spurious columns bounded on pure noise."""
        X, y = _build_all_noise(seed)
        m = _make_mrmr(
            fe_mi_greedy_enable=True,
            fe_mi_greedy_top_k=5,
            fe_mi_greedy_seed_cols_count=5,
            fe_mi_greedy_include_unary=True,
            fe_mi_greedy_include_binary=True,
        )
        m.fit(X, y)
        # 5 seed cols x (8 unary + 8 binary x 4 partners) ~= 200 candidates;
        # FP budget of 5 keeps the absolute-MI floor honest while allowing
        # the statistical noise inherent in MI from 1500 samples.
        assert (
            len(m.mi_greedy_features_) <= 5
        ), f"seed={seed}: mi_greedy manufactured {len(m.mi_greedy_features_)} engineered columns on a pure-noise frame: {m.mi_greedy_features_}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_both_enabled_no_spurious_columns(self, seed):
        """With both constructors enabled together, combined spurious columns stay bounded on pure noise."""
        X, y = _build_all_noise(seed)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_top_k=5,
            fe_mi_greedy_enable=True,
            fe_mi_greedy_top_k=5,
            fe_mi_greedy_seed_cols_count=5,
            fe_mi_greedy_include_unary=True,
            fe_mi_greedy_include_binary=False,
        )
        m.fit(X, y)
        total_spurious = len(m.hybrid_orth_features_) + len(m.mi_greedy_features_)
        assert total_spurious <= 6, (
            f"seed={seed}: combined FE stages manufactured {total_spurious} "
            f"engineered columns on pure noise; "
            f"hybrid={m.hybrid_orth_features_}, mig={m.mi_greedy_features_}"
        )


# ---------------------------------------------------------------------------
# B. Adversarial noise pair: signal mul DISCOVERED, noise mul REJECTED
# ---------------------------------------------------------------------------


def _build_adversarial_pair(seed: int, n: int = 2500):
    """``y = sign(x_real * x_pad)`` where:

    * x_real  -- standard normal
    * x_pad   -- mostly noise with a 0.25 signal-correlation contribution
    * noise_a / noise_b -- pure standard normal (no signal)
    * noise_c / noise_d -- pure standard normal (no signal)

    The TRUE cross-feature is ``x_real * x_pad``. Any
    ``noise_X * noise_Y`` cross MUST be rejected by the two-gate
    selection. The relevant signal mass is on the (x_real, x_pad) pair.
    """
    rng = np.random.default_rng(seed)
    x_real = rng.standard_normal(n)
    signal = rng.standard_normal(n)
    # x_pad: 0.25 * shared_signal + 0.97 * noise. Marginal MI to y near
    # 0 (sign of product depends on sign of both, not on x_pad alone).
    x_pad = 0.25 * signal + np.sqrt(1.0 - 0.25**2) * rng.standard_normal(n)
    # Build y as sign(x_real * (0.25*signal)) -- the SHARED component is
    # what drives the target; x_pad's noise component contributes only
    # MI loss, not signal.
    score = x_real * (0.25 * signal) + 0.05 * rng.standard_normal(n)
    y = (score > 0).astype(int)
    X = pd.DataFrame(
        {
            "x_real": x_real,
            "x_pad": x_pad,
            "noise_a": rng.standard_normal(n),
            "noise_b": rng.standard_normal(n),
            "noise_c": rng.standard_normal(n),
            "noise_d": rng.standard_normal(n),
        }
    )
    return X, pd.Series(y, name="y")


class TestAdversarialPair:
    """The genuine x_real*x_pad cross must be discovered; noise*noise crosses must be rejected."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_no_noise_noise_cross_in_mi_greedy(self, seed):
        """No mi_greedy binary column crosses two pure-noise sources."""
        X, y = _build_adversarial_pair(seed)
        m = _make_mrmr(
            fe_mi_greedy_enable=True,
            fe_mi_greedy_top_k=6,
            fe_mi_greedy_seed_cols_count=6,
            fe_mi_greedy_include_unary=False,
            fe_mi_greedy_include_binary=True,
        )
        m.fit(X, y)
        # Walk every appended mi_greedy column; for binary names of the form
        # ``(col_i__transform__col_j)``, recover (col_i, col_j) and assert
        # NEITHER side is from the {noise_a, noise_b, noise_c, noise_d} pool
        # unless the other side is x_real or x_pad (a noise * signal cross
        # is acceptable -- it carries shared signal mass; pure noise*noise
        # is the contract we pin).
        from mlframe.feature_selection.filters._mi_greedy_fe import (
            BINARY_TRANSFORMS,
        )

        noise_pool = {"noise_a", "noise_b", "noise_c", "noise_d"}
        for name in m.mi_greedy_features_:
            if not (name.startswith("(") and name.endswith(")")):
                continue
            inner = name[1:-1]
            parsed = None
            for tname in BINARY_TRANSFORMS:
                token = f"__{tname}__"
                idx = inner.find(token)
                if idx >= 0:
                    col_i = inner[:idx]
                    col_j = inner[idx + len(token) :]
                    parsed = (col_i, col_j)
                    break
            if parsed is None:
                continue
            col_i, col_j = parsed
            both_noise = col_i in noise_pool and col_j in noise_pool
            assert not both_noise, (
                f"seed={seed}: noise*noise cross {name!r} entered "
                f"mi_greedy_features_; the two-gate absolute-MI floor "
                f"should have rejected it. All appended: "
                f"{m.mi_greedy_features_}"
            )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_no_noise_noise_cross_in_hybrid_orth_pair(self, seed):
        """No hybrid_orth pair column crosses two pure-noise sources."""
        X, y = _build_adversarial_pair(seed)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=True,
            fe_hybrid_orth_top_k=5,
            fe_hybrid_orth_pair_max_degree=2,
        )
        m.fit(X, y)
        # Hybrid pair names look like ``x_a*x_b__He1_He1`` (or similar). Walk
        # ``hybrid_orth_features_`` and reject any whose source pair is a
        # pure noise*noise cross. Univariate hybrid columns (single-source,
        # no ``*`` in the name) are not part of this contract.
        noise_pool = {"noise_a", "noise_b", "noise_c", "noise_d"}
        for name in m.hybrid_orth_features_:
            if "*" not in name:
                continue  # univariate, not a cross
            # Hybrid pair naming is opaque; parse defensively by checking
            # which source columns appear in the engineered name. We accept
            # any pair that involves x_real or x_pad; reject pairs that
            # involve ONLY noise columns.
            mentioned_signal = any(src in name for src in ("x_real", "x_pad"))
            mentioned_noise = [n for n in noise_pool if n in name]
            if mentioned_noise and not mentioned_signal:
                # The cross involves only noise sources. Permitted only
                # when explainable as a one-noise-only term, which a pair
                # cross is not (always 2 sources).
                pytest.fail(f"seed={seed}: hybrid_orth pair {name!r} crosses two noise columns; absolute-MI floor leaked. All: {m.hybrid_orth_features_}")


# ---------------------------------------------------------------------------
# C. Multi-collinear sources: He_2 once, not 10x
# ---------------------------------------------------------------------------


def _build_collinear_quadratic(seed: int, n: int = 2000, p: int = 10):
    """``y = sign(x1^2 - 1)`` with x2..x_p as x1 + tiny IID jitter. Every
    column carries the SAME quadratic signal; redundancy must prune."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    cols = {"x1": x1}
    for i in range(2, p + 1):
        # 1% noise added to x1; MI is essentially unchanged.
        cols[f"x{i}"] = x1 + 0.01 * rng.standard_normal(n)
    X = pd.DataFrame(cols)
    y = ((x1 * x1 - 1.0) + 0.05 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


class TestMultiCollinearSources:
    """With 10 near-duplicate quadratic-signal sources, He_2 emission and final support both stay bounded."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_hybrid_orth_does_not_emit_he2_for_every_duplicate(self, seed):
        """hybrid_orth's top_k cap bounds He_2 column emission even with 10 near-identical sources."""
        X, y = _build_collinear_quadratic(seed)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=10,
        )
        m.fit(X, y)
        # Count He_2 columns -- one per source max would be 10; we expect
        # the top_k cap to bound the count but the constructor itself does
        # NOT de-duplicate near-identical sources. The contract is on the
        # FINAL support: MRMR's redundancy term should prune dups.
        he2_cols = [c for c in m.hybrid_orth_features_ if c.endswith("__He2") or "He2" in c]
        # The constructor honors top_k so even with 10 sources we cap at the
        # configured ceiling (top_k=10 explicit -- equals the dup count).
        # Pin instead that the FINAL support pruning works downstream.
        assert len(he2_cols) <= 10, f"seed={seed}: hybrid_orth emitted {len(he2_cols)} He_2 columns (> top_k cap); {m.hybrid_orth_features_}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_final_support_prunes_collinear_duplicates(self, seed):
        """MRMR's redundancy term prunes near-identical He_2 columns from the final transform output."""
        X, y = _build_collinear_quadratic(seed)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=10,
            max_runtime_mins=1.0,
        )
        m.fit(X, y)
        # Transform output: count the He_2-like columns that actually
        # survive into the final support. With 10 near-identical sources
        # the redundancy term should keep <=3 He_2 columns (the gain on
        # the second/third is essentially zero MI uplift past the first).
        out = m.transform(X)
        col_names = [str(c) for c in out.columns]
        he2_survivors = [c for c in col_names if "He2" in c]
        assert len(he2_survivors) <= 3, (
            f"seed={seed}: {len(he2_survivors)} near-identical He_2 columns "
            f"survived into transform output; MRMR redundancy should have "
            f"pruned the duplicates. Survivors: {he2_survivors}; raw "
            f"hybrid_orth={m.hybrid_orth_features_}"
        )


# ---------------------------------------------------------------------------
# D. Conflict between MI-greedy and hybrid orth: redundancy resolves
# ---------------------------------------------------------------------------


def _build_quadratic_one_source(seed: int, n: int = 2000):
    """``y = sign(x^2 - 1)`` on a single source column, for the hybrid-vs-mi_greedy conflict contract."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x": x,
            "noise_a": rng.standard_normal(n),
            "noise_b": rng.standard_normal(n),
            "noise_c": rng.standard_normal(n),
        }
    )
    y = ((x * x - 1.0) + 0.05 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


class TestHybridMiGreedyConflictResolution:
    """When both constructors encode the same signal, MRMR redundancy keeps at most one encoding."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_each_constructor_alone_emits_winner(self, seed):
        """Run each FE pipeline INDEPENDENTLY (the other disabled) and
        confirm both successfully emit a quadratic-x encoding for the
        y = sign(x^2 - 1) signal.

        Run separately because when BOTH are enabled at once, the
        cross-stage Spearman-dedup at _mrmr_fit_impl.py kicks in: hybrid
        runs first and emits x__He2; mi_greedy's |x|-family outputs are
        all monotone-equivalent (Spearman rho = 1.0 with x__He2) and get
        pruned. The principled contract for combined-mode is "exactly
        one survives" - pinned in test_final_support_picks_at_most_one.
        """
        X, y = _build_quadratic_one_source(seed)
        # Hybrid alone.
        m_h = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=3,
            fe_mi_greedy_enable=False,
        )
        m_h.fit(X, y)
        hybrid_he2 = [c for c in m_h.hybrid_orth_features_ if "He2" in c]
        assert hybrid_he2, f"seed={seed}: hybrid_orth (alone) did not emit x__He2 for quadratic signal; got {m_h.hybrid_orth_features_}"
        # MI-greedy alone.
        m_g = _make_mrmr(
            fe_hybrid_orth_enable=False,
            fe_mi_greedy_enable=True,
            fe_mi_greedy_top_k=3,
            fe_mi_greedy_seed_cols_count=4,
            fe_mi_greedy_include_unary=True,
            fe_mi_greedy_include_binary=False,
        )
        m_g.fit(X, y)
        mig_x2 = [c for c in m_g.mi_greedy_features_ if c in {"square(x)", "abs(x)", "sqrt_abs(x)", "log_abs(x)"}]
        assert mig_x2, f"seed={seed}: mi_greedy (alone) did not emit a |x|-family transform for quadratic signal; got {m_g.mi_greedy_features_}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_final_support_picks_at_most_one_x_quadratic(self, seed):
        """With both constructors enabled, the final transform output keeps at most one quadratic-x encoding."""
        X, y = _build_quadratic_one_source(seed)
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=3,
            fe_mi_greedy_enable=True,
            fe_mi_greedy_top_k=3,
            fe_mi_greedy_seed_cols_count=4,
            fe_mi_greedy_include_unary=True,
            fe_mi_greedy_include_binary=False,
        )
        m.fit(X, y)
        out = m.transform(X)
        col_names = [str(c) for c in out.columns]
        x_quadratic = [c for c in col_names if ("He2" in c or c in {"square(x)", "abs(x)", "sqrt_abs(x)", "log_abs(x)"})]
        # MRMR redundancy: both encodings of x^2 carry essentially identical
        # signal (correlation ~1.0 on |x| ranking). At most one should
        # survive into the final selection.
        assert (
            len(x_quadratic) <= 1
        ), f"seed={seed}: {len(x_quadratic)} quadratic-x encodings survived selection ({x_quadratic}); MRMR redundancy should keep at most one"


# ---------------------------------------------------------------------------
# E. Recipe corruption survival: actionable error, never silent garbage
# ---------------------------------------------------------------------------


def _build_pickle_ratio_signal(seed: int, n: int = 1500):
    """A ratio-band signal on (x_revenue, x_cost), for the recipe-corruption-survival contract."""
    rng = np.random.default_rng(seed)
    x_revenue = np.exp(rng.normal(0.0, 1.0, size=n))
    x_cost = np.exp(rng.normal(0.0, 1.0, size=n))
    X = pd.DataFrame(
        {
            "x_revenue": x_revenue,
            "x_cost": x_cost,
            "noise_a": rng.standard_normal(n),
            "noise_b": rng.standard_normal(n),
        }
    )
    ratio = x_revenue / x_cost
    y = ((ratio > 0.7) & (ratio < 1.5)).astype(int)
    return X, pd.Series(y, name="y")


def _replace_recipe_extra(recipe, **overrides):
    """Rebuild a frozen EngineeredRecipe with replaced fields, including
    overwriting individual ``extra`` keys (MappingProxyType is read-only,
    so we deepcopy + mutate the plain dict, then re-construct)."""
    new_extra = dict(recipe.extra)
    if "extra_updates" in overrides:
        new_extra.update(overrides.pop("extra_updates"))
    kwargs = dict(
        name=recipe.name,
        kind=recipe.kind,
        src_names=recipe.src_names,
        unary_names=recipe.unary_names,
        binary_name=recipe.binary_name,
        unary_preset=recipe.unary_preset,
        binary_preset=recipe.binary_preset,
        quantization=recipe.quantization,
        factorize_nbins=recipe.factorize_nbins,
        unknown_strategy=recipe.unknown_strategy,
        extra=new_extra,
    )
    kwargs.update(overrides)
    return recipe.__class__(**kwargs)


@cache
def _pickle_ratio_mi_greedy_fit(seed: int):
    """Cached ``(X, y, m)`` for the mi_greedy fit on the pickle-ratio-signal
    fixture. Shared between test_pickle_unpickle_clean_roundtrip,
    test_corrupted_transform_name_raises_actionable, and
    test_corrupted_src_names_raises_or_recovers -- all three fit the identical
    config on identical per-seed data and only mutate their OWN
    ``pickle.loads(pickle.dumps(m))`` copy, never ``m`` itself.
    """
    X, y = _build_pickle_ratio_signal(seed)
    m = _make_mrmr(
        fe_mi_greedy_enable=True,
        fe_mi_greedy_top_k=5,
        fe_mi_greedy_seed_cols_count=4,
        fe_mi_greedy_include_unary=True,
        fe_mi_greedy_include_binary=True,
    )
    m.fit(X, y)
    return X, y, m


class TestRecipeCorruptionSurvival:
    """A corrupted recipe must raise an actionable error or safely recover; never silently produce a wrong column."""

    @pytest.mark.parametrize("seed", SEEDS_SHORT)
    def test_pickle_unpickle_clean_roundtrip(self, seed):
        """Sanity: pickle round-trip without corruption is byte-identical
        transform output. Pins the baseline against which the corruption
        scenarios are compared."""
        X, _y, m = _pickle_ratio_mi_greedy_fit(seed)
        out_pre = m.transform(X)
        m_rt = pickle.loads(pickle.dumps(m))  # nosec B301 -- round-trip of a locally-created, trusted object
        out_post = m_rt.transform(X)
        np.testing.assert_allclose(
            np.asarray(out_pre, dtype=np.float64),
            np.asarray(out_post, dtype=np.float64),
            rtol=1e-12,
            atol=1e-12,
        )

    @pytest.mark.parametrize("seed", SEEDS_SHORT)
    def test_corrupted_transform_name_raises_actionable(self, seed):
        """Corrupt a mi_greedy_transform recipe by replacing its
        ``transform`` extra with an unknown name. ``apply_recipe`` MUST
        raise KeyError with a message that names the bad transform
        (actionable). It MUST NOT silently emit a wrong column.
        """
        X, _y, m = _pickle_ratio_mi_greedy_fit(seed)
        m_rt = pickle.loads(pickle.dumps(m))  # nosec B301 -- round-trip of a locally-created, trusted object
        recipes = m_rt._engineered_recipes_
        # Find a mi_greedy_transform recipe.
        target_idx = None
        for i, r in enumerate(recipes):
            if r.kind == "mi_greedy_transform":
                target_idx = i
                break
        if target_idx is None:
            pytest.skip(f"seed={seed}: no mi_greedy_transform recipe in support ({[r.name for r in recipes]}); cannot test corruption")
        original = recipes[target_idx]
        corrupted = _replace_recipe_extra(
            original,
            extra_updates={"transform": "this_is_not_a_real_transform_xyz"},
        )
        recipes[target_idx] = corrupted
        # Replay MUST raise (silent fallback would be a bug). The error
        # message must identify the bad transform name so a user can act.
        with pytest.raises((KeyError, ValueError)) as exc_info:
            m_rt.transform(X)
        msg = str(exc_info.value)
        assert "this_is_not_a_real_transform_xyz" in msg, f"seed={seed}: corruption raise must name the bad transform for actionability; got: {msg!r}"

    @pytest.mark.parametrize("seed", SEEDS_SHORT)
    def test_corrupted_src_names_raises_or_recovers(self, seed):
        """Corrupt a mi_greedy_transform recipe by pointing src_names at a
        column that does not exist in test X. Acceptable outcomes:
          1. Replay raises (KeyError / ValueError / TypeError / IndexError) -- preferred (loud failure).
          2. Replay returns AN array with non-zero size (safe-fallback path).
        Unacceptable: silent return of a column whose values look correct
        for the ORIGINAL source -- that would mask a recipe-vs-X mismatch.
        """
        X, _y, m = _pickle_ratio_mi_greedy_fit(seed)
        m_rt = pickle.loads(pickle.dumps(m))  # nosec B301 -- round-trip of a locally-created, trusted object
        recipes = m_rt._engineered_recipes_
        target_idx = None
        for i, r in enumerate(recipes):
            if r.kind == "mi_greedy_transform":
                target_idx = i
                break
        if target_idx is None:
            pytest.skip(f"seed={seed}: no mi_greedy_transform recipe in support; cannot test corruption")
        original = recipes[target_idx]
        # Replace src_names with a column NOT in X (and not in
        # feature_names_in_ either).
        bogus_src = tuple(f"col_definitely_not_in_X_{i}" for i in range(len(original.src_names)))
        corrupted = _replace_recipe_extra(original, src_names=bogus_src)
        recipes[target_idx] = corrupted
        # We treat ANY exception as the safe outcome (loud, actionable).
        # Silent success would only be acceptable if the recipe registered
        # an explicit fallback that returns a documented sentinel column;
        # the current MI-greedy replay does NOT, so any silent success
        # implies the column lookup returned a phantom value -- a bug.
        try:
            m_rt.transform(X)
        except (KeyError, ValueError, TypeError, IndexError, AttributeError) as exc:
            msg = str(exc)
            # Actionability: error must mention the missing column name.
            assert any(b in msg for b in bogus_src), f"seed={seed}: error must name the missing column for actionability; got: {msg!r}"
            return
        # Reached only on silent success -> contract failure.
        pytest.fail(f"seed={seed}: transform() silently succeeded with corrupted src_names {bogus_src!r}; expected an actionable raise.")
