"""
Comprehensive tests for feature_selection/filters.py

Tests include:
- Property-based tests for helper functions using hypothesis
- MRMR feature selection tests for classification and regression
- Feature engineering capability tests
- Edge cases and integration tests
"""

import re

import pytest
import numpy as np
import pandas as pd
import warnings

from hypothesis import given, settings, strategies as st, assume, HealthCheck
from hypothesis.extra.numpy import arrays

from sklearn.datasets import make_classification, make_regression

# Import the module under test
from mlframe.feature_selection.filters import (
    MRMR,
    entropy,
    categorize_dataset,
    discretize_array,
    compute_mi_from_classes,
)


_IDENT = re.compile(r"[A-Za-z_]\w*")


def _referenced_columns(mrmr) -> set:
    """Set of identifier tokens that appear anywhere in a fitted selector's
    ``get_feature_names_out()`` (raw column names AND the names of source
    columns folded into engineered features such as ``div(a,reciproc(b))``).

    Under the full-mode default a synergy like ``y=a*b`` is recovered as one
    engineered feature with an empty raw ``support_``; this helper lets the
    membership assertions credit ``a`` and ``b`` as "selected" because they are
    genuinely used by the surviving engineered feature. Function-name tokens
    (div/add/log/...) are also returned but are harmless for an
    ``expected_feature in referenced`` check since real column names never
    collide with the FE op vocabulary in these fixtures.
    """
    names = list(mrmr.get_feature_names_out())
    referenced: set = set()
    for nm in names:
        referenced |= set(_IDENT.findall(str(nm)))
    return referenced

class TestMRMRFeatureEngineering:
    """Test MRMR's feature engineering capabilities."""

    @pytest.mark.slow
    def test_synergistic_feature_detection(self, synergistic_features_data):
        """
        Test that MRMR detects synergistic features.
        y = a^2/b + log(c)*sin(d)

        MRMR should:
        1. Select a, b, c, d (not e)
        2. Potentially recommend engineered features
        """
        df, y, expected_features = synergistic_features_data

        mrmr = MRMR(
            full_npermutations=5,
            baseline_npermutations=10,
            fe_max_steps=1,  # Enable feature engineering
            fe_min_pair_mi_prevalence=1.1,
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X=df, y=y)

        # Re-baselined for the full-mode default (use_simple_mode=False, prewarp
        # ON by default): on this synergy target ``y=a**2/b + log(c)*sin(d)`` the FE
        # pipeline recovers BOTH terms as engineered features with an EMPTY raw
        # ``support_`` -- measured ``div(sqr(a),abs(b))`` (== a**2/|b|) and
        # ``mul(prewarp(c),sin(d))`` (== log(c)*sin(d) via the learned pre-warp). The
        # old "informative feature in support_ names" check then fails despite a
        # PERFECT recovery (all 4 informative columns folded into 2 engineered
        # survivors, noise 'e' dropped). Credit each source column when referenced by
        # ANY selected feature (raw or engineered), mirroring the multiplicative /
        # additive synergy tests. Still falsifiable: if FE failed to combine the
        # pairs neither the informative columns nor a recovery would appear, and 'e'
        # leaking into any survivor would still trip the noise assertion.
        referenced = _referenced_columns(mrmr)

        # Noise feature 'e' must not be referenced by any selected feature.
        assert 'e' not in referenced, (
            f"Noise feature 'e' should not be selected; "
            f"names={list(mrmr.get_feature_names_out())}"
        )

        # At least 3 of the 4 informative features must be referenced by a survivor.
        informative_selected = sum(1 for f in expected_features if f in referenced)
        assert informative_selected >= 3, (
            f"Expected at least 3 informative features referenced, got "
            f"{informative_selected}: {list(mrmr.get_feature_names_out())}"
        )

    @pytest.mark.slow
    def test_feature_engineering_example(self, feature_engineering_example_data):
        """
        Test the user's exact example from the ticket:
        y = a^2/b + log(c)*sin(d)

        MRMR should select a, b, c, d and potentially recommend:
        - mul(log(c), sin(d))
        - mul(sqr(a), reciproc(b))
        """
        df, y, expected_features = feature_engineering_example_data

        mrmr = MRMR(
            full_npermutations=10,
            baseline_npermutations=20,
            fe_max_steps=2,
            fe_min_pair_mi_prevalence=1.05,
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X=df, y=y)

        # Re-baselined for the full-mode default (use_simple_mode=False, prewarp ON):
        # the user's ticket target ``y=a**2/b + log(c)*sin(d)`` is recovered as
        # ENGINEERED features (e.g. div(sqr(a),abs(b)) and mul(prewarp(c),sin(d)))
        # with an empty raw ``support_``, so 'a'/'b' live INSIDE the engineered
        # survivors rather than in raw support_ -- the old raw-name membership check
        # then fails despite the intended recovery. Credit a/b when referenced by any
        # selected feature, mirroring the synergy / multiplicative / additive tests.
        referenced = _referenced_columns(mrmr)

        # Check that noise is not referenced by any selected feature.
        assert 'e' not in referenced, (
            f"Noise feature 'e' should not be selected; "
            f"names={list(mrmr.get_feature_names_out())}"
        )

        # Check that the key source columns are referenced by a survivor (raw or engineered).
        for feat in ['a', 'b']:
            assert feat in referenced, (
                f"Feature '{feat}' should be selected (raw or engineered); "
                f"names={list(mrmr.get_feature_names_out())}"
            )

    def test_feature_engineering_example_single_compound(self):
        """STRENGTHENED canonical gate on the user's EXACT ticket formula INCLUDING the f/5 noise term:

            y = a**2/b + f/5 + log(c)*sin(d)     ('f' is NOT a feature -> f/5 is irreducible noise)

        (NB: the shared ``feature_engineering_example_data`` fixture intentionally drops f/5, testing the
        easier noiseless target; this test uses the real noisy one so the redundancy gate is exercised
        under realistic conditions.) The signal must be recovered as essentially ONE fused compound that
        carries BOTH structural halves (an a/b term AND a c/d term), with the redundancy gate dropping any
        standalone sub-fragment -- including raw operands like 'c' whose marginal signal toward y is ~0
        (E[log(c)*sin(d)] over d~U(0,2pi) is ~0). Catches the fragmentation regression (full compound PLUS
        redundant siblings such as raw 'c' / mul(log(c),sin(d)) / div(sqr(a),...)).
        """
        rng = np.random.default_rng(42)
        n = 10_000
        a = rng.random(n) + 0.1; b = rng.random(n) + 0.1; c = rng.random(n) + 0.1
        d = rng.random(n) * 2 * np.pi; e = rng.random(n); f = rng.random(n)
        df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
        y = a**2 / b + f / 5 + np.log(c) * np.sin(d)

        mrmr = MRMR(
            full_npermutations=10, baseline_npermutations=20, fe_max_steps=2,
            fe_min_pair_mi_prevalence=1.05, verbose=0, n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X=df, y=y)

        names = [str(n) for n in mrmr.get_feature_names_out()]
        assert names, "selector returned no features"

        # Classify each selected feature by which signal half(s) it references. The two structural halves
        # are {a,b} (the a**2/b term) and {c,d} (the log(c)*sin(d) term); 'e' is pure noise.
        def _cols(nm):
            return set(_IDENT.findall(nm)) & {"a", "b", "c", "d", "e"}

        assert all("e" not in _cols(nm) for nm in names), f"noise 'e' referenced: {names}"

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

        # (1) the full compound (both halves fused into ONE feature) must be recovered.
        assert len(full) >= 1, f"no single feature fuses both a/b and c/d halves: {names}"
        # (2) it must be THE selection, not one of several: no standalone half-fragment may co-survive
        #     with the full compound (those are redundant given it). This is the "only 1 compound" property.
        assert not frag_cd, f"redundant c/d-only sub-term(s) survived alongside the full compound: {frag_cd} in {names}"
        assert not frag_ab, f"redundant a/b-only sub-term(s) survived alongside the full compound: {frag_ab} in {names}"
        # (3) exactly one fused compound (no duplicate near-identical compounds).
        assert len(full) == 1, f"expected exactly ONE fused compound, got {len(full)}: {full}"

        # (4) the compound must be the CLEAN ADDITIVE form add(div(sqr(a),b), mul(log(c),sin(d))) -- NOT a
        #     monotone-distorted variant. MI is monotone-invariant, so the search can pick a warped form
        #     (log(a**2/b), double-prewarp, sub instead of add) that is MI-equivalent but ugly + less
        #     linearly usable. The target is ADDITIVE (a**2/b + log(c)*sin(d)), so the recovered compound
        #     must be additive with each half left CLEAN. Enforce:
        compound = full[0]
        #   (4a) no learned prewarp warp anywhere -- the inner log(c)*sin(d) is library-expressible, so a
        #        prewarp form is only monotone-equivalent and must lose to the clean library form.
        assert "prewarp" not in compound, f"compound uses a monotone-distorting prewarp warp: {compound}"
        #   (4b) additive top-level combination (a**2/b + log(c)*sin(d)), not a subtraction/quotient of halves.
        assert compound.startswith("add("), f"compound is not an additive (add) combination: {compound}"
        #   (4c) the c/d half is the CLEAN log(c)*sin(d) (no warp), and (4d) the a/b half is NOT log-wrapped
        #        (the a**2/b term must stay magnitude-carrying, not collapsed to log(a**2/b)).
        assert "log(c)" in compound and "sin(d)" in compound, f"c/d half is not the clean log(c)*sin(d): {compound}"
        assert "sqr(a)" in compound, f"a/b half is not the clean a**2 term: {compound}"
        for _bad in ("log(div", "log(mul", "log(neg", "log(sub", "log(add"):
            assert _bad not in compound, f"a/b half is log-distorted ({_bad}...): {compound}"

    # ----------------------------------------------------------------------------------------------
    # PREWARP vs CLEAN-LIBRARY-FORM enforcement pins (2026-06-20).
    # MI is monotone-invariant, so on a target whose structure the elementary library ALREADY expresses
    # (e.g. log(c)*sin(d)), a learned per-operand prewarp warp is only MONOTONE-equivalent and must LOSE
    # to the clean library form (the magnitude-carrying a**2/b is more LINEARLY usable for an additive
    # target than log(a**2/b)). But prewarp must STILL win where it is genuinely needed -- a NON-monotone
    # inner the library cannot express. These two pins lock both sides so the enforcement can never
    # silently regress into either "prewarp distorts a clean recovery" or "prewarp disabled".
    # ----------------------------------------------------------------------------------------------

    def test_clean_library_form_preferred_over_monotone_prewarp(self):
        """ENFORCEMENT PIN: on a target the library already expresses (y = a**2/b + f/5 + log(c)*sin(d)),
        NO selected feature may use a learned ``prewarp`` warp -- the clean library compound
        ``add(div(sqr(a),b), mul(log(c),sin(d)))`` is more linearly usable and must win over the
        monotone-MI-equivalent prewarp distortion (``sub(log(div(sqr(a),neg(b))),prewarp(mul(prewarp(c),
        sin(d))))``). Guards the exact regression the user flagged."""
        rng = np.random.default_rng(42)
        n = 10_000
        a = rng.random(n) + 0.1; b = rng.random(n) + 0.1; c = rng.random(n) + 0.1
        d = rng.random(n) * 2 * np.pi; e = rng.random(n); f = rng.random(n)
        df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
        y = a**2 / b + f / 5 + np.log(c) * np.sin(d)
        mrmr = MRMR(full_npermutations=10, baseline_npermutations=20, fe_max_steps=2,
                    fe_min_pair_mi_prevalence=1.05, verbose=0, n_jobs=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X=df, y=y)
        names = [str(nm) for nm in mrmr.get_feature_names_out()]
        offenders = [nm for nm in names if "prewarp" in nm]
        assert not offenders, (
            f"prewarp warp used on a library-expressible target -- a monotone-equivalent distortion that "
            f"should lose to the clean library form: {offenders} in {names}"
        )

    def test_prewarp_retained_on_genuine_nonmonotone_inner(self):
        """NON-REGRESSION PIN (other side of the enforcement): on a GENUINELY non-monotone inner the
        elementary library CANNOT express -- y = (a**3 - 2a)*(b**2 - b) -- a learned prewarp form
        (``mul(prewarp(a),prewarp(b))``) MUST still be selected. This guards the linear-usability
        enforcement from over-correcting into "prewarp disabled"."""
        rng = np.random.default_rng(7)
        n = 20_000
        a = rng.uniform(-2, 2, n); b = rng.uniform(-2, 2, n); e = rng.uniform(-2, 2, n)
        df = pd.DataFrame({"a": a, "b": b, "e": e})
        y = (a**3 - 2 * a) * (b**2 - b)  # non-monotone inner in a (and b) -> library is blind, prewarp wins
        mrmr = MRMR(fe_max_steps=1, verbose=0, n_jobs=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X=df, y=y)
        names = [str(nm) for nm in mrmr.get_feature_names_out()]
        assert any("prewarp" in nm for nm in names), (
            f"prewarp recovery LOST on a genuinely non-monotone inner the library cannot express -- the "
            f"linear-usability enforcement over-corrected and disabled prewarp's intended case: {names}"
        )

    def test_multiplicative_synergy(self, multiplicative_synergy_data):
        """Test that MRMR detects multiplicative synergy: y = a * b."""
        df, y, expected_features = multiplicative_synergy_data

        mrmr = MRMR(
            full_npermutations=5,
            baseline_npermutations=5,
            fe_max_steps=1,
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X=df, y=y)

        # Re-baselined for full-mode default (use_simple_mode=False): on the
        # multiplicative target y=a*b full mode recovers the synergy as a SINGLE
        # engineered feature (e.g. div(a,reciproc(b)) == a*b) with an EMPTY raw
        # ``support_``, so the old "a and b both in support_ names" check fails
        # despite a perfect recovery. Credit both source columns when they are
        # referenced by ANY selected feature (raw or engineered). Still
        # falsifiable: if FE failed to combine the pair, neither a nor b would
        # appear in any survivor.
        referenced = _referenced_columns(mrmr)
        for feat in expected_features:
            assert feat in referenced, (
                f"Feature '{feat}' should be selected (raw or engineered); "
                f"names={list(mrmr.get_feature_names_out())}"
            )

        # Noise feature should ideally not be selected
        # (though with limited permutations it might be)

    def test_additive_synergy(self, additive_synergy_data):
        """Test that MRMR detects additive relationships: y = a + b."""
        df, y, expected_features = additive_synergy_data

        mrmr = MRMR(
            full_npermutations=5,
            baseline_npermutations=5,
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X=df, y=y)

        # Re-baselined for full-mode default (use_simple_mode=False): on the
        # additive target y=a+b full mode recovers the synergy as a SINGLE
        # engineered feature add(a,b) with an EMPTY raw ``support_``; the old
        # "a and b both in support_ names" check then fails despite a perfect
        # recovery. Credit both source columns when referenced by ANY selected
        # feature (raw or engineered). Falsifiable as above.
        referenced = _referenced_columns(mrmr)
        for feat in expected_features:
            assert feat in referenced, (
                f"Feature '{feat}' should be selected (raw or engineered); "
                f"names={list(mrmr.get_feature_names_out())}"
            )

    @pytest.mark.parametrize("transform_name,transform_func,feature_gen", [
        ("squared", lambda x: x**2, lambda rng: rng.standard_normal(3000)),
        ("log", lambda x: np.log(np.abs(x) + 1), lambda rng: rng.random(3000) + 0.1),
        ("sin", np.sin, lambda rng: rng.random(3000) * 2 * np.pi),
    ])
    def test_unary_transform_detection(self, transform_name, transform_func, feature_gen):
        """Test that MRMR can detect features with unary transforms."""
        rng = np.random.default_rng(42)

        a = feature_gen(rng)
        b = rng.standard_normal(len(a))  # Noise

        y = transform_func(a) + rng.standard_normal(len(a)) * 0.1

        df = pd.DataFrame({'a': a, 'b': b})

        mrmr = MRMR(
            full_npermutations=5,
            baseline_npermutations=5,
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X=df, y=y)

        selected_indices = mrmr.support_.tolist()

        # Feature 'a' (index 0) should be selected
        assert 0 in selected_indices, f"Feature 'a' should be selected for {transform_name} transform"

    def test_no_false_positives_independent_features(self):
        """Test that MRMR doesn't over-select when features are independent."""
        rng = np.random.default_rng(42)
        n = 5000

        # All independent features
        df = pd.DataFrame({
            'a': rng.standard_normal(n),
            'b': rng.standard_normal(n),
            'c': rng.standard_normal(n),
            'd': rng.standard_normal(n),
            'e': rng.standard_normal(n),
        })

        # Target only depends on 'a'
        y = df['a'] + rng.standard_normal(n) * 0.1

        mrmr = MRMR(
            full_npermutations=10,
            baseline_npermutations=10,
            min_relevance_gain=0.01,  # Higher threshold
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X=df, y=y)

        # Should select very few features (ideally just 'a')
        assert mrmr.n_features_ <= 3, \
            f"Expected few features for simple relationship, got {mrmr.n_features_}"


# ================================================================================================
# MRMR Edge Cases
# ================================================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
