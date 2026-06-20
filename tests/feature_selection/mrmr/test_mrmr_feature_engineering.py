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
