"""
Comprehensive tests for feature_selection/filters.py

Tests include:
- Property-based tests for helper functions using hypothesis
- MRMR feature selection tests for classification and regression
- Feature engineering capability tests
- Edge cases and integration tests
"""

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


class TestMRMRClassification:
    """Test MRMR on classification tasks."""

    def test_binary_classification(self, simple_classification_data):
        """Test MRMR identifies informative features in binary classification."""
        X, y, informative_indices = simple_classification_data

        mrmr = MRMR(full_npermutations=5, baseline_npermutations=5, min_relevance_gain=0.001, verbose=0, n_jobs=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)

        # Re-baselined for full-mode default (use_simple_mode=False): full mode
        # de-duplicates the correlated informative columns into ENGINEERED
        # combinations (e.g. add(informative_1,neg(informative_2))) that live in
        # get_feature_names_out(), NOT in the raw integer ``support_`` array, so
        # the old ``support_ & informative`` overlap undercounts a correct
        # selection (it sees support_==[]). Credit engineered features that
        # reference an informative column. Still falsifiable: an all-noise
        # selection references none of the informative columns.
        from tests.feature_selection._biz_val_synth import signal_recovery_count

        overlap = signal_recovery_count(mrmr, list(informative_indices), prefix="informative_")
        assert overlap >= 1, f"No informative features detected (raw or engineered); names={list(mrmr.get_feature_names_out())}"

    def test_imbalanced_classes(self, imbalanced_classification_data):
        """Test MRMR on imbalanced classification data.

        B15 (post-plan default flip): with the new ``fe_fallback_to_all=False``
        default, MRMR returns ``n_features_=0`` when the screening pass
        rejects every candidate at the configured permutation budget on
        a 95/5 imbalance. The legacy fallback masked this rejection by
        running FE on all features. We pin ``fe_fallback_to_all=True``
        here to keep asserting that the legacy path still works for
        callers who opt in.
        """
        X, y, informative_indices = imbalanced_classification_data

        mrmr = MRMR(full_npermutations=5, baseline_npermutations=5, fe_fallback_to_all=True, verbose=0, n_jobs=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)

        assert hasattr(mrmr, "n_features_")
        assert mrmr.n_features_ > 0

    def test_multiclass_classification(self, multiclass_data):
        """Test MRMR on multiclass classification data."""
        X, y, informative_indices = multiclass_data

        mrmr = MRMR(full_npermutations=5, baseline_npermutations=5, verbose=0, n_jobs=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)

        assert hasattr(mrmr, "n_features_")
        assert mrmr.n_features_ > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
