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

class TestMRMRRegression:
    """Test MRMR on regression tasks."""

    def test_linear_regression(self, simple_regression_data):
        """Test MRMR identifies informative features in regression."""
        X, y, informative_indices = simple_regression_data

        mrmr = MRMR(
            full_npermutations=5,
            baseline_npermutations=5,
            min_relevance_gain=0.001,
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)

        # Re-baselined for full-mode default (use_simple_mode=False): full mode
        # de-duplicates the linear informative columns into a single ENGINEERED
        # combination (e.g. add(informative_0,log(informative_1))) that lives in
        # get_feature_names_out(), NOT in the raw integer ``support_`` array, so
        # the old ``support_ & informative`` overlap sees support_==[] and
        # undercounts. Credit engineered references to informative columns.
        # Still falsifiable: an all-noise selection references no informative col.
        from tests.feature_selection._biz_val_synth import signal_recovery_count
        overlap = signal_recovery_count(mrmr, list(informative_indices),
                                        prefix="informative_")
        assert overlap >= 1, (
            f"No informative features detected (raw or engineered); "
            f"names={list(mrmr.get_feature_names_out())}"
        )

    def test_nonlinear_regression(self, nonlinear_transform_data):
        """Test MRMR on data with nonlinear relationships."""
        X, y, informative_names = nonlinear_transform_data

        mrmr = MRMR(
            full_npermutations=5,
            baseline_npermutations=5,
            verbose=0,
            n_jobs=1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)

        assert hasattr(mrmr, 'n_features_')
        assert mrmr.n_features_ > 0


# ================================================================================================
# MRMR Feature Engineering Tests
# ================================================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
