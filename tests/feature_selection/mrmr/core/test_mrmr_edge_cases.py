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

class TestMRMREdgeCases:
    """Test MRMR edge cases and error handling."""

    # PR-11 dedup: test_single_feature and test_all_noise_features migrated
    # to test_selectors_shared.py::TestSharedTrivialInputs (parametrized
    # over both RFECV and MRMR).

    def test_constant_feature(self):
        """Test MRMR with constant feature."""
        rng = np.random.default_rng(42)
        n = 200
        X = pd.DataFrame({
            'informative': rng.standard_normal(n),
            'constant': np.ones(n),
        })
        y = (X['informative'] > 0).astype(int)

        mrmr = MRMR(full_npermutations=3, baseline_npermutations=3, verbose=0, n_jobs=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)

        # Should handle constant feature gracefully
        assert hasattr(mrmr, "n_features_")

    def test_highly_correlated_features(self, correlated_features_data):
        """Test MRMR with highly correlated features."""
        X, y, _ = correlated_features_data

        mrmr = MRMR(full_npermutations=5, baseline_npermutations=5, verbose=0, n_jobs=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)

        # Should complete and select features
        assert mrmr.n_features_ > 0

    # PR-11 dedup: test_ndarray_input migrated to
    # test_selectors_shared.py::TestSharedInputTypes::test_numpy_array_input.

    def test_skip_retraining_parameter_exists(self, simple_classification_data):
        """Test skip_retraining_on_same_content parameter can be set."""
        X, y, _ = simple_classification_data

        # Just test that the parameter can be set without error
        mrmr = MRMR(full_npermutations=3, baseline_npermutations=3, skip_retraining_on_same_content=True, verbose=0, n_jobs=1)

        mrmr.fit(X, y)
        assert hasattr(mrmr, "n_features_")

    def test_target_container_normalization(self):
        """MRMR target normalization accepts numpy, pandas and polars
        carriers. The pre-fix code handled only numpy and pandas-like
        ``.values``."""
        pl = pytest.importorskip("polars")
        from mlframe.feature_selection.filters.mrmr import _target_to_numpy_values

        base = np.array([0, 1, 1, 0], dtype=np.int64)
        cases = [
            base,
            pd.Series(base),
            pd.DataFrame({"target": base}),
            pl.Series("target", base),
            pl.DataFrame({"target": base}),
            base.tolist(),
        ]

        for target in cases:
            vals = _target_to_numpy_values(target)
            assert isinstance(vals, np.ndarray)
            assert vals.shape[0] == base.shape[0]
            assert np.asarray(vals).reshape(base.shape[0], -1)[:, 0].tolist() == base.tolist()

        assert _target_to_numpy_values(base) is base

    @pytest.mark.parametrize(
        "x_backend,target_backend",
        [
            ("pandas", "polars_series"),
            ("polars", "polars_series"),
            ("polars", "polars_frame"),
        ],
    )
    def test_target_containers_fit(self, x_backend, target_backend):
        """Regression: Polars target carriers have no ``.values`` attr.

        The suite path ``FeatureSelectionConfig(use_mrmr_fs=True)`` passes a
        Polars target through sklearn's supervised ``fit_transform`` call.
        MRMR must normalize that target via ``to_numpy`` rather than assuming
        pandas-like ``y.values``.
        """
        pl = pytest.importorskip("polars")
        from mlframe.feature_selection.filters import CatFEConfig

        rng = np.random.default_rng(0)
        n = 120
        signal = rng.normal(size=n)
        data = {
            "signal": signal,
            "noise": rng.normal(size=n),
        }
        X = pd.DataFrame(data) if x_backend == "pandas" else pl.DataFrame(data)
        target = (signal > 0).astype(np.int64)
        if target_backend == "polars_series":
            y = pl.Series("target", target)
        elif target_backend == "polars_frame":
            y = pl.DataFrame({"target": target})
        else:
            raise AssertionError(target_backend)

        mrmr = MRMR(
            full_npermutations=1,
            baseline_npermutations=1,
            quantization_nbins=5,
            fe_max_steps=0,
            min_features_fallback=1,
            cat_fe_config=CatFEConfig(enable=False),
            verbose=0,
            n_jobs=1,
            n_workers=1,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)

        assert hasattr(mrmr, "support_")
        assert hasattr(mrmr, "n_features_")

    def test_no_features_selected_transform(self):
        """Test MRMR transform when no features are selected (empty selection).

        This tests the edge case where MRMR finds no useful features due to
        very strict thresholds. The transform should still work without errors.
        Regression test for IndexError with empty numpy array indexing.
        """
        rng = np.random.default_rng(42)
        # Create data where features have zero correlation with target
        X = pd.DataFrame(rng.standard_normal((200, 5)), columns=["a", "b", "c", "d", "e"])
        y = rng.integers(0, 2, 200)

        mrmr = MRMR(
            full_npermutations=2,
            baseline_npermutations=2,
            min_relevance_gain=10.0,  # Extremely high threshold - nothing will pass
            min_nonzero_confidence=0.99,  # Very strict confidence requirement
            verbose=0,
            n_jobs=1,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)

        # Transform should work even when no RAW features survive the strict thresholds.
        X_transformed = mrmr.transform(X)

        # Completed without IndexError; rows preserved and column count == n_features_
        # (engineered features may still be produced even when no raw column is kept).
        assert hasattr(mrmr, "n_features_")
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == mrmr.n_features_
        assert len(mrmr.get_feature_names_out()) == X_transformed.shape[1]

    def test_perfect_feature_detection(self):
        """Test MRMR detects a feature with perfect correlation to target.

        When one feature is perfectly correlated with the target, MRMR
        should identify and select it.
        """
        rng = np.random.default_rng(42)
        n = 500
        # Create noise features
        X = pd.DataFrame({
            'noise1': rng.standard_normal(n),
            'noise2': rng.standard_normal(n),
            'noise3': rng.standard_normal(n),
        })
        # Perfect feature: target is directly derived from it
        X["perfect"] = rng.standard_normal(n)
        y = (X["perfect"] > 0).astype(int)  # Binary classification from perfect feature

        mrmr = MRMR(full_npermutations=5, baseline_npermutations=5, verbose=0, n_jobs=1)

        mrmr.fit(X, y)

        # The perfect feature should be selected - check via support_ mask
        selected_features = X.columns[mrmr.support_].tolist() if hasattr(mrmr, "support_") else []
        assert "perfect" in selected_features, f"Perfect feature not selected. Selected: {selected_features}"
        # It should be among the top features
        assert mrmr.n_features_ >= 1


# ================================================================================================
# MRMR Parameter Coverage Tests
# ================================================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
