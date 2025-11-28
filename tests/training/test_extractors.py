"""
Tests for training/extractors.py module.

Covers:
- FeaturesAndTargetsExtractor base class
- SimpleFeaturesAndTargetsExtractor concrete implementation
- Helper functions: get_dataframe_info, intize_targets, get_sample_weights_by_recency
- Edge cases and error handling
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta
from typing import Dict, Any, Union

from mlframe.training.extractors import (
    FeaturesAndTargetsExtractor,
    SimpleFeaturesAndTargetsExtractor,
    get_dataframe_info,
    intize_targets,
    get_sample_weights_by_recency,
)
from mlframe.training.configs import TargetTypes


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestGetDataframeInfo:
    """Tests for get_dataframe_info function."""

    def test_pandas_dataframe(self):
        """Test info extraction from pandas DataFrame."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.0, 2.0, 3.0],
            'str_col': ['a', 'b', 'c']
        })

        info = get_dataframe_info(df)

        assert isinstance(info, str), "Info should be a string"
        assert len(info) > 0, "Info should not be empty"
        # Should contain column count or entry information
        assert '3' in info or 'entries' in info.lower() or 'columns' in info.lower(), "Should contain DataFrame info"

    def test_polars_dataframe(self):
        """Test info extraction from Polars DataFrame."""
        df = pl.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.0, 2.0, 3.0],
            'str_col': ['a', 'b', 'c']
        })

        info = get_dataframe_info(df)

        assert isinstance(info, str), "Info should be a string"
        assert len(info) > 0, "Info should not be empty"

    def test_unsupported_type_raises_error(self):
        """Test that unsupported types raise TypeError."""
        with pytest.raises(TypeError, match="Unsupported DataFrame type"):
            get_dataframe_info([1, 2, 3])

        with pytest.raises(TypeError, match="Unsupported DataFrame type"):
            get_dataframe_info({"a": 1})


class TestIntizeTargets:
    """Tests for intize_targets function."""

    def test_pandas_series(self):
        """Test conversion of pandas Series."""
        targets = {'target1': pd.Series([0.0, 1.0, 0.0, 1.0])}

        intize_targets(targets)

        assert isinstance(targets['target1'], np.ndarray), "Should convert to numpy array"
        assert targets['target1'].dtype == np.int8, "Should be int8"
        np.testing.assert_array_equal(targets['target1'], [0, 1, 0, 1])

    def test_polars_series(self):
        """Test conversion of Polars Series."""
        targets = {'target1': pl.Series([0, 1, 0, 1])}

        intize_targets(targets)

        assert isinstance(targets['target1'], np.ndarray), "Should convert to numpy array"
        assert targets['target1'].dtype == np.int8, "Should be int8"
        np.testing.assert_array_equal(targets['target1'], [0, 1, 0, 1])

    def test_numpy_array(self):
        """Test conversion of numpy array."""
        targets = {'target1': np.array([0.0, 1.0, 0.0, 1.0])}

        intize_targets(targets)

        assert isinstance(targets['target1'], np.ndarray), "Should remain numpy array"
        assert targets['target1'].dtype == np.int8, "Should be int8"

    def test_multiple_targets(self):
        """Test conversion of multiple targets."""
        targets = {
            'target1': pd.Series([0, 1, 0]),
            'target2': np.array([1, 0, 1]),
            'target3': pl.Series([0, 0, 1])
        }

        intize_targets(targets)

        for name, arr in targets.items():
            assert isinstance(arr, np.ndarray), f"{name} should be numpy array"
            assert arr.dtype == np.int8, f"{name} should be int8"

    def test_unsupported_type_raises_error(self):
        """Test that unsupported types raise TypeError."""
        targets = {'target1': [0, 1, 0]}  # List is not supported

        with pytest.raises(TypeError, match="Unsupported target type"):
            intize_targets(targets)


class TestGetSampleWeightsByRecency:
    """Tests for get_sample_weights_by_recency function."""

    def test_basic_recency_weights(self):
        """Test that more recent samples get higher weights."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        date_series = pd.Series(dates)

        weights = get_sample_weights_by_recency(date_series)

        assert isinstance(weights, (np.ndarray, pd.Series)), "Should return array-like"
        assert len(weights) == len(date_series), "Should have same length as input"

        # More recent dates (later indices) should have higher weights
        # (comparing first and last portions)
        early_weights = weights[:10].mean() if isinstance(weights, np.ndarray) else weights.iloc[:10].mean()
        late_weights = weights[-10:].mean() if isinstance(weights, np.ndarray) else weights.iloc[-10:].mean()
        assert late_weights >= early_weights, "Recent samples should have higher or equal weights"

    def test_custom_min_weight(self):
        """Test custom minimum weight parameter."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        date_series = pd.Series(dates)

        weights_default = get_sample_weights_by_recency(date_series, min_weight=1.0)
        weights_higher = get_sample_weights_by_recency(date_series, min_weight=2.0)

        # Higher min_weight should increase all weights
        assert np.all(weights_higher >= weights_default), "Higher min_weight should increase weights"

    def test_custom_weight_drop(self):
        """Test custom weight drop per year parameter."""
        dates = pd.date_range('2020-01-01', periods=365 * 3, freq='D')  # 3 years
        date_series = pd.Series(dates)

        weights_low_drop = get_sample_weights_by_recency(date_series, weight_drop_per_year=0.05)
        weights_high_drop = get_sample_weights_by_recency(date_series, weight_drop_per_year=0.2)

        # Higher drop rate should create larger spread
        spread_low = np.max(weights_low_drop) - np.min(weights_low_drop)
        spread_high = np.max(weights_high_drop) - np.min(weights_high_drop)
        assert spread_high >= spread_low, "Higher drop rate should create larger weight spread"

    def test_single_date(self):
        """Test with single date (edge case)."""
        date_series = pd.Series([pd.Timestamp('2023-01-01')])

        # Should not crash with single date
        weights = get_sample_weights_by_recency(date_series)
        assert len(weights) == 1, "Should return single weight"


# =============================================================================
# FeaturesAndTargetsExtractor Base Class Tests
# =============================================================================


class TestFeaturesAndTargetsExtractorBase:
    """Tests for FeaturesAndTargetsExtractor base class."""

    def test_init_default_values(self):
        """Test default initialization."""
        extractor = FeaturesAndTargetsExtractor()

        assert extractor.ts_field is None
        assert extractor.datetime_features is None
        assert extractor.group_field is None
        assert extractor.columns_to_drop == set()
        assert extractor.allowed_targets is None
        assert extractor.verbose == 0

    def test_init_with_params(self):
        """Test initialization with parameters."""
        extractor = FeaturesAndTargetsExtractor(
            ts_field='timestamp',
            group_field='group_id',
            columns_to_drop={'col1', 'col2'},
            verbose=1
        )

        assert extractor.ts_field == 'timestamp'
        assert extractor.group_field == 'group_id'
        assert extractor.columns_to_drop == {'col1', 'col2'}
        assert extractor.verbose == 1

    def test_add_features_default_passthrough(self):
        """Test that default add_features just returns the input."""
        extractor = FeaturesAndTargetsExtractor()
        df = pd.DataFrame({'a': [1, 2, 3]})

        result = extractor.add_features(df)

        pd.testing.assert_frame_equal(result, df)

    def test_build_targets_default_empty(self):
        """Test that default build_targets returns empty dict."""
        extractor = FeaturesAndTargetsExtractor()
        df = pd.DataFrame({'a': [1, 2, 3]})

        result = extractor.build_targets(df)

        assert result == {}

    def test_prepare_artifacts_default_empty(self):
        """Test that default prepare_artifacts returns empty dict."""
        extractor = FeaturesAndTargetsExtractor()
        df = pd.DataFrame({'a': [1, 2, 3]})

        result = extractor.prepare_artifacts(df)

        assert result == {}

    def test_get_sample_weights_default_empty(self):
        """Test that default get_sample_weights returns empty dict."""
        extractor = FeaturesAndTargetsExtractor()
        df = pd.DataFrame({'a': [1, 2, 3]})

        result = extractor.get_sample_weights(df)

        assert result == {}

    def test_transform_basic(self):
        """Test basic transform without any features/targets."""
        extractor = FeaturesAndTargetsExtractor()
        df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})

        result = extractor.transform(df)

        # Should return tuple of 8 elements
        assert len(result) == 8, "Transform should return 8 elements"
        df_out, target_by_type, group_ids_raw, group_ids, timestamps, artifacts, cols_to_drop, sample_weights = result

        pd.testing.assert_frame_equal(df_out, df)
        assert target_by_type == {}
        assert group_ids_raw is None
        assert group_ids is None
        assert timestamps is None
        assert artifacts == {}
        assert cols_to_drop == set()
        assert sample_weights == {}

    def test_transform_with_ts_field(self):
        """Test transform with timestamp field."""
        extractor = FeaturesAndTargetsExtractor(ts_field='timestamp')
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'timestamp': pd.date_range('2023-01-01', periods=3)
        })

        result = extractor.transform(df)
        _, _, _, _, timestamps, _, _, _ = result

        assert timestamps is not None
        assert len(timestamps) == 3

    def test_transform_with_group_field(self):
        """Test transform with group field."""
        extractor = FeaturesAndTargetsExtractor(group_field='group_id')
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'group_id': ['A', 'A', 'B', 'B']
        })

        result = extractor.transform(df)
        _, _, group_ids_raw, group_ids, _, _, _, _ = result

        assert group_ids_raw is not None
        assert group_ids is not None
        assert len(np.unique(group_ids)) == 2  # Two groups: A and B


class TestFeaturesAndTargetsExtractorSubclass:
    """Test subclassing FeaturesAndTargetsExtractor."""

    def test_custom_add_features(self):
        """Test custom add_features implementation."""
        class CustomExtractor(FeaturesAndTargetsExtractor):
            def add_features(self, df):
                df = df.copy()
                df['new_feature'] = df['feature1'] * 2
                return df

        extractor = CustomExtractor()
        df = pd.DataFrame({'feature1': [1, 2, 3]})

        result = extractor.transform(df)
        df_out = result[0]

        assert 'new_feature' in df_out.columns
        np.testing.assert_array_equal(df_out['new_feature'].values, [2, 4, 6])

    def test_custom_build_targets(self):
        """Test custom build_targets implementation."""
        class CustomExtractor(FeaturesAndTargetsExtractor):
            def build_targets(self, df):
                return {
                    TargetTypes.REGRESSION: {'my_target': df['target'].values}
                }

        extractor = CustomExtractor()
        df = pd.DataFrame({'feature1': [1, 2, 3], 'target': [10, 20, 30]})

        result = extractor.transform(df)
        target_by_type = result[1]

        assert TargetTypes.REGRESSION in target_by_type
        assert 'my_target' in target_by_type[TargetTypes.REGRESSION]


# =============================================================================
# SimpleFeaturesAndTargetsExtractor Tests
# =============================================================================


class TestSimpleFeaturesAndTargetsExtractorRegression:
    """Tests for SimpleFeaturesAndTargetsExtractor regression targets."""

    def test_single_regression_target(self):
        """Test extraction of single regression target."""
        extractor = SimpleFeaturesAndTargetsExtractor(
            regression_targets=['target']
        )
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0],
            'target': [10.0, 20.0, 30.0]
        })

        result = extractor.transform(df)
        _, target_by_type, _, _, _, _, cols_to_drop, _ = result

        assert TargetTypes.REGRESSION in target_by_type
        assert 'target' in target_by_type[TargetTypes.REGRESSION]
        assert 'target' in cols_to_drop

    def test_multiple_regression_targets(self):
        """Test extraction of multiple regression targets."""
        extractor = SimpleFeaturesAndTargetsExtractor(
            regression_targets=['target1', 'target2']
        )
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'target1': [10.0, 20.0, 30.0],
            'target2': [100.0, 200.0, 300.0]
        })

        result = extractor.transform(df)
        _, target_by_type, _, _, _, _, cols_to_drop, _ = result

        assert TargetTypes.REGRESSION in target_by_type
        assert 'target1' in target_by_type[TargetTypes.REGRESSION]
        assert 'target2' in target_by_type[TargetTypes.REGRESSION]
        assert {'target1', 'target2'} <= cols_to_drop

    def test_missing_regression_target_raises_error(self):
        """Test that missing regression target column raises KeyError."""
        extractor = SimpleFeaturesAndTargetsExtractor(
            regression_targets=['nonexistent_target']
        )
        df = pd.DataFrame({'feature1': [1.0, 2.0, 3.0]})

        with pytest.raises(KeyError, match="Regression target column 'nonexistent_target' not found"):
            extractor.transform(df)


class TestSimpleFeaturesAndTargetsExtractorClassification:
    """Tests for SimpleFeaturesAndTargetsExtractor classification targets."""

    def test_single_classification_target(self):
        """Test extraction of single classification target."""
        extractor = SimpleFeaturesAndTargetsExtractor(
            classification_targets=['target']
        )
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'target': [0, 1, 0]
        })

        result = extractor.transform(df)
        _, target_by_type, _, _, _, _, cols_to_drop, _ = result

        assert TargetTypes.BINARY_CLASSIFICATION in target_by_type
        targets = target_by_type[TargetTypes.BINARY_CLASSIFICATION]
        assert 'target' in targets
        assert targets['target'].dtype == np.int8
        assert 'target' in cols_to_drop

    def test_classification_with_threshold(self):
        """Test classification target with threshold."""
        extractor = SimpleFeaturesAndTargetsExtractor(
            classification_targets=['score'],
            classification_thresholds={'score': 0.5}
        )
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'score': [0.2, 0.4, 0.6, 0.8]
        })

        result = extractor.transform(df)
        _, target_by_type, _, _, _, _, _, _ = result

        targets = target_by_type[TargetTypes.BINARY_CLASSIFICATION]
        # Target name should include threshold
        assert 'score_above_0.5' in targets
        expected = np.array([0, 0, 1, 1], dtype=np.int8)
        np.testing.assert_array_equal(targets['score_above_0.5'], expected)

    def test_classification_with_exact_values(self):
        """Test classification target with exact value matching."""
        extractor = SimpleFeaturesAndTargetsExtractor(
            classification_targets=['status'],
            classification_exact_values={'status': 'active'}
        )
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'status': ['active', 'inactive', 'active']
        })

        result = extractor.transform(df)
        _, target_by_type, _, _, _, _, _, _ = result

        targets = target_by_type[TargetTypes.BINARY_CLASSIFICATION]
        assert 'status_eq_active' in targets
        expected = np.array([1, 0, 1], dtype=np.int8)
        np.testing.assert_array_equal(targets['status_eq_active'], expected)

    def test_missing_classification_target_raises_error(self):
        """Test that missing classification target column raises KeyError."""
        extractor = SimpleFeaturesAndTargetsExtractor(
            classification_targets=['nonexistent']
        )
        df = pd.DataFrame({'feature1': [1.0, 2.0, 3.0]})

        with pytest.raises(KeyError, match="Classification target column 'nonexistent' not found"):
            extractor.transform(df)


class TestSimpleFeaturesAndTargetsExtractorMixed:
    """Tests for mixed regression and classification targets."""

    def test_both_regression_and_classification(self):
        """Test extraction of both regression and classification targets."""
        extractor = SimpleFeaturesAndTargetsExtractor(
            regression_targets=['reg_target'],
            classification_targets=['cls_target'],
            classification_thresholds={'cls_target': 50}
        )
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'reg_target': [10.0, 20.0, 30.0],
            'cls_target': [40, 60, 80]
        })

        result = extractor.transform(df)
        _, target_by_type, _, _, _, _, cols_to_drop, _ = result

        assert TargetTypes.REGRESSION in target_by_type
        assert TargetTypes.BINARY_CLASSIFICATION in target_by_type
        assert 'reg_target' in target_by_type[TargetTypes.REGRESSION]
        assert 'cls_target_above_50' in target_by_type[TargetTypes.BINARY_CLASSIFICATION]
        assert {'reg_target', 'cls_target'} <= cols_to_drop


class TestSimpleFeaturesAndTargetsExtractorSampleWeights:
    """Tests for sample weights functionality."""

    def test_recency_sample_weights(self):
        """Test recency-based sample weights with ts_field."""
        extractor = SimpleFeaturesAndTargetsExtractor(
            ts_field='timestamp',
            regression_targets=['target']
        )
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'target': [10.0, 20.0, 30.0, 40.0, 50.0],
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='D')
        })

        result = extractor.transform(df)
        _, _, _, _, _, _, _, sample_weights = result

        assert 'recency' in sample_weights
        weights = sample_weights['recency']
        assert len(weights) == 5

    def test_no_sample_weights_without_ts_field(self):
        """Test that no sample weights are generated without ts_field."""
        extractor = SimpleFeaturesAndTargetsExtractor(
            regression_targets=['target']
        )
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'target': [10.0, 20.0, 30.0]
        })

        result = extractor.transform(df)
        _, _, _, _, _, _, _, sample_weights = result

        assert sample_weights == {}


class TestSimpleFeaturesAndTargetsExtractorPolars:
    """Tests for Polars DataFrame support."""

    def test_polars_regression_target(self):
        """Test regression target extraction from Polars DataFrame."""
        extractor = SimpleFeaturesAndTargetsExtractor(
            regression_targets=['target']
        )
        df = pl.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'target': [10.0, 20.0, 30.0]
        })

        result = extractor.transform(df)
        _, target_by_type, _, _, _, _, _, _ = result

        assert TargetTypes.REGRESSION in target_by_type
        assert 'target' in target_by_type[TargetTypes.REGRESSION]

    def test_polars_classification_target(self):
        """Test classification target extraction from Polars DataFrame."""
        extractor = SimpleFeaturesAndTargetsExtractor(
            classification_targets=['target']
        )
        df = pl.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'target': [0, 1, 0]
        })

        result = extractor.transform(df)
        _, target_by_type, _, _, _, _, _, _ = result

        targets = target_by_type[TargetTypes.BINARY_CLASSIFICATION]
        assert targets['target'].dtype == np.int8

    def test_polars_with_ts_field(self):
        """Test Polars DataFrame with timestamp field conversion."""
        extractor = SimpleFeaturesAndTargetsExtractor(
            ts_field='timestamp',
            regression_targets=['target']
        )
        df = pl.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'target': [10.0, 20.0, 30.0],
            'timestamp': pl.date_range(datetime(2023, 1, 1), datetime(2023, 1, 3), eager=True)
        })

        result = extractor.transform(df)
        _, _, _, _, timestamps, _, _, _ = result

        # Timestamps should be converted to pandas Series
        assert isinstance(timestamps, pd.Series)


class TestSimpleFeaturesAndTargetsExtractorEdgeCases:
    """Edge case tests for SimpleFeaturesAndTargetsExtractor."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        extractor = SimpleFeaturesAndTargetsExtractor(
            regression_targets=['target']
        )
        df = pd.DataFrame({'feature1': [], 'target': []})

        result = extractor.transform(df)
        _, target_by_type, _, _, _, _, _, _ = result

        assert TargetTypes.REGRESSION in target_by_type
        assert len(target_by_type[TargetTypes.REGRESSION]['target']) == 0

    def test_no_targets_specified(self):
        """Test when no targets are specified."""
        extractor = SimpleFeaturesAndTargetsExtractor()
        df = pd.DataFrame({'feature1': [1.0, 2.0, 3.0]})

        result = extractor.transform(df)
        _, target_by_type, _, _, _, _, _, _ = result

        # Should have no targets
        assert TargetTypes.REGRESSION not in target_by_type
        assert TargetTypes.BINARY_CLASSIFICATION not in target_by_type

    def test_classification_without_threshold_or_exact(self):
        """Test classification target without threshold or exact value (direct use)."""
        extractor = SimpleFeaturesAndTargetsExtractor(
            classification_targets=['binary_col']
        )
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'binary_col': [0, 1, 1]
        })

        result = extractor.transform(df)
        _, target_by_type, _, _, _, _, _, _ = result

        targets = target_by_type[TargetTypes.BINARY_CLASSIFICATION]
        # Should use column name directly
        assert 'binary_col' in targets
        np.testing.assert_array_equal(targets['binary_col'], [0, 1, 1])

    def test_single_row_dataframe(self):
        """Test with single row DataFrame."""
        extractor = SimpleFeaturesAndTargetsExtractor(
            regression_targets=['target']
        )
        df = pd.DataFrame({'feature1': [1.0], 'target': [10.0]})

        result = extractor.transform(df)
        _, target_by_type, _, _, _, _, _, _ = result

        assert len(target_by_type[TargetTypes.REGRESSION]['target']) == 1

    def test_columns_to_drop_accumulates(self):
        """Test that columns_to_drop accumulates target columns."""
        initial_cols = {'col_to_remove'}
        extractor = SimpleFeaturesAndTargetsExtractor(
            columns_to_drop=initial_cols,
            regression_targets=['target1'],
            classification_targets=['target2']
        )
        df = pd.DataFrame({
            'feature1': [1.0, 2.0],
            'col_to_remove': [1, 2],
            'target1': [10.0, 20.0],
            'target2': [0, 1]
        })

        result = extractor.transform(df)
        _, _, _, _, _, _, cols_to_drop, _ = result

        assert 'col_to_remove' in cols_to_drop
        assert 'target1' in cols_to_drop
        assert 'target2' in cols_to_drop
