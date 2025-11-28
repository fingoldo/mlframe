"""Tests for helper functions in training/trainer.py."""

import pytest
import numpy as np
import pandas as pd

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

# Import from the new training module locations
from mlframe.training.trainer import (
    _extract_target_subset,
    _subset_dataframe,
    _prepare_df_for_model,
    _setup_sample_weight,
)
from mlframe.config import TABNET_MODEL_TYPES


class TestExtractTargetSubset:
    """Tests for _extract_target_subset helper."""
    
    def test_pandas_series(self):
        """Test extraction from pandas Series."""
        target = pd.Series([10, 20, 30, 40, 50], index=[0, 1, 2, 3, 4])
        idx = np.array([0, 2, 4])
        result = _extract_target_subset(target, idx)
        
        assert isinstance(result, pd.Series)
        assert list(result.values) == [10, 30, 50]
    
    def test_numpy_array(self):
        """Test extraction from numpy array."""
        target = np.array([10, 20, 30, 40, 50])
        idx = np.array([1, 3])
        result = _extract_target_subset(target, idx)
        
        assert isinstance(result, np.ndarray)
        assert list(result) == [20, 40]
    
    def test_none_idx_returns_target(self):
        """Test that None idx returns original target."""
        target = pd.Series([1, 2, 3])
        result = _extract_target_subset(target, None)
        
        assert result is target
    
    @pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
    def test_polars_series(self):
        """Test extraction from polars Series."""
        target = pl.Series([10, 20, 30, 40, 50])
        idx = np.array([0, 2, 4])
        result = _extract_target_subset(target, idx)
        
        assert isinstance(result, pl.Series)
        assert result.to_list() == [10, 30, 50]


class TestSubsetDataframe:
    """Tests for _subset_dataframe helper."""
    
    def test_pandas_basic(self):
        """Test basic subsetting of pandas DataFrame."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": [100, 200, 300, 400, 500],
        })
        idx = np.array([0, 2, 4])
        result = _subset_dataframe(df, idx)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result["a"].values) == [1, 3, 5]
    
    def test_pandas_with_drop_columns(self):
        """Test subsetting with column dropping."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [10, 20, 30],
            "c": [100, 200, 300],
        })
        idx = np.array([0, 2])
        result = _subset_dataframe(df, idx, drop_columns=["b"])
        
        assert list(result.columns) == ["a", "c"]
        assert len(result) == 2
    
    def test_none_df_returns_none(self):
        """Test that None df returns None."""
        result = _subset_dataframe(None, np.array([0, 1]))
        assert result is None
    
    def test_none_idx_returns_df(self):
        """Test that None idx returns original df."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = _subset_dataframe(df, None)
        assert result is df
    
    @pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
    def test_polars_basic(self):
        """Test basic subsetting of polars DataFrame."""
        df = pl.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
        })
        idx = np.array([1, 3])
        result = _subset_dataframe(df, idx)
        
        assert isinstance(result, pl.DataFrame)
        assert result["a"].to_list() == [2, 4]


class TestPrepareDfForModel:
    """Tests for _prepare_df_for_model helper."""
    
    def test_non_tabnet_returns_df(self):
        """Test that non-TabNet models return DataFrame unchanged."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = _prepare_df_for_model(df, "CatBoostClassifier")
        
        assert result is df
        assert hasattr(result, "columns")
    
    def test_tabnet_converts_to_numpy(self):
        """Test that TabNet models convert DataFrame to numpy."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = _prepare_df_for_model(df, "TabNetClassifier")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)
    
    def test_none_df_returns_none(self):
        """Test that None df returns None."""
        result = _prepare_df_for_model(None, "TabNetClassifier")
        assert result is None
    
    def test_all_tabnet_types(self):
        """Test conversion for all TabNet model types."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        
        for model_type in TABNET_MODEL_TYPES:
            result = _prepare_df_for_model(df, model_type)
            assert isinstance(result, np.ndarray), f"Failed for {model_type}"


class TestSetupSampleWeight:
    """Tests for _setup_sample_weight helper."""
    
    def test_none_weight_no_change(self):
        """Test that None sample_weight does not modify fit_params."""
        fit_params = {}
        
        class MockModel:
            def fit(self, X, y, sample_weight=None):
                pass
        
        _setup_sample_weight(None, None, MockModel(), fit_params)
        assert "sample_weight" not in fit_params
    
    def test_unsupported_model_no_change(self):
        """Test that models without sample_weight support are skipped."""
        fit_params = {}
        
        class MockModel:
            def fit(self, X, y):  # No sample_weight param
                pass
        
        _setup_sample_weight(np.array([1, 2, 3]), None, MockModel(), fit_params)
        assert "sample_weight" not in fit_params
    
    def test_numpy_weight_with_idx(self):
        """Test numpy weight with index subsetting."""
        fit_params = {}
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        train_idx = np.array([0, 2, 4])
        
        class MockModel:
            def fit(self, X, y, sample_weight=None):
                pass
        
        _setup_sample_weight(weights, train_idx, MockModel(), fit_params)
        assert "sample_weight" in fit_params
        np.testing.assert_array_equal(fit_params["sample_weight"], [1.0, 3.0, 5.0])
    
    def test_pandas_weight_with_idx(self):
        """Test pandas Series weight with index subsetting."""
        fit_params = {}
        weights = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        train_idx = np.array([1, 3])
        
        class MockModel:
            def fit(self, X, y, sample_weight=None):
                pass
        
        _setup_sample_weight(weights, train_idx, MockModel(), fit_params)
        assert "sample_weight" in fit_params
        np.testing.assert_array_equal(fit_params["sample_weight"], [2.0, 4.0])
