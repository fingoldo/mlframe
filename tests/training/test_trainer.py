"""
Tests for training/trainer.py module.

Covers:
- train_and_evaluate_model main function
- Helper functions for training
- Config building functions
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from mlframe.training.configs import (
    DataConfig,
    TrainingControlConfig,
    MetricsConfig,
    DisplayConfig,
    NamingConfig,
    PredictionsContainer,
)


def call_train_and_evaluate(
    model,
    df=None,
    target=None,
    train_df=None,
    val_df=None,
    test_df=None,
    train_target=None,
    val_target=None,
    test_target=None,
    train_idx=None,
    val_idx=None,
    test_idx=None,
    sample_weight=None,
    test_preds=None,
    compute_trainset_metrics=False,
    compute_valset_metrics=True,
    compute_testset_metrics=True,
    just_evaluate=False,
    pre_pipeline=None,
    print_report=True,
    show_perf_chart=True,
    show_fi=True,
    verbose=False,
    **kwargs,
):
    """Helper to call train_and_evaluate_model with config objects."""
    from mlframe.training.trainer import train_and_evaluate_model

    data = DataConfig(
        df=df,
        target=target,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_target=train_target,
        val_target=val_target,
        test_target=test_target,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        sample_weight=sample_weight,
    )
    control = TrainingControlConfig(
        verbose=verbose,
        compute_trainset_metrics=compute_trainset_metrics,
        compute_valset_metrics=compute_valset_metrics,
        compute_testset_metrics=compute_testset_metrics,
        just_evaluate=just_evaluate,
        pre_pipeline=pre_pipeline,
    )
    metrics = MetricsConfig()
    display = DisplayConfig(
        print_report=print_report,
        show_perf_chart=show_perf_chart,
        show_fi=show_fi,
    )
    naming = NamingConfig()
    predictions = PredictionsContainer(test_preds=test_preds)

    return train_and_evaluate_model(
        model=model,
        data=data,
        control=control,
        metrics=metrics,
        display=display,
        naming=naming,
        predictions=predictions,
    )


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestExtractTargetSubset:
    """Tests for _extract_target_subset helper function."""

    def test_pandas_series(self):
        """Test extraction from pandas Series."""
        from mlframe.training.trainer import _extract_target_subset

        target = pd.Series([1, 2, 3, 4, 5])
        idx = np.array([0, 2, 4])

        result = _extract_target_subset(target, idx)

        assert isinstance(result, pd.Series)
        assert list(result.values) == [1, 3, 5]

    def test_numpy_array(self):
        """Test extraction from numpy array."""
        from mlframe.training.trainer import _extract_target_subset

        target = np.array([1, 2, 3, 4, 5])
        idx = np.array([0, 2, 4])

        result = _extract_target_subset(target, idx)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 3, 5])

    def test_none_idx_returns_target(self):
        """Test that None idx returns full target."""
        from mlframe.training.trainer import _extract_target_subset

        target = np.array([1, 2, 3])

        result = _extract_target_subset(target, None)

        np.testing.assert_array_equal(result, target)

    def test_polars_series(self):
        """Test extraction from Polars Series."""
        from mlframe.training.trainer import _extract_target_subset

        target = pl.Series([1, 2, 3, 4, 5])
        idx = np.array([0, 2, 4])

        result = _extract_target_subset(target, idx)

        assert isinstance(result, pl.Series)
        assert result.to_list() == [1, 3, 5]


class TestSubsetDataframe:
    """Tests for _subset_dataframe helper function."""

    def test_pandas_basic(self):
        """Test basic pandas DataFrame subsetting."""
        from mlframe.training.trainer import _subset_dataframe

        df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40]})
        idx = np.array([0, 2])

        result = _subset_dataframe(df, idx)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result['a']) == [1, 3]

    def test_pandas_with_drop_columns(self):
        """Test subsetting with column dropping."""
        from mlframe.training.trainer import _subset_dataframe

        df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30], 'c': [100, 200, 300]})
        idx = np.array([0, 1])

        result = _subset_dataframe(df, idx, drop_columns=['c'])

        assert 'c' not in result.columns
        assert 'a' in result.columns
        assert 'b' in result.columns

    def test_none_df_returns_none(self):
        """Test that None DataFrame returns None."""
        from mlframe.training.trainer import _subset_dataframe

        result = _subset_dataframe(None, np.array([0, 1]))

        assert result is None

    def test_none_idx_returns_full_df(self):
        """Test that None idx returns full DataFrame."""
        from mlframe.training.trainer import _subset_dataframe

        df = pd.DataFrame({'a': [1, 2, 3]})

        result = _subset_dataframe(df, None)

        pd.testing.assert_frame_equal(result, df)

    def test_polars_basic(self):
        """Test Polars DataFrame subsetting."""
        from mlframe.training.trainer import _subset_dataframe

        df = pl.DataFrame({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40]})
        idx = np.array([0, 2])

        result = _subset_dataframe(df, idx)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2

    def test_drop_columns_as_string_warning(self):
        """Test that string drop_columns logs warning and converts."""
        from mlframe.training.trainer import _subset_dataframe

        df = pd.DataFrame({'a': [1, 2], 'b': [10, 20]})

        # Should convert string to list and work
        result = _subset_dataframe(df, None, drop_columns='a')

        assert 'a' not in result.columns


class TestPrepareDfForModel:
    """Tests for _prepare_df_for_model helper function."""

    def test_non_tabnet_returns_df(self):
        """Test non-TabNet model returns DataFrame unchanged."""
        from mlframe.training.trainer import _prepare_df_for_model

        df = pd.DataFrame({'a': [1, 2, 3]})

        result = _prepare_df_for_model(df, 'CatBoostRegressor')

        pd.testing.assert_frame_equal(result, df)

    def test_none_df_returns_none(self):
        """Test None DataFrame returns None."""
        from mlframe.training.trainer import _prepare_df_for_model

        result = _prepare_df_for_model(None, 'TabNetClassifier')

        assert result is None


class TestSetupSampleWeight:
    """Tests for _setup_sample_weight helper function."""

    def test_adds_sample_weight_to_fit_params(self):
        """Test sample weight is added to fit_params."""
        from mlframe.training.trainer import _setup_sample_weight
        from sklearn.linear_model import Ridge

        sample_weight = np.array([1.0, 2.0, 3.0, 4.0])
        train_idx = np.array([0, 1])

        # Use a real sklearn model that supports sample_weight
        model_obj = Ridge()

        fit_params = {}
        _setup_sample_weight(sample_weight, train_idx, model_obj, fit_params)

        assert 'sample_weight' in fit_params
        np.testing.assert_array_equal(fit_params['sample_weight'], [1.0, 2.0])

    def test_none_sample_weight_does_nothing(self):
        """Test None sample weight doesn't modify fit_params."""
        from mlframe.training.trainer import _setup_sample_weight

        model_obj = MagicMock()
        fit_params = {}

        _setup_sample_weight(None, np.array([0, 1]), model_obj, fit_params)

        assert 'sample_weight' not in fit_params


class TestInitializeMutableDefaults:
    """Tests for _initialize_mutable_defaults helper function."""

    def test_none_values_become_empty(self):
        """Test None values are converted to empty containers."""
        from mlframe.training.trainer import _initialize_mutable_defaults

        drop_columns, default_drop_columns, fi_kwargs, confidence_kwargs = \
            _initialize_mutable_defaults(None, None, None, None)

        assert drop_columns == []
        assert default_drop_columns == []
        assert fi_kwargs == {}
        assert confidence_kwargs == {}

    def test_existing_values_preserved(self):
        """Test existing values are preserved."""
        from mlframe.training.trainer import _initialize_mutable_defaults

        drop_columns, default_drop_columns, fi_kwargs, confidence_kwargs = \
            _initialize_mutable_defaults(['a'], ['b'], {'k': 'v'}, {'c': 'd'})

        assert drop_columns == ['a']
        assert default_drop_columns == ['b']
        assert fi_kwargs == {'k': 'v'}
        assert confidence_kwargs == {'c': 'd'}


# =============================================================================
# Config Building Tests
# =============================================================================


class TestBuildConfigsFromParams:
    """Tests for _build_configs_from_params function."""

    def test_returns_all_config_objects(self):
        """Test that all config objects are returned."""
        from mlframe.training.trainer import _build_configs_from_params

        result = _build_configs_from_params()

        assert len(result) == 7
        data_config, control_config, metrics_config, display_config, naming_config, confidence_config, predictions = result

        # Check types (these are dataclass/Pydantic models or similar)
        assert hasattr(data_config, 'df')
        assert hasattr(control_config, 'verbose')
        assert hasattr(metrics_config, 'nbins')
        assert hasattr(display_config, 'figsize')
        assert hasattr(naming_config, 'model_name')
        assert hasattr(confidence_config, 'include')
        assert hasattr(predictions, 'train_preds')

    def test_merges_drop_columns(self):
        """Test drop_columns and default_drop_columns are merged."""
        from mlframe.training.trainer import _build_configs_from_params

        data_config, *_ = _build_configs_from_params(
            drop_columns=['col1', 'col2'],
            default_drop_columns=['col3']
        )

        assert 'col1' in data_config.drop_columns
        assert 'col3' in data_config.drop_columns


# =============================================================================
# Train and Evaluate Model Tests (Integration)
# =============================================================================


class TestTrainAndEvaluateModelBasic:
    """Basic tests for train_and_evaluate_model function."""

    @pytest.fixture
    def simple_data(self):
        """Create simple training data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y = 2 * X[:, 0] + np.random.randn(n_samples) * 0.5

        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(n_features)])

        train_idx = np.arange(80)
        test_idx = np.arange(80, 100)

        return df, y, train_idx, test_idx

    def test_basic_regression_training(self, simple_data):
        """Test basic regression model training."""
        df, y, train_idx, test_idx = simple_data
        target = pd.Series(y)
        model = Ridge(alpha=1.0)

        result, train_df, val_df, test_df = call_train_and_evaluate(
            model=model,
            df=df,
            target=target,
            train_idx=train_idx,
            test_idx=test_idx,
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
            verbose=False,
        )

        assert result is not None
        assert hasattr(result, 'model')
        assert hasattr(result, 'test_preds')
        assert hasattr(result, 'metrics')
        assert result.test_preds is not None
        assert len(result.test_preds) == len(test_idx)

    def test_just_evaluate_mode(self, simple_data):
        """Test just_evaluate mode doesn't train."""
        df, y, train_idx, test_idx = simple_data
        target = pd.Series(y)

        # Pre-train model
        model = Ridge(alpha=1.0)
        train_df_subset = df.iloc[train_idx]
        train_target = y[train_idx]
        model.fit(train_df_subset, train_target)

        # Use just_evaluate
        result, *_ = call_train_and_evaluate(
            model=model,
            df=df,
            target=target,
            train_idx=train_idx,
            test_idx=test_idx,
            just_evaluate=True,
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
            verbose=False,
        )

        assert result is not None
        assert result.test_preds is not None

    def test_returns_simplenamespace(self, simple_data):
        """Test that result is SimpleNamespace with expected attributes."""
        df, y, train_idx, test_idx = simple_data
        target = pd.Series(y)
        model = Ridge(alpha=1.0)

        result, *_ = call_train_and_evaluate(
            model=model,
            df=df,
            target=target,
            train_idx=train_idx,
            test_idx=test_idx,
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
            verbose=False,
        )

        assert isinstance(result, SimpleNamespace)
        expected_attrs = ['model', 'test_preds', 'test_probs', 'val_preds', 'val_probs',
                          'train_preds', 'train_probs', 'metrics', 'columns', 'pre_pipeline']
        for attr in expected_attrs:
            assert hasattr(result, attr), f"Missing attribute: {attr}"


class TestTrainAndEvaluateModelMetrics:
    """Tests for metrics computation in train_and_evaluate_model."""

    @pytest.fixture
    def classification_data(self):
        """Create classification data."""
        np.random.seed(42)
        n_samples = 200

        X = np.random.randn(n_samples, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])

        train_idx = np.arange(150)
        val_idx = np.arange(150, 175)
        test_idx = np.arange(175, 200)

        return df, y, train_idx, val_idx, test_idx

    def test_metrics_dict_populated(self, classification_data):
        """Test that metrics dict is populated."""
        df, y, train_idx, val_idx, test_idx = classification_data
        target = pd.Series(y)
        model = LogisticRegression(max_iter=1000)

        result, *_ = call_train_and_evaluate(
            model=model,
            df=df,
            target=target,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            compute_trainset_metrics=True,
            compute_valset_metrics=True,
            compute_testset_metrics=True,
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
            verbose=False,
        )

        assert 'train' in result.metrics
        assert 'val' in result.metrics
        assert 'test' in result.metrics

    def test_can_disable_trainset_metrics(self, classification_data):
        """Test that trainset metrics computation can be disabled."""
        df, y, train_idx, val_idx, test_idx = classification_data
        target = pd.Series(y)
        model = LogisticRegression(max_iter=1000)

        result, *_ = call_train_and_evaluate(
            model=model,
            df=df,
            target=target,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            compute_trainset_metrics=False,
            compute_valset_metrics=False,
            compute_testset_metrics=True,
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
            verbose=False,
        )

        # Train and val should be empty, test should have metrics
        assert result.train_preds is None or len(result.metrics.get('train', {})) == 0


class TestTrainAndEvaluateModelPrePipeline:
    """Tests for pre_pipeline functionality."""

    def test_with_pre_pipeline(self):
        """Test training with pre_pipeline transformation."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = 2 * X[:, 0] + np.random.randn(100) * 0.5

        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
        target = pd.Series(y)

        pre_pipeline = Pipeline([('scaler', StandardScaler())])
        model = Ridge(alpha=1.0)

        train_idx = np.arange(80)
        test_idx = np.arange(80, 100)

        result, *_ = call_train_and_evaluate(
            model=model,
            df=df,
            target=target,
            train_idx=train_idx,
            test_idx=test_idx,
            pre_pipeline=pre_pipeline,
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
            verbose=False,
        )

        assert result is not None
        assert result.pre_pipeline is pre_pipeline


class TestTrainAndEvaluateModelEdgeCases:
    """Edge case tests for train_and_evaluate_model."""

    def test_model_none_just_evaluate(self):
        """Test with model=None in just_evaluate mode (uses pre-computed preds)."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0] + np.random.randn(100) * 0.5

        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
        target = pd.Series(y)
        test_idx = np.arange(80, 100)

        # Pre-computed predictions
        test_preds_arr = y[test_idx] + np.random.randn(20) * 0.1

        result, *_ = call_train_and_evaluate(
            model=None,
            df=df,
            target=target,
            test_idx=test_idx,
            test_preds=test_preds_arr,
            just_evaluate=True,
            compute_trainset_metrics=False,
            compute_valset_metrics=False,
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
            verbose=False,
        )

        assert result is not None
        np.testing.assert_array_equal(result.test_preds, test_preds_arr)

    def test_with_separate_train_val_test_dfs(self):
        """Test with separate train/val/test DataFrames."""
        np.random.seed(42)

        # Create separate DataFrames
        train_df = pd.DataFrame(np.random.randn(80, 5), columns=[f'f{i}' for i in range(5)])
        val_df = pd.DataFrame(np.random.randn(10, 5), columns=[f'f{i}' for i in range(5)])
        test_df = pd.DataFrame(np.random.randn(10, 5), columns=[f'f{i}' for i in range(5)])

        train_target = pd.Series(2 * train_df['f0'] + np.random.randn(80) * 0.5)
        val_target = pd.Series(2 * val_df['f0'] + np.random.randn(10) * 0.5)
        test_target = pd.Series(2 * test_df['f0'] + np.random.randn(10) * 0.5)

        model = Ridge(alpha=1.0)

        result, *_ = call_train_and_evaluate(
            model=model,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_target=train_target,
            val_target=val_target,
            test_target=test_target,
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
            verbose=False,
        )

        assert result is not None
        assert result.test_preds is not None
        assert len(result.test_preds) == 10


# =============================================================================
# Early Stopping Callback Tests
# =============================================================================


class TestSetupEarlyStoppingCallback:
    """Tests for _setup_early_stopping_callback function.

    These tests verify that:
    1. XGBoost callbacks are replaced (not accumulated) between training runs
    2. User-provided callbacks are preserved
    3. LightGBM/CatBoost get fresh callback lists each call
    """

    def test_xgboost_callback_replaced_not_accumulated(self):
        """Test that XGBoost early stopping callback is replaced between iterations.

        When training the same XGBoost model with multiple weight schemas,
        old XGBoostCallback instances should be removed to avoid stale time budgets.
        """
        from xgboost import XGBClassifier
        from mlframe.training.trainer import _setup_early_stopping_callback
        from mlframe.training.helpers import XGBoostCallback

        model_obj = XGBClassifier(n_estimators=10, verbosity=0)
        callback_params = {"time_budget_mins": 60, "patience": 10}
        fit_params = {}

        # Simulate first weight schema - set up callback
        _setup_early_stopping_callback("xgb", fit_params, callback_params, model_obj)

        callbacks_after_first = model_obj.get_params().get("callbacks", [])
        assert len(callbacks_after_first) == 1
        assert isinstance(callbacks_after_first[0], XGBoostCallback)
        first_callback = callbacks_after_first[0]

        # Simulate second weight schema - set up callback again
        _setup_early_stopping_callback("xgb", fit_params, callback_params, model_obj)

        callbacks_after_second = model_obj.get_params().get("callbacks", [])
        # Should still have only 1 callback (old one replaced, not accumulated)
        assert len(callbacks_after_second) == 1
        assert isinstance(callbacks_after_second[0], XGBoostCallback)
        # Should be a different (new) callback instance
        assert callbacks_after_second[0] is not first_callback

    def test_xgboost_user_callbacks_preserved(self):
        """Test that user-provided XGBoost callbacks are preserved.

        When setting up early stopping, any user-provided callbacks that are
        NOT XGBoostCallback instances should be preserved.
        """
        from xgboost import XGBClassifier
        from xgboost.callback import TrainingCallback
        from mlframe.training.trainer import _setup_early_stopping_callback
        from mlframe.training.helpers import XGBoostCallback

        # Create a custom user callback
        class CustomUserCallback(TrainingCallback):
            """Custom callback for testing."""
            def after_iteration(self, model, epoch, evals_log):
                return False  # Continue training

        user_callback = CustomUserCallback()
        model_obj = XGBClassifier(n_estimators=10, verbosity=0, callbacks=[user_callback])
        callback_params = {"time_budget_mins": 60, "patience": 10}
        fit_params = {}

        # Set up early stopping callback
        _setup_early_stopping_callback("xgb", fit_params, callback_params, model_obj)

        callbacks = model_obj.get_params().get("callbacks", [])

        # Should have 2 callbacks: user callback + XGBoostCallback
        assert len(callbacks) == 2

        # User callback should be preserved
        assert user_callback in callbacks

        # XGBoostCallback should be added
        xgb_callbacks = [cb for cb in callbacks if isinstance(cb, XGBoostCallback)]
        assert len(xgb_callbacks) == 1

    def test_lgb_catboost_fresh_callback_list(self):
        """Test that LightGBM/CatBoost get fresh callback lists each call.

        Unlike XGBoost, LightGBM and CatBoost use fit_params["callbacks"] which
        should be created fresh each time (not accumulated).
        """
        from mlframe.training.trainer import _setup_early_stopping_callback
        from mlframe.training.helpers import LightGBMCallback, CatBoostCallback

        callback_params = {"time_budget_mins": 60, "patience": 10}

        # Test LightGBM
        fit_params_lgb = {}
        _setup_early_stopping_callback("lgb", fit_params_lgb, callback_params, None)
        assert len(fit_params_lgb["callbacks"]) == 1
        assert isinstance(fit_params_lgb["callbacks"][0], LightGBMCallback)
        first_lgb_callback = fit_params_lgb["callbacks"][0]

        # Call again - should create fresh list with new callback
        fit_params_lgb2 = {}
        _setup_early_stopping_callback("lgb", fit_params_lgb2, callback_params, None)
        assert len(fit_params_lgb2["callbacks"]) == 1
        assert fit_params_lgb2["callbacks"][0] is not first_lgb_callback

        # Test CatBoost
        fit_params_cb = {}
        _setup_early_stopping_callback("cb", fit_params_cb, callback_params, None)
        assert len(fit_params_cb["callbacks"]) == 1
        assert isinstance(fit_params_cb["callbacks"][0], CatBoostCallback)
        first_cb_callback = fit_params_cb["callbacks"][0]

        # Call again - should create fresh list with new callback
        fit_params_cb2 = {}
        _setup_early_stopping_callback("cb", fit_params_cb2, callback_params, None)
        assert len(fit_params_cb2["callbacks"]) == 1
        assert fit_params_cb2["callbacks"][0] is not first_cb_callback
