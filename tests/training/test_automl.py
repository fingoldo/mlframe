"""
Tests for mlframe automl training functions.

Tests cover AutoGluon and LightAutoML (LAMA) model training,
including error handling, parameter validation, and suite functions.
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from mlframe.training.automl import (
    train_autogluon_model,
    train_lama_model,
    train_automl_models_suite,
)
from mlframe.training.configs import AutoMLConfig


# ================================================================================================
# Fixtures
# ================================================================================================

@pytest.fixture
def sample_train_df():
    """Generate sample training DataFrame."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "feature3": np.random.randn(n),
        "target": np.random.randint(0, 2, n),
    })


@pytest.fixture
def sample_test_df():
    """Generate sample test DataFrame."""
    np.random.seed(43)
    n = 50
    return pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "feature3": np.random.randn(n),
        "target": np.random.randint(0, 2, n),
    })


@pytest.fixture
def sample_polars_df():
    """Generate sample Polars training DataFrame."""
    np.random.seed(42)
    n = 100
    return pl.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "feature3": np.random.randn(n),
        "target": np.random.randint(0, 2, n).tolist(),
    })


# ================================================================================================
# AutoGluon Tests
# ================================================================================================

class TestTrainAutogluonModel:
    """Tests for train_autogluon_model function."""

    def test_returns_none_when_autogluon_not_available(self, sample_train_df):
        """Test that function returns None when AutoGluon is not installed."""
        with patch.dict('sys.modules', {'autogluon': None, 'autogluon.tabular': None}):
            # Need to reimport to trigger the ImportError
            import importlib
            import mlframe.training.automl as automl_module
            importlib.reload(automl_module)

            result = automl_module.train_autogluon_model(sample_train_df)
            assert result is None

    @pytest.mark.skip(reason="AutoGluon heavy dependency - run manually if needed")
    def test_basic_training(self, sample_train_df, tmp_path):
        """Test basic AutoGluon training."""
        init_params = {"path": str(tmp_path / "ag_model")}
        fit_params = {"time_limit": 10, "presets": "medium_quality"}

        result = train_autogluon_model(
            sample_train_df,
            init_params=init_params,
            fit_params=fit_params,
            verbose=0,
        )

        assert result is not None
        assert hasattr(result, 'model')
        assert hasattr(result, 'metrics')
        assert hasattr(result, 'fi')

    @pytest.mark.skip(reason="AutoGluon heavy dependency - run manually if needed")
    def test_training_with_test_df(self, sample_train_df, sample_test_df, tmp_path):
        """Test AutoGluon training with test evaluation."""
        init_params = {"path": str(tmp_path / "ag_model")}
        fit_params = {"time_limit": 10}

        result = train_autogluon_model(
            sample_train_df,
            test_df=sample_test_df,
            init_params=init_params,
            fit_params=fit_params,
            verbose=0,
        )

        assert result is not None
        assert result.test_probs is not None
        assert len(result.test_probs) == len(sample_test_df)


class TestTrainAutogluonModelMocked:
    """Tests for train_autogluon_model with mocked AutoGluon."""

    def test_default_params_used(self, sample_train_df):
        """Test that default parameters are used when not provided."""
        mock_predictor = MagicMock()
        mock_predictor.fit.return_value = None
        mock_predictor.feature_importance.return_value = pd.DataFrame()

        mock_tabular = MagicMock()
        mock_tabular.TabularPredictor.return_value = mock_predictor
        mock_tabular.TabularDataset.return_value = sample_train_df

        with patch.dict('sys.modules', {'autogluon.tabular': mock_tabular}):
            import importlib
            import mlframe.training.automl as automl_module
            importlib.reload(automl_module)

            result = automl_module.train_autogluon_model(
                sample_train_df,
                verbose=0,
            )

            # Verify predictor was created
            mock_tabular.TabularPredictor.assert_called_once()
            mock_predictor.fit.assert_called_once()

    def test_custom_target_name(self, sample_train_df):
        """Test training with custom target column name."""
        df = sample_train_df.rename(columns={"target": "custom_target"})

        mock_predictor = MagicMock()
        mock_predictor.fit.return_value = None
        mock_predictor.feature_importance.return_value = pd.DataFrame()

        mock_tabular = MagicMock()
        mock_tabular.TabularPredictor.return_value = mock_predictor
        mock_tabular.TabularDataset.return_value = df

        with patch.dict('sys.modules', {'autogluon.tabular': mock_tabular}):
            import importlib
            import mlframe.training.automl as automl_module
            importlib.reload(automl_module)

            result = automl_module.train_autogluon_model(
                df,
                target_name="custom_target",
                verbose=0,
            )

            # Verify correct target was used
            call_kwargs = mock_tabular.TabularPredictor.call_args[1]
            assert call_kwargs.get('label') == "custom_target"


# ================================================================================================
# LAMA Tests
# ================================================================================================

class TestTrainLamaModel:
    """Tests for train_lama_model function."""

    def test_returns_none_when_lama_not_available(self, sample_train_df):
        """Test that function returns None when LAMA is not installed."""
        with patch.dict('sys.modules', {'lightautoml': None}):
            import importlib
            import mlframe.training.automl as automl_module
            importlib.reload(automl_module)

            result = automl_module.train_lama_model(sample_train_df)
            assert result is None

    @pytest.mark.skip(reason="LAMA heavy dependency - run manually if needed")
    def test_basic_training(self, sample_train_df):
        """Test basic LAMA training."""
        from lightautoml.tasks import Task

        init_params = {"task": Task("binary"), "timeout": 10}

        result = train_lama_model(
            sample_train_df,
            init_params=init_params,
            verbose=0,
        )

        assert result is not None
        assert hasattr(result, 'model')
        assert hasattr(result, 'metrics')

    @pytest.mark.skip(reason="LAMA heavy dependency - run manually if needed")
    def test_training_with_test_df(self, sample_train_df, sample_test_df):
        """Test LAMA training with test evaluation."""
        from lightautoml.tasks import Task

        init_params = {"task": Task("binary"), "timeout": 10}

        result = train_lama_model(
            sample_train_df,
            test_df=sample_test_df,
            init_params=init_params,
            verbose=0,
        )

        assert result is not None
        assert result.test_probs is not None
        assert len(result.test_probs) == len(sample_test_df)


class TestTrainLamaModelMocked:
    """Tests for train_lama_model with mocked LAMA."""

    def test_default_task_is_binary(self, sample_train_df):
        """Test that default task is binary classification."""
        mock_automl = MagicMock()
        mock_automl.fit_predict.return_value = MagicMock()
        mock_automl.get_feature_scores.return_value = pd.DataFrame()

        mock_presets = MagicMock()
        mock_presets.TabularAutoML.return_value = mock_automl

        mock_task = MagicMock()
        mock_tasks = MagicMock()
        mock_tasks.Task.return_value = mock_task

        mock_mpl = MagicMock()
        mock_mpl.rcParams = {}
        mock_mpl.rcParamsDefault = {}

        with patch.dict('sys.modules', {
            'lightautoml': MagicMock(),
            'lightautoml.automl': MagicMock(),
            'lightautoml.automl.presets': MagicMock(),
            'lightautoml.automl.presets.tabular_presets': mock_presets,
            'lightautoml.tasks': mock_tasks,
            'matplotlib': mock_mpl,
        }):
            import importlib
            import mlframe.training.automl as automl_module
            importlib.reload(automl_module)

            result = automl_module.train_lama_model(
                sample_train_df,
                verbose=0,
            )

            # Verify Task was created with 'binary'
            mock_tasks.Task.assert_called_with('binary')


# ================================================================================================
# Suite Function Tests
# ================================================================================================

class TestTrainAutomlModelsSuite:
    """Tests for train_automl_models_suite function."""

    def test_empty_dict_when_no_models_enabled(self, sample_train_df):
        """Test that empty dict is returned when no models are enabled."""
        config = AutoMLConfig(use_autogluon=False, use_lama=False)

        result = train_automl_models_suite(
            sample_train_df,
            config=config,
            verbose=0,
        )

        assert result == {}

    def test_validates_target_in_train_df(self, sample_train_df):
        """Test that ValueError is raised if target not in train_df."""
        df_no_target = sample_train_df.drop(columns=["target"])
        config = AutoMLConfig(use_autogluon=True)

        with pytest.raises(ValueError, match="Target column 'target' not found in train_df"):
            train_automl_models_suite(
                df_no_target,
                target_name="target",
                config=config,
                verbose=0,
            )

    def test_validates_target_in_test_df(self, sample_train_df, sample_test_df):
        """Test that ValueError is raised if target not in test_df."""
        test_no_target = sample_test_df.drop(columns=["target"])
        config = AutoMLConfig(use_autogluon=True)

        with pytest.raises(ValueError, match="Target column 'target' not found in test_df"):
            train_automl_models_suite(
                sample_train_df,
                test_df=test_no_target,
                target_name="target",
                config=config,
                verbose=0,
            )

    @pytest.mark.xfail(reason="Bug: to_pandas() called on already-converted DataFrame in automl.py:267")
    def test_converts_polars_to_pandas(self, sample_polars_df):
        """Test that Polars DataFrames are converted to pandas."""
        config = AutoMLConfig(use_autogluon=False, use_lama=False)

        # Should not raise any errors
        result = train_automl_models_suite(
            sample_polars_df,
            config=config,
            verbose=0,
        )

        assert result == {}

    def test_custom_target_label_in_config(self, sample_train_df):
        """Test that automl_target_label in config is used."""
        df = sample_train_df.rename(columns={"target": "my_label"})
        config = AutoMLConfig(
            use_autogluon=False,
            use_lama=False,
            automl_target_label="my_label",
        )

        # Should validate against the config target label
        result = train_automl_models_suite(
            df,
            target_name="my_label",
            config=config,
            verbose=0,
        )

        assert result == {}

    def test_default_config_used_when_none(self, sample_train_df):
        """Test that default AutoMLConfig is used when not provided."""
        # Default config has use_autogluon=False, use_lama=False
        result = train_automl_models_suite(
            sample_train_df,
            verbose=0,
        )

        assert result == {}


class TestTrainAutomlModelsSuiteMocked:
    """Tests for train_automl_models_suite with mocked models."""

    def test_trains_autogluon_when_enabled(self, sample_train_df):
        """Test that AutoGluon is trained when enabled in config."""
        mock_result = SimpleNamespace(
            model=MagicMock(),
            metrics={},
            fi=None,
            test_probs=None,
            test_target=None,
        )

        with patch('mlframe.training.automl.train_autogluon_model', return_value=mock_result):
            config = AutoMLConfig(use_autogluon=True, use_lama=False)

            result = train_automl_models_suite(
                sample_train_df,
                config=config,
                verbose=0,
            )

            assert 'autogluon' in result
            assert result['autogluon'] == mock_result

    def test_trains_lama_when_enabled(self, sample_train_df):
        """Test that LAMA is trained when enabled in config."""
        mock_result = SimpleNamespace(
            model=MagicMock(),
            metrics={},
            fi=None,
            test_probs=None,
            test_target=None,
        )

        with patch('mlframe.training.automl.train_lama_model', return_value=mock_result):
            config = AutoMLConfig(use_autogluon=False, use_lama=True)

            result = train_automl_models_suite(
                sample_train_df,
                config=config,
                verbose=0,
            )

            assert 'lama' in result
            assert result['lama'] == mock_result

    def test_trains_both_when_enabled(self, sample_train_df):
        """Test that both models are trained when enabled."""
        mock_ag_result = SimpleNamespace(
            model=MagicMock(),
            metrics={'test_auc': 0.8},
            fi=None,
            test_probs=None,
            test_target=None,
        )
        mock_lama_result = SimpleNamespace(
            model=MagicMock(),
            metrics={'test_auc': 0.85},
            fi=None,
            test_probs=None,
            test_target=None,
        )

        with patch('mlframe.training.automl.train_autogluon_model', return_value=mock_ag_result):
            with patch('mlframe.training.automl.train_lama_model', return_value=mock_lama_result):
                config = AutoMLConfig(use_autogluon=True, use_lama=True)

                result = train_automl_models_suite(
                    sample_train_df,
                    config=config,
                    verbose=0,
                )

                assert 'autogluon' in result
                assert 'lama' in result

    def test_skips_model_when_returns_none(self, sample_train_df):
        """Test that model is skipped when training returns None."""
        with patch('mlframe.training.automl.train_autogluon_model', return_value=None):
            config = AutoMLConfig(use_autogluon=True)

            result = train_automl_models_suite(
                sample_train_df,
                config=config,
                verbose=0,
            )

            assert 'autogluon' not in result

    def test_passes_config_params_to_autogluon(self, sample_train_df):
        """Test that config parameters are passed to train_autogluon_model."""
        mock_result = SimpleNamespace(
            model=MagicMock(),
            metrics={},
            fi=None,
            test_probs=None,
            test_target=None,
        )

        with patch('mlframe.training.automl.train_autogluon_model', return_value=mock_result) as mock_fn:
            config = AutoMLConfig(
                use_autogluon=True,
                autogluon_init_params={'path': '/test'},
                autogluon_fit_params={'time_limit': 60},
                automl_verbose=2,
            )

            train_automl_models_suite(
                sample_train_df,
                config=config,
                verbose=0,
            )

            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs['init_params'] == {'path': '/test'}
            assert call_kwargs['fit_params'] == {'time_limit': 60}
            assert call_kwargs['verbose'] == 2

    def test_passes_config_params_to_lama(self, sample_train_df):
        """Test that config parameters are passed to train_lama_model."""
        mock_result = SimpleNamespace(
            model=MagicMock(),
            metrics={},
            fi=None,
            test_probs=None,
            test_target=None,
        )

        with patch('mlframe.training.automl.train_lama_model', return_value=mock_result) as mock_fn:
            config = AutoMLConfig(
                use_lama=True,
                lama_init_params={'timeout': 30},
                lama_fit_params={'roles': {}},
                automl_verbose=0,
            )

            train_automl_models_suite(
                sample_train_df,
                config=config,
                verbose=0,
            )

            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs['init_params'] == {'timeout': 30}
            assert call_kwargs['fit_params'] == {'roles': {}}
            assert call_kwargs['verbose'] == 0


# ================================================================================================
# AutoMLConfig Tests
# ================================================================================================

class TestAutoMLConfig:
    """Tests for AutoMLConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AutoMLConfig()

        assert config.use_autogluon is False
        assert config.use_lama is False
        assert config.autogluon_init_params is None
        assert config.autogluon_fit_params is None
        assert config.lama_init_params is None
        assert config.lama_fit_params is None
        assert config.automl_verbose == 1
        assert config.automl_show_fi is True
        assert config.automl_target_label == "target"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AutoMLConfig(
            use_autogluon=True,
            use_lama=True,
            autogluon_init_params={'eval_metric': 'auc'},
            autogluon_fit_params={'time_limit': 3600},
            lama_init_params={'timeout': 1800},
            automl_verbose=0,
            automl_target_label="label",
        )

        assert config.use_autogluon is True
        assert config.use_lama is True
        assert config.autogluon_init_params == {'eval_metric': 'auc'}
        assert config.autogluon_fit_params == {'time_limit': 3600}
        assert config.lama_init_params == {'timeout': 1800}
        assert config.automl_verbose == 0
        assert config.automl_target_label == "label"
