"""
Tests for training configuration classes.
"""

import pytest
from pydantic import ValidationError

from mlframe.training.configs import (
    PreprocessingConfig,
    TrainingSplitConfig,
    LinearModelConfig,
    AutoMLConfig,
    TrainingConfig,
    config_from_dict,
)


class TestPreprocessingConfig:
    """Test PreprocessingConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PreprocessingConfig()
        assert config.fillna_value is None
        assert config.fix_infinities is True
        assert config.ensure_float32_dtypes is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PreprocessingConfig(
            fillna_value=0.0,
            fix_infinities=False,
            drop_columns=['col1', 'col2']
        )
        assert config.fillna_value == 0.0
        assert config.fix_infinities is False
        assert config.drop_columns == ['col1', 'col2']

    def test_from_dict(self):
        """Test creation from dictionary."""
        params = {'fillna_value': -999.0, 'fix_infinities': True}
        config = config_from_dict(PreprocessingConfig, params)
        assert config.fillna_value == -999.0
        assert config.fix_infinities is True


class TestTrainingSplitConfig:
    """Test TrainingSplitConfig class."""

    def test_default_values(self):
        """Test default split configuration."""
        config = TrainingSplitConfig()
        assert config.test_size == 0.1
        assert config.val_size == 0.1
        assert config.shuffle_val is False
        assert config.random_seed == 42

    def test_validation_size_bounds(self):
        """Test validation of size parameters."""
        # Valid values
        config = TrainingSplitConfig(test_size=0.2, val_size=0.15)
        assert config.test_size == 0.2
        assert config.val_size == 0.15

        # Invalid values should raise validation error
        with pytest.raises(ValidationError):
            TrainingSplitConfig(test_size=1.5)  # > 1.0

        with pytest.raises(ValidationError):
            TrainingSplitConfig(val_size=-0.1)  # < 0.0

    def test_sequential_fraction(self):
        """Test sequential fraction parameters."""
        config = TrainingSplitConfig(
            val_sequential_fraction=0.7,
            test_sequential_fraction=0.8
        )
        assert config.val_sequential_fraction == 0.7
        assert config.test_sequential_fraction == 0.8


class TestLinearModelConfig:
    """Test LinearModelConfig class."""

    def test_all_model_types(self):
        """Test configuration for all linear model types."""
        model_types = ['linear', 'ridge', 'lasso', 'elasticnet', 'huber', 'ransac', 'sgd']

        for model_type in model_types:
            config = LinearModelConfig(model_type=model_type)
            assert config.model_type == model_type

    def test_regularization_params(self):
        """Test regularization parameter configuration."""
        config = LinearModelConfig(
            model_type='ridge',
            alpha=0.5,
            l1_ratio=0.3
        )
        assert config.alpha == 0.5
        assert config.l1_ratio == 0.3

    def test_sgd_params(self):
        """Test SGD-specific parameters."""
        config = LinearModelConfig(
            model_type='sgd',
            loss='squared_error',
            penalty='l2',
            learning_rate='invscaling',
            max_iter=2000
        )
        assert config.loss == 'squared_error'
        assert config.penalty == 'l2'
        assert config.learning_rate == 'invscaling'
        assert config.max_iter == 2000


class TestAutoMLConfig:
    """Test AutoMLConfig class."""

    def test_default_values(self):
        """Test default AutoML configuration."""
        config = AutoMLConfig()
        assert config.use_autogluon is False
        assert config.use_lama is False
        assert config.automl_target_label == 'target'

    def test_autogluon_config(self):
        """Test AutoGluon configuration."""
        config = AutoMLConfig(
            use_autogluon=True,
            autogluon_init_params={'eval_metric': 'log_loss'},
            autogluon_fit_params={'time_limit': 3600}
        )
        assert config.use_autogluon is True
        assert config.autogluon_init_params['eval_metric'] == 'log_loss'
        assert config.autogluon_fit_params['time_limit'] == 3600

    def test_lama_config(self):
        """Test LAMA configuration."""
        config = AutoMLConfig(
            use_lama=True,
            lama_init_params={'timeout': 3600},
            lama_fit_params={'verbose': 2}
        )
        assert config.use_lama is True
        assert config.lama_init_params['timeout'] == 3600
        assert config.lama_fit_params['verbose'] == 2


class TestTrainingConfig:
    """Test TrainingConfig class."""

    def test_aggregated_config(self):
        """Test aggregated training configuration."""
        config = TrainingConfig(
            target_name='my_target',
            model_name='experiment_1',
            mlframe_models=['linear', 'ridge', 'cb'],
            preprocessing=PreprocessingConfig(fillna_value=0.0),
            split=TrainingSplitConfig(test_size=0.15),
        )

        assert config.target_name == 'my_target'
        assert config.model_name == 'experiment_1'
        assert 'linear' in config.mlframe_models
        assert config.preprocessing.fillna_value == 0.0
        assert config.split.test_size == 0.15

    def test_default_sub_configs(self):
        """Test default sub-configurations."""
        config = TrainingConfig(
            target_name='target',
            model_name='test',
        )

        assert isinstance(config.preprocessing, PreprocessingConfig)
        assert isinstance(config.split, TrainingSplitConfig)
        assert config.mlframe_models == ["cb", "lgb", "xgb", "mlp"]
