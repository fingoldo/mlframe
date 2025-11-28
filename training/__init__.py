"""
mlframe.training - Modular training pipeline for machine learning models.

This package provides a clean, RAM-efficient training pipeline with support for:
- Linear models (linear, ridge, lasso, elasticnet, huber, ransac, sgd)
- Tree-based models (CatBoost, LightGBM, XGBoost, HistGradientBoosting)
- Neural networks (MLP with PyTorch Lightning)
- AutoML (AutoGluon, LightAutoML)

## Main Functions

- `train_mlframe_models_suite`: Train regular ML models with full pipeline
- `train_automl_models_suite`: Train AutoML models (separate function for RAM efficiency)

## Configuration Classes

All configuration uses Pydantic models but supports dict-like instantiation:

- `PreprocessingConfig`: Data preprocessing settings
- `TrainingSplitConfig`: Train/val/test splitting settings
- `PolarsPipelineConfig`: Polars-ds pipeline settings
- `LinearModelConfig`: Linear model hyperparameters
- `TreeModelConfig`: Tree model hyperparameters
- `MLPConfig`: Neural network settings
- `AutoMLConfig`: AutoML settings
- `TrainingConfig`: Aggregated configuration

## Example Usage

```python
from mlframe.training import train_mlframe_models_suite, PreprocessingConfig, TrainingSplitConfig

# Train a suite of models including new linear models
models, metadata = train_mlframe_models_suite(
    df="data.parquet",
    target_name="target",
    model_name="experiment_1",
    preprocessor=my_preprocessor,
    mlframe_models=["linear", "ridge", "lasso", "cb", "lgb", "xgb"],
    preprocessing_config=PreprocessingConfig(fillna_value=0.0),
    split_config=TrainingSplitConfig(test_size=0.1, val_size=0.1),
)
```

## AutoML Usage

```python
from mlframe.training import train_automl_models_suite, AutoMLConfig

# Train AutoML models separately (more RAM-efficient)
automl_models = train_automl_models_suite(
    train_df=train_df,  # Must include target column
    test_df=test_df,    # Must include target column
    target_name="target",
    config=AutoMLConfig(
        use_autogluon=True,
        use_lama=False,
        autogluon_fit_params=dict(time_limit=3600),
    ),
)
```
"""

# Lazy imports using __getattr__ to avoid circular dependencies with training_old.py
# All imports are deferred until actually accessed

_LAZY_IMPORTS = {
    # Core functions
    'train_mlframe_models_suite': ('.core', 'train_mlframe_models_suite'),
    'train_automl_models_suite': ('.automl', 'train_automl_models_suite'),

    # Configuration classes
    'PreprocessingConfig': ('.configs', 'PreprocessingConfig'),
    'TrainingSplitConfig': ('.configs', 'TrainingSplitConfig'),
    'PolarsPipelineConfig': ('.configs', 'PolarsPipelineConfig'),
    'FeatureSelectionConfig': ('.configs', 'FeatureSelectionConfig'),
    'LinearModelConfig': ('.configs', 'LinearModelConfig'),
    'TreeModelConfig': ('.configs', 'TreeModelConfig'),
    'MLPConfig': ('.configs', 'MLPConfig'),
    'NGBConfig': ('.configs', 'NGBConfig'),
    'AutoMLConfig': ('.configs', 'AutoMLConfig'),
    'TrainingConfig': ('.configs', 'TrainingConfig'),
    'config_from_dict': ('.configs', 'config_from_dict'),
    'TargetTypes': ('.configs', 'TargetTypes'),

    # Model utilities
    'create_linear_model': ('.models', 'create_linear_model'),
    'is_linear_model': ('.models', 'is_linear_model'),
    'is_tree_model': ('.models', 'is_tree_model'),
    'is_neural_model': ('.models', 'is_neural_model'),
    'LINEAR_MODEL_TYPES': ('.models', 'LINEAR_MODEL_TYPES'),

    # Preprocessing utilities
    'load_and_prepare_dataframe': ('.preprocessing', 'load_and_prepare_dataframe'),
    'preprocess_dataframe': ('.preprocessing', 'preprocess_dataframe'),
    'save_split_artifacts': ('.preprocessing', 'save_split_artifacts'),
    'create_split_dataframes': ('.preprocessing', 'create_split_dataframes'),

    # Pipeline utilities
    'create_polarsds_pipeline': ('.pipeline', 'create_polarsds_pipeline'),
    'fit_and_transform_pipeline': ('.pipeline', 'fit_and_transform_pipeline'),
    'prepare_df_for_catboost': ('.pipeline', 'prepare_df_for_catboost'),

    # Utility functions
    'log_ram_usage': ('.utils', 'log_ram_usage'),
    'log_phase': ('.utils', 'log_phase'),
    'drop_columns_from_dataframe': ('.utils', 'drop_columns_from_dataframe'),
    'get_pandas_view_of_polars_df': ('.utils', 'get_pandas_view_of_polars_df'),
    'save_series_or_df': ('.utils', 'save_series_or_df'),
    'process_nans': ('.utils', 'process_nans'),
    'process_nulls': ('.utils', 'process_nulls'),
    'process_infinities': ('.utils', 'process_infinities'),
    'remove_constant_columns': ('.utils', 'remove_constant_columns'),

    # IO utilities
    'save_mlframe_model': ('.io', 'save_mlframe_model'),
    'load_mlframe_model': ('.io', 'load_mlframe_model'),

    # Splitting utilities
    'make_train_test_split': ('.splitting', 'make_train_test_split'),

    # Evaluation utilities
    'evaluate_model': ('.evaluation', 'evaluate_model'),

    # Training execution functions
    'select_target': ('.train_eval', 'select_target'),
    'process_model': ('.train_eval', 'process_model'),

    # Core trainer functions (migrated from training_old.py)
    'train_and_evaluate_model': ('.trainer', 'train_and_evaluate_model'),
    'configure_training_params': ('.trainer', 'configure_training_params'),
    'DataConfig': ('.configs', 'DataConfig'),
    'TrainingControlConfig': ('.configs', 'TrainingControlConfig'),
    'MetricsConfig': ('.configs', 'MetricsConfig'),
    'DisplayConfig': ('.configs', 'DisplayConfig'),
    'NamingConfig': ('.configs', 'NamingConfig'),
    'ConfidenceAnalysisConfig': ('.configs', 'ConfidenceAnalysisConfig'),
    'PredictionsContainer': ('.configs', 'PredictionsContainer'),
    'FairnessConfig': ('.configs', 'FairnessConfig'),

    # Helper functions (migrated from training_old.py)
    'get_trainset_features_stats': ('.helpers', 'get_trainset_features_stats'),
    'get_trainset_features_stats_polars': ('.helpers', 'get_trainset_features_stats_polars'),
    'get_training_configs': ('.helpers', 'get_training_configs'),
    'parse_catboost_devices': ('.helpers', 'parse_catboost_devices'),
    'LightGBMCallback': ('.helpers', 'LightGBMCallback'),
    'CatBoostCallback': ('.helpers', 'CatBoostCallback'),
    'XGBoostCallback': ('.helpers', 'XGBoostCallback'),
}

_cache = {}


def __getattr__(name):
    """Lazy import handler for module attributes."""
    if name in _LAZY_IMPORTS:
        if name not in _cache:
            module_name, attr_name = _LAZY_IMPORTS[name]
            import importlib
            # Import the submodule directly, not through the package
            full_module = f"mlframe.training.{module_name[1:]}"  # Remove leading dot and add separator
            module = importlib.import_module(full_module)
            _cache[name] = getattr(module, attr_name)
        return _cache[name]
    raise AttributeError(f"module 'mlframe.training' has no attribute {name!r}")


# train_and_evaluate_model is now in trainer.py and loaded via _LAZY_IMPORTS


__all__ = [
    # Core functions
    'train_mlframe_models_suite',
    'train_automl_models_suite',
    'make_train_test_split',
    'train_and_evaluate_model',
    'configure_training_params',

    # Configuration
    'PreprocessingConfig',
    'TrainingSplitConfig',
    'PolarsPipelineConfig',
    'FeatureSelectionConfig',
    'LinearModelConfig',
    'TreeModelConfig',
    'MLPConfig',
    'NGBConfig',
    'AutoMLConfig',
    'TrainingConfig',
    'config_from_dict',
    'TargetTypes',
    'DataConfig',
    'TrainingControlConfig',
    'MetricsConfig',
    'DisplayConfig',
    'NamingConfig',
    'ConfidenceAnalysisConfig',
    'PredictionsContainer',
    'FairnessConfig',

    # Models
    'create_linear_model',
    'is_linear_model',
    'is_tree_model',
    'is_neural_model',
    'LINEAR_MODEL_TYPES',

    # Preprocessing
    'load_and_prepare_dataframe',
    'preprocess_dataframe',
    'save_split_artifacts',
    'create_split_dataframes',

    # Pipeline
    'create_polarsds_pipeline',
    'fit_and_transform_pipeline',
    'prepare_df_for_catboost',

    # Utils
    'log_ram_usage',
    'log_phase',
    'drop_columns_from_dataframe',
    'save_mlframe_model',
    'load_mlframe_model',
    'get_pandas_view_of_polars_df',
    'save_series_or_df',
    'process_nans',
    'process_nulls',
    'process_infinities',
    'remove_constant_columns',

    # Evaluation
    'evaluate_model',

    # Training execution
    'select_target',
    'process_model',

    # Helper functions (from helpers.py)
    'get_trainset_features_stats',
    'get_trainset_features_stats_polars',
    'get_training_configs',
    'parse_catboost_devices',
    'LightGBMCallback',
    'CatBoostCallback',
    'XGBoostCallback',
]


__version__ = '2.0.0'  # Major refactoring version
