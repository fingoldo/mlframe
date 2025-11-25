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

# Core training functions
from .core import train_mlframe_models_suite
from .automl import train_automl_models_suite

# Lazy import from original training_old.py module (not this package)
def _get_training_old_module():
    """Lazy-load training_old.py module to avoid slow startup."""
    import os
    import importlib.util
    # Load training_old.py file directly (not the training/ package)
    training_old_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training_old.py')
    spec = importlib.util.spec_from_file_location("mlframe.training_old_module", training_old_path)
    training_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training_module)
    return training_module

# Create wrapper function for lazy loading
def make_train_test_split(*args, **kwargs):
    """Train/val/test split function from original training_old.py (lazy-loaded)."""
    # Remove verbose parameter if present (not supported by original function)
    kwargs = {k: v for k, v in kwargs.items() if k != 'verbose'}
    training_module = _get_training_old_module()
    return training_module.make_train_test_split(*args, **kwargs)

def train_and_evaluate_model(*args, **kwargs):
    """Train and evaluate model function from original training_old.py (lazy-loaded)."""
    training_module = _get_training_old_module()
    return training_module.train_and_evaluate_model(*args, **kwargs)

# Configuration classes
from .configs import (
    PreprocessingConfig,
    TrainingSplitConfig,
    PolarsPipelineConfig,
    FeatureSelectionConfig,
    LinearModelConfig,
    TreeModelConfig,
    MLPConfig,
    NGBConfig,
    AutoMLConfig,
    TrainingConfig,
    config_from_dict,
)

# Model utilities
from .models import (
    create_linear_model,
    is_linear_model,
    is_tree_model,
    is_neural_model,
    LINEAR_MODEL_TYPES,
)

# Preprocessing utilities
from .preprocessing import (
    load_and_prepare_dataframe,
    preprocess_dataframe,
    save_split_artifacts,
    create_split_dataframes,
)

# Pipeline utilities
from .pipeline import (
    create_polarsds_pipeline,
    fit_and_transform_pipeline,
    prepare_df_for_catboost,
)

# Utility functions
from .utils import (
    log_ram_usage,
    log_phase,
    drop_columns_from_dataframe,
    save_mlframe_model,
    load_mlframe_model,
    get_pandas_view_of_polars_df,
    save_series_or_df,
    process_nans,
    process_nulls,
    process_infinities,
    remove_constant_columns,
)

# Evaluation utilities
from .evaluation import evaluate_model


__all__ = [
    # Core functions
    'train_mlframe_models_suite',
    'train_automl_models_suite',
    'make_train_test_split',
    'train_and_evaluate_model',

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
]


__version__ = '2.0.0'  # Major refactoring version
