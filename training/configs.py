"""
Configuration classes for mlframe training pipeline.

Uses Pydantic for validation while supporting dict-like instantiation for backward compatibility.
"""

from typing import Optional, Dict, Any, List, Callable, Union
from pydantic import BaseModel, Field, ConfigDict
import numpy as np


class BaseConfig(BaseModel):
    """Base configuration class with flexible dict support."""

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for flexibility
        arbitrary_types_allowed=True,  # Allow numpy, torch, etc.
        validate_assignment=True,
        protected_namespaces=(),  # Allow model_ prefix for field names
    )


class PreprocessingConfig(BaseConfig):
    """Configuration for data preprocessing."""

    fillna_value: Optional[float] = None
    fix_infinities: bool = True
    ensure_float32_dtypes: bool = True
    skip_infinity_checks: bool = True
    drop_columns: Optional[List[str]] = None
    n_rows: Optional[int] = Field(
        default=None,
        ge=1,
    )
    tail: Optional[int] = Field(
        default=None,
        ge=1,
    )
    columns: Optional[List[str]] = None


class TrainingSplitConfig(BaseConfig):
    """Configuration for train/val/test splitting."""

    test_size: float = Field(default=0.1, ge=0.0, le=1.0)
    val_size: float = Field(default=0.1, ge=0.0, le=1.0)
    shuffle_val: bool = False
    shuffle_test: bool = False
    val_sequential_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    test_sequential_fraction: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    trainset_aging_limit: Optional[float] = None
    wholeday_splitting: bool = True
    random_seed: int = 42


class PolarsPipelineConfig(BaseConfig):
    """Configuration for Polars-ds pipeline."""

    use_polarsds_pipeline: bool = True
    scaler_name: str = "standard"  # standard, minmax, robust, etc.
    imputer_strategy: str = "mean"
    categorical_encoding: str = "ordinal"  # ordinal, onehot, target, etc.


class FeatureSelectionConfig(BaseConfig):
    """Configuration for feature selection."""

    use_mrmr_fs: bool = False
    mrmr_kwargs: Optional[Dict[str, Any]] = None
    rfecv_models: Optional[List[str]] = None
    rfecv_kwargs: Optional[Dict[str, Any]] = None


class ModelConfig(BaseConfig):
    """Base configuration for ML models."""

    verbose: int = 1
    random_state: int = 42
    n_jobs: Optional[int] = None


class LinearModelConfig(ModelConfig):
    """Configuration for linear models."""

    model_type: str = "linear"  # linear, ridge, lasso, elasticnet, huber, ransac, sgd

    # Regularization parameters
    alpha: float = 1.0  # For Ridge, Lasso, ElasticNet, SGD
    l1_ratio: float = 0.5  # For ElasticNet (mix of L1 and L2)

    # Robust regression parameters
    epsilon: float = 1.35  # For Huber
    max_trials: int = 100  # For RANSAC
    residual_threshold: Optional[float] = None  # For RANSAC

    # SGD parameters
    loss: str = "squared_error"  # For SGD: squared_error, huber, epsilon_insensitive
    penalty: str = "l2"  # For SGD: l2, l1, elasticnet
    max_iter: int = 1000
    tol: float = 1e-3
    learning_rate: str = "invscaling"  # constant, optimal, invscaling, adaptive
    eta0: float = 0.01

    # Classification-specific
    C: float = 1.0  # For LogisticRegression (inverse of alpha)
    solver: str = "lbfgs"  # lbfgs, liblinear, sag, saga, newton-cg

    # Calibration
    use_calibrated_classifier: bool = True


class TreeModelConfig(ModelConfig):
    """Configuration for tree-based models (CB, LGB, XGB, etc.)."""

    iterations: int = 5000
    learning_rate: float = 0.1
    max_depth: Optional[int] = None
    early_stopping_rounds: int = 0  # 0 means auto (iterations // 3)

    # GPU settings
    task_type: str = "CPU"  # CPU or GPU
    devices: Optional[str] = None

    # Model-specific kwargs
    cb_kwargs: Optional[Dict[str, Any]] = None
    lgb_kwargs: Optional[Dict[str, Any]] = None
    xgb_kwargs: Optional[Dict[str, Any]] = None
    hgb_kwargs: Optional[Dict[str, Any]] = None


class MLPConfig(ModelConfig):
    """Configuration for Multi-Layer Perceptron (PyTorch Lightning)."""

    model_params: Optional[Dict[str, Any]] = None
    network_params: Optional[Dict[str, Any]] = None
    trainer_params: Optional[Dict[str, Any]] = None
    dataloader_params: Optional[Dict[str, Any]] = None
    datamodule_params: Optional[Dict[str, Any]] = None

    use_swa: bool = True
    swa_params: Optional[Dict[str, Any]] = None
    tune_params: bool = False
    float32_matmul_precision: str = "medium"


class NGBConfig(ModelConfig):
    """Configuration for NGBoost."""

    n_estimators: int = 500
    learning_rate: float = 0.01
    minibatch_frac: float = 1.0
    Dist: Optional[Any] = None
    Score: Optional[Any] = None


class AutoMLConfig(BaseConfig):
    """Configuration for AutoML models (AutoGluon, LAMA)."""

    # AutoGluon settings
    use_autogluon: bool = False
    autogluon_init_params: Optional[Dict[str, Any]] = None
    autogluon_fit_params: Optional[Dict[str, Any]] = None

    # LAMA settings
    use_lama: bool = False
    lama_init_params: Optional[Dict[str, Any]] = None
    lama_fit_params: Optional[Dict[str, Any]] = None

    # Common settings
    automl_verbose: int = 1
    automl_show_fi: bool = True
    automl_target_label: str = "target"
    time_limit: Optional[int] = None  # In seconds


class TrainingConfig(BaseConfig):
    """Main configuration aggregating all training settings."""

    # Core inputs
    target_name: str
    model_name: str

    # Model selection
    mlframe_models: Optional[List[str]] = Field(default_factory=lambda: ["cb", "lgb", "xgb", "mlp"])
    use_ordinary_models: bool = True
    use_mlframe_ensembles: bool = True

    # Sub-configurations
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    split: TrainingSplitConfig = Field(default_factory=TrainingSplitConfig)
    pipeline: PolarsPipelineConfig = Field(default_factory=PolarsPipelineConfig)
    feature_selection: FeatureSelectionConfig = Field(default_factory=FeatureSelectionConfig)

    # Model-specific configs (can be overridden per model)
    linear_config: Optional[LinearModelConfig] = None
    tree_config: Optional[TreeModelConfig] = None
    mlp_config: Optional[MLPConfig] = None
    ngb_config: Optional[NGBConfig] = None

    # Directory paths
    data_dir: str = ""
    models_dir: str = "models"

    # Control parameters
    config_params_override: Optional[Dict[str, Any]] = None
    control_params_override: Optional[Dict[str, Any]] = None
    init_common_params: Optional[Dict[str, Any]] = None

    # Misc
    verbose: int = 1

    # Transformers (sklearn-like)
    imputer: Optional[Any] = None
    scaler: Optional[Any] = None
    category_encoder: Optional[Any] = None

    # Meta-model for target transformation
    metamodel_func: Optional[Callable] = None


# Helper function to create config from dict (backward compatibility)
def config_from_dict(config_class: type[BaseConfig], params: Dict[str, Any]) -> BaseConfig:
    """Create config from dict, handling nested dicts."""
    return config_class(**params)


# Export all configs
__all__ = [
    "BaseConfig",
    "PreprocessingConfig",
    "TrainingSplitConfig",
    "PolarsPipelineConfig",
    "FeatureSelectionConfig",
    "ModelConfig",
    "LinearModelConfig",
    "TreeModelConfig",
    "MLPConfig",
    "NGBConfig",
    "AutoMLConfig",
    "TrainingConfig",
    "config_from_dict",
]
