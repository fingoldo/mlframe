"""
Configuration classes for mlframe training pipeline.

Uses Pydantic for validation while supporting dict-like instantiation for backward compatibility.
All config classes support lenient validation - inputs are normalized to canonical forms.
"""

from typing import Optional, Dict, Any, List, Callable, Tuple
from enum import StrEnum

from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator


# =============================================================================
# Constants
# =============================================================================

DEFAULT_RANDOM_SEED = 42
"""Default random seed for reproducibility across all operations."""

DEFAULT_TREE_ITERATIONS = 5000
"""Default number of iterations for tree-based models (CB, LGB, XGB)."""

DEFAULT_CALIBRATION_BINS = 10
"""Default number of bins for calibration reports."""

VALID_MODEL_TYPES = {"cb", "lgb", "xgb", "mlp", "ngb", "linear", "ridge", "lasso", "elasticnet", "huber", "ransac", "sgd"}
"""Valid model type identifiers for mlframe_models parameter."""

VALID_LINEAR_MODEL_TYPES = {"linear", "ridge", "lasso", "elasticnet", "huber", "ransac", "sgd"}
"""Valid linear model type identifiers."""

VALID_SCALER_NAMES = {"standard", "min_max", "abs_max", "robust", None}
"""Valid scaler names for Polars pipeline."""

VALID_TASK_TYPES = {"CPU", "GPU"}
"""Valid task types for tree-based models (uppercase)."""

VALID_MATMUL_PRECISIONS = {"high", "medium", "highest"}
"""Valid float32 matmul precision settings for PyTorch."""


class TargetTypes(StrEnum):
    """Enumeration for ML task types.

    Attributes
    ----------
    REGRESSION : str
        Regression task type for continuous targets.
    BINARY_CLASSIFICATION : str
        Binary classification task type for two-class targets.
    """

    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"


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
    """Configuration for train/val/test data splitting.

    Controls how data is partitioned into training, validation, and test sets.
    Supports both random shuffling and sequential (time-based) splitting.

    Parameters
    ----------
    test_size : float
        Fraction of data for test set (default: 0.1).
    val_size : float
        Fraction of data for validation set (default: 0.1).
    shuffle_val : bool
        Whether to shuffle validation data (default: False).
    shuffle_test : bool
        Whether to shuffle test data (default: False).
    val_sequential_fraction : float
        Fraction of validation data taken sequentially from end (default: 0.5).
    test_sequential_fraction : float, optional
        Fraction of test data taken sequentially from end.
    trainset_aging_limit : float, optional
        Maximum age (fraction) of training samples to keep.
    wholeday_splitting : bool
        Whether to split on day boundaries (default: True).
    random_seed : int
        Random seed for reproducible splits (default: 42).

    Raises
    ------
    ValueError
        If test_size + val_size > 1.0.
    """

    test_size: float = Field(default=0.1, ge=0.0, le=1.0)
    val_size: float = Field(default=0.1, ge=0.0, le=1.0)
    shuffle_val: bool = False
    shuffle_test: bool = False
    val_sequential_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    test_sequential_fraction: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    trainset_aging_limit: Optional[float] = None
    wholeday_splitting: bool = True
    random_seed: int = DEFAULT_RANDOM_SEED

    @model_validator(mode="after")
    def validate_split_sizes(self) -> "TrainingSplitConfig":
        """Ensure test_size + val_size <= 1.0 to leave room for training data."""
        if self.test_size + self.val_size > 1.0:
            raise ValueError(f"test_size ({self.test_size}) + val_size ({self.val_size}) = " f"{self.test_size + self.val_size} must be <= 1.0")
        return self


class PolarsPipelineConfig(BaseConfig):
    """Configuration for Polars-ds preprocessing pipeline.

    Controls data preprocessing using Polars-based transformations including
    scaling, imputation, and categorical encoding.

    Parameters
    ----------
    use_polarsds_pipeline : bool
        Whether to use the Polars-ds pipeline (default: True).
    scaler_name : str, optional
        Scaler type: "standard", "min_max", "abs_max", "robust", or None.
        Case-insensitive, normalized to lowercase (default: "standard").
    imputer_strategy : str
        Strategy for imputing missing values: "mean", "median", etc. (default: "mean").
    categorical_encoding : str
        Encoding for categorical features: "ordinal", "onehot", "target" (default: "ordinal").
    robust_q_low : float
        Lower quantile for robust scaling (default: 0.01).
    robust_q_high : float
        Upper quantile for robust scaling (default: 0.99).
    """

    use_polarsds_pipeline: bool = True
    scaler_name: Optional[str] = "standard"
    imputer_strategy: str = "mean"
    categorical_encoding: str = "ordinal"
    robust_q_low: float = 0.01
    robust_q_high: float = 0.99

    @field_validator("scaler_name", mode="before")
    @classmethod
    def normalize_scaler_name(cls, v: Optional[str]) -> Optional[str]:
        """Normalize scaler_name to lowercase and validate."""
        if v is None:
            return None
        v_lower = v.lower()
        valid_names = {"standard", "min_max", "abs_max", "robust"}
        if v_lower not in valid_names:
            raise ValueError(f"scaler_name must be one of {valid_names} or None, got '{v}'")
        return v_lower


class FeatureSelectionConfig(BaseConfig):
    """Configuration for feature selection methods.

    Controls mRMR (minimum Redundancy Maximum Relevance) and RFECV
    (Recursive Feature Elimination with Cross-Validation) feature selection.

    Parameters
    ----------
    use_mrmr_fs : bool
        Whether to use mRMR feature selection (default: False).
    mrmr_kwargs : dict, optional
        Arguments for mRMR. Expected keys: features_to_select, show_progress, redundancy_metric.
    rfecv_models : list of str, optional
        Model types for RFECV feature selection (e.g., ["cb", "lgb"]).
    rfecv_kwargs : dict, optional
        Arguments for RFECV. Expected keys: step, min_features_to_select, cv, scoring.
    """

    use_mrmr_fs: bool = False
    mrmr_kwargs: Optional[Dict[str, Any]] = None  # keys: features_to_select, show_progress, redundancy_metric
    rfecv_models: Optional[List[str]] = None
    rfecv_kwargs: Optional[Dict[str, Any]] = None  # keys: step, min_features_to_select, cv, scoring


class ModelConfig(BaseConfig):
    """Base configuration for all ML models.

    Common parameters shared across all model types.

    Parameters
    ----------
    verbose : int
        Verbosity level for model training (default: 1).
    random_state : int
        Random seed for reproducibility (default: 42).
    n_jobs : int, optional
        Number of parallel jobs (-1 for all CPUs).
    """

    verbose: int = 1
    random_state: int = DEFAULT_RANDOM_SEED
    n_jobs: Optional[int] = None


class LinearModelConfig(ModelConfig):
    """Configuration for linear models (Ridge, Lasso, ElasticNet, etc.).

    Supports multiple linear model types with their respective parameters.
    The model_type is case-insensitive and normalized to lowercase.

    Parameters
    ----------
    model_type : str
        Type of linear model: "linear", "ridge", "lasso", "elasticnet",
        "huber", "ransac", "sgd". Case-insensitive (default: "linear").
    alpha : float
        Regularization strength for Ridge, Lasso, ElasticNet, SGD (default: 1.0).
    l1_ratio : float
        Mix of L1/L2 for ElasticNet (0=L2, 1=L1) (default: 0.5).
    epsilon : float
        Threshold for Huber loss (default: 1.35).
    max_trials : int
        Maximum iterations for RANSAC (default: 100).
    residual_threshold : float, optional
        Threshold for inliers in RANSAC.
    loss : str
        Loss function for SGD: "squared_error", "huber" (default: "squared_error").
    penalty : str
        Regularization penalty for SGD: "l2", "l1", "elasticnet" (default: "l2").
    max_iter : int
        Maximum iterations for iterative solvers (default: 1000).
    tol : float
        Convergence tolerance (default: 1e-3).
    learning_rate : str
        Learning rate schedule for SGD (default: "invscaling").
    eta0 : float
        Initial learning rate for SGD (default: 0.01).
    C : float
        Inverse regularization for LogisticRegression (default: 1.0).
    solver : str
        Solver for LogisticRegression (default: "lbfgs").
    use_calibrated_classifier : bool
        Whether to use probability calibration (default: True).
    """

    model_type: str = "linear"

    # Regularization parameters
    alpha: float = 1.0
    l1_ratio: float = 0.5

    # Robust regression parameters
    epsilon: float = 1.35
    max_trials: int = 100
    residual_threshold: Optional[float] = None

    # SGD parameters
    loss: str = "squared_error"
    penalty: str = "l2"
    max_iter: int = 1000
    tol: float = 1e-3
    learning_rate: str = "invscaling"
    eta0: float = 0.01

    # Classification-specific
    C: float = 1.0
    solver: str = "lbfgs"

    # Calibration
    use_calibrated_classifier: bool = False

    @field_validator("model_type", mode="before")
    @classmethod
    def normalize_model_type(cls, v: str) -> str:
        """Normalize model_type to lowercase and validate."""
        v_lower = v.lower()
        if v_lower not in VALID_LINEAR_MODEL_TYPES:
            raise ValueError(f"model_type must be one of {VALID_LINEAR_MODEL_TYPES}, got '{v}'")
        return v_lower


class TreeModelConfig(ModelConfig):
    """Configuration for tree-based models (CatBoost, LightGBM, XGBoost, etc.).

    Controls hyperparameters for gradient boosting models including
    iterations, learning rate, tree depth, and GPU settings.

    Parameters
    ----------
    iterations : int
        Number of boosting iterations (default: 5000).
    learning_rate : float
        Step size for gradient descent (default: 0.1).
    max_depth : int, optional
        Maximum tree depth (None for unlimited).
    early_stopping_rounds : int
        Rounds without improvement before stopping. 0 = auto (iterations // 3).
    task_type : str
        Computation device: "CPU" or "GPU". Case-insensitive, normalized to uppercase.
    devices : str, optional
        GPU device specification (e.g., "0", "0-3").
    cb_kwargs : dict, optional
        CatBoost-specific parameters. Keys: cat_features, od_type, od_wait, etc.
    lgb_kwargs : dict, optional
        LightGBM-specific parameters. Keys: num_leaves, min_child_samples, etc.
    xgb_kwargs : dict, optional
        XGBoost-specific parameters. Keys: tree_method, grow_policy, etc.
    hgb_kwargs : dict, optional
        HistGradientBoosting-specific parameters.
    """

    iterations: int = DEFAULT_TREE_ITERATIONS
    learning_rate: float = 0.1
    max_depth: Optional[int] = None
    early_stopping_rounds: int = 0  # 0 means auto (iterations // 3)

    # GPU settings
    task_type: str = "CPU"
    devices: Optional[str] = None

    # Model-specific kwargs
    cb_kwargs: Optional[Dict[str, Any]] = None  # keys: cat_features, od_type, od_wait, border_count
    lgb_kwargs: Optional[Dict[str, Any]] = None  # keys: num_leaves, min_child_samples, feature_fraction
    xgb_kwargs: Optional[Dict[str, Any]] = None  # keys: tree_method, grow_policy, max_bin
    hgb_kwargs: Optional[Dict[str, Any]] = None  # keys: max_leaf_nodes, min_samples_leaf

    @field_validator("task_type", mode="before")
    @classmethod
    def normalize_task_type(cls, v: str) -> str:
        """Normalize task_type to uppercase and validate."""
        v_upper = v.upper()
        if v_upper not in VALID_TASK_TYPES:
            raise ValueError(f"task_type must be one of {VALID_TASK_TYPES}, got '{v}'")
        return v_upper


class MLPConfig(ModelConfig):
    """Configuration for Multi-Layer Perceptron (PyTorch Lightning).

    Controls neural network architecture, training, and optimization settings.

    Parameters
    ----------
    model_params : dict, optional
        Model initialization parameters. Keys: hidden_dims, activation, dropout.
    network_params : dict, optional
        Network architecture parameters. Keys: layers, batch_norm.
    trainer_params : dict, optional
        PyTorch Lightning Trainer parameters. Keys: max_epochs, accelerator.
    dataloader_params : dict, optional
        DataLoader parameters. Keys: batch_size, num_workers.
    datamodule_params : dict, optional
        DataModule parameters. Keys: train_split, val_split.
    use_swa : bool
        Whether to use Stochastic Weight Averaging (default: True).
    swa_params : dict, optional
        SWA callback parameters. Keys: swa_lrs, swa_epoch_start.
    tune_params : bool
        Whether to tune hyperparameters (default: False).
    float32_matmul_precision : str
        PyTorch matmul precision: "high", "medium", "highest". Case-insensitive.
    """

    model_params: Optional[Dict[str, Any]] = None  # keys: hidden_dims, activation, dropout
    network_params: Optional[Dict[str, Any]] = None  # keys: layers, batch_norm
    trainer_params: Optional[Dict[str, Any]] = None  # keys: max_epochs, accelerator, devices
    dataloader_params: Optional[Dict[str, Any]] = None  # keys: batch_size, num_workers
    datamodule_params: Optional[Dict[str, Any]] = None  # keys: train_split, val_split

    use_swa: bool = True
    swa_params: Optional[Dict[str, Any]] = None  # keys: swa_lrs, swa_epoch_start
    tune_params: bool = False
    float32_matmul_precision: str = "medium"

    @field_validator("float32_matmul_precision", mode="before")
    @classmethod
    def normalize_precision(cls, v: str) -> str:
        """Normalize float32_matmul_precision to lowercase and validate."""
        v_lower = v.lower()
        if v_lower not in VALID_MATMUL_PRECISIONS:
            raise ValueError(f"float32_matmul_precision must be one of {VALID_MATMUL_PRECISIONS}, got '{v}'")
        return v_lower


class NGBConfig(ModelConfig):
    """Configuration for NGBoost probabilistic regression/classification.

    NGBoost outputs full probability distributions rather than point predictions.

    Parameters
    ----------
    n_estimators : int
        Number of boosting stages (default: 500).
    learning_rate : float
        Boosting learning rate (default: 0.01).
    minibatch_frac : float
        Fraction of data per boosting iteration (default: 1.0).
    Dist : Any, optional
        NGBoost distribution class (e.g., ngboost.distns.Normal, Bernoulli).
    Score : Any, optional
        NGBoost scoring rule class (e.g., ngboost.scores.LogScore, CRPS).
    """

    n_estimators: int = 500
    learning_rate: float = 0.01
    minibatch_frac: float = 1.0
    Dist: Optional[Any] = None  # ngboost.distns distribution class (Normal, Bernoulli, etc.)
    Score: Optional[Any] = None  # ngboost.scores scoring rule (LogScore, CRPS, etc.)


class AutoMLConfig(BaseConfig):
    """Configuration for AutoML frameworks (AutoGluon, LAMA).

    Supports automatic model selection and hyperparameter tuning.

    Parameters
    ----------
    use_autogluon : bool
        Whether to use AutoGluon (default: False).
    autogluon_init_params : dict, optional
        AutoGluon predictor initialization params. Keys: eval_metric, path.
    autogluon_fit_params : dict, optional
        AutoGluon fit params. Keys: time_limit, presets, hyperparameters.
    use_lama : bool
        Whether to use LightAutoML (default: False).
    lama_init_params : dict, optional
        LAMA initialization params.
    lama_fit_params : dict, optional
        LAMA fit params.
    automl_verbose : int
        Verbosity level (default: 1).
    automl_show_fi : bool
        Whether to show feature importances (default: True).
    automl_target_label : str
        Target column name for AutoML (default: "target").
    time_limit : int, optional
        Maximum training time in seconds.
    """

    # AutoGluon settings
    use_autogluon: bool = False
    autogluon_init_params: Optional[Dict[str, Any]] = None  # keys: eval_metric, path, problem_type
    autogluon_fit_params: Optional[Dict[str, Any]] = None  # keys: time_limit, presets, hyperparameters

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
    """Main configuration aggregating all training settings.

    This is the top-level config that combines preprocessing, splitting,
    feature selection, and model configurations.

    Parameters
    ----------
    target_name : str
        Name of the target column.
    model_name : str
        Identifier for the trained model.
    mlframe_models : list of str, optional
        Model types to train: "cb", "lgb", "xgb", "mlp", "ngb", "linear", etc.
        Case-insensitive, normalized to lowercase.
    use_ordinary_models : bool
        Whether to train individual models (default: True).
    use_mlframe_ensembles : bool
        Whether to train ensemble models (default: True).
    preprocessing : PreprocessingConfig
        Data preprocessing settings.
    split : TrainingSplitConfig
        Train/val/test split settings.
    pipeline : PolarsPipelineConfig
        Polars pipeline settings.
    feature_selection : FeatureSelectionConfig
        Feature selection settings.
    linear_config : LinearModelConfig, optional
        Linear model hyperparameters.
    tree_config : TreeModelConfig, optional
        Tree model hyperparameters.
    mlp_config : MLPConfig, optional
        MLP hyperparameters.
    ngb_config : NGBConfig, optional
        NGBoost hyperparameters.
    data_dir : str
        Base directory for data files.
    models_dir : str
        Directory for saved models (default: "models").
    config_params_override : dict, optional
        Override model config parameters. Keys depend on model type.
    control_params_override : dict, optional
        Override training control parameters.
    init_common_params : dict, optional
        Common parameters for all models.
    verbose : int
        Verbosity level (default: 1).
    imputer : Any, optional
        sklearn-compatible imputer transformer.
    scaler : Any, optional
        sklearn-compatible scaler transformer.
    category_encoder : Any, optional
        sklearn-compatible category encoder.
    metamodel_func : Callable, optional
        Function to wrap models (e.g., for target transformation).

    Raises
    ------
    ValueError
        If both use_ordinary_models and use_mlframe_ensembles are False.
        If mlframe_models contains invalid model types.
    """

    # Core inputs
    target_name: str
    model_name: str

    # Model selection
    mlframe_models: Optional[List[str]] = Field(default_factory=lambda: ["linear", "cb", "lgb", "xgb", "mlp"])
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
    config_params_override: Optional[Dict[str, Any]] = None  # keys: model-specific overrides
    control_params_override: Optional[Dict[str, Any]] = None  # keys: verbose, use_cache, just_evaluate
    init_common_params: Optional[Dict[str, Any]] = None  # keys: common params for all models

    # Misc
    verbose: int = 1

    # Transformers (sklearn-like)
    imputer: Optional[Any] = None  # sklearn imputer (SimpleImputer, etc.)
    scaler: Optional[Any] = None  # sklearn scaler (StandardScaler, etc.)
    category_encoder: Optional[Any] = None  # sklearn encoder (OrdinalEncoder, etc.)

    # Meta-model for target transformation
    metamodel_func: Optional[Callable] = None

    @field_validator("mlframe_models", mode="before")
    @classmethod
    def normalize_mlframe_models(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Normalize model names to lowercase and validate."""
        if v is None:
            return None
        normalized = [m.lower() for m in v]
        invalid = set(normalized) - VALID_MODEL_TYPES
        if invalid:
            raise ValueError(f"Invalid model types: {invalid}. Valid types: {VALID_MODEL_TYPES}")
        return normalized

    @model_validator(mode="after")
    def validate_model_selection(self) -> "TrainingConfig":
        """Ensure at least one model training option is enabled."""
        if not self.use_ordinary_models and not self.use_mlframe_ensembles:
            raise ValueError("At least one of use_ordinary_models or use_mlframe_ensembles must be True")
        return self


# =====================================================================================
# train_and_evaluate_model Configuration Classes
# =====================================================================================


class DataConfig(BaseConfig):
    """Input data configuration for train_and_evaluate_model.

    Holds DataFrames, targets, indices, and auxiliary data for model training.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Full dataset (used when train/val/test_df not provided).
    train_df : pd.DataFrame, optional
        Training features DataFrame.
    val_df : pd.DataFrame, optional
        Validation features DataFrame.
    test_df : pd.DataFrame, optional
        Test features DataFrame.
    target : np.ndarray or pd.Series, optional
        Full target array (used with indices).
    train_target : np.ndarray or pd.Series, optional
        Training target values.
    val_target : np.ndarray or pd.Series, optional
        Validation target values.
    test_target : np.ndarray or pd.Series, optional
        Test target values.
    train_idx : np.ndarray, optional
        Indices for training rows in df.
    val_idx : np.ndarray, optional
        Indices for validation rows in df.
    test_idx : np.ndarray, optional
        Indices for test rows in df.
    group_ids : np.ndarray, optional
        Group identifiers for per-group AUC computation.
    sample_weight : np.ndarray or pd.Series, optional
        Sample weights for weighted training.
    drop_columns : list of str
        Columns to drop from features.
    target_label_encoder : LabelEncoder, optional
        Encoder for multiclass target labels.
    skip_infinity_checks : bool
        Whether to skip infinity value checks (default: False).
    n_features : int, optional
        Feature count for display when df not provided (e.g., ensembles).
    """

    # DataFrames
    df: Optional[Any] = None  # pd.DataFrame - full dataset
    train_df: Optional[Any] = None  # pd.DataFrame
    val_df: Optional[Any] = None  # pd.DataFrame
    test_df: Optional[Any] = None  # pd.DataFrame

    # Targets
    target: Optional[Any] = None  # np.ndarray or pd.Series
    train_target: Optional[Any] = None  # np.ndarray or pd.Series
    val_target: Optional[Any] = None  # np.ndarray or pd.Series
    test_target: Optional[Any] = None  # np.ndarray or pd.Series

    # Indices
    train_idx: Optional[Any] = None  # np.ndarray
    val_idx: Optional[Any] = None  # np.ndarray
    test_idx: Optional[Any] = None  # np.ndarray

    # Additional data
    group_ids: Optional[Any] = None  # np.ndarray - for per-group AUC computation
    sample_weight: Optional[Any] = None  # np.ndarray or pd.Series

    # Data configuration
    drop_columns: List[str] = Field(default_factory=list)
    target_label_encoder: Optional[Any] = None  # sklearn.preprocessing.LabelEncoder
    skip_infinity_checks: bool = False

    # Feature count for display (optional - used when df not provided, e.g., ensembles)
    n_features: Optional[int] = None


class TrainingControlConfig(BaseConfig):
    """Training control flags and settings for train_and_evaluate_model.

    Controls training behavior, caching, and metrics computation.

    Parameters
    ----------
    verbose : bool
        Whether to print verbose output (default: False).
    use_cache : bool
        Whether to load cached models if available (default: False).
    just_evaluate : bool
        Skip training, only evaluate pre-computed predictions (default: False).
    compute_trainset_metrics : bool
        Whether to compute metrics on training set (default: False).
    compute_valset_metrics : bool
        Whether to compute metrics on validation set (default: True).
    compute_testset_metrics : bool
        Whether to compute metrics on test set (default: True).
    pre_pipeline : TransformerMixin, optional
        sklearn-compatible preprocessing pipeline.
    skip_pre_pipeline_transform : bool
        Whether to skip pre_pipeline transform (default: False).
    fit_params : dict, optional
        Additional parameters passed to model.fit(). Keys depend on model type.
    callback_params : dict, optional
        Parameters for training callbacks. Keys: patience, verbose.
    """

    verbose: bool = False
    use_cache: bool = False
    just_evaluate: bool = False

    # Metrics computation flags
    compute_trainset_metrics: bool = False
    compute_valset_metrics: bool = True
    compute_testset_metrics: bool = True

    # Pipeline
    pre_pipeline: Optional[Any] = None  # sklearn TransformerMixin
    skip_pre_pipeline_transform: bool = False
    fit_params: Optional[Dict[str, Any]] = None  # keys: eval_set, early_stopping_rounds, etc.
    callback_params: Optional[Dict[str, Any]] = None  # keys: patience, verbose

    # Model category for early stopping callback setup (cb, xgb, lgb, etc.)
    model_category: Optional[str] = None


class MetricsConfig(BaseConfig):
    """Metrics and evaluation configuration for train_and_evaluate_model.

    Controls calibration metrics, custom metrics, and fairness subgroups.

    Parameters
    ----------
    nbins : int
        Number of bins for calibration plots (default: 10).
    custom_ice_metric : Callable, optional
        Custom Integral Calibration Error metric function.
    custom_rice_metric : Callable, optional
        Custom Robust Integral Calibration Error metric function.
    subgroups : dict, optional
        Fairness subgroup definitions for fairness metrics.
    train_details : str
        Additional details for training set report.
    val_details : str
        Additional details for validation set report.
    test_details : str
        Additional details for test set report.
    """

    nbins: int = DEFAULT_CALIBRATION_BINS
    custom_ice_metric: Optional[Callable] = None
    custom_rice_metric: Optional[Callable] = None
    subgroups: Optional[Dict] = None  # keys: subgroup_name -> column_name or criteria

    # Split descriptions for reporting
    train_details: str = ""
    val_details: str = ""
    test_details: str = ""


class DisplayConfig(BaseConfig):
    """Display and plotting configuration for train_and_evaluate_model.

    Controls figure sizes, report printing, and output paths.

    Parameters
    ----------
    figsize : tuple of int
        Figure size for plots (width, height) (default: (15, 5)).
    print_report : bool
        Whether to print performance report (default: True).
    show_perf_chart : bool
        Whether to display performance charts (default: True).
    show_fi : bool
        Whether to show feature importances (default: True).
    fi_kwargs : dict
        Additional kwargs for feature importance plots.
    plot_file : str
        Base path for saving plot files.
    data_dir : str
        Base directory for data files.
    models_subdir : str
        Subdirectory for saved models (default: "models").
    display_sample_size : int
        Number of sample rows to display during training (default: 0 = disabled).
    show_feature_names : bool
        Whether to show feature names in training report (default: False).
    """

    figsize: Tuple[int, int] = (15, 5)
    print_report: bool = True
    show_perf_chart: bool = True
    show_fi: bool = True
    fi_kwargs: Dict[str, Any] = Field(default_factory=dict)  # keys: max_features, plot_type

    plot_file: str = ""
    data_dir: str = ""
    models_subdir: str = "models"
    display_sample_size: int = 0
    show_feature_names: bool = False


class ConfidenceAnalysisConfig(BaseConfig):
    """Confidence analysis configuration for train_and_evaluate_model.

    Controls SHAP-based confidence analysis of model predictions.

    Parameters
    ----------
    include : bool
        Whether to include confidence analysis (default: False).
    use_shap : bool
        Whether to use SHAP for explanations (default: True).
    max_features : int
        Maximum features to show in plots (default: 6).
    cmap : str
        Colormap for plots (default: "bwr").
    alpha : float
        Transparency for plot points (default: 0.9).
    ylabel : str
        Y-axis label for plots.
    title : str
        Plot title.
    model_kwargs : dict
        Additional kwargs for confidence model. Keys: n_estimators, max_depth.
    """

    include: bool = False
    use_shap: bool = True
    max_features: int = 6
    cmap: str = "bwr"
    alpha: float = 0.9
    ylabel: str = "Feature value"
    title: str = "Confidence of correct Test set predictions"
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)  # keys: n_estimators, max_depth


class NamingConfig(BaseConfig):
    """Model naming configuration for train_and_evaluate_model.

    Controls model naming for reports and saved files.

    Parameters
    ----------
    model_name : str
        Name of the model for reports.
    model_name_prefix : str
        Prefix to add before model type in names.
    """

    model_name: str = ""
    model_name_prefix: str = ""


class PredictionsContainer(BaseConfig):
    """Container for pre-computed predictions (used in just_evaluate mode).

    Holds predictions and probabilities for train/val/test splits.

    Parameters
    ----------
    train_preds : np.ndarray, optional
        Training set predictions.
    train_probs : np.ndarray, optional
        Training set probabilities.
    val_preds : np.ndarray, optional
        Validation set predictions.
    val_probs : np.ndarray, optional
        Validation set probabilities.
    test_preds : np.ndarray, optional
        Test set predictions.
    test_probs : np.ndarray, optional
        Test set probabilities.
    """

    train_preds: Optional[Any] = None  # np.ndarray
    train_probs: Optional[Any] = None  # np.ndarray
    val_preds: Optional[Any] = None  # np.ndarray
    val_probs: Optional[Any] = None  # np.ndarray
    test_preds: Optional[Any] = None  # np.ndarray
    test_probs: Optional[Any] = None  # np.ndarray


class FairnessConfig(BaseConfig):
    """Fairness analysis configuration.

    Controls fairness metric computation across demographic subgroups.

    Parameters
    ----------
    enabled : bool
        Whether to enable fairness analysis (default: False).
    protected_attributes : list of str, optional
        Column names of protected attributes.
    fairness_metrics : list of str, optional
        Fairness metrics to compute (e.g., "demographic_parity", "equalized_odds").
    """

    enabled: bool = False
    protected_attributes: Optional[List[str]] = None
    fairness_metrics: Optional[List[str]] = None


# Helper function to create config from dict (backward compatibility)
def config_from_dict(config_class: type[BaseConfig], params: Dict[str, Any]) -> BaseConfig:
    """Create config from dict, handling nested dicts.

    Parameters
    ----------
    config_class : type[BaseConfig]
        The config class to instantiate.
    params : dict
        Dictionary of parameters to pass to the config.

    Returns
    -------
    BaseConfig
        Instantiated config object.
    """
    return config_class(**params)


# Export all configs and constants
__all__ = [
    # Constants
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_TREE_ITERATIONS",
    "DEFAULT_CALIBRATION_BINS",
    "VALID_MODEL_TYPES",
    "VALID_LINEAR_MODEL_TYPES",
    "VALID_SCALER_NAMES",
    "VALID_TASK_TYPES",
    "VALID_MATMUL_PRECISIONS",
    # Enums
    "TargetTypes",
    # Base
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
    # train_and_evaluate_model configs
    "DataConfig",
    "TrainingControlConfig",
    "MetricsConfig",
    "DisplayConfig",
    "ConfidenceAnalysisConfig",
    "NamingConfig",
    "PredictionsContainer",
    "FairnessConfig",
]
