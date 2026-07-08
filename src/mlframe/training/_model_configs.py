"""Model + hyperparameter + training-behavior configs for ``mlframe.training.configs``.

Split out from ``configs.py`` to keep that file below the 1k-line monolith
threshold. Behaviour preserved bit-for-bit; every class is re-exported from
``configs`` so existing ``from mlframe.training.configs import ModelConfig``
(and the other moved names) imports continue to resolve.

What lives here:
  - ``ModelConfig`` (base) and subclasses: ``LinearModelConfig``,
    ``TreeModelConfig``, ``MLPConfig``, ``NGBConfig``.
  - ``AutoMLConfig``, ``ModelHyperparamsConfig``, ``TrainingBehaviorConfig``.
  - ``MultilabelDispatchConfig``, ``LearningToRankConfig``,
    ``QuantileRegressionConfig``, ``EnsemblingConfig``.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, FrozenSet, List, Optional

from pydantic import Field, field_validator, model_validator

from ._configs_base import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_RFECV_CV_SPLITS,
    DEFAULT_RFECV_MAX_NOIMPROVING_ITERS,
    DEFAULT_RFECV_MAX_RUNTIME_MINS,
    DEFAULT_TREE_ITERATIONS,
    VALID_LINEAR_MODEL_TYPES,
    VALID_MATMUL_PRECISIONS,
    VALID_TASK_TYPES,
    BaseConfig,
)


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
        Can also be set via `iterations` for consistency with tree models.
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

    # Regularization parameters. Range guards catch garbage at construction
    # (alpha=-1 / l1_ratio=1.5) -- sklearn Ridge/Lasso/ElasticNet reject these
    # too, but only deep inside fit; the earlier the better. alpha>=0 (0 == OLS);
    # l1_ratio in [0,1] (sklearn ElasticNet contract: 0=L2, 1=L1).
    alpha: float = Field(default=1.0, ge=0.0)
    l1_ratio: float = Field(default=0.5, ge=0.0, le=1.0)

    # Robust regression parameters
    epsilon: float = 1.35
    max_trials: int = Field(default=100, ge=1)
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

    @model_validator(mode="before")
    @classmethod
    def map_iterations_to_max_iter(cls, data: Any) -> Any:
        """Map 'iterations' to 'max_iter' for consistency with tree models."""
        if isinstance(data, dict) and "iterations" in data:
            # Only set max_iter from iterations if max_iter wasn't explicitly provided
            if "max_iter" not in data:
                data["max_iter"] = data.pop("iterations")
            else:
                # Remove iterations if max_iter is also present (max_iter takes precedence)
                data.pop("iterations")
        return data

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
    early_stopping_rounds : int or None
        Rounds without improvement before stopping. 0 = auto (iterations // 3).
        None disables early stopping entirely.
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

    # Range guards mirror ModelHyperparamsConfig: iterations=0 trains zero trees
    # (LightGBM/XGB silently predict the init constant -- a degenerate, no-error
    # run); a negative / >1 learning_rate propagates straight to the booster.
    iterations: int = Field(default=DEFAULT_TREE_ITERATIONS, ge=1)
    learning_rate: float = Field(default=0.1, gt=0.0, le=1.0)
    # None = unlimited; when set must be >=1 (max_depth=0 is a degenerate stump).
    max_depth: Optional[int] = Field(default=None, ge=1)
    # 0 = auto (iterations // 3); None = disabled; a negative value is nonsense.
    early_stopping_rounds: Optional[int] = Field(default=0, ge=0)

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

    # Range guards: n_estimators=0 yields a no-stage NGBoost (degenerate);
    # minibatch_frac must be a (0,1] fraction (NGBoost subsamples that share
    # of rows per stage -- 0 / >1 / negative all misbehave silently).
    n_estimators: int = Field(default=500, ge=1)
    learning_rate: float = Field(default=0.01, gt=0.0, le=1.0)
    minibatch_frac: float = Field(default=1.0, gt=0.0, le=1.0)
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


class ModelHyperparamsConfig(BaseConfig):
    """Model hyperparameters for the training pipeline.

    Replaces the legacy untyped ``config_params`` / ``config_params_override`` dicts.
    All fields have sensible defaults; pass only what you want to change.

    Parameters
    ----------
    has_time : bool
        Whether the dataset has a time column (ordered splitting).
    learning_rate : float
        Global learning rate for tree models.
    iterations : int
        Number of boosting iterations.
    early_stopping_rounds : int or None
        Patience for early stopping. None disables early stopping entirely.
    catboost_custom_classif_metrics : list of str, optional
        Custom CatBoost classification metrics.
    rfecv_kwargs : dict, optional
        RFECV parameters (max_runtime_mins, cv_n_splits, max_noimproving_iters).
    cb_kwargs : dict, optional
        Extra CatBoost constructor kwargs.
    lgb_kwargs : dict, optional
        Extra LightGBM constructor kwargs.
    xgb_kwargs : dict, optional
        Extra XGBoost constructor kwargs.
    hgb_kwargs : dict, optional
        Extra HistGradientBoosting constructor kwargs.
    mlp_kwargs : dict, optional
        Extra MLP constructor kwargs.
    ngb_kwargs : dict, optional
        Extra NGBoost constructor kwargs.
    """

    # Legitimate pass-through extras consumed by ``get_training_configs``
    # via ``**config_params``. Adding a name here silences the
    # "unknown field" warning from BaseConfig when users pass it through
    # ``hyperparams_config={"mae_weight": 2.0, ...}``.
    _known_extras: ClassVar[FrozenSet[str]] = frozenset({
        # ICE-metric weights (see metrics.integral_calibration_error_from_metrics)
        "mae_weight", "std_weight", "roc_auc_weight", "pr_auc_weight",
        "brier_loss_weight", "min_roc_auc", "roc_auc_penalty",
        # Robustness / integral-error bin config
        "robustness_num_ts_splits", "robustness_std_coeff",
        "robustness_greater_is_better",
        "nbins", "cont_nbins", "method", "use_weighted_calibration",
        "weight_by_class_npositives",
        # Scoring + metric defaults
        "def_classif_metric", "def_regr_metric",
        # Training infra knobs
        "validation_fraction", "use_explicit_early_stopping",
        "random_seed", "verbose",
        # Non-classif extras
        "catboost_custom_regr_metrics",
    })

    has_time: bool = False
    # Range validators catch garbage (learning_rate=-0.1, iterations=0, etc.) at construction; otherwise they propagate silently to the tree backends and surface as confusing errors much later.
    learning_rate: float = Field(default=0.2, gt=0.0, le=1.0)
    iterations: int = Field(default=700, ge=1)
    early_stopping_rounds: Optional[int] = Field(default=100, ge=1)
    catboost_custom_classif_metrics: Optional[List[str]] = None

    # 2026-05-26: promoted from ``_known_extras`` passthrough to a
    # first-class field so users see it in IDE auto-complete + docs.
    # Default "RMSE" matches the competition-canonical metric printed
    # in chart titles ("MAE=... RMSE=... R2=..."). Applied uniformly
    # across CB / LGB / XGB regression paths:
    #   CB:  ``eval_metric=def_regr_metric``                 (native names)
    #   LGB: ``metric=`` mapped {RMSE->l2, MAE->l1, Huber->huber, ...}
    #   XGB: ``eval_metric=`` mapped {RMSE->rmse, MAE->mae, Huber->mphe, ...}
    # Heavy-kurt route via ``_apply_loss_recommendation_in_place``
    # overrides this for the affected target only.
    def_regr_metric: str = Field(default="RMSE")
    def_classif_metric: str = Field(default="AUC")
    # Deprecated: prefer FeatureSelectionConfig.rfecv_kwargs which carries
    # field-level validation against RFECV.__init__. This field remains for
    # backward compatibility with callers that thread rfecv params through
    # get_training_configs (see helpers.py:778); both fields stay live until
    # downstream callers migrate. When both are populated, FSC's value
    # should win (resolution policy enforced at the call site, not here).
    rfecv_kwargs: Dict[str, Any] = Field(default_factory=lambda: {
        "max_runtime_mins": DEFAULT_RFECV_MAX_RUNTIME_MINS,
        "cv_n_splits": DEFAULT_RFECV_CV_SPLITS,
        "max_noimproving_iters": DEFAULT_RFECV_MAX_NOIMPROVING_ITERS,
    })

    # Per-model kwargs
    cb_kwargs: Optional[Dict[str, Any]] = None
    lgb_kwargs: Optional[Dict[str, Any]] = None
    xgb_kwargs: Optional[Dict[str, Any]] = None
    hgb_kwargs: Optional[Dict[str, Any]] = None
    mlp_kwargs: Optional[Dict[str, Any]] = None
    ngb_kwargs: Optional[Dict[str, Any]] = None

    # First-class predict-time MLP batch size. When None (default) the wrapper auto-adapts to free memory + input width via ``mlp_runtime_defaults.resolve_mlp_predict_batch_size``; a hardcoded small batch makes 4M-row predict paths spend minutes on DataLoader overhead. Set explicitly to an int to lock a specific batch (eg ``mlp_predict_batch_size=512`` on memory-constrained boxes with wide dataframes; ``8192`` on slim-row narrow-width predictions).
    mlp_predict_batch_size: Optional[int] = None


# TrainingBehaviorConfig / MultilabelDispatchConfig / LearningToRankConfig / QuantileRegressionConfig carved to
# ``_model_configs_behavior.py`` (1k-LOC ceiling); re-exported so existing import paths keep resolving.
from ._model_configs_behavior import (
    LearningToRankConfig,
    MultilabelDispatchConfig,
    QuantileRegressionConfig,
    TrainingBehaviorConfig,
)

# EnsemblingConfig carved to ``_model_configs_ensembling.py`` to keep this
# parent below the 1k LOC monolith threshold. Re-export preserves the
# canonical ``from mlframe.training.configs import EnsemblingConfig`` import
# and the historic ``from mlframe.training._model_configs import EnsemblingConfig``
# bottom-of-monolith pattern (class identity is preserved by the
# re-export, so ``isinstance`` checks downstream keep working).
from ._model_configs_ensembling import EnsemblingConfig
