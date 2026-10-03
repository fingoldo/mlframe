"""Model + hyperparameter + training-behavior configs for ``mlframe.training.configs``.

Split out from ``configs.py`` to keep that file below the 1k-line monolith
threshold. Behaviour preserved bit-for-bit; every class is re-exported from
``configs`` so existing ``from mlframe.training.configs import ModelConfig``
(and the other moved names) imports continue to resolve.

What lives here:
  - ``ModelConfig`` (base) and subclasses: ``LinearModelConfig``,
    ``MLPConfig`` (the strict validator of ``ModelHyperparamsConfig.mlp_kwargs``).
  - ``AutoMLConfig``, ``ModelHyperparamsConfig``, ``TrainingBehaviorConfig``.
  - ``MultilabelDispatchConfig``, ``LearningToRankConfig``,
    ``QuantileRegressionConfig``, ``EnsemblingConfig``.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional

from pydantic import ConfigDict, Field, field_validator, model_validator

from ._inert_fields import InertFieldsWarningMixin
from ._configs_base import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_RFECV_CV_SPLITS,
    DEFAULT_RFECV_MAX_NOIMPROVING_ITERS,
    DEFAULT_RFECV_MAX_RUNTIME_MINS,
    VALID_LINEAR_MODEL_TYPES,
    VALID_MATMUL_PRECISIONS,
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
        Wrap the classifier in ``CalibratedClassifierCV`` (default: False). Off because it refits the model k times
        and the suite calibrates probabilities post hoc on its own calibration slice when one is configured; stacking
        the two would calibrate twice.
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
    # sklearn SGD early stopping: hold out ``validation_fraction`` of the training rows and stop after ``n_iter_no_change`` epochs without improvement.
    early_stopping: bool = False
    validation_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    n_iter_no_change: int = Field(default=5, ge=1)
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


class MLPConfig(BaseConfig):
    """The sections the suite reads from ``ModelHyperparamsConfig.mlp_kwargs`` on its regular (non-LTR) path, validated strictly.

    ``mlp_kwargs`` is a nested dict; a misspelled section name (``trainer_param``) used to be ignored silently. ``validate_nested_mlp_kwargs``
    builds this model from it, so an unknown top-level key, a section that is not a dict, or an unsupported ``float32_matmul_precision`` raises
    before any model is built. The sections themselves stay plain dicts: Lightning, the DataLoader and the SWA callback own their parameters.
    Every default is what the suite does when the key is absent (``use_swa`` off, no explicit matmul precision).

    Parameters
    ----------
    model_params : dict, optional
        Overrides of the MLP module's initialization parameters (hidden sizes, optimizer, learning rate, ...).
    network_params : dict, optional
        Overrides of the network architecture (``nlayers``, neuron sizes, layer norm, ...).
    trainer_params : dict, optional
        PyTorch Lightning ``Trainer`` parameters (``max_epochs``, ``precision``, ``max_time``, ...).
    dataloader_params : dict, optional
        DataLoader parameters (``batch_size``, ``num_workers``, ...).
    datamodule_params : dict, optional
        DataModule parameters.
    use_swa : bool
        Stochastic Weight Averaging (default: off).
    swa_params : dict, optional
        SWA callback parameters (``swa_lrs``, ``swa_epoch_start``, ``annealing_epochs``).
    tune_params : bool
        Tune hyperparameters before fitting (default: off).
    float32_matmul_precision : str, optional
        ``"high"``, ``"medium"`` or ``"highest"`` (case-insensitive); ``None`` leaves the torch default.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model_params: Optional[Dict[str, Any]] = None
    network_params: Optional[Dict[str, Any]] = None
    trainer_params: Optional[Dict[str, Any]] = None
    dataloader_params: Optional[Dict[str, Any]] = None
    datamodule_params: Optional[Dict[str, Any]] = None

    use_swa: bool = False
    swa_params: Optional[Dict[str, Any]] = None
    tune_params: bool = False
    float32_matmul_precision: Optional[str] = None

    @field_validator("float32_matmul_precision", mode="before")
    @classmethod
    def normalize_precision(cls, v: Any) -> Any:
        """Normalize float32_matmul_precision to lowercase and validate (``None`` passes through)."""
        if v is None:
            return None
        v_lower = str(v).lower()
        if v_lower not in VALID_MATMUL_PRECISIONS:
            raise ValueError(f"float32_matmul_precision must be one of {VALID_MATMUL_PRECISIONS}, got '{v}'")
        return v_lower


def validate_nested_mlp_kwargs(mlp_kwargs: Optional[Dict[str, Any]]) -> None:
    """Raise when ``mlp_kwargs`` (the nested form the regular suite reads) has an unknown section or a malformed value; ``None`` / ``{}`` pass.

    Not applied to the learning-to-rank path, where the same field is a flat set of ``MLPRanker`` constructor arguments.
    """
    if mlp_kwargs:
        MLPConfig(**mlp_kwargs)


class AutoMLConfig(InertFieldsWarningMixin, BaseConfig):
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
        Maximum training time in seconds, passed to AutoGluon as ``fit(time_limit=...)`` and to LightAutoML as
        ``TabularAutoML(timeout=...)``; a ``time_limit`` / ``timeout`` key in the per-library params dict takes precedence.
    """

    # Accepted for back-compat, read by nothing: a non-default value warns instead of silently doing nothing.
    INERT_FIELDS: ClassVar[dict[str, str]] = {
        "automl_show_fi": "the AutoML branch reads FeatureSelectionConfig.show_fi",
    }

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
    time_limit: Optional[int] = Field(default=None, ge=1)  # seconds; forwarded to both libraries (see _with_time_budget)


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
        Patience for early stopping, >= 1. None disables early stopping entirely. 0 is rejected rather than meaning
        "auto": give the patience explicitly.
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

    # Knobs forwarded to ``get_training_configs`` that used to be accepted as undeclared extras. ``None`` means "not set": the suite drops
    # None-valued fields (``model_dump(exclude_none=True)``), so ``get_training_configs`` keeps its own default; a test pins the two together.
    # Integral-calibration-error weights of the early-stopping metric (see metrics.integral_calibration_error_from_metrics):
    method: Optional[str] = None
    mae_weight: Optional[float] = None
    std_weight: Optional[float] = None
    roc_auc_weight: Optional[float] = None
    pr_auc_weight: Optional[float] = None
    brier_loss_weight: Optional[float] = None
    min_roc_auc: Optional[float] = None
    roc_auc_penalty: Optional[float] = None
    use_weighted_calibration: Optional[bool] = None
    weight_by_class_npositives: Optional[bool] = None
    nbins: Optional[int] = Field(default=None, ge=1)
    # Robustness term of the early-stopping metric (0 time splits = disabled):
    robustness_num_ts_splits: Optional[int] = Field(default=None, ge=0)
    robustness_std_coeff: Optional[float] = None
    robustness_greater_is_better: Optional[bool] = None
    # Early-stopping infrastructure and run-level knobs:
    validation_fraction: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    use_explicit_early_stopping: Optional[bool] = None
    random_seed: Optional[int] = None
    verbose: Optional[int] = None
    catboost_custom_regr_metrics: Optional[List[str]] = None

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
from ._model_configs_behavior import (  # noqa: F401
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
from ._model_configs_ensembling import EnsemblingConfig  # noqa: F401
