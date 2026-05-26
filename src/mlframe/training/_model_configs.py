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

from typing import Any, Callable, ClassVar, Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Union

from pydantic import Field, field_validator, model_validator

from ._configs_base import BaseConfig, DEFAULT_CALIBRATION_BINS, DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH, DEFAULT_RANDOM_SEED, DEFAULT_RFECV_CV_SPLITS, DEFAULT_RFECV_MAX_NOIMPROVING_ITERS, DEFAULT_RFECV_MAX_RUNTIME_MINS, DEFAULT_TREE_ITERATIONS, VALID_LINEAR_MODEL_TYPES, VALID_MATMUL_PRECISIONS, VALID_TASK_TYPES


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

    iterations: int = DEFAULT_TREE_ITERATIONS
    learning_rate: float = 0.1
    max_depth: Optional[int] = None
    early_stopping_rounds: Optional[int] = 0  # 0 = auto (iterations // 3); None = disabled

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


class TrainingBehaviorConfig(BaseConfig):
    """Training behavior flags and control settings.

    Replaces the legacy untyped ``control_params`` / ``control_params_override`` dicts.
    Controls *how* training runs (GPU, calibration, fairness, verbosity) rather than
    model hyperparameters.

    Parameters
    ----------
    prefer_gpu_configs : bool
        Whether to prefer GPU model configurations.
    prefer_cpu_for_lightgbm : bool
        Force LightGBM to CPU even when GPU is available.
    prefer_calibrated_classifiers : bool
        Use calibrated classifier variants (CalibratedClassifierCV wrappers).
    use_robust_eval_metric : bool
        Use robust evaluation metrics.
    nbins : int
        Number of bins for calibration reports.
    xgboost_verbose : int
        Verbosity level for XGBoost training.
    rfecv_model_verbose : int
        Verbosity level for RFECV models.
    fairness_features : list of str, optional
        Feature names for fairness analysis.
    fairness_min_pop_cat_thresh : int
        Minimum population per category for fairness analysis.
    metamodel_func : Callable, optional
        Function to wrap models (e.g., for target transformation).
    default_classification_scoring : dict, optional
        Custom classification scoring configuration.
    default_regression_scoring : dict, optional
        Custom regression scoring configuration.
    callback_params : dict, optional
        Parameters for training callbacks (patience, verbose).
    prefer_cpu_for_xgboost : bool
        Force XGBoost to CPU even when GPU is available.
    cont_nbins : int
        Number of bins for continuous features in fairness subgroups.
    cb_fit_params : dict, optional
        Extra kwargs passed to CatBoost .fit() (e.g. early_stopping_rounds, custom callbacks).
    use_flaml_zeroshot : bool
        Use FLAML zero-shot models for XGBoost/LightGBM.
    enable_crash_reporting : bool
        Default True. At suite start, enable faulthandler (SIGSEGV /
        SIGABRT -> Python traceback) and on Windows suppress the
        "Python has stopped working" WER popup so Jupyter kernels exit
        cleanly instead of hanging. No-op if already enabled in the
        process.
    continue_on_model_failure : bool
        If True, catch exceptions from individual per-model training
        (e.g. XGBoost ``bad_malloc`` on too-large frames) and continue
        the suite with the next model/weighting instead of aborting
        the whole run. Crashes that kill the process at the OS level
        (access violation in a worker thread that faulthandler can't
        catch) will still terminate -- for true isolation use subprocess
        training, which this flag does NOT provide.
    """

    prefer_gpu_configs: bool = True
    prefer_cpu_for_lightgbm: bool = True
    prefer_cpu_for_xgboost: bool = False
    prefer_calibrated_classifiers: bool = True
    use_robust_eval_metric: bool = True
    nbins: int = DEFAULT_CALIBRATION_BINS
    xgboost_verbose: int = 0
    rfecv_model_verbose: int = 0
    fairness_features: Optional[List[str]] = None
    fairness_min_pop_cat_thresh: int = DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH
    cont_nbins: int = 6
    metamodel_func: Optional[Callable] = None
    default_classification_scoring: Optional[Dict[str, Any]] = None
    default_regression_scoring: Optional[Dict[str, Any]] = None
    callback_params: Optional[Dict[str, Any]] = None
    cb_fit_params: Optional[Dict[str, Any]] = None
    use_flaml_zeroshot: bool = False
    # Default True: faulthandler + Windows WER suppression are pure
    # diagnostics -- they don't change training behavior, only replace
    # the "Python has stopped working" modal with a Python traceback.
    # Users who rely on the WER popup (rare) can opt out.
    enable_crash_reporting: bool = True
    # Default False: silently skipping a failed model is a semantic
    # shift that users must opt into explicitly.
    continue_on_model_failure: bool = False
    # Default False: feature-drift-driven per-target MLP HPT override
    # is OFF by default. The drift sensor still runs and stamps the
    # recommendation into metadata + emits a WARN log line so the
    # operator sees the recommendation, but the MLP config the user
    # passed is NOT mutated. Set True to enable auto-apply (regression
    # only by default; classification requires the shape-detector gate
    # below to also be satisfied).
    #
    # Rationale: the override is a black-box config rewrite and the
    # paired study showed classification doesn't have a clean trigger
    # (Pearson r=-0.101 overall; interaction-rich classification
    # targets are actively hurt by the override). Operators who want
    # the override can opt in after reading the docs; everyone else
    # gets the MLP they configured.
    feature_drift_auto_apply_neural_overrides: bool = False

    # Default True: align Polars Categorical dicts across
    # train/val/test via shared pl.Enum(union_of_categories) before
    # model training. Mechanism not fully understood but empirically
    # prevents a silent process kill on Windows when XGB constructs
    # val IterativeDMatrix with ref=train on large frames (7.3M+ rows,
    # 15+ cat features).
    # Theory: pl.Categorical assigns physical codes per-Series
    # (order-of-first-occurrence), so the same string can have
    # different physical codes in train vs val vs test. XGB's native
    # layer at scale appears to treat val's physical codes as indices
    # into train's bin structure without re-reading the dict,
    # corrupting memory. pl.Enum(list) enforces a shared dict
    # by construction so physical codes are consistent across splits.
    # Disable to reproduce the pre-fix behavior or if the alignment
    # cost (O(n_rows) per cat column) is prohibitive.
    align_polars_categorical_dicts: bool = True

    # Silencing knobs for verbose report blocks.
    #
    # ``report_residual_audit``: when False, ``report_model_perf`` skips the multi-line residual-audit footer (moments / shape / hetero / hypothesis / suggested-loss block). Default True (informative for regression diagnostics); set False on production runs where the block adds 6-8 noisy lines per (model x split).
    #
    # ``confidence_ensemble_quantile``: top-quantile of MOST-CONFIDENT rows used by the "Conf Ensemble" flavors. Default 0.1 (= top 10%); set 0.0 to disable Conf Ensembles entirely (saves ~6 flavor x 2 split = 12 log blocks + their charts per ensemble pass). The raw ensemble metrics still print - only the confidence-subset variant is suppressed.
    report_residual_audit: bool = True
    confidence_ensemble_quantile: float = 0.1

    # When True (default), the simple-ensembling blends (arithm / harm / quad / qube / geo / median) consume AP12-calibrated probs stamped by ``post_calibrate_model`` (``member.calibrated_val_probs`` / ``calibrated_test_probs``) instead of raw ``member.val_probs`` / ``test_probs``. This dampens the heterogeneous-scale dominance bug flagged by ensembling-critique A3#3 (well-calibrated tree probs in [0.1, 0.9] dominated by raw sigmoid in [0.005, 0.01] under arithmetic mean). When False, every blend uses raw probs (legacy pre-W16D behaviour). RRF is rank-based and is unaffected either way (scale-invariant). Members without the AP12 stamp transparently fall back to raw probs -- the knob never raises on missing calibration.
    use_ap12_calibrated_probs_in_ensemble: bool = True

    # Pre-pipeline LRU bound. Default 4 covers the common Linear+MLP+RFECV+catboost suite without thrashing; long-running services with bigger model rosters can bump this without monkey-patching the module global.
    pre_pipeline_cache_max: int = 4

    # Fix 8 (2026-04-21): append a per-model input-schema fingerprint
    # (``__sch_<10 hex>``) to model filenames so two runs with different
    # feature-type configs (text vs cat promotion, encoding, alignment)
    # don't silently overwrite each other. Default True. Set False to
    # restore the pre-2026-04-21 naming scheme (``{model}_{weight}.dump``);
    # load-time schema verification is also skipped for those artefacts.
    model_file_hash_suffix: bool = True

    # 2026-04-26 Session 7: temporal target audit. When set, per-target
    # the suite computes a time-series view of the target (P(y=1) for
    # binary, mean(y) for regression) at the configured granularity,
    # detects change points / regime shifts, and warns when the rate
    # diverges across segments. Saves a chart to the per-target charts
    # folder. Skipped silently when the timestamp column is absent or
    # not datetime-typed.
    target_temporal_audit_column: Optional[str] = None
    """Column name (datetime-typed) used as the time axis for the
    per-target temporal audit. ``None`` (default) disables the audit.
    Set to e.g. ``'job_posted_at'`` to enable."""

    target_temporal_audit_granularity: str = "auto"
    """One of ``"auto"`` (default; picks granularity that yields 30-50
    bins) or one of ``"minute"`` / ``"hour"`` / ``"day"`` / ``"week"`` /
    ``"month"`` / ``"quarter"`` / ``"year"``."""

    target_temporal_audit_save_plot: bool = True
    """Save the time-series chart to the per-target charts folder."""

    # Extreme-AR + group-aware MLP skip. When set, skips the MLP fit
    # on targets where lag1_corr >= mlp_extreme_ar_threshold AND the
    # split is group-aware. Default FALSE: turning off MLP is a poor
    # solution; the user has asked the framework to make the MLP
    # ACTUALLY WORK on this regime, not silently skip. The defensive
    # protections that DO ship by default (and bound the damage when
    # MLP is allowed to train):
    #   * ``_TTRWithEvalSetScaling.predict`` clips inverse-transformed
    #     y_hat to [y_train_min - 3*std, y_train_max + 3*std]. Bounds
    #     the catastrophic blow-up; the model still learns badly but
    #     predictions stay within ~3 sigma of train range.
    #   * Ensemble dummy-floor gate drops MLP from the blend if its
    #     OOF RMSE exceeds the strongest dummy. So even a bad MLP
    #     doesn't poison the final ensemble RMSE.
    # Substantive fix paths (see ticket TBD) that this knob does NOT
    # address:
    #   * Residual-target MLP (train on y - alpha*lag, predict residual).
    #   * Output activation bounding (tanh-scaled to train target range).
    #   * Drop per-group aggregate features from MLP input set
    #     (group-level features extrapolate on unseen groups).
    mlp_extreme_ar_group_aware_skip: bool = False
    mlp_extreme_ar_threshold: float = 0.99

    # Drop per-group AGGREGATE features from the MLP's view of
    # X. Pattern matches columns like
    # ``group_<feature>_mean`` / ``group_<feature>_std`` / ``group_*_(mean|std|min|max)``:
    # these encode the train-only group mean of some other feature and
    # are CONSTANT within a group. On an unseen-group test row the
    # value is necessarily extrapolated (the test group never appears
    # in the train aggregate) and the MLP, which composes to a near-
    # affine map on whitened inputs, picks up the resulting train-vs-
    # test direction as the dominant signal -> catastrophic rank
    # inversion on unseen groups.
    # TREE models still see these features (they handle the OOD
    # categorical signal via leaves, not via affine slope). Only the
    # MLP fit-path drops them. Default False (opt-in): enable when the
    # calling project ships per-group aggregate columns matching the
    # pattern; benign no-op otherwise.
    mlp_drop_per_group_constants: bool = False
    # Regex pattern. Match is case-INSENSITIVE on column names. Default
    # captures a generic ``group_*_<reducer>`` naming; tweak for the
    # calling project's aggregate convention (e.g. ``well_.*_(mean|std)``
    # or ``rig_.*_(mean|std)``).
    mlp_drop_per_group_constants_pattern: str = r"^group_.*_(mean|std|min|max)$"

    # L2 weight-decay auto-bump for MLP on extreme-AR + group-aware
    # regimes (Fix 3, 2026-05-26). When the trigger fires
    # (lag1_corr_per_group >= mlp_extreme_ar_threshold AND the active
    # split sets ``prefer_group_aware=True``), multiply the MLP
    # optimizer's ``weight_decay`` by this factor. AdamW is forced ON
    # (Adam ignores weight_decay) and weight_decay defaults are bumped
    # from 0.0 -> base * factor. Heavier L2 bounds the effective slope
    # of the MLP's affine composition, capping extrapolation magnitude
    # on unseen-group test rows.
    mlp_extreme_ar_weight_decay_factor: float = 100.0
    # Base weight_decay when bumping. The default (1e-4) * factor=100
    # produces 1e-2 -- the upper end of "moderate" L2 for tabular
    # MLPs. Override for very high-noise regimes.
    mlp_extreme_ar_weight_decay_base: float = 1e-4


class MultilabelDispatchConfig(BaseConfig):
    """Configuration for multilabel-classification dispatch.

    Bundles every multilabel-only knob so per-strategy code only sees one
    parameter (instead of an exploding ``_maybe_wrap_multilabel(...)``
    signature) and so adding a new strategy choice (e.g. ``stacking``)
    doesn't require touching every dispatch site.

    Only consulted when ``target_type == MULTILABEL_CLASSIFICATION``.

    Strategy choices
    ----------------
    auto      : let the strategy pick -- CatBoost uses native MultiLogloss,
                everyone else uses ``MultiOutputClassifier(estimator)`` (OvR)
    wrapper   : force ``MultiOutputClassifier(estimator)`` even on CB
                (degrades CB native to OvR -- useful for A/B vs native)
    chain     : ``_ChainEnsemble`` of ``n_chains`` random-ordered
                ``ClassifierChain(estimator, cv=cv)`` instances; averages
                ``predict_proba`` outputs. Empirically +2-5% Jaccard on
                correlated labels (sklearn ``plot_classifier_chain_yeast``).
    native    : assert strategy supports native multilabel; raise if not.
                For users who explicitly want CB MultiLogloss and want to
                fail loud if mis-configured.
    """

    strategy: str = "auto"  # Literal["auto","wrapper","chain","native"]
    n_chains: int = 3
    chain_order_strategy: str = "random"  # Literal["random","by_frequency","user"]
    chain_order_user: Optional[List[List[int]]] = None  # one ordering per chain
    chain_seeds: Optional[List[int]] = None
    cv: Optional[int] = 5  # ClassifierChain.cv -- 5 cross-validates chain features
    per_label_thresholds: Optional[List[float]] = None  # decision-rule thresholds
    wrapper_n_jobs: Union[int, str] = "auto"  # MultiOutputClassifier n_jobs
    allow_uncalibrated_multi: bool = False  # downgrade post-hoc calib skip from raise to warn
    # 2026-04-24 Session-2: opt-in for native XGB multilabel (multi_strategy=
    # 'multi_output_tree' + objective='binary:logistic'). XGB 3.x ships this
    # as experimental -- vector-output trees share structure across labels
    # (smaller model, integrated GPU/SHAP, faster inference). Marked WIP
    # by upstream until v3.1; default False uses MultiOutputClassifier
    # wrapper. Set True to opt in (only takes effect with strategy='native'
    # or 'auto' + XGBoostStrategy with the flag set). Combined with
    # XGBoostStrategy.supports_native_multilabel which is gated on this
    # flag at runtime.
    force_native_xgb_multilabel: bool = False

    @model_validator(mode="after")
    def _check_chain_strategy_invariants(self):
        """Validate strategy choice + chain_order_user shape.

        Pre-2026-05-20 a typo ``strategy="wrappr"`` was silently accepted (no
        Literal validation on the string), and ``chain_order_strategy="user"``
        with missing chain_order_user was accepted as well -- ClassifierChain
        silently fell back to a default ordering, the operator's hand-crafted
        order was ignored with no log line.
        """
        _STRATEGY = {"auto", "wrapper", "chain", "native"}
        _ORDER = {"random", "by_frequency", "user"}
        if self.strategy not in _STRATEGY:
            raise ValueError(
                f"MultilabelDispatchConfig.strategy={self.strategy!r} not in {sorted(_STRATEGY)}"
            )
        if self.chain_order_strategy not in _ORDER:
            raise ValueError(
                f"MultilabelDispatchConfig.chain_order_strategy={self.chain_order_strategy!r} "
                f"not in {sorted(_ORDER)}"
            )
        if self.chain_order_strategy == "user":
            if self.chain_order_user is None:
                raise ValueError(
                    "MultilabelDispatchConfig: chain_order_strategy='user' but "
                    "chain_order_user is None. Either supply chain_order_user=[[...], ...] "
                    "with one ordering per chain, or pick chain_order_strategy='random' / "
                    "'by_frequency'."
                )
            if len(self.chain_order_user) != self.n_chains:
                raise ValueError(
                    f"MultilabelDispatchConfig: chain_order_user has {len(self.chain_order_user)} "
                    f"orderings but n_chains={self.n_chains}. Sizes must match."
                )
        return self


class LearningToRankConfig(BaseConfig):
    """Configuration for ``LEARNING_TO_RANK`` target dispatch.

    Holds knobs that are LTR-only so per-strategy ranking code sees one
    parameter (mirrors ``MultilabelDispatchConfig`` for multilabel).

    Library defaults verified empirically on the installed stack
    (CatBoost 1.2.10, XGBoost 3.x, LightGBM 4.6.0):

    - **CB** ``YetiRankPairwise`` is the listwise default; alternatives
      via ``cb_loss_fn``: ``YetiRank``, ``QuerySoftMax``, ``PairLogit``,
      ``PairLogitPairwise``, ``StochasticRank:metric=NDCG``.
    - **XGB** ``rank:ndcg`` works on graded relevance. ``rank:map``
      requires binary y (``is_binary`` C++ check). The dispatcher
      auto-falls-back to ``rank:ndcg`` with WARN if y.max()>1 even when
      user pinned ``rank:map``.
    - **LGB** ``lambdarank`` (default) or ``rank_xendcg``.

    Ensemble: RRF default (TREC standard, scale-invariant — survives
    softmax/sigmoid/raw-score divergence across CB/XGB/LGB). Borda is a
    simpler scale-invariant alternative; ``score_mean`` requires the
    user to assert ``assume_comparable_scales=True``.
    """

    cb_loss_fn: str = "YetiRankPairwise"
    """CatBoost loss_function for the ranker. Listwise pairwise default."""

    xgb_objective: str = "rank:ndcg"
    """XGBoost objective. ``rank:map`` is rejected at fit-time when
    ``y.max() > 1`` -- use ``rank:ndcg`` for graded relevance."""

    lgb_objective: str = "lambdarank"
    """LightGBM objective. ``lambdarank`` is robust on both binary and
    graded labels; ``rank_xendcg`` is an alternative."""

    mlp_loss_fn: str = "ranknet"
    """MLPRanker loss. ``ranknet`` (default; pairwise BCE on score
    differences, Burges 2005) or ``listnet`` (listwise softmax
    cross-entropy, Cao 2007). Both handle binary + graded relevance."""

    eval_at: tuple = (1, 5, 10)
    """Cutoffs for NDCG@k / MAP@k metrics. Mirrors LightGBM ``eval_at``."""

    ensemble_method: str = "rrf"
    """Ensembling method for combining ranker scores. ``rrf`` (Reciprocal
    Rank Fusion, TREC default) is invariant to monotone score transforms
    -- safe for cross-library blends. ``borda`` per-query rank averaging.
    ``score_mean`` requires comparable scales (asserted via
    ``assume_comparable_scales``)."""

    ltr_ensemble_method: Literal["rrf", "borda"] = "rrf"
    """Typed rank-fusion choice for LTR ensembling: ``rrf`` (scale-invariant,
    TREC default) or ``borda`` (per-query rank averaging, simpler and also
    scale-invariant but underweights long lists). Distinct from
    ``ensemble_method`` because this field is restricted to the two
    rank-fusion strategies that survive cross-library score-scale divergence
    without external calibration; ``score_mean`` is intentionally excluded
    here (use ``ensemble_method=score_mean`` with ``assume_comparable_scales``
    if you have calibrated scores)."""

    rrf_k: int = 60
    """RRF damping constant. 60 is the TREC default. Larger ``k`` flattens
    the position weight; smaller emphasises top-1."""

    assume_comparable_scales: bool = False
    """When True, ``ensemble_method=score_mean`` is allowed without warn.
    Set this only after externally calibrating model scores onto a
    comparable scale (e.g. via Platt / isotonic per-model)."""

    autodetect_label_format: bool = True
    """When True, dispatcher inspects ``y`` at fit-time:
    ``y.max() > 1`` -> graded (force XGB to ``rank:ndcg``);
    ``y in {0,1}`` -> binary (XGB ``rank:map`` allowed). When False,
    pass user-pinned objectives through unchanged (will crash on
    mismatched format -- caller takes responsibility)."""


class QuantileRegressionConfig(BaseConfig):
    """Configuration for ``QUANTILE_REGRESSION`` target dispatch.

    Holds quantile-regression-specific knobs: which alphas to predict,
    crossing-fix strategy, point-estimate alpha, coverage pairs for
    interval reports.

    Library support matrix (verified 2026-05-08 against installed stack
    CB 1.2.10 / XGB 3.x / LGB 4.6 / sklearn 1.7+):

    - **CatBoost** ``loss_function="MultiQuantile:alpha=0.1,0.5,0.9"``
      single fit, returns (N, K).
    - **XGBoost >=2.0** ``objective="reg:quantileerror",
      quantile_alpha=[0.1,0.5,0.9]`` single fit, returns (N, K).
    - **LightGBM** ``objective="quantile", alpha=0.5`` -- scalar only;
      multi-quantile via K independent fits stacked
      (_QuantileMultiOutputWrapper).
    - **HGB** ``loss="quantile", quantile=0.5`` -- scalar only; same
      wrapper path.
    - **Linear** ``QuantileRegressor(quantile=0.5)`` -- scalar only;
      same wrapper path. Slow on n>100K (LP solver O(n^2)).
    - **MLP / Recurrent** K-output head + summed pinball loss; single
      fit, returns (N, K).
    """

    alphas: tuple = (0.1, 0.5, 0.9)
    """Quantile levels to predict. Must be sorted ascending and all
    strictly between 0 and 1. Default targets the 10/50/90 percentiles
    (80% prediction interval + median)."""

    crossing_fix: str = "sort"
    """Post-prediction crossing-fix strategy:
    - ``sort``: ``np.sort(preds, axis=1)`` -- cheap, idempotent, default
    - ``isotonic``: per-row IsotonicRegression(increasing=True) -- more
      accurate when crossings are frequent, slower
    - ``none``: leave predictions unchanged (caller handles crossings)
    No library natively enforces non-crossing; even CB MultiQuantile and
    XGB quantile_alpha=[...] can produce crossings on rare configurations.
    """

    point_estimate_alpha: float = 0.5
    """Which alpha to use as the point-prediction (for downstream
    consumers that need a single y_hat). Must be present in ``alphas``
    -- default 0.5 (median). Mean-of-alphas is the alternative if
    user picks an alpha not in the set; validator enforces membership.
    """

    coverage_pairs: tuple = ((0.1, 0.9),)
    """List of (alpha_low, alpha_high) pairs for interval-coverage
    reporting. Each pair must be present in ``alphas`` and lo < hi.
    Default reports the (0.1, 0.9) -> nominal-80% interval.
    """

    wrapper_n_jobs: Any = "auto"
    """Joblib n_jobs for ``_QuantileMultiOutputWrapper`` (LGB / HGB /
    Linear paths). ``"auto"`` -> ``min(K, os.cpu_count() // 2)`` to
    avoid nested-parallelism thrashing when the inner estimator has
    its own thread pool. Set to 1 to serialise."""

    @model_validator(mode="after")
    def _validate_alphas(self) -> "QuantileRegressionConfig":
        alphas = self.alphas
        if not alphas:
            raise ValueError("QuantileRegressionConfig.alphas must be non-empty.")
        if any(not (0.0 < a < 1.0) for a in alphas):
            raise ValueError(
                f"QuantileRegressionConfig.alphas must be in (0, 1) "
                f"strict; got {list(alphas)}"
            )
        if list(alphas) != sorted(alphas):
            raise ValueError(
                f"QuantileRegressionConfig.alphas must be sorted ascending; "
                f"got {list(alphas)}"
            )
        if len(set(alphas)) != len(alphas):
            raise ValueError(
                f"QuantileRegressionConfig.alphas must be unique; "
                f"got {list(alphas)}"
            )
        if self.crossing_fix not in ("sort", "isotonic", "none"):
            raise ValueError(
                f"crossing_fix must be one of sort/isotonic/none; "
                f"got {self.crossing_fix!r}"
            )
        # point_estimate_alpha membership is enforced loosely (closest match)
        # so callers don't need to update both fields in lockstep.
        if self.point_estimate_alpha not in alphas:
            # Find closest alpha (silent snap to nearest grid point).
            closest = min(alphas, key=lambda a: abs(a - self.point_estimate_alpha))
            object.__setattr__(self, "point_estimate_alpha", closest)
        for lo, hi in self.coverage_pairs:
            if lo not in alphas or hi not in alphas:
                raise ValueError(
                    f"coverage_pair ({lo}, {hi}) not in alphas {list(alphas)}"
                )
            if lo >= hi:
                raise ValueError(
                    f"coverage_pair lo={lo} must be < hi={hi}"
                )
        return self


class EnsemblingConfig(BaseConfig):
    """Configuration for ensembling behaviour, including streaming-vs-legacy
    aggregation choice and quantile-fallback budget.

    Replaces the env-var ``ENSEMBLE_FORCE_LEGACY_MATERIALISATION=1`` knob
    (which is invisible in function signatures, untestable, and global)
    with a structured config. Env var is still honoured as the default
    for one release for back-compat.
    """

    force_legacy: bool = False
    """If True, use the pre-streaming materialised-aggregation path
    (allocates ``(M, N, K)`` tensors). Default False uses streaming Welford."""

    quantile_budget_bytes: int = 500 * 1024 * 1024
    """Skip quantile-bucket aggregation with warn when ``M*N*K*8 > budget``.
    500 MB default. Override per environment to taste."""

    accumulator: str = "welford"
    """Streaming accumulator implementation. ``welford`` is the only
    impl today; ``tdigest`` / ``p2_quantile`` planned (registered via
    ``StreamingAccumulator`` Protocol)."""

    flag_degenerate_conf_subset: bool = True
    """If True, prepend ``[DEGENERATE]`` to the Conf Ensemble model_name
    when the confidence-filtered subset's class balance collapses
    (``min(class_support) / max(class_support) < degenerate_class_ratio``).

    Why: a uniform-quantile confidence filter often keeps only the
    rows the ensemble is most confident about, which on imbalanced data
    means "almost-all-positive" or "almost-all-negative" subsets.
    Metrics computed on that subset are deceptively pristine
    (BR=0.026 %, LL=0.002 — observed in one prod log) and easy to
    misread as headlines. The marker is a one-glance hint that the
    block is reporting on a degenerate slice.

    Binary classification only — for regression there is no class
    balance to check; the flag has no effect."""

    degenerate_class_ratio: float = 0.01
    """Threshold below which a confidence-filtered subset is flagged
    as degenerate. ``0.01`` means a class balance worse than 1:100
    (e.g. 21 negatives vs 81 815 positives, observed in one prod log)
    triggers the marker. Has no effect when
    ``flag_degenerate_conf_subset=False``."""


