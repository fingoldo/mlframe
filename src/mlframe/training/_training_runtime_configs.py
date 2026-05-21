"""Training-runtime + IO configs for ``mlframe.training.configs``.

Split out from ``configs.py`` to keep that file below the 1k-line monolith
threshold. Behaviour preserved bit-for-bit; every class is re-exported from
``configs`` so existing
``from mlframe.training.configs import TrainingConfig`` (and the other
moved names) imports continue to resolve.

What lives here:
  - ``TrainingConfig`` (top-level aggregate)
  - ``DataConfig``, ``TrainingControlConfig``, ``MetricsConfig``
  - ``FeatureImportanceConfig``, ``OutputConfig``, ``OutlierDetectionConfig``
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import Field, field_validator, model_validator

from ._configs_base import BaseConfig, DEFAULT_CALIBRATION_BINS, VALID_MODEL_TYPES
from ._feature_selection_config import FeatureSelectionConfig
from ._preprocessing_configs import (
    PreprocessingConfig,
    TrainingSplitConfig,
    PreprocessingBackendConfig,
)
from ._model_configs import (
    LinearModelConfig,
    TreeModelConfig,
    MLPConfig,
    NGBConfig,
    ModelHyperparamsConfig,
    TrainingBehaviorConfig,
)
from ._composite_target_discovery_config import CompositeTargetDiscoveryConfig
# ``ReportingConfig`` / ``PredictionsContainer`` are referenced only in
# docstrings here and would create a circular import (_reporting_configs imports
# FeatureImportanceConfig from this module). Do NOT add a top-level import for
# them.


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
    pipeline : PreprocessingBackendConfig
        Preprocessing backend settings (polars-ds vs sklearn dispatch + scaler/imputer/encoder).
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
    hyperparams : ModelHyperparamsConfig
        Model hyperparameters (iterations, learning rate, per-model kwargs).
    behavior : TrainingBehaviorConfig
        Training behavior flags (GPU preference, calibration, fairness).
    verbose : int
        Verbosity level (default: 1).
    metamodel_func : Callable, optional
        Function to wrap models (e.g., for target transformation).
        Note: ``imputer`` / ``scaler`` / ``category_encoder`` overrides moved to
        ``PreprocessingConfig`` in 2026-04-27 (the dict-typed pass-through that
        previously held them was deleted).

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
    pipeline: PreprocessingBackendConfig = Field(default_factory=PreprocessingBackendConfig)
    feature_selection: FeatureSelectionConfig = Field(default_factory=FeatureSelectionConfig)

    # Model-specific configs (can be overridden per model)
    linear_config: Optional[LinearModelConfig] = None
    tree_config: Optional[TreeModelConfig] = None
    mlp_config: Optional[MLPConfig] = None
    ngb_config: Optional[NGBConfig] = None

    # Directory paths
    data_dir: str = ""
    models_dir: str = "models"

    # Typed configuration objects
    hyperparams: ModelHyperparamsConfig = Field(default_factory=ModelHyperparamsConfig)
    behavior: TrainingBehaviorConfig = Field(default_factory=TrainingBehaviorConfig)

    # Misc
    verbose: int = 1

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

    # 2026-05-10: target_type for downstream-correct chart dispatch.
    # When set (caller knows the target_type), gates
    # ``render_multi_target_panels`` to fire ONLY the matching branch
    # (regression suppresses LTR / multilabel / multiclass panels;
    # learning_to_rank suppresses regression-specific panels; etc.).
    # Default None preserves shape-based heuristic behavior for
    # back-compat callers.
    target_type: Optional[str] = None

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
    skip_preprocessing : bool
        Whether to skip only preprocessing (scaler/imputer/encoder) while still
        running feature selectors (default: False). Used when polars-ds pipeline
        was already applied globally.
    fit_params : dict, optional
        Additional parameters passed to model.fit(). Keys depend on model type.
    callback_params : dict, optional
        Parameters for training callbacks. Keys: patience, verbose.
    """

    verbose: Union[bool, int] = False
    # 2026-04-27: default flipped False -> True. Cache loading is almost
    # always faster than retraining; the previous False default was
    # inconsistent with train_eval.py:664 which already read the
    # internal common_params dict with .get("use_cache", True). Making
    # both ends agree on True; users who want force-retrain pass False
    # explicitly via TrainingControlConfig (suite-level wiring deferred -
    # remains internal-only).
    use_cache: bool = True
    just_evaluate: bool = False

    # Metrics computation flags
    compute_trainset_metrics: bool = False
    compute_valset_metrics: bool = True
    compute_testset_metrics: bool = True

    # Pipeline
    pre_pipeline: Optional[Any] = None  # sklearn TransformerMixin
    skip_pre_pipeline_transform: bool = False
    skip_preprocessing: bool = False  # Skip only preprocessing, still run feature selectors
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


class FeatureImportanceConfig(BaseConfig):
    """Configuration for feature-importance plots.

    Replaces the dict-typed ``fi_kwargs`` that previously lived on
    pre-2026-04-27 ``ReportingConfig`` (it was a separate dict field then) and
    was reachable from the suite layer only via the deleted dict-typed
    pass-through. Fields mirror the kwargs of
    ``mlframe.training.evaluation.plot_model_feature_importances``.
    """

    # 2026-05-12: default 40 -> 10. Plots/log lines become readable on
    # the common feature counts (10-50) without horizontal scroll, and
    # the user can still bump it via FeatureImportanceConfig(num_factors=...)
    # when they want a wider view.
    num_factors: int = 10
    # 2026-05-13 (user request): default figsize reduced to half the
    # 3-panel regression diagnostic chart (DEFAULT_FIGSIZE=(15, 5)). The
    # previous unified (15, 5) FI plot still dominated suite reports.
    figsize: Tuple[float, float] = (7.5, 2.5)
    positive_fi_only: bool = False
    show_plots: bool = True
    # 2026-05-12 (user request): cap zero-FI bars so the chart stays
    # compact when most features were pruned by the model (eg an XGB on a
    # residual target where ``TVT_prev=0.99`` and everything else is 0).
    # Shows AT MOST this many bars with |FI| ~ 0 in the magnitude-ranked
    # plot; non-zero bars always render in full.
    max_zero_fi_to_plot: int = 4


class OutputConfig(BaseConfig):
    """Filesystem destinations for saved artifacts.

    Holds path/output knobs that previously lived on the pre-2026-04-27
    ReportingConfig (``plot_file``) or as top-level kwargs of ``train_mlframe_models_suite``
    (``data_dir``, ``models_dir``, ``save_charts``). Pulled out so
    ``ReportingConfig`` covers only "look of the report" and not "where
    to write things".

    ``models_dir`` was previously named ``models_subdir`` on the report config
    and ``models_dir`` at the suite level. Renamed to ``models_dir`` for
    symmetry with ``data_dir`` (both are typed peers of the same noun
    pattern).
    """

    # Optional[str] (not bare str) so callers can pass None to disable saving:
    # `data_dir=None` -> no charts/artifacts; `models_dir=None` -> no model
    # files. Falsy paths short-circuit the save_split_artifacts /
    # _setup_model_directories branches at training-loop level. Was the
    # established suite-level semantics for the deleted top-level kwargs;
    # preserved here.
    data_dir: Optional[str] = ""
    models_dir: Optional[str] = "models"
    plot_file: Optional[str] = ""
    save_charts: bool = True

    @model_validator(mode="after")
    def _check_save_charts_has_destination(self):
        """Raise when the user EXPLICITLY set save_charts=True but left data_dir empty.

        Pre-2026-05-20 the save branch silently short-circuited on falsy data_dir
        (per the comment at L2040-2044), so every chart the suite rendered got dropped
        on the floor. An operator who explicitly opted into save_charts but forgot to
        configure data_dir saw no saved artifacts and no log line.

        Fires when save_charts=True was explicitly passed and data_dir is empty AND
        was NOT itself explicitly passed by the caller. A caller that supplies BOTH
        save_charts and data_dir (including data_dir="" via model_dump round-trip)
        has made a deliberate choice and is not the misconfiguration we want to catch.
        """
        if (
            "save_charts" in self.model_fields_set
            and self.save_charts
            and not self.data_dir
            and "data_dir" not in self.model_fields_set
        ):
            raise ValueError(
                "OutputConfig: save_charts=True was explicitly set but data_dir is empty; "
                f"got data_dir={self.data_dir!r}. The save branch silently short-circuits "
                "on falsy data_dir and every chart would be dropped on the floor. Either "
                "set data_dir='<path>' or drop the save_charts=True override to fall back "
                "to the default-no-save behaviour."
            )
        return self


class OutlierDetectionConfig(BaseConfig):
    """Configuration for the once-per-suite outlier-detection pass.

    ``apply_to_val`` was previously named ``od_val_set`` at the suite
    level - renamed for clarity.
    """

    detector: Optional[Any] = None  # sklearn OutlierMixin or None
    apply_to_val: bool = True
