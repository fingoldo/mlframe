"""
Configuration classes for mlframe training pipeline.

Uses Pydantic for validation while supporting dict-like instantiation for backward compatibility.
All config classes support lenient validation - inputs are normalized to canonical forms.
"""

from typing import Optional, Dict, Any, List, Callable, Tuple, Literal, Union, ClassVar, FrozenSet
from enum import StrEnum

from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator


# =============================================================================
# Preprocessing extensions (Audit #02 phase 3) — single shared pipeline surface.
#
# Wired into `fit_and_transform_pipeline` so every model in the suite reuses one
# transformed frame. A None config preserves the existing polars-native fastpath
# byte-for-byte. See `PreprocessingExtensionsConfig` below for field details.
# =============================================================================


# =============================================================================
# Constants
# =============================================================================

DEFAULT_RANDOM_SEED = 42
"""Default random seed for reproducibility across all operations."""

DEFAULT_TREE_ITERATIONS = 5000
"""Default number of iterations for tree-based models (CB, LGB, XGB)."""

DEFAULT_CALIBRATION_BINS = 10
"""Default number of bins for calibration reports."""

DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH = 1000
"""Default minimum population per category for fairness analysis."""

DEFAULT_RFECV_MAX_RUNTIME_MINS = 180
"""Default RFECV max runtime in minutes (3 hours)."""

DEFAULT_RFECV_CV_SPLITS = 4
"""Default number of CV splits for RFECV."""

DEFAULT_RFECV_MAX_NOIMPROVING_ITERS = 15
"""Default max non-improving iterations for RFECV early stopping."""

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
    MULTICLASS_CLASSIFICATION : str
        K>2 single-label classification (exclusive labels via softmax).
        Target shape is (N,) integer in {0, ..., K-1}.
    MULTILABEL_CLASSIFICATION : str
        K>=1 independent binary outputs (per-label sigmoid).
        Target shape is (N, K) binary matrix.
    """

    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"

    @property
    def is_classification(self) -> bool:
        """True for binary, multiclass, and multilabel; False for regression.

        Use this instead of `target_type == BINARY_CLASSIFICATION` so new
        classification flavours route correctly without touching every
        call site (8 sites previously hardcoded the binary equality check).
        """
        return self in (
            TargetTypes.BINARY_CLASSIFICATION,
            TargetTypes.MULTICLASS_CLASSIFICATION,
            TargetTypes.MULTILABEL_CLASSIFICATION,
        )

    @property
    def is_regression(self) -> bool:
        return self == TargetTypes.REGRESSION

    @property
    def is_binary(self) -> bool:
        return self == TargetTypes.BINARY_CLASSIFICATION

    @property
    def is_multiclass(self) -> bool:
        return self == TargetTypes.MULTICLASS_CLASSIFICATION

    @property
    def is_multilabel(self) -> bool:
        return self == TargetTypes.MULTILABEL_CLASSIFICATION

    @property
    def is_multi_output(self) -> bool:
        """True when probability output is (N, K) with K>2 logically.

        Convenience predicate for ``[:, 1]`` slicing sites that should
        bail out / dispatch differently for multi-* targets.
        """
        return self in (
            TargetTypes.MULTICLASS_CLASSIFICATION,
            TargetTypes.MULTILABEL_CLASSIFICATION,
        )


class BaseConfig(BaseModel):
    """Base configuration class with flexible dict support.

    Uses ``extra="allow"`` so user-supplied kwargs flow through to downstream
    callees (e.g. ``hyperparams_config={"mae_weight": 1.0}`` is not declared
    on ``ModelHyperparamsConfig`` but is consumed by ``get_training_configs``
    via ``**config_params``). Downside: typos like ``iteratoins=100`` get
    silently absorbed. The ``_warn_on_unknown_extras`` validator below issues
    a WARNING so typos are noticed (unless a subclass sets the
    ``_known_extras`` class attribute to list the legitimate extras).
    """

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for flexibility
        arbitrary_types_allowed=True,  # Allow numpy, torch, etc.
        validate_assignment=True,
        protected_namespaces=(),  # Allow model_ prefix for field names
    )

    #: Subclasses may list extra kwargs that are legitimately consumed
    #: downstream (e.g. ``ModelHyperparamsConfig`` -> ICE metric weights
    #: ``mae_weight`` / ``std_weight`` / ...). Entries here do not emit
    #: the "unknown extra" warning. Declared on the subclass like:
    #:     _known_extras: ClassVar[FrozenSet[str]] = frozenset({"mae_weight", ...})
    _known_extras: "ClassVar[FrozenSet[str]]" = frozenset()

    @model_validator(mode="after")
    def _warn_on_unknown_extras(self) -> "BaseConfig":
        """Log a WARNING for each extra field that is not a known pass-through.

        Catches the common typo class (``iteratoins`` for ``iterations``,
        ``prefer_calibrated_classifer`` missing an ``i``, etc.) that
        ``extra="allow"`` otherwise swallows without feedback.
        """
        extras = self.model_extra or {}
        if not extras:
            return self
        known = type(self)._known_extras
        unknown = [k for k in extras if k not in known]
        if unknown:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "%s received unknown field(s) %s — these are accepted (extra='allow') "
                "but NOT declared on the model. If this is a typo for a real field, "
                "the value will have no effect. Known pass-through extras: %s",
                type(self).__name__, sorted(unknown), sorted(known) or "(none declared)",
            )
        return self


class PreprocessingConfig(BaseConfig):
    """Configuration for data preprocessing.

    Strict validation (``extra="forbid"``): this config has a small,
    stable, well-known surface with no pass-through kwargs. Unknown
    fields raise immediately so typos like ``fillna_vlue=0`` surface
    at construction time instead of silently doing nothing.
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_assignment=True,
        protected_namespaces=(),
    )

    fillna_value: Optional[float] = None
    fix_infinities: bool = True
    ensure_float32_dtypes: bool = True
    skip_infinity_checks: bool = True
    drop_columns: Optional[List[str]] = None
    # 2026-04-21: promoted from implicit always-on behaviour to an
    # explicit toggle. Default True preserves the pre-flag behaviour
    # (constant columns dropped during preprocess_dataframe). Set False
    # to keep constant columns — useful for downstream consumers that
    # rely on a fixed column layout across train/val/test splits.
    remove_constant_columns: bool = True
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
    val_placement : {"forward", "backward"}
        Temporal placement of the validation set relative to train/test
        (default: "forward"). "forward" = conventional [train][val][test];
        "backward" = [val][train][test] ("First test then train", Mazzanti
        2024). Backward testing gives a better proxy of deployment error
        under drift but conflicts with recency weighting — see the field
        comments below for the full trade-off analysis.

    Raises
    ------
    ValueError
        If test_size + val_size > 1.0, or if an unknown field is passed
        (strict validation via ``extra="forbid"``).

    Notes
    -----
    Strict validation: this config declares all split-related fields
    explicitly. There is no reason for a pass-through kwarg here, so
    typos (``trainset_agng_limit``, ``wholeday_spliting``) raise at
    construction instead of silently doing nothing.
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_assignment=True,
        protected_namespaces=(),
    )

    test_size: float = Field(default=0.1, ge=0.0, le=1.0)
    val_size: float = Field(default=0.1, ge=0.0, le=1.0)
    shuffle_val: bool = False
    shuffle_test: bool = False
    val_sequential_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    test_sequential_fraction: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # ``None`` = no aging. When set, must be strictly in (0, 1). Previously
    # this field was unvalidated, letting -0.5 / 1.5 / 0 propagate silently
    # to make_train_test_split (which now rejects them too, but the earlier
    # the better).
    trainset_aging_limit: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    wholeday_splitting: bool = True
    random_seed: int = DEFAULT_RANDOM_SEED

    # "First test then train" — Mazzanti 2024 (Medium, 58-dataset benchmark).
    # When ``val_placement="backward"`` with time-indexed data, val is placed
    # BEFORE train on the timeline:
    #
    #   forward  (default):  [ train ] [ val ]   [ test ]   ← conventional
    #   backward          :  [ val   ] [ train ] [ test ]   ← Mazzanti
    #
    # Rationale: in forward-testing the val→train temporal gap is ~0 while
    # the train→prod gap is large (weeks / months), so val-metric is sampled
    # from the "near" edge of the drift trajectory and overstates prod
    # performance. The 2026-04-23 prod log on jobsdetails showed this
    # vividly — VAL ROC AUC 0.999 vs TEST 0.71. Backward-testing mirrors
    # the val→train gap against the train→prod gap, so val-metric is
    # sampled from the same drift-distance regime as deployment — an
    # empirically better proxy (38 % vs 51 % mean deviation over 58
    # datasets in Mazzanti's benchmark).
    #
    # Trade-offs:
    #   - Conflicts with recency weighting (train weighted toward the most
    #     recent rows while validating on the oldest rows is conceptually
    #     inverted). ``core.py`` emits a WARN if a non-uniform weighting
    #     schema is active alongside ``val_placement="backward"``; disable
    #     ``use_recency_weighting`` on the extractor to silence, or accept
    #     the conflict deliberately.
    #   - Early-stopping against a backward val optimizes for "regenerates
    #     to an earlier regime", which approximates "projects into a
    #     future regime" only when drift is roughly symmetric in time. If
    #     past and future regimes differ (post-COVID retail, post-2008
    #     finance), neither forward nor backward val is fully trustworthy.
    val_placement: Literal["forward", "backward"] = "forward"

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
    imputer_strategy : str, optional
        Strategy for imputing missing values: "mean", "median", etc. (default: "mean").
        Pass None to skip imputation.
    categorical_encoding : str, optional
        Encoding for categorical features: "ordinal", "onehot", "target" (default: "ordinal").
        Pass None to skip categorical encoding.
    skip_categorical_encoding : bool
        If True, skip categorical encoding even when ``categorical_encoding`` is set.
        Auto-set by ``train_mlframe_models_suite`` when all requested models handle
        categoricals natively (e.g. CatBoost, XGBoost, HGB on Polars input).
        Default: False.
    robust_q_low : float
        Lower quantile for robust scaling (default: 0.01).
    robust_q_high : float
        Upper quantile for robust scaling (default: 0.99).
    """

    use_polarsds_pipeline: bool = True
    scaler_name: Optional[str] = "standard"
    imputer_strategy: Optional[str] = "mean"
    categorical_encoding: Optional[str] = "ordinal"
    skip_categorical_encoding: bool = False
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


class PreprocessingExtensionsConfig(BaseConfig):
    """Optional shared-pipeline extensions applied once, reused by every model.

    When ``None`` is passed to ``train_mlframe_models_suite``, no extension runs
    and the Polars-native fastpath is preserved. Setting any field here
    activates the sklearn bridge inside ``fit_and_transform_pipeline`` — even
    tree models will then consume the shared transformed frame.

    Order of application (each step is optional):
      1. TF-IDF on declared text columns.
      2. Scaler (overrides the Polars-ds scaler when set).
      3. Binarizer OR KBinsDiscretizer (mutually exclusive).
      4. PolynomialFeatures (guarded by ``memory_safety_max_features``).
      5. Non-linear feature map (RBFSampler / Nystroem / …).
      6. Dim reducer (PCA / UMAP / …).
    """

    scaler: Optional[Literal[
        "StandardScaler", "StandardScaler_nomean",
        "RobustScaler", "MinMaxScaler", "MaxAbsScaler",
        "PowerTransformer_yj", "PowerTransformer_yj_nostd",
        "QuantileTransformer_uniform", "QuantileTransformer_normal",
        "Normalizer_l2",
    ]] = None
    binarization_threshold: Optional[float] = None
    kbins: Optional[int] = None
    kbins_encode: Literal["ordinal", "onehot"] = "ordinal"
    polynomial_degree: Optional[int] = None
    polynomial_interaction_only: bool = True
    nonlinear_features: Optional[Literal[
        "RBFSampler", "Nystroem", "AdditiveChi2Sampler", "SkewedChi2Sampler"
    ]] = None
    nonlinear_n_components: int = 100
    tfidf_columns: List[str] = Field(default_factory=list)
    tfidf_max_features: int = 5000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    dim_reducer: Optional[Literal[
        "PCA", "KernelPCA", "LDA", "NMF", "TruncatedSVD", "FastICA",
        "Isomap", "UMAP", "GaussianRandomProjection",
        "SparseRandomProjection", "RandomTreesEmbedding", "BernoulliRBM",
    ]] = None
    dim_n_components: int = 50
    memory_safety_max_features: int = 100_000
    verbose_logging: bool = True

    @model_validator(mode="after")
    def _check_mutual_exclusion(self) -> "PreprocessingExtensionsConfig":
        if self.binarization_threshold is not None and self.kbins is not None:
            raise ValueError(
                "binarization_threshold and kbins are mutually exclusive; set at most one."
            )
        if self.polynomial_degree is not None and self.polynomial_degree < 2:
            raise ValueError("polynomial_degree must be >= 2 when set")
        if self.kbins is not None and self.kbins < 2:
            raise ValueError("kbins must be >= 2 when set")
        return self


class FeatureTypesConfig(BaseConfig):
    """Configuration for special feature types (text, embedding).

    Controls how text and embedding columns are detected and routed to models
    that support them (currently CatBoost only). Columns are dropped for
    models that don't support them.

    Parameters
    ----------
    text_features : list of str, optional
        Explicit list of text feature column names. These are free-text string
        columns passed to CatBoost via ``fit(text_features=...)``.
    embedding_features : list of str, optional
        Explicit list of embedding feature column names. These are columns
        containing list-of-float vectors, passed via ``fit(embedding_features=...)``.
    auto_detect_feature_types : bool
        Whether to auto-detect text and embedding features from DataFrame schema
        (default: True). Embeddings detected via ``pl.List(pl.Float32/64)``.
        Text vs categorical split by cardinality threshold.
    use_text_features : bool
        Master opt-out for text-feature routing (default: True). When False,
        auto-detection never promotes any column to text_features regardless
        of cardinality, and any explicit ``text_features`` list is also
        cleared before downstream consumption. All high-cardinality string
        columns stay as cat_features (for models that support them) or are
        dropped by models that don't. Useful when CB's text-feature TF-IDF
        pipeline is the training bottleneck (e.g. ``skills_text`` with 2M
        unique values) and the user prefers cat-feature treatment across
        the whole suite. Same caveat as other flags: changing this between
        runs invalidates cached models — see Fix 8 schema fingerprint.
    cat_text_cardinality_threshold : int
        String columns with ``n_unique <= threshold`` are treated as categorical
        (existing pipeline). Columns with ``n_unique > threshold`` are treated
        as text features. Only applies when ``auto_detect_feature_types=True``
        (default: 300).

        Default raised from 50 → 300 on 2026-04-19 after a prod incident
        (round 12): two columns with ``n_unique`` just above the old
        50 floor (``job_post_source:71``, ``_raw_countries:2196``) got
        promoted to text_features, crashing CatBoost's TF-IDF estimator
        (``Dictionary size is 0``). CatBoost's text pipeline only pays
        off on actual free-text blobs (hundreds to thousands of tokens
        like ``skills_text``), not on 50-300-cardinality enumerations
        like country codes or source categories. Setting the floor at
        300 keeps the common "enum-like" columns as cat_features where
        CB handles them efficiently via its native categorical split
        logic, and reserves text_features for columns where the text
        estimator's TF-IDF / n-gram extractors actually add signal.

        If you have a genuinely mid-cardinality column you want treated
        as text (100-300 unique tokens with repetitive content), override
        per-call via ``FeatureTypesConfig(cat_text_cardinality_threshold=100)``.

    Notes
    -----
    Strict validation (``extra="forbid"``): small, stable surface with
    no pass-through extras. Typos like ``embeding_features`` or
    ``cat_txt_cardinality_threshold`` raise at construction.
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_assignment=True,
        protected_namespaces=(),
    )

    text_features: Optional[List[str]] = None
    embedding_features: Optional[List[str]] = None
    auto_detect_feature_types: bool = True
    use_text_features: bool = True
    cat_text_cardinality_threshold: int = Field(default=300, ge=1)
    # 2026-04-21: per-column "honor the user's explicit dtype" signal.
    # When False (default, current behaviour), any text-like column
    # (pl.String / pl.Utf8 / pl.Categorical / pl.Enum / pandas object|
    # string|category) with n_unique > threshold gets auto-promoted to
    # text_features — even if the user explicitly cast it to
    # pl.Categorical / pl.Enum / pd.Categorical. When True, a column
    # whose incoming dtype ALREADY encodes a categorical intent
    # (pl.Categorical, pl.Enum, pandas ``category``) is treated as
    # user-declared: it stays in cat_features regardless of cardinality.
    # Only raw pl.String / pl.Utf8 / pandas object/string columns remain
    # candidates for auto-promotion under this flag. Use case: a
    # high-cardinality column (e.g. 10k-unique ``skills_text`` already
    # cast to ``pl.Categorical`` upstream) that the operator wants
    # handled by CatBoost's categorical path, not its TF-IDF text path.
    # Default stays False to preserve existing behaviour; flip per
    # config when the workload has user-curated Categorical intent.
    honor_user_dtype: bool = False


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
    # Range validators added 2026-04-19 — previously any garbage
    # (learning_rate=-0.1, iterations=0, etc.) propagated silently to the
    # tree backends and surfaced as confusing errors much later.
    learning_rate: float = Field(default=0.2, gt=0.0, le=1.0)
    iterations: int = Field(default=700, ge=1)
    early_stopping_rounds: Optional[int] = Field(default=100, ge=1)
    catboost_custom_classif_metrics: Optional[List[str]] = None
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
        SIGABRT → Python traceback) and on Windows suppress the
        "Python has stopped working" WER popup so Jupyter kernels exit
        cleanly instead of hanging. No-op if already enabled in the
        process.
    continue_on_model_failure : bool
        If True, catch exceptions from individual per-model training
        (e.g. XGBoost ``bad_malloc`` on too-large frames) and continue
        the suite with the next model/weighting instead of aborting
        the whole run. Crashes that kill the process at the OS level
        (access violation in a worker thread that faulthandler can't
        catch) will still terminate — for true isolation use subprocess
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
    # diagnostics — they don't change training behavior, only replace
    # the "Python has stopped working" modal with a Python traceback.
    # Users who rely on the WER popup (rare) can opt out.
    enable_crash_reporting: bool = True
    # Default False: silently skipping a failed model is a semantic
    # shift that users must opt into explicitly.
    continue_on_model_failure: bool = False
    # Default True: align Polars Categorical dicts across
    # train/val/test via shared pl.Enum(union_of_categories) before
    # model training. Mechanism not fully understood but empirically
    # prevents a silent process kill on Windows when XGB constructs
    # val IterativeDMatrix with ref=train on large frames (7.3M+ rows,
    # 15+ cat features) — observed 2026-04-20 on prod_jobsdetails.
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

    # Fix 8 (2026-04-21): append a per-model input-schema fingerprint
    # (``__sch_<10 hex>``) to model filenames so two runs with different
    # feature-type configs (text vs cat promotion, encoding, alignment)
    # don't silently overwrite each other. Default True. Set False to
    # restore the pre-2026-04-21 naming scheme (``{model}_{weight}.dump``);
    # load-time schema verification is also skipped for those artefacts.
    model_file_hash_suffix: bool = True


class MultilabelDispatchConfig(BaseConfig):
    """Configuration for multilabel-classification dispatch.

    Bundles every multilabel-only knob so per-strategy code only sees one
    parameter (instead of an exploding ``_maybe_wrap_multilabel(...)``
    signature) and so adding a new strategy choice (e.g. ``stacking``)
    doesn't require touching every dispatch site.

    Only consulted when ``target_type == MULTILABEL_CLASSIFICATION``.

    Strategy choices
    ----------------
    auto      : let the strategy pick — CatBoost uses native MultiLogloss,
                everyone else uses ``MultiOutputClassifier(estimator)`` (OvR)
    wrapper   : force ``MultiOutputClassifier(estimator)`` even on CB
                (degrades CB native to OvR — useful for A/B vs native)
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
    cv: Optional[int] = 5  # ClassifierChain.cv — 5 cross-validates chain features
    per_label_thresholds: Optional[List[float]] = None  # decision-rule thresholds
    wrapper_n_jobs: Union[int, str] = "auto"  # MultiOutputClassifier n_jobs
    allow_uncalibrated_multi: bool = False  # downgrade post-hoc calib skip from raise to warn


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
    hyperparams : ModelHyperparamsConfig
        Model hyperparameters (iterations, learning rate, per-model kwargs).
    behavior : TrainingBehaviorConfig
        Training behavior flags (GPU preference, calibration, fairness).
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

    # Typed configuration objects
    hyperparams: ModelHyperparamsConfig = Field(default_factory=ModelHyperparamsConfig)
    behavior: TrainingBehaviorConfig = Field(default_factory=TrainingBehaviorConfig)
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
    use_cache: bool = False
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
    "DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH",
    "DEFAULT_RFECV_MAX_RUNTIME_MINS",
    "DEFAULT_RFECV_CV_SPLITS",
    "DEFAULT_RFECV_MAX_NOIMPROVING_ITERS",
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
    "FeatureTypesConfig",
    "FeatureSelectionConfig",
    "ModelConfig",
    "LinearModelConfig",
    "TreeModelConfig",
    "MLPConfig",
    "NGBConfig",
    "AutoMLConfig",
    "ModelHyperparamsConfig",
    "TrainingBehaviorConfig",
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
