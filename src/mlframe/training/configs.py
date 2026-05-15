"""
Configuration classes for mlframe training pipeline.

Uses Pydantic for validation while supporting dict-like instantiation for backward compatibility.
All config classes support lenient validation - inputs are normalized to canonical forms.
"""

from __future__ import annotations


from typing import Optional, Dict, Any, List, Callable, Tuple, Literal, Union, ClassVar, FrozenSet, Set

import sys
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    # Polyfill for Python 3.9 / 3.10 (StrEnum landed in 3.11). The (str, Enum)
    # MRO gives the same equality + hashability + string-coercion behaviour
    # downstream code relies on (e.g. models.get(str_key) hash-matches
    # models.get(enum_key)).
    from enum import Enum
    class StrEnum(str, Enum):  # type: ignore[no-redef]
        def __str__(self) -> str:
            return str(self.value)

from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator


DEFAULT_RANDOM_SEED = 42
"""Random seed for reproducibility across all operations."""

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

VALID_MODEL_TYPES = {"cb", "lgb", "xgb", "hgb", "mlp", "ngb", "linear", "ridge", "lasso", "elasticnet", "huber", "ransac", "sgd"}
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
    LEARNING_TO_RANK : str
        Pairwise / listwise ranking with per-row group_id. Targets are
        per-document graded relevance (graded 0..K) or binary clicks
        (0/1). Output is a per-row score; ordering within each query
        group is what matters. CB / XGB / LGB have native rankers
        (CatBoostRanker / XGBRanker / LGBMRanker); HGB / Linear are
        not supported and skipped with NotImplementedError.
    QUANTILE_REGRESSION : str
        Predict K conditional quantiles instead of a single
        conditional mean. Target shape (N,); model output shape (N, K)
        where K = len(alphas). Use cases: prediction intervals,
        risk modelling, time-series uncertainty quantification.
        CatBoost (MultiQuantile) and XGBoost (>=2.0, quantile_alpha)
        support single-fit multi-quantile natively; LGB/HGB/Linear
        fan out to K independent fits via _QuantileMultiOutputWrapper;
        MLP / Recurrent use a K-output head + summed pinball loss.
    """

    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    LEARNING_TO_RANK = "learning_to_rank"
    QUANTILE_REGRESSION = "quantile_regression"

    @property
    def is_classification(self) -> bool:
        """True for binary, multiclass, and multilabel; False for regression
        and ranking.

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
    def is_ranking(self) -> bool:
        """True only for ``LEARNING_TO_RANK``.

        LTR is its own class — neither classification nor regression.
        Ranking outputs are scores (not probabilities, not real-valued
        regression targets), evaluated per-query (NDCG/MAP/MRR).
        Sites that branch on classification-vs-regression must add an
        explicit LTR branch when LTR is in scope.
        """
        return self == TargetTypes.LEARNING_TO_RANK

    @property
    def is_multi_output(self) -> bool:
        """True when probability output is (N, K) with K>2 logically.

        Convenience predicate for ``[:, 1]`` slicing sites that should
        bail out / dispatch differently for multi-* targets. LTR is NOT
        multi-output (output is a single score per row).
        """
        return self in (
            TargetTypes.MULTICLASS_CLASSIFICATION,
            TargetTypes.MULTILABEL_CLASSIFICATION,
        )

    @property
    def is_quantile(self) -> bool:
        """True only for ``QUANTILE_REGRESSION``.

        Quantile-regression output is (N, K) where K = len(alphas) and
        each column is a conditional-quantile estimate. NOT classification
        (no class probabilities), NOT plain regression (no single point
        prediction); branches that gate on regression vs classification
        must add an explicit QR branch when QR is in scope.
        """
        return self == TargetTypes.QUANTILE_REGRESSION


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
                "%s received unknown field(s) %s -- these are accepted (extra='allow') "
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
    # Default True drops constant columns during preprocess_dataframe. Set False to keep constant columns - useful for downstream consumers that rely on a fixed column layout across train/val/test splits.
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

    # None preserves the suite's context-aware default selection (different transformers per model type / cat-feature presence / polars-vs-pandas path); concrete defaults can't express that branching honestly, and sklearn imports at config-class load are a real cold-start tax we'd pay on every ``import mlframe.training.configs``.
    category_encoder: Optional[Any] = None
    imputer: Optional[Any] = None
    scaler: Optional[Any] = None


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
        under drift but conflicts with recency weighting -- see the field
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

    # "First test then train" - Mazzanti 2024 (Medium, 58-dataset benchmark).
    # When ``val_placement="backward"`` with time-indexed data, val is placed BEFORE train on the timeline:
    #
    #   forward  (default):  [ train ] [ val ]   [ test ]   <- conventional
    #   backward          :  [ val   ] [ train ] [ test ]   <- Mazzanti
    #
    # In forward-testing the val->train temporal gap is ~0 while the train->prod gap is large (weeks / months), so val-metric is sampled from the "near" edge of the drift trajectory and overstates prod performance (eg observed VAL ROC AUC 0.999 vs TEST 0.71 on jobsdetails). Backward-testing mirrors the val->train gap against the train->prod gap, so val-metric is sampled from the same drift-distance regime as deployment - an empirically better proxy (38% vs 51% mean deviation over 58 datasets in Mazzanti's benchmark).
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

    # When True (default), if the FeaturesAndTargetsExtractor produced
    # ``group_ids`` (e.g. ``SimpleFeaturesAndTargetsExtractor(group_field="well_id")``),
    # the splitter routes through ``GroupShuffleSplit`` so that no group
    # straddles train/val/test. Critical for non-IID data: wells, users,
    # patients, sessions. Without it, an unlucky shuffle leaks rows from
    # the same well into both train and val -- the model memorises the
    # well rather than the underlying signal, val metric inflates, and
    # the gap between val and held-out test (let alone production) is
    # the kind of silent failure that gets caught only after deploy.
    # Set to False to ignore an existing ``group_ids`` and fall back to
    # the historical IID path -- e.g. when groups are present for some
    # downstream purpose (sample weighting) but should not constrain
    # the split.
    use_groups: bool = True

    @model_validator(mode="after")
    def validate_split_sizes(self) -> "TrainingSplitConfig":
        """Ensure test_size + val_size <= 1.0 to leave room for training data."""
        if self.test_size + self.val_size > 1.0:
            raise ValueError(f"test_size ({self.test_size}) + val_size ({self.val_size}) = " f"{self.test_size + self.val_size} must be <= 1.0")
        return self


class PreprocessingBackendConfig(BaseConfig):
    """Selects the engine and parameters for *basic* preprocessing - scaling,
    imputation, categorical encoding - and is consumed both by the
    legacy ``train_mlframe_models_suite`` path and by the
    ``FeatureHandlingConfig``-driven path.

    The name describes the responsibility (which BACKEND - polars-native vs sklearn - is preferred), which is closer to what the field actually controls. ``prefer_polarsds`` is the boolean dispatch flag. Pandas inputs always fall back to sklearn; the flag only controls the polars-input path.

    ``imputer_strategy`` is connected through to the polars-ds ``Blueprint.impute()`` step in ``create_polarsds_pipeline``. Behavioural sensor tests live in ``tests/training/test_imputer_wiring.py``.

    Parameters
    ----------
    prefer_polarsds : bool
        Prefer polars-ds Blueprint operations over sklearn equivalents
        when the input is a polars DataFrame and polars-ds exposes the
        operation. Default ``True``. Pandas inputs always fall back to
        sklearn; this flag only controls the polars-input path.
    fallback_to_sklearn : bool
        If polars-ds lacks a requested operation (e.g. TF-IDF), fall
        back to sklearn rather than failing. Default ``True``.
    scaler_name : str, optional
        Scaler type: "standard", "min_max", "abs_max", "robust", or None.
        Case-insensitive, normalized to lowercase (default: "standard").
    imputer_strategy : str, optional
        Strategy for imputing missing values in *numeric* columns.
        Accepted values: "mean", "median", "most_frequent" (mapped to
        polars-ds "mode"), or None. Default "mean". Numeric-only -- text
        and string columns are imputed by the categorical encoding step,
        not here. Pass None to skip the imputer entirely.
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

    prefer_polarsds: bool = True
    fallback_to_sklearn: bool = True
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

    @field_validator("imputer_strategy", mode="before")
    @classmethod
    def normalize_imputer_strategy(cls, v: Optional[str]) -> Optional[str]:
        """Normalize imputer_strategy to a value the wiring layer can route.

        Accepts sklearn-style names (``most_frequent``) and maps them to
        the equivalent polars-ds value (``mode``) so the rest of the
        codebase only sees one canonical form.
        """
        if v is None:
            return None
        v_lower = v.lower().strip()
        # Map sklearn convention to polars-ds convention.
        alias = {"most_frequent": "mode"}
        v_lower = alias.get(v_lower, v_lower)
        valid = {"mean", "median", "mode"}
        if v_lower not in valid:
            raise ValueError(
                f"imputer_strategy must be one of {{'mean','median','most_frequent','mode'}} or None, "
                f"got '{v}'"
            )
        return v_lower


class PreprocessingExtensionsConfig(BaseConfig):
    """Optional shared-pipeline extensions applied once, reused by every model.

    When ``None`` is passed to ``train_mlframe_models_suite``, no extension runs
    and the Polars-native fastpath is preserved. Setting any field here
    activates the sklearn bridge inside ``fit_and_transform_pipeline`` -- even
    tree models will then consume the shared transformed frame.

    Order of application (each step is optional):
      0. PySR symbolic regression (``pysr_enabled``) -- generates
         new numeric features from discovered equations. Runs FIRST
         so downstream scalers / polynomial features can consume
         the engineered columns.
      1. TF-IDF on declared text columns.
      2. Scaler (overrides the Polars-ds scaler when set).
      3. Binarizer OR KBinsDiscretizer (mutually exclusive).
      4. PolynomialFeatures (guarded by ``memory_safety_max_features``).
      5. Non-linear feature map (RBFSampler / Nystroem / ...).
      6. Dim reducer (PCA / UMAP / ...).
    """

    scaler: Optional[Literal[
        "StandardScaler", "StandardScaler_nomean",
        "RobustScaler", "MinMaxScaler", "MaxAbsScaler",
        "PowerTransformer_yj", "PowerTransformer_yj_nostd",
        "QuantileTransformer_uniform", "QuantileTransformer_normal",
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
    # When True (default), TF-IDF outputs survive the extensions stage as
    # ``pd.DataFrame.sparse.from_spmatrix(...)``; sparse-aware backends
    # (cb / xgb / lgb / linear / ridge / sgd) read csr via ``.sparse.to_coo()``
    # while dense-only backends densify implicitly on ``.to_numpy()``.
    # At ``max_features=5000`` and 1M rows this is the difference between
    # ~40 GB dense float64 and ~hundreds of MB sparse. Set False to restore
    # the pre-2026-05-15 unconditional ``.toarray()`` path.
    tfidf_keep_sparse: bool = True
    dim_reducer: Optional[Literal[
        "PCA", "KernelPCA", "LDA", "NMF", "TruncatedSVD", "FastICA",
        "Isomap", "UMAP", "GaussianRandomProjection",
        "SparseRandomProjection", "RandomTreesEmbedding", "BernoulliRBM",
    ]] = None
    dim_n_components: int = 50
    memory_safety_max_features: int = 100_000
    verbose_logging: bool = True
    # PySR symbolic regression -- discovers human-readable equations
    # from the data and adds their output as new numeric features.
    # Requires Julia + SymbolicRegression.jl (installed automatically
    # via PySR / juliacall). Off by default; enable with
    # pysr_enabled=True plus a pysr_params dict for budget control.
    pysr_enabled: bool = False
    pysr_params: Optional[Dict] = Field(
        default=None,
        description="passed to PySRRegressor() as constructor kwargs"
    )

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
        runs invalidates cached models - see schema-fingerprint suffix.
    cat_text_cardinality_threshold : int
        String columns with ``n_unique <= threshold`` are treated as categorical
        (existing pipeline). Columns with ``n_unique > threshold`` are treated
        as text features. Only applies when ``auto_detect_feature_types=True``
        (default: 300).

        The 300 floor keeps "enum-like" columns (country codes, source categories) as cat_features where CB handles them efficiently via its native categorical split logic, and reserves text_features for columns where the text estimator's TF-IDF / n-gram extractors actually add signal. CatBoost's text pipeline only pays off on actual free-text blobs (hundreds to thousands of tokens like ``skills_text``), not on 50-300-cardinality enumerations; a lower floor crashes CatBoost's TF-IDF estimator with ``Dictionary size is 0``.

        If you have a genuinely mid-cardinality column you want treated as text (100-300 unique tokens with repetitive content), override per-call via ``FeatureTypesConfig(cat_text_cardinality_threshold=100)``.

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
    # Per-column "honor the user's explicit dtype" signal. When False (default), any text-like column (pl.String / pl.Utf8 / pl.Categorical / pl.Enum / pandas object|string|category) with n_unique > threshold gets auto-promoted to text_features even if the user explicitly cast it. When True, a column whose incoming dtype already encodes a categorical intent (pl.Categorical, pl.Enum, pandas ``category``) stays in cat_features regardless of cardinality; only raw pl.String / pl.Utf8 / pandas object/string columns remain candidates for auto-promotion. Use case: a high-cardinality column (eg 10k-unique ``skills_text`` already cast to ``pl.Categorical`` upstream) that the operator wants handled by CatBoost's categorical path, not its TF-IDF text path.
    honor_user_dtype: bool = False
    # Minimum non-null FRACTION required to promote a high-cardinality string column to text_features. Below this, CatBoost's TF-IDF estimator yields an empty dictionary and raises "Dictionary size is 0" (text_feature_estimators.cpp). Fraction (not absolute count) so the floor scales with dataset size. Default 0.01 = 1% of rows.
    min_non_null_fraction_for_text_promotion: float = Field(default=0.01, ge=0.0, le=1.0)
    # Datetime decomposition methods applied to detected datetime columns
    # before the pre-pipeline clone. Each entry is a polars/pandas ``.dt``
    # accessor name; ``create_date_features`` casts the result to int8.
    # Backward-compat default keeps the legacy {day, weekday, month, hour}
    # quad. Per the richness-first policy, the set is exposed so callers can
    # request richer decompositions (year / ordinal_day / minute / second).
    # The historical name "dayofyear" maps to the polars accessor
    # ``ordinal_day``; pandas exposes both names. To stay portable across
    # backends, use ``ordinal_day``.
    datetime_methods: Set[str] = Field(
        default_factory=lambda: {"day", "weekday", "month", "hour"},
    )
    # Phase ordering: run auto-detect-feature-types BEFORE fit_pipeline so
    # pandas-input datasets with high-cardinality string columns can be promoted
    # to text_features before the ordinal encoder ingests them. Pre-fix (default
    # False) the call order was fit_pipeline -> auto_detect_feature_types, which
    # silently ordinal-encoded text columns (the encoder ran first and the
    # detect step saw integer codes). Default True per audit row FE-P1-2; flip
    # to False to restore legacy ordering for byte-for-byte reproducibility.
    feature_types_first: bool = True


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
    mrmr_kwargs: Optional[Dict[str, Any]] = None
    rfecv_models: Optional[List[str]] = None
    rfecv_kwargs: Optional[Dict[str, Any]] = None
    custom_pre_pipelines: Dict[str, Any] = Field(default_factory=dict)

    # When a feature-selection pipeline (MRMR / RFECV / custom) is identity-equivalent - keeps every input column and creates no new ones - training models on it duplicates the ordinary (no-pipeline) branch. Set False to still train both (eg for ensembling diversities from different random seeds). Default True skips the duplicate branch, logging a [Dedup] info.
    skip_identity_equivalent_pre_pipelines: bool = True

    @field_validator("mrmr_kwargs")
    @classmethod
    def _validate_mrmr_kwargs(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not v:
            return v
        import inspect
        from mlframe.feature_selection.filters import MRMR
        valid_keys = set(inspect.signature(MRMR.__init__).parameters) - {"self"}
        unknown = sorted(set(v) - valid_keys)
        if unknown:
            raise ValueError(
                f"FeatureSelectionConfig.mrmr_kwargs: unknown key(s) {unknown}. "
                f"Valid keys: {sorted(valid_keys)}"
            )
        return v

    @field_validator("rfecv_kwargs")
    @classmethod
    def _validate_rfecv_kwargs(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not v:
            return v
        import inspect
        from mlframe.feature_selection.wrappers._rfecv import RFECV
        # ``cv_n_splits`` is consumed by get_training_configs to construct a CV splitter; not a direct RFECV.__init__ arg.
        valid_keys = (set(inspect.signature(RFECV.__init__).parameters) - {"self"}) | {"cv_n_splits"}
        unknown = sorted(set(v) - valid_keys)
        if unknown:
            raise ValueError(
                f"FeatureSelectionConfig.rfecv_kwargs: unknown key(s) {unknown}. "
                f"Valid keys: {sorted(valid_keys)}"
            )
        return v


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


class OutlierDetectionConfig(BaseConfig):
    """Configuration for the once-per-suite outlier-detection pass.

    ``apply_to_val`` was previously named ``od_val_set`` at the suite
    level - renamed for clarity.
    """

    detector: Optional[Any] = None  # sklearn OutlierMixin or None
    apply_to_val: bool = True


class CompositeTargetDiscoveryConfig(BaseConfig):
    """Configuration for composite-target discovery.

    Discovery looks for transformations of the target ``y`` of the form
    ``T = f(y, base)`` such that the model trained on ``T`` (and a
    feature set excluding ``base``) generalises better than the model
    trained on raw ``y``. Typical case: ``y = TVT`` with ``base = TVT_prev``
    where the autoregressive lag dominates feature importance.

    All fitted parameters (alpha/beta for linear_residual, MAD bounds
    for logratio, etc.) are computed from rows passed via ``train_idx``
    only. Validation and test rows are NEVER touched at fit time.

    Default OFF: opt in by setting ``enabled=True`` and configuring
    base candidates explicitly OR leaving ``base_candidates="auto"``
    for automatic discovery via MI-gain ranking.
    """

    enabled: bool = False

    # Base candidate selection.
    # - "auto": rank all numeric features by structural MI gain
    #   (MI(y - LinearFit(x), X \ {x}) on train) and take the top
    #   ``auto_base_top_k`` after applying forbidden-pattern + corr
    #   + ptp filters.
    # - list[str]: explicit list of column names. Still passes through
    #   the forbidden / corr / ptp guards; columns failing the guard
    #   are skipped with a warning.
    base_candidates: Union[List[str], str] = "auto"
    auto_base_top_k: int = 3

    # Priority-base hint -- features that should be treated as base
    # candidates regardless of pairwise ``MI(y, x)`` ranking. When
    # populated, ``_auto_base`` puts these first (in given order) and
    # fills remaining slots up to ``auto_base_top_k`` with the top
    # MI-ranked features.
    #
    # The hint exists because pairwise MI is fooled by features that
    # have global trend with y but no structural residual signal
    # (e.g. spatial coordinates on geographically-trended targets).
    # ``BaselineDiagnostics``'s ablation (drop feature -> measure RMSE
    # delta) is a much more reliable signal for "which feature
    # actually drives prediction": a feature whose removal hurts RMSE
    # by 500% is unambiguously dominant, regardless of MI estimation
    # noise. ``train_mlframe_models_suite`` populates the hint from
    # the ablation output automatically; users can also pass it
    # explicitly.
    #
    # Hint features still go through the standard filters
    # (forbidden_pattern / non_numeric / constant / corr_threshold);
    # any that fail are logged and dropped.
    dominant_features_hint: Optional[List[str]] = None

    # Transform names from the registry (mlframe.training.composite).
    #
    # 2026-05-11 (R10c brainstorm rollout): default extended from the original 4 to include the SINGLE-BASE, DROP-IN transforms shipped in commits 9e05955 + 0894369:
    #   - ``quantile_residual`` -- conditional-on-bin centering + scaling.
    #   - ``monotonic_residual`` -- monotone PCHIP spline residual.
    # These accept the standard ``(y, base)`` signature and need no special orchestration -- discovery evaluates them like ``linear_residual``.
    #
    # NOT in default list (require orchestration the discovery loop does not yet provide):
    #   - ``linear_residual_multi`` -- needs multi-column base selection (forward stepwise); single-base mode is identical to ``linear_residual``.
    #   - ``linear_residual_grouped`` -- needs ``group_column`` extraction + groups kwarg through fit/forward/inverse.
    #   - ``ewma_residual`` / ``rolling_quantile_ratio`` / ``frac_diff`` -- require chronological row order which most datasets lack at the discovery stage.
    # All four are accessible via explicit user configuration (``CompositeTargetEstimator(...)`` directly) and ship with their own tests; auto-discovery integration is the open item beyond this PR.
    transforms: List[str] = Field(
        default_factory=lambda: [
            "diff", "ratio", "logratio", "linear_residual",
            "quantile_residual", "monotonic_residual",
        ]
    )

    # OPEN-1 integration (2026-05-12): multi-base forward-stepwise auto-promotion of kept ``linear_residual`` specs. After single-base discovery + raw-y baseline gate + tiny-model rerank, Discovery picks each kept linear_residual spec and tries greedily ADDING bases from the auto-base candidate pool. When the marginal CV-RMSE reduction clears ``multi_base_min_marginal_rmse_gain`` (default 2%), the spec is upgraded to ``linear_residual_multi`` with the expanded base list.
    #
    # Default ON: measure-first benchmark (``benchmarks/composite_multi_base_benchmark.py``) confirms geo-mean gain 83% on positive scenarios (S2: y = b1+b2+eps, S3: y = b1+b2+b3+eps) AND no-harm on negative scenarios (S1: single dominant b1 + noise candidates; S4: collinear b1+b1_dup pool). Decision rule met. Opt-out by setting ``multi_base_enabled=False`` if production data violates the benchmark's assumptions (highly correlated candidate pool, very small n_train, etc.).
    multi_base_enabled: bool = True
    multi_base_max_k: int = 3
    multi_base_min_marginal_rmse_gain: float = 0.02

    # MI screening. Sample to keep the diagnostic under one minute on
    # 4M-row datasets; mi_sample_n=None uses full train.
    mi_sample_n: Optional[int] = 200_000
    top_k_after_mi: int = 8
    # Pre-filter threshold for ``mi_gain = MI(T, X_no_base) - MI(y, X_no_base)``.
    # Default lowered from +0.01 -> -0.5 on 2026-05-11 (R10c bug #3)
    # after a production TVT regression run discovered 0 specs despite
    # BaselineDiagnostics ablation correctly identifying ``TVT_prev``
    # as the dominant feature. Root cause: pure-lag composite
    # ``T = y - y_prev = noise`` has ``MI(T, X_no_base) ~ 0`` while
    # ``MI(y, X_no_base) > 0``, so ``mi_gain`` is structurally
    # NEGATIVE for the correct composite -- a sign of a clean lag fit,
    # not a sign the composite is useless. The MI-gain pre-filter
    # was rejecting LEGITIMATE compositions.
    #
    # The actual "is this composite predictively useful" decision is
    # made downstream by the raw-y baseline gate (Phase B; compares
    # tiny CV-RMSE of composite vs raw-y on the same screening folds).
    # With ``eps_mi_gain=-0.5`` the pre-filter only drops composites
    # whose mi_gain is MUCH worse than raw -- typical "transform broke
    # the target" cases (logratio on negative y, ratio on near-zero
    # base). Pure-lag composites pass through to the raw-y gate where
    # they are correctly evaluated.
    eps_mi_gain: float = -0.5
    mi_n_neighbors: int = 3  # sklearn mutual_info_regression k.

    # MI estimator. "knn" uses the Kraskov estimator (sklearn default,
    # accurate but slow on n>10k); "bin" uses a quantile-binning
    # estimator (5-10x faster, biased low on heavy-tail).
    #
    # Default flipped from "knn" -> "bin" 2026-05-10 after a
    # statistical review noted that:
    # 1. kNN is biased high on heavy-tail / mixed-density distributions
    #    and the bias scales DIFFERENTLY for raw y (potentially fat-
    #    tailed) vs T = transform(y, base) (sub-Gaussian after
    #    linear_residual). That asymmetric bias inflates apparent
    #    mi_gain even when the true gain is zero -- which matches the
    #    production failure mode where MI passes but RMSE doesn't.
    # 2. bin (quantile) estimator is approximately bias-free under
    #    monotone transforms because the bin edges follow the
    #    transformed distribution -- exactly what the registry's
    #    transforms (diff/ratio/logratio/linear_residual) do.
    # 3. bin is 10x faster on the 200K-row screening sample we
    #    typically run.
    #
    # Set to "knn" explicitly for non-monotone transforms or when
    # n < 5*nbins (bin floor needs ~80 rows at default nbins=16).
    mi_estimator: str = "bin"
    mi_nbins: int = 16  # Bin count when ``mi_estimator == "bin"``.

    # R10b statistician #1: aggregation across feature columns when
    # comparing MI(T, X_no_base) against MI(y, X_no_base). Legacy
    # ``"sum"`` is biased (overcounts shared information when X is
    # correlated, and the over-count differs between numerator and
    # denominator). Mean is invariant to feature count and is the
    # cleaner default; users on existing benchmarks can pin
    # ``"sum"`` for reproducibility.
    mi_aggregation: str = "mean"

    # MI sampling strategy. "random" is the cheap default; switch to
    # "stratified_quantile" on heavy-tail targets (financial returns,
    # fraud scores, queue lengths) where random sampling can miss the
    # tail rows that carry most of the signal. Stratified sampling
    # bins y into ``mi_n_strata`` quantile bins and samples equally
    # from each, guaranteeing per-bin coverage.
    mi_sample_strategy: str = "random"
    mi_n_strata: int = 10

    # Phase B: tiny-model rerank. After MI screening narrows to top-K,
    # train a tiny model (LightGBM or per-family) per surviving
    # candidate and re-rank by CV-RMSE measured on the y-scale (after
    # inverse). This is the "true objective" -- MI is a proxy. Skip
    # by setting ``screening`` = ``"mi"``.
    #
    # Default raised from "mi" -> "hybrid" in 2026-05-10 after a
    # production case where MI-only screening kept composites whose
    # bases (spatial coordinates) had trivial pairwise MI(y, x) but
    # zero structural signal for residual learning. The MI-gain test
    # passed barely (mi_gain ~ 0.01) but the resulting models had
    # WORSE OOF RMSE than raw-y because subtracting the base added
    # noise to the target. Phase B's CV-RMSE-on-y-scale catches this
    # directly. Cost: ~0.5-2 min per target on a 4M-row dataset.
    screening: str = "hybrid"  # "mi" | "tiny_model" | "hybrid"
    tiny_model_n_estimators: int = 60
    tiny_model_num_leaves: int = 15
    tiny_model_learning_rate: float = 0.1
    tiny_model_cv_folds: int = 3
    tiny_model_sample_n: int = 20_000  # rows used per tiny-model fit
    top_m_after_tiny: int = 3  # final top-M after Phase B re-rank
    tiny_model_n_jobs: int = 1  # >1 = parallelise CV folds via joblib

    # Force deterministic mode on the tiny models built INSIDE Phase B
    # (``_build_tiny_model``). When True, injects the well-known
    # determinism flags per family:
    # - LightGBM: ``deterministic=True``, ``force_row_wise=True``
    # - XGBoost: explicit ``tree_method="hist"``, ``predictor="auto"``
    # - CatBoost: ``boosting_type="Plain"`` (Plain is deterministic;
    #   Ordered is the non-deterministic default)
    # Bit-exact run-to-run reproducibility on the rerank stage at a
    # 5-10% per-fit cost. Default OFF.
    # Scope: this controls the tiny models we BUILD ourselves for
    # rerank. The actual composite-target inner training (the K
    # LightGBM/XGB models that train the per-spec composite targets)
    # is configured via ``hyperparams_config``, not this flag.
    deterministic_screening_models: bool = False

    # Per-family screening: instead of one tiny LightGBM, train a
    # tiny model of each family in the user's ``mlframe_models``
    # list (cb / lgb / xgb / linear). Different families pick
    # different top features on the same data, so a candidate that
    # wins for one family may lose for another. Aggregation via
    # ``tiny_consensus``:
    # - "single_lgbm" (default): one LightGBM, fastest.
    # - "per_family": train ``tiny_screening_models`` per family;
    #   aggregate by ``tiny_consensus`` ("union": top-M from each
    #   family; "borda": Borda-count rank aggregation).
    tiny_screening_models: str = "single_lgbm"  # "single_lgbm" | "per_family"
    tiny_screening_families: Tuple[str, ...] = ("lightgbm",)
    tiny_consensus: str = "union"  # "union" | "borda"

    # Raw-y baseline gate. During tiny-model rerank, also train a tiny
    # model on the RAW target (no composite transform) on the same
    # screening sample / folds and use its CV-RMSE as a hard floor:
    # any composite whose CV-RMSE >= raw_baseline * tolerance is
    # rejected as a regression. Catches the "wrong base" case where
    # MI-gain passes but the resulting target is actually harder to
    # predict (e.g. subtracting a spatial coordinate that has global
    # trend with y but no structural residual signal).
    #
    # Tolerance > 1.0 allows composites that are *slightly* worse on
    # the screening sample but might still help in the cross-target
    # ensemble. 1.0 = strict (composite MUST beat raw). Default 1.02
    # = composite kept if within 2% of raw, rejected if worse.
    require_beats_raw_baseline: bool = True
    raw_baseline_tolerance: float = 1.02

    # R10b improvement #1: regime-aware gate. In addition to the
    # global mean RMSE comparison, also check per-quintile-of-base
    # RMSE: a spec is rejected if its tiny CV-RMSE in any quintile
    # exceeds raw_baseline-in-that-quintile by ``raw_baseline_per_bin_tolerance``.
    # This catches "two-regime" failure modes where logratio is
    # correct on multiplicative-regime rows but actively wrong on
    # additive-regime rows; mean RMSE hides this and the spec
    # ships even though it's miscalibrated half the time.
    #
    # Tolerance defaults looser than the global gate (1.10 vs 1.02)
    # because per-bin estimates have higher variance on small
    # screening samples. Set ``per_bin_n_bins=0`` to disable the
    # per-bin check.
    raw_baseline_per_bin_tolerance: float = 1.10
    per_bin_n_bins: int = 5

    # R10b improvement #10: median-of-seeds gate. Tiny CV-RMSE with
    # 3 folds is variance-prone (one unlucky split can drag the mean).
    # Optionally repeat the K-fold split with multiple seeds and take
    # the MEDIAN across (folds × seeds) for both raw-y and per-spec
    # CV-RMSE. The gate then compares median composite vs median raw,
    # which is more stable than the mean. Default 1 = backwards-
    # compatible (single-seed). Set to e.g. 3 or 5 on small screening
    # samples where gate decisions are noisy. Compute scales linearly.
    tiny_model_n_seed_repeats: int = 1

    # R10b statistician #4: paired one-sided Wilcoxon signed-rank
    # test on per-fold-pair RMSE differences (composite minus raw).
    # Replaces the static ``raw_baseline * tolerance`` threshold with
    # a non-parametric significance test: spec is rejected unless
    # the median of per-fold differences is significantly negative
    # (composite < raw) at level ``gate_alpha``. Scipy must be
    # available; falls back to threshold-only gate if not.
    #
    # Cost: requires per-fold RMSE pairs from BOTH composite and
    # raw runs, which we already collect when
    # ``tiny_model_n_seed_repeats > 1``. With n_seed_repeats=1 the
    # test has 3 fold pairs total -- the test will be too low-power
    # to reject anything except egregious cases. Recommended:
    # n_seed_repeats=5 for the test to have meaningful power.
    use_wilcoxon_gate: bool = False
    gate_alpha: float = 0.05

    # R10b statistician #6: detect alpha-drift in linear_residual.
    # Fit alpha on first half of train and on second half; compare
    # via Chow-style |Δα| / pooled SE. If the absolute z-score
    # exceeds ``alpha_drift_z_threshold`` (default 3.0), the
    # linear_residual spec for that base is flagged in metadata
    # with reason ``alpha_drift_detected`` and (optionally) rejected.
    # Catches concept-drift / non-stationary y/base relationships
    # that LR's point-estimate alpha silently degrades on at test.
    detect_linear_residual_alpha_drift: bool = True
    alpha_drift_z_threshold: float = 3.0
    # When True, drop linear_residual specs that fail the drift
    # check; when False, keep them but log a warning + record in
    # metadata. Default False -- drift is informational only by
    # default; flag to True on series with known non-stationarity.
    reject_on_alpha_drift: bool = False

    # R10b stat #8: bootstrap CI on mi_gain. The point-estimate
    # mi_gain has noise floor that scales with the screening sample
    # size and the heaviness of the y-tail; the eps_mi_gain absolute
    # threshold misses this. Optional bootstrap (resample the
    # screening sample, recompute MI, take 2.5/97.5 percentiles)
    # produces an honest CI; the gate then compares ``eps_mi_gain``
    # against the lower CI bound, not the point estimate.
    #
    # Cost: ``mi_gain_bootstrap_n`` extra MI evaluations per spec
    # (default 0 = disabled; recommended 50 for confidence band).
    mi_gain_bootstrap_n: int = 0
    mi_gain_bootstrap_random_state: int = 12345

    # R10b stat #8 (continued): boost n_strata on heavy-tail targets
    # when stratified MI sampling is enabled. Default 10 strata is
    # too few for tail-driven signal -- tail rows get one bin each
    # and MI estimates become unstable. Auto-detection: when y skew
    # > 2.0 OR kurtosis > 5.0, boost ``mi_n_strata`` to
    # ``mi_n_strata_heavy_tail``. Manual override via setting
    # ``mi_n_strata`` explicitly.
    mi_n_strata_heavy_tail: int = 30

    @field_validator("screening", mode="before")
    @classmethod
    def _normalise_screening(cls, v: str) -> str:
        v_lower = str(v).lower()
        valid = {"mi", "tiny_model", "hybrid"}
        if v_lower not in valid:
            raise ValueError(f"screening must be one of {valid}, got '{v}'")
        return v_lower

    @field_validator("mi_estimator", mode="before")
    @classmethod
    def _normalise_mi_estimator(cls, v: str) -> str:
        v_lower = str(v).lower()
        valid = {"knn", "bin"}
        if v_lower not in valid:
            raise ValueError(f"mi_estimator must be one of {valid}, got '{v}'")
        return v_lower

    @field_validator("mi_sample_strategy", mode="before")
    @classmethod
    def _normalise_mi_sample_strategy(cls, v: str) -> str:
        v_lower = str(v).lower()
        valid = {"random", "stratified_quantile"}
        if v_lower not in valid:
            raise ValueError(
                f"mi_sample_strategy must be one of {valid}, got '{v}'"
            )
        return v_lower

    @field_validator("tiny_screening_models", mode="before")
    @classmethod
    def _normalise_tiny_screening_models(cls, v: str) -> str:
        v_lower = str(v).lower()
        valid = {"single_lgbm", "per_family"}
        if v_lower not in valid:
            raise ValueError(
                f"tiny_screening_models must be one of {valid}, got '{v}'"
            )
        return v_lower

    @field_validator("tiny_consensus", mode="before")
    @classmethod
    def _normalise_tiny_consensus(cls, v: str) -> str:
        v_lower = str(v).lower()
        valid = {"union", "borda"}
        if v_lower not in valid:
            raise ValueError(f"tiny_consensus must be one of {valid}, got '{v}'")
        return v_lower

    # Forbidden base filters. Block columns whose names match any of
    # these regex patterns (target leakage via target encoding /
    # rolling target stats / etc.).
    forbidden_base_patterns: List[str] = Field(
        default_factory=lambda: [
            r"^target_enc_",
            r"^mean_target_",
            r"_te$",
            r"^lagged_target_",
            r"^y_smooth_",
        ]
    )

    # Block columns whose Pearson |corr(base, y)| exceeds this threshold.
    # Intent: catch literal copies / trivial linear transforms of y
    # (e.g. ``y_renamed = y``, ``y_scaled = y / 1000``). NOT intended
    # to catch autoregressive lag features such as ``TVT_prev`` --
    # those legitimately reach corr ~ 0.999 on slow-moving series due
    # to autocorrelation, and they are exactly the kind of dominant
    # feature composite-target discovery exists to handle.
    #
    # The primary defence against target-encoding leakage is the regex
    # patterns above (``forbidden_base_patterns``); the corr threshold
    # is just a backstop. Default raised from 0.999 to 0.99999 in
    # 2026-05-10 after observing it filtered out legitimate
    # ``TVT_prev`` (lag-1) on a real production run.
    forbidden_base_corr_threshold: float = 0.99999

    # Block constant or near-constant base columns (zero variance ->
    # OLS in linear_residual is degenerate; ratio / logratio are
    # uninformative).
    constant_base_eps: float = 1e-12

    # Domain validity. Drop a (base, transform) candidate entirely if
    # fewer than this fraction of train rows pass the transform's
    # domain_check (e.g. logratio requires y, base > 0).
    min_valid_domain_frac: float = 0.7

    # Behaviour when no candidate clears eps_mi_gain.
    # - "fallback_raw": warn and emit no composite targets (caller
    #   trains on raw target only).
    # - "raise": raise RuntimeError -- useful in CI / scripted modes
    #   to flag degenerate inputs.
    # - "warn": warn but emit the best-of-bad candidates anyway.
    fail_on_no_gain: str = "fallback_raw"

    random_state: int = DEFAULT_RANDOM_SEED

    # Cross-target ensemble strategy. Run after each composite-target
    # model is wrapped to y-scale, builds one combined predictor over
    # all (raw + K composite) wrappers.
    # - "off": no ensemble; ``models[type][f"_CT_ENSEMBLE__{target}"]`` not created.
    # - "mean": equal-weight average over all components.
    # - "oof_weighted": gain-over-baseline weighting using per-component
    #   RMSE (train-RMSE proxy by default; honest holdout RMSE when
    #   ``oof_holdout_frac > 0``); auto-falls-back to best-single
    #   component if no component clears the baseline.
    # - "linear_stack": Ridge regression on per-component predictions.
    # - "nnls_stack": non-negative least squares on per-component preds.
    #
    # Default flipped from "off" -> "oof_weighted" 2026-05-10 (R10b
    # improvement #5), then "oof_weighted" -> "nnls_stack" 2026-05-10
    # (R10c) after the wide ensemble shootout
    # (``mlframe/benchmarks/composite_ensemble_shootout.py``):
    # 6 scenarios x 3 seeds = 18 (scenario, seed) datapoints, 11
    # ensemble strategies tested. Results (mean improvement %
    # vs best_single_by_train, sorted):
    #
    #   nnls_stack            +1.24%  (13/18 wins)  <- WINNER
    #   best_single_by_train  +0.00%  (baseline)
    #   bma_softmax           -0.60%
    #   inverse_variance      -1.61%
    #   linear_stack_ridge    -1.71%
    #   inverse_rmse          -7.17%
    #   stacked_gbdt         -12.44%
    #   oof_weighted         -18.42%  (previous default!)
    #   median               -19.07%
    #   trimmed_mean         -19.07%
    #   mean                 -23.20%
    #
    # NNLS is the only strategy with positive mean improvement and
    # majority wins. Single-spec case is handled by the ensemble class
    # via best-single fallback (no overhead). Set to "off" explicitly
    # to skip ensemble construction.
    cross_target_ensemble_strategy: str = "nnls_stack"

    # When True AND the per-target ``baseline_diagnostics`` reports
    # ``composite_recommendation == "unlikely_to_help"``, discovery
    # short-circuits with a warning and produces no specs. Saves
    # the MI / tiny-model / re-fit cost on targets where composite
    # mode is unlikely to add value (init_score baseline already
    # captures the dominance, or no feature dominates strongly).
    # Default False so explicit opt-ins don't get silently overridden.
    auto_skip_on_baseline_optimal: bool = False

    # Use BaselineDiagnostics ablation top-K as priority base
    # candidates (``dominant_features_hint``) instead of relying on
    # pairwise MI(y, x) ranking alone. Pairwise MI gets fooled by
    # features with global trend but no structural residual signal
    # (spatial coords on geographically-trended y); ablation directly
    # measures predictive contribution and is much more reliable.
    #
    # When True, ``train_mlframe_models_suite`` runs BaselineDiagnostics
    # inline (cached) before discovery and injects the top-K
    # ablation-ranked features as the hint. When the inline
    # diagnostic fails or returns no dominant features, falls back
    # silently to MI-only ranking.
    #
    # Default True since it strictly improves auto-base on the
    # production failure mode and the inline BD cost is amortised
    # (the same diagnostic runs in the per-target loop later;
    # caching reuses it).
    use_baseline_diagnostics_hint: bool = True
    baseline_diagnostics_hint_top_k: int = 3

    # R10c bug #5: hint-strength threshold for the adaptive hint cap.
    # When the top hint feature has BaselineDiagnostics ablation
    # ``delta_pct >= hint_strength_threshold_pct``, ``_auto_base``
    # uses the FULL hint list (no cap) instead of capping at
    # ``max(1, top_k // 2)``. Set to a high value (e.g. 1000) to
    # effectively disable the strong-hint shortcut.
    hint_strength_threshold_pct: float = 50.0

    # Cross-base correlation dedup (R10b improvement #9). After
    # auto-base ranking, drop a candidate base if its absolute Pearson
    # correlation against any already-kept candidate exceeds this
    # threshold on the screening sample. Stops near-duplicate lag
    # variants (``TVT_prev``, ``TVT_prev_lag2``, ``TVT_smooth_3``) from
    # all surviving into Phase B and inflating ensemble correlation.
    # Set to 1.0 to disable.
    auto_base_dedup_corr_threshold: float = 0.95

    # R10b improvement #2: permutation-MI null distribution test in
    # ``_auto_base``. For each candidate feature compute MI(y, x) AND
    # MI(y, shuffle(x)) on ``auto_base_null_perms`` shuffles, then
    # require the candidate's MI to exceed ``mean_null + n_sigma *
    # std_null``. Catches features whose MI(y, x) is non-trivial only
    # because of a shared monotonic component (time/spatial trend),
    # not structural information about y.
    #
    # Cost: ``auto_base_null_perms`` extra MI evaluations per
    # candidate (default 20 × ~1ms each on bin-MI estimator = ~20ms
    # per feature on the screening sample). Set
    # ``auto_base_null_perms=0`` to disable.
    auto_base_null_perms: int = 20
    auto_base_null_z_threshold: float = 3.0
    # Block-shuffle length for temporal datasets so the null
    # preserves marginal autocorrelation. ``"auto"`` uses
    # ``int(sqrt(n))``; explicit int for fixed length; ``1`` for
    # row-level shuffle (i.i.d. assumption).
    auto_base_null_block_length: Union[str, int] = "auto"

    # R10b improvement #7: structural detectors for time-index and
    # spatial-coordinate features. Demote (push to bottom of MI
    # ranking) features that look like:
    # - **Time index**: |Spearman(rank(x), arange(n))| > 0.95.
    #   Catches a row-counter or timestamp masquerading as a base
    #   candidate; on temporal data the row index correlates with
    #   y purely from drift, no structural information.
    # - **Spatial coordinate block**: pairwise correlations among
    #   3+ numeric features form a block where each pair has
    #   |corr| > 0.5. Catches X/Y/Z lat-lon-altitude triplets where
    #   pairwise MI(y, coord) is high purely from spatial drift,
    #   not structural information.
    # Demoted features are ALSO available as bases when their MI
    # genuinely exceeds non-demoted candidates (defensive, not
    # blocking). Set to False to disable.
    auto_base_demote_time_index: bool = True
    auto_base_demote_spatial_coords: bool = True

    # Collapse ``linear_residual`` -> ``diff`` when the fitted alpha
    # is approximately 1.0 (R10b improvement #6). ``linear_residual``
    # is a strict generalisation of ``diff`` (diff = linear_residual
    # with alpha=1, beta=0). When OLS lands at alpha~1 on stationary
    # lag features, the two transforms produce numerically identical
    # T columns -- but ``linear_residual`` carries TWO learned
    # parameters with train-time variance. ``diff`` is the lower-
    # variance answer. The threshold compares the scale-invariant
    # ratio ``|alpha - 1| * std(base) / std(y)``; below this value,
    # the linear_residual spec is considered redundant with diff and
    # dropped if a diff spec for the same base also kept. Set to 0.0
    # to disable (always keep both).
    collapse_linear_residual_alpha_eps: float = 0.05

    # R3.18: handling multilabel (multi-output) regression targets,
    # i.e. ``target_by_type[regression][name]`` is a 2-D array of
    # shape ``(n_rows, n_outputs)``.
    # - ``"per_target"`` (default): expand into ``n_outputs`` separate
    #   1-D regression targets named ``{name}_out{j}``; discovery
    #   runs independently per output, naming composites
    #   ``{name}_out{j}__{transform}__{base}``. Per-target training
    #   loop downstream sees them as ordinary 1-D targets.
    # - ``"skip"``: legacy behaviour -- mark with metadata note,
    #   produce no composites for that target. Useful when the
    #   caller knows they don't want the per-output expansion (e.g.
    #   the training loop downstream expects the 2-D shape intact).
    multilabel_strategy: str = "per_target"

    # Cap the number of components combined at predict time. Useful
    # for online single-row latency-sensitive serving where running
    # K=8 wrappers per row blows the SLA. When > 0, the ensemble
    # keeps only the top-N components by weight (after the standard
    # weight computation), drops the rest, and re-normalises. None
    # / 0 means keep all components (default).
    max_inference_components: Optional[int] = None

    # Honest OOF for the ensemble gate / stacking.
    #
    # When > 0, the suite carves an extra holdout slice (this fraction
    # of filtered_train_idx) and at ensemble-build time re-fits a
    # clone of every component on the (1-frac) stack_train slice,
    # then predicts on the held-out slice. The honest holdout
    # predictions feed the stacking solvers (linear_stack /
    # nnls_stack) and the gain-over-baseline weights, replacing the
    # train-RMSE proxy that overstates accuracy. Cost: re-fits every
    # component once on (1-frac) of train rows.
    #
    # Default flipped from 0.0 -> 0.2 on 2026-05-15 because the default
    # cross_target_ensemble_strategy is ``nnls_stack``. Fitting NNLS on
    # in-sample component predictions is a stacking leak: every
    # component has effectively memorised its training rows, so NNLS
    # picks weights that overweight whichever component fits noise
    # best. A 20% honest holdout is the standard "stacking on OOF" cure
    # documented in Sill et al. 2009 (Feature-Weighted Linear Stacking)
    # and removes the leak at the cost of one extra fit per component
    # on 80% of train rows. Set to 0.0 explicitly to opt out (e.g. when
    # using a non-stacking strategy like ``mean`` where the train-RMSE
    # proxy is harmless).
    oof_holdout_frac: float = 0.2
    oof_random_state: int = DEFAULT_RANDOM_SEED

    # Stacking-aware gate (measure-first NNLS gate). When True AND
    # ``cross_target_ensemble_strategy`` is ``linear_stack`` or
    # ``nnls_stack``, the ensemble-build path first runs
    # :func:`stacking_aware_gate` over the component predictions to drop
    # components whose NNLS weight falls below
    # ``stacking_aware_gate_min_weight``. The surviving subset feeds the
    # actual stacker. Skipped when fewer than 2 components survive (the
    # stacker handles single-component falls back on its own).
    stacking_aware_gate_enabled: bool = False
    stacking_aware_gate_min_weight: float = 0.05

    # Composite-feature stacking. When True, ``run_composite_target_discovery``
    # produces an opt-in stub call to ``composite_oof_predictions`` /
    # ``composite_predictions_as_feature`` on the discovered specs so
    # downstream code can attach the predictions as engineered features.
    # Default False; full wiring requires the downstream FE pipeline to
    # consume the new column, which is caller-specific.
    composite_feature_stacking_enabled: bool = False

    @field_validator("cross_target_ensemble_strategy", mode="before")
    @classmethod
    def _normalise_ensemble_strategy(cls, v: str) -> str:
        v_lower = str(v).lower()
        valid = {"off", "mean", "oof_weighted", "linear_stack", "nnls_stack"}
        if v_lower not in valid:
            raise ValueError(
                f"cross_target_ensemble_strategy must be one of {valid}, got '{v}'"
            )
        return v_lower

    @field_validator("fail_on_no_gain", mode="before")
    @classmethod
    def _normalise_fail_mode(cls, v: str) -> str:
        v_lower = str(v).lower()
        valid = {"fallback_raw", "raise", "warn"}
        if v_lower not in valid:
            raise ValueError(f"fail_on_no_gain must be one of {valid}, got '{v}'")
        return v_lower

    @field_validator("multilabel_strategy", mode="before")
    @classmethod
    def _normalise_multilabel_strategy(cls, v: str) -> str:
        v_lower = str(v).lower()
        valid = {"per_target", "skip"}
        if v_lower not in valid:
            raise ValueError(
                f"multilabel_strategy must be one of {valid}, got '{v}'")
        return v_lower


class BaselineDiagnosticsConfig(BaseConfig):
    """Configuration for the auto-baseline diagnostics pass.

    Runs once per (target_type, target_name) before per-target training
    starts. Cheap (~30-90 s on a sampled view of train+val) and reports:

    1. Headline baseline metric (RMSE/MAE/R^2 for regression;
       AUC/logloss for binary). One quick model.
    2. Sequential ablation: drop top-K features by feature_importances_,
       retrain, measure metric delta. Surfaces dominant features.
    3. ``init_score`` baseline for regression: refits a quick LightGBM
       with the top-1 dominant feature passed via init_score (model
       learns only the residual). If this baseline already matches raw
       within ``init_score_optimal_threshold_pct``, composite-target
       discovery is unlikely to add value (recommendation downgrade).
    4. ``composite_recommendation`` flag in {high_potential, marginal,
       unlikely_to_help} consumed by composite-target discovery to gate
       expensive screening loops.

    Default ON for regression and binary classification; multiclass /
    multilabel / LtR / quantile_regression skipped (init_score semantics
    don't carry).
    """

    enabled: bool = True

    # Ablation: drop top-K features ranked by quick-model FI.
    ablation_top_k: int = 5

    # Quick-model knobs. LightGBM is the workhorse: fast, supports
    # init_score natively for regression, robust on cold caches.
    quick_model_family: Literal["lightgbm"] = "lightgbm"
    quick_model_n_estimators: int = 200
    quick_model_num_leaves: int = 31
    quick_model_learning_rate: float = 0.05

    # init_score baseline: regression-only in MVP. Top-K dominant
    # features are summed; for K=1 the single base is passed as
    # init_score, for K>1 a quick OLS combines them first.
    init_score_top_k: int = 1
    # init_score baseline supports both regression and binary classification.
    # For binary the init_score lives in logit space: top-K dominant
    # features are LR-combined into a probability-scale score, then
    # converted to logit and passed as LightGBM's ``init_score=`` so the
    # booster learns the residual logit.
    init_score_apply_to_target_types: Tuple[str, ...] = (
        "regression", "binary_classification",
    )

    # Sample size for ablation/init-score fits. Capped well below the
    # full 4M-row regime so the diagnostic stays under ~1 minute on
    # large datasets. None means "use full train".
    sample_n: Optional[int] = 50_000

    # Recommendation thresholds (in PERCENT of headline metric).
    # Ablation Δ% is computed as (metric_after_drop / metric_raw - 1) * 100.
    # Higher Δ% means the dropped feature contributed more.
    high_potential_min_dominance_pct: float = 5.0  # >5pct from any top-K feature -> dominant
    init_score_optimal_threshold_pct: float = 1.0  # init_score within 1pct of raw -> already optimal
    marginal_threshold_pct: float = 2.0  # 2pct <= dominance < 5pct -> marginal

    # Higher-is-better metrics (AUC) flip the sign convention for
    # ablation Δ% computation. Auto-derived from target_type at runtime.
    apply_to_target_types: Tuple[str, ...] = ("regression", "binary_classification")

    random_state: int = DEFAULT_RANDOM_SEED


class DummyBaselinesConfig(BaseConfig):
    """Configuration for the pre-training Dummy-baseline report.

    Runs once per (target_type, target_name) AFTER ``BaselineDiagnostics``
    and BEFORE the per-model training loop. Computes a tabular comparison
    of trivial / dummy baselines (mean / median / prior / per_group_mean
    / TS-naive / seasonal-naive / ...) on val + test, picks the strongest
    by a target-type-specific primary metric, and emits one overlay plot
    for the strongest baseline only.

    Sit-alongside relationship with ``BaselineDiagnosticsConfig``: that
    class answers "is the target predictable from these features at
    all?" (LightGBM quick fit + feature ablation); this class answers
    "is the task even hard?" (vs trivial reference predictors).

    Operator contract (per plan v3):
    - Default INFO output: ≤ 2 lines per target (verdict + plot path);
      full table demoted to DEBUG.
    - Suite-end summary block with cross-target verdict table.
    - Four canonical UPPERCASE WARN tokens for grep-able alerts:
      ``BEST_MODEL_BELOW_DUMMY``, ``ALL_BASELINES_BELOW_RANDOM``,
      ``TS_BEATS_TREES``, ``PARTIAL_FAILURE``.
    """

    enabled: bool = True

    # 2026-05-11: render the pre-training "strongest baseline overlay"
    # chart (predictions-vs-actual scatter + residual histogram for
    # regression; class-prior bar for classification). Renders BEFORE
    # the per-model training loop fires so the operator sees the
    # no-model floor next to the verdict line. Default ON per the
    # user's repeated request. Set False to suppress (e.g. headless
    # CI / fuzz where the chart is irrelevant).
    plot_strongest: bool = True

    # Per-target-type opt-out. Default: every supported target type
    # gets baselines. Operator can disable for specific types via
    # ``apply_to_target_types - {"learning_to_rank"}`` etc.
    apply_to_target_types: FrozenSet[str] = frozenset({
        "regression", "binary_classification", "multiclass_classification",
        "multilabel_classification", "learning_to_rank", "quantile_regression",
    })

    # Time-series baseline knobs (only fire when ``ts_field`` is set on
    # the FTE AND the train/val/test split is temporally monotonic).
    # ``ts_extra_periods`` lets the user inject domain-known seasonal
    # periods (e.g. 17-day biological cycles, 90-day quarterly cycles)
    # that the auto-step-size + ACF detector would miss.
    ts_extra_periods: Tuple[int, ...] = ()

    # Per-group baseline (per_group_mean / per_group_prior) leakage
    # defenses (round-3 audit D1).
    # - ``per_group_max_cardinality_ratio``: skip the baseline if the
    #   chosen categorical's unique-count > (n_train * this ratio).
    #   Default 0.5 catches row-id-like keys (user_id, transaction_id,
    #   hash) that would silently produce perfect-prediction oracles.
    # - ``per_group_min_val_coverage_pct``: exclude per_group_* from
    #   strongest-pick eligibility if val coverage of the chosen cat
    #   falls below this. Below 50%, the metric is dominated by
    #   unseen-category fallback (= train_y.mean()) and not by the
    #   group-conditioning effect.
    # - ``per_group_high_overlap_threshold``: if more than this fraction
    #   of val rows have a group with ≥5 train labels, log the
    #   row-label annotation "(high entity overlap — measures
    #   re-appearance, not generalization)".
    per_group_max_cardinality_ratio: float = 0.5
    per_group_min_val_coverage_pct: float = 50.0
    per_group_high_overlap_threshold: float = 0.5

    # n_repeats for stochastic baselines (round-3 audit C#2, C#5).
    # Single-realization variance dominates the AUC / NDCG estimate at
    # small n_val; reporting mean ± std across deterministic seeds
    # gives the operator a noise-floor anchor.
    stratified_n_repeats: int = 20
    random_within_query_n_repeats: int = 10

    # Strongest-pick robustness gate (round-3 D2).
    # Paired bootstrap on the strongest-vs-runner-up baseline pair;
    # if P(strongest beats runner-up) falls below this, annotate the
    # log line as "(TIE)" and suppress the overlay plot.
    paired_bootstrap_n_resamples: int = 1000
    strongest_min_beat_runner_up_prob: float = 0.7

    # Bootstrap CI on the strongest baseline's primary metric, fired
    # only when ``min(n_val, n_test) < bootstrap_ci_threshold`` (point
    # estimate is accurate to <1% above this threshold; CI suppressed
    # to keep output uncluttered).
    bootstrap_ci_threshold: int = 2000
    bootstrap_ci_n_resamples: int = 1000

    # Auto-WARN trigger: model lift below this multiplier vs strongest
    # dummy baseline → ``BEST_MODEL_BELOW_DUMMY`` warning emitted in
    # the suite-end summary. 1.5x is the canonical "your model isn't
    # better than random" Kaggle threshold; can be tightened or
    # loosened per deployment.
    best_model_min_lift: float = 1.5

    # Random seed for stochastic baselines + bootstrap (combined with
    # per-target hash internally to ensure independence across
    # targets — round-3 D13).
    random_state: int = DEFAULT_RANDOM_SEED


# Title-metrics token grammar - mirrors metrics.TITLE_METRIC_TOKENS but kept
# duplicated here to avoid importing from metrics.py at config-class
# definition time (import-cost concern, plus configs is a foundational
# module). Keep these two sets in sync; the validator in ReportingConfig
# falls back gracefully if a new token is added in metrics.py without here.
_REPORTING_ALLOWED_TITLE_TOKENS: FrozenSet[str] = frozenset({
    "ICE", "BR", "BR_DECOMP", "ECE", "CMAEW",
    "COV", "LL", "ROC_AUC", "PR_AUC", "DENS",
})


class ReportingConfig(BaseConfig):
    """Look of the calibration / training performance report.

    Scope: report appearance + per-metric title composition + histogram
    subplot toggles. Filesystem paths live on ``OutputConfig``;
    feature-importance plot parameters live on ``FeatureImportanceConfig``
    (referenced via ``feature_importance_config``).

    Title-metric composition is an ordered string template
    ``title_metrics_template``. The grammar is validated at config
    construction time so an invalid template fails before training
    starts, not mid-figure.

    Token grammar (closed set, case-insensitive on input):
      - ``ICE``: integral calibration error
      - ``BR``: Brier loss (bare)
      - ``BR_DECOMP``: Brier with REL/RES/UNC decomposition parenthetical
        (mutually exclusive with ``BR``)
      - ``ECE``: standard expected calibration error
      - ``CMAEW``: mlframe-native power-weighted calibration MAE
      - ``COV``: bin coverage
      - ``LL``: log loss
      - ``ROC_AUC``: ROC AUC (with grouped variant in brackets when
        group_ids supplied)
      - ``PR_AUC``: PR AUC (followed by PR/RE/F1 trailing)
      - ``DENS``: bin density [max;min]

    Tokens render in the order given. Whitespace-separated. Duplicates
    rejected. Unknown tokens rejected. Empty string is legal (title gets
    only the user-supplied prefix).

    Histogram subplot (``show_prob_histogram``, default True) draws a
    predicted-probability histogram under the reliability scatter, sharing
    the X axis. Y-scale auto-picks log when ``max(hits)/max(min(hits),1) >
    100`` and linear otherwise; override via
    ``prob_histogram_yscale="log" | "linear"``. Inline per-bin population
    text labels next to scatter points are independently controlled by
    ``show_inline_population_labels`` so users can keep both, drop both, or
    keep only one.
    """

    figsize: Tuple[int, int] = (15, 5)
    print_report: bool = True
    show_perf_chart: bool = True
    show_fi: bool = True
    feature_importance_config: Optional[FeatureImportanceConfig] = None
    display_sample_size: int = 0
    show_feature_names: bool = False

    # Per-split metric computation gates (lifted from the trainer-internal
    # TrainingControlConfig so suite users can disable train-set metrics for
    # speed, or enable them for overfit diagnostics). Defaults match the
    # historical trainer-internal hardcoded defaults.
    compute_trainset_metrics: bool = False
    compute_valset_metrics: bool = True
    compute_testset_metrics: bool = True

    # Custom ICE / RICE metric callables (signature: (y_true, y_score) -> float).
    # When None, the trainer falls back to the mlframe-native ICE built from
    # compute_probabilistic_multiclass_error.
    custom_ice_metric: Optional[Callable] = None
    custom_rice_metric: Optional[Callable] = None

    # Histogram subplot - independent toggles for the histogram itself and
    # for the inline population annotations on the scatter plot.
    show_prob_histogram: bool = True
    prob_histogram_yscale: Literal["auto", "log", "linear"] = "auto"
    show_inline_population_labels: bool = True

    # Title-metrics template. Validator parses + populates title_metrics_tokens.
    title_metrics_template: str = "ICE BR_DECOMP ECE CMAEW LL ROC_AUC PR_AUC"
    # Populated by the model_validator after title_metrics_template is validated.
    # Stored as a tuple so downstream hot-path code (fast_calibration_report)
    # never has to re-parse the string. Do not set directly - it is overwritten
    # at construction.
    title_metrics_tokens: Tuple[str, ...] = ()

    # backend x output-format DSL. See ``mlframe.reporting.output.parse_plot_output_dsl`` for grammar.
    #
    # Default keeps interactive plotly HTML (for sharing / jupyter) + matplotlib PNG (10-20x faster, no Chromium). Routing PNG export through kaleido spends 12-15s per figure on a Chromium ``page.reload()``; on a 4-model x VAL+TEST x N-ensemble suite this ballooned to MINUTES of pure chart-export wall-time. Users who need plotly PNG explicitly set ``"plotly[html,png]"``.
    plot_outputs: str = "plotly[html] + matplotlib[png]"

    # Opt-out for jupyter inline plot display.
    # ``None`` (default): auto-detect via ``__IPYTHON__`` / ``sys.ps1`` in ``render_and_save`` - inside a notebook kernel, figures render inline in the cell output AFTER on-disk save (the saved file is the artifact; the inline render is the operator-feedback path).
    # ``True``: force inline display (useful for non-standard runtimes where auto-detection misfires).
    # ``False``: save-to-disk only - skips ``renderer.show(fig)`` even when running inside a kernel. Use for batch jupyter runs (papermill, nbconvert, scheduled notebooks) that don't need cell-output renders AND want to skip the per-figure inline render cost (~50-200ms / figure for plotly, ~20-50ms for matplotlib; accumulates to seconds on a 4-model x VAL+TEST x 6-ensemble suite). Also useful when the inline backend is broken (eg plotly.io renderer misconfigured) and the operator wants the on-disk PNG/HTML without cell-output errors.
    plot_inline_display: Optional[bool] = None

    # Matplotlib style + rcParams override.
    #
    # Use cases:
    # - ``matplotlib_style="ggplot"`` -> use the "ggplot" style sheet for all charts the suite emits. Accepts any name resolvable by ``plt.style.use(...)`` (eg ``"seaborn-v0_8-darkgrid"``, ``"dark_background"``, ``"fivethirtyeight"``, ``"_classic_test_patch"``, or a path to a user-written ``.mplstyle`` file).
    # - ``matplotlib_style=["seaborn-v0_8", "dark_background"]`` - list to layer multiple styles (matplotlib stacks them; later wins on conflict).
    # - ``matplotlib_rcparams={"font.size": 12, "axes.grid": True, ...}`` - direct rcParams dict; merged ON TOP of any style sheet so the user can fine-tune specific keys without writing a full .mplstyle file.
    #
    # Application: both fields are applied to the PROCESS-WIDE matplotlib state at suite entry (mirrors the existing ``plot_inline_display`` plumbing). When ``None`` (default), the user's script-level ``plt.style.use(...)`` / ``plt.rcParams`` settings are preserved untouched - so a one-line ``plt.style.use("ggplot")`` before the suite invocation also works for callers who don't want to thread the field through a config object.
    #
    # The fields are NOT reverted on suite exit; matches the ``plot_inline_display`` semantics (operators expect "set once, see everywhere" for plot styling in a long-running notebook session).
    matplotlib_style: Optional[Union[str, List[str]]] = None
    matplotlib_rcparams: Optional[Dict[str, Any]] = None

    # Plotly template override - separate from the matplotlib style because plotly has its own template system. Common values: ``"plotly"`` (default), ``"plotly_white"``, ``"plotly_dark"``, ``"ggplot2"``, ``"seaborn"``, ``"simple_white"``, ``"presentation"``. Applied via ``plotly.io.templates.default = ...`` at suite entry, process-wide (mirrors matplotlib_style semantics). ``None`` (default) keeps the user's pre-suite plotly setting.
    #
    # Ergonomic note: to unify the look across both backends, set BOTH ``matplotlib_style`` and ``plotly_template`` to matching themes, eg ``matplotlib_style="ggplot"`` + ``plotly_template="ggplot2"``. There is no single "theme" knob that targets both because the available style names and rcParams keys differ between backends.
    plotly_template: Optional[str] = None

    # Per-figure DPI for saved PNG / inline rendering. matplotlib's default is 100. Lowering to 80 cuts savefig wall-time ~30% linearly (verified on a 6-panel multiclass figure: 1330ms -> ~900ms) at a visible-but-acceptable resolution loss; raising to 150 sharpens for publication / slides at a ~2.25x cost. ``None`` (default) defers to matplotlib's global default. Honoured by the matplotlib renderer (``MatplotlibRenderer.save``) and by the legacy ``show_calibration_plot`` save path; plotly path (``plot_outputs`` with ``[png]``) routes through kaleido which has its own DPI knob - when both plotly+matplotlib are emitted, only the matplotlib PNG honours this flag.
    plot_dpi: Optional[int] = None

    # Per-target_type panel templates. Same DSL grammar as ``title_metrics_template`` (space-separated tokens, validator checks against frozen allowed set, no duplicates). All-by-default; operator removes tokens to skip individual panels.
    multiclass_panels: str = "CONFUSION PR_F1 ROC CALIB_GRID PROB_DIST TOP_K_ACC"
    multilabel_panels: str = "PR_F1 CALIB_GRID COOCCURRENCE CARDINALITY JACCARD_DIST"
    ltr_panels: str = "NDCG_K NDCG_DIST LIFT MRR_DIST SCORE_BY_REL"
    quantile_panels: str = "RELIABILITY PINBALL_BY_ALPHA INTERVAL_BAND WIDTH_DIST PIT_HIST"

    @field_validator("title_metrics_template")
    @classmethod
    def _validate_title_template(cls, v: str) -> str:
        toks = [t.strip().upper() for t in v.split() if t.strip()]
        unknown = [t for t in toks if t not in _REPORTING_ALLOWED_TITLE_TOKENS]
        if unknown:
            raise ValueError(
                f"Unknown title-metrics tokens {unknown}. "
                f"Allowed: {sorted(_REPORTING_ALLOWED_TITLE_TOKENS)}"
            )
        if len(toks) != len(set(toks)):
            dupes = sorted({t for t in toks if toks.count(t) > 1})
            raise ValueError(f"Duplicate title-metrics tokens: {dupes}")
        if "BR" in toks and "BR_DECOMP" in toks:
            raise ValueError(
                "BR and BR_DECOMP are mutually exclusive in title_metrics_template"
            )
        return v

    @field_validator("plot_outputs")
    @classmethod
    def _validate_plot_outputs(cls, v: str) -> str:
        # Defer to the DSL parser; it raises ValueError on any malformed
        # / unsupported / duplicate clause. We don't store the parsed
        # spec on the config -- callers re-parse at render time (cheap;
        # parser is regex-based and runs once per chart).
        from mlframe.reporting.output import parse_plot_output_dsl
        parse_plot_output_dsl(v)
        return v

    @field_validator("multiclass_panels", "multilabel_panels", "ltr_panels", "quantile_panels")
    @classmethod
    def _validate_panel_template(cls, v: str, info) -> str:
        # Per-target_type allowed token sets. PR2 will populate the actual
        # panel-builder dispatch; for now we allow the documented token
        # vocabulary (validator catches typos at config-construction time).
        _ALLOWED = {
            "multiclass": frozenset({
                "CONFUSION", "PR_F1", "ROC", "PR_CURVES",
                "CALIB_GRID", "PROB_DIST", "TOP_K_ACC",
            }),
            "multilabel": frozenset({
                "PR_F1", "ROC", "CALIB_GRID", "COOCCURRENCE",
                "CARDINALITY", "JACCARD_DIST", "HAMMING_DIST",
            }),
            "ltr": frozenset({
                "NDCG_K", "NDCG_DIST", "LIFT", "MRR_DIST",
                "SCORE_BY_REL", "TOP1_BY_QSIZE",
            }),
            "quantile": frozenset({
                "RELIABILITY", "PINBALL_BY_ALPHA", "INTERVAL_BAND",
                "WIDTH_DIST", "PIT_HIST",
            }),
        }
        target_key = info.field_name.replace("_panels", "")
        allowed = _ALLOWED[target_key]
        toks = [t.strip().upper() for t in v.split() if t.strip()]
        unknown = [t for t in toks if t not in allowed]
        if unknown:
            raise ValueError(
                f"Unknown {target_key} panel tokens {unknown}. "
                f"Allowed: {sorted(allowed)}"
            )
        if len(toks) != len(set(toks)):
            dupes = sorted({t for t in toks if toks.count(t) > 1})
            raise ValueError(
                f"Duplicate {target_key} panel tokens: {dupes}"
            )
        return v

    @field_validator("feature_importance_config", mode="before")
    @classmethod
    def _coerce_feature_importance_config(cls, v):
        """Accept ``FeatureImportanceConfig`` instances even when the Python
        class identity has diverged.

        Pydantic v2's ``model_type`` validator strictly checks
        ``type(instance) is FeatureImportanceConfig``. Two practical scenarios
        break that without any code bug on either side:
          1) ``%autoreload 2`` in a Jupyter session re-imports ``configs.py``
             after a code edit -- new ``FeatureImportanceConfig`` class is
             defined, but ``trainer.py`` (already imported earlier) still
             references the OLD class and instantiates from it.
          2) Two separate working copies of mlframe sit on ``sys.path`` (e.g.
             a recovery checkout + the canonical one) and import resolution
             picks one for ``configs`` and the other for ``trainer``.

        Both produce ``input_value=FeatureImportanceConfig(...),
        input_type=FeatureImportanceConfig`` errors that are confusing
        because the names match. Round-tripping through ``model_dump()``
        rebuilds the instance against THIS module's class identity and
        recovers transparently. Same-class instances pass through.
        """
        if v is None:
            return None
        if isinstance(v, FeatureImportanceConfig):
            return v
        # Stale-class shim: anything pydantic-shaped with the right name.
        if hasattr(v, "model_dump") and type(v).__name__ == "FeatureImportanceConfig":
            return FeatureImportanceConfig(**v.model_dump())
        # Dicts pass through normal pydantic validation (handled by the
        # default validator after this one returns a dict).
        return v

    @model_validator(mode="after")
    def _populate_title_tokens(self) -> "ReportingConfig":
        toks = tuple(t.strip().upper() for t in self.title_metrics_template.split() if t.strip())
        # Bypass validate_assignment for this derived field so we don't recurse.
        object.__setattr__(self, "title_metrics_tokens", toks)
        return self


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
    "PreprocessingBackendConfig",
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
    "ReportingConfig",
    "FeatureImportanceConfig",
    "OutputConfig",
    "OutlierDetectionConfig",
    "ConfidenceAnalysisConfig",
    "NamingConfig",
    "PredictionsContainer",
    "FairnessConfig",
]
