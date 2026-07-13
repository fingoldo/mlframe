"""Preprocessing-side configs for ``mlframe.training.configs``.

Split out from ``configs.py`` to keep that file below the 1k-line monolith
threshold. Behaviour preserved bit-for-bit; every class is re-exported from
``configs`` so existing
``from mlframe.training.configs import PreprocessingConfig`` (and the other
moved names) imports continue to resolve.

What lives here:
  - ``PreprocessingConfig``
  - ``TrainingSplitConfig``
  - ``PreprocessingBackendConfig``
  - ``PreprocessingExtensionsConfig``
  - ``FeatureTypesConfig``
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from pydantic import ConfigDict, Field, field_validator, model_validator

from ._configs_base import BaseConfig, DEFAULT_RANDOM_SEED


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
    # Fraction of the whole dataset carved as a DISJOINT calibration slice from the TRAIN portion only
    # (never val=early-stop budget, never test=honest holdout). The splitter (make_train_test_split with
    # return_calib=True) carves a group-aware, time-ordered ``calib_idx``; the base model is fit on the
    # shrunk train (train-minus-calib) so calib rows are leakage-free, and finalize auto-fits a post-hoc
    # isotonic calibrator per per-target model on this slice. None/0 -> no carve, behaviour unchanged.
    calib_size: Optional[float] = Field(default=None, ge=0.0, lt=1.0)
    # Second DISJOINT holdout (from the TRAIN portion, like calib) reserved for CONFORMAL residuals of the
    # already-recalibrated shipped predictor: calib_size fits the recalibration map g, conformal_size scores
    # g(model) so the interval reflects what ships (sharing one slice makes residuals in-sample for g ->
    # optimistic coverage). None/0 -> finalize falls back to the calib slice (regression-safe until g exists)
    # or to CV+/OOF residuals. Carved by the SAME structure-aware splitter (group-disjoint / forward-walk).
    conformal_size: Optional[float] = Field(default=None, ge=0.0, lt=1.0)
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
    # ``group_ids`` (e.g. ``SimpleFeaturesAndTargetsExtractor(group_field="group_id")``),
    # the splitter routes through ``GroupShuffleSplit`` so that no group
    # straddles train/val/test. Critical for non-IID data: groups, users,
    # patients, sessions. Without it, an unlucky shuffle leaks rows from
    # the same group into both train and val -- the model memorises the
    # group rather than the underlying signal, val metric inflates, and
    # the gap between val and held-out test (let alone production) is
    # the kind of silent failure that gets caught only after deploy.
    # Set to False to ignore an existing ``group_ids`` and fall back to
    # the historical IID path -- e.g. when groups are present for some
    # downstream purpose (sample weighting) but should not constrain
    # the split.
    use_groups: bool = True

    # Cap on number of distinct composite-stratify classes (multi-head classification). sklearn StratifiedShuffleSplit allocates O(n_classes) buckets per split and requires at least 2 samples per class; >200 classes typically means most classes have <2 rows and the splitter rejects (silent UN-stratified fallback before WARN was added). Users with many-head problems and sufficient n per row-tuple can opt-in to a larger cap; the WARN-on-skip path continues to surface the abandoned stratification.
    composite_cardinality_cap: int = Field(default=200, ge=2)

    # Bucket-stratify ALL split modes (regression by decile/quartile bins, classification by class). When True (default), regression targets are binned and stratified the same way classification targets are -- prevents heavy-tail / multimodal regression from concentrating tail rows in val or test. Combine with groups via iterative-stratification when installed; with timestamps the temporal split wins and a chi-square check on bucket distribution per fold logs a WARN on imbalance.
    bucket_stratify: bool = True

    # Optional chronological-order column (timestamp / monotone index). When set, downstream consumers treat the
    # split as TEMPORAL: the conformal finalize structure-inference (``infer_split_structure``) flags split-conformal
    # marginal coverage as INVALID for this split (it needs online/blocked conformal), and the time-series CV
    # routing (``cv_strategy``) reads it. ``None`` = cross-sectional (legacy behaviour). This is the config surface;
    # routing the suite's MAIN split through ``composite/cv.py`` (purged forward-walk) is wired in a follow-up.
    time_column: Optional[str] = None
    # CV strategy for the main split: "random" (default, legacy) / "timeseries" (forward-walk) / "purged"
    # (forward-walk + purge/embargo). Consumed by the conformal structure-inference today; the make_train_test_split
    # routing lands in the E2 follow-up. Kept here so the intent is declarable now and conformal validity is honest.
    cv_strategy: Literal["random", "timeseries", "purged"] = "random"
    # Embargo gap (rows) for cv_strategy="purged": drop the most-recent ``cv_purge`` TRAIN rows (closest in
    # time to the future val/test block) so a windowed/recurrent label near the boundary cannot leak into the
    # holdout (Lopez de Prado). Default 0 -> no gap (purged == timeseries until set); applied post-split as a
    # pure train-index trim (train only shrinks), so it never reorders or touches val/test.
    cv_purge: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_split_sizes(self) -> "TrainingSplitConfig":
        """Ensure test_size + val_size + calib_size + conformal_size <= 1.0 to leave room for training data."""
        _calib = self.calib_size if self.calib_size is not None else 0.0
        _conformal = self.conformal_size if self.conformal_size is not None else 0.0
        _total = self.test_size + self.val_size + _calib + _conformal
        if _total > 1.0:
            raise ValueError(
                f"test_size ({self.test_size}) + val_size ({self.val_size}) + calib_size ({_calib}) + "
                f"conformal_size ({_conformal}) = {_total} must be <= 1.0"
            )
        if self.cv_strategy in ("timeseries", "purged") and self.val_placement == "backward":
            raise ValueError(
                f"cv_strategy={self.cv_strategy!r} (forward-walk) conflicts with val_placement='backward'; "
                "backward val places val BEFORE train on the timeline, which a forward-walk split cannot honour. "
                "Use val_placement='forward' with a time-series cv_strategy."
            )
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
    # Maximum output column count for the polynomial-feature expansion in ``PolynomialFeatureExpander``.
    # The expander auto-tunes its config (flips ``interaction_only`` -> True, then decrements ``degree``,
    # finally skips the step) until ``projected <= polynomial_max_features``. Default 10_000 keeps the
    # fp32 working set under ~40MB at 1M rows. Set to 0 or None to disable the auto-tune (legacy
    # unbounded behaviour); the warn-only soft floor at polynomial.py:84 still fires.
    polynomial_max_features: Optional[int] = 10_000

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
            raise ValueError(f"imputer_strategy must be one of {{'mean','median','most_frequent','mode'}} or None, " f"got '{v}'")
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
      1.5. Row-wise summary stats (``row_wise_summary_stats_enabled``, default ON) and
           row-wise top-k extreme columns (``row_wise_extreme_columns_enabled``, default ON) --
           purely additive per-row aggregates over the already-numeric column subset, computed
           AFTER the numeric-only filter so they see the same columns the rest of the bridge does
           and BEFORE scaler/kbins/polynomial/dim_reducer so the new columns participate in every
           later step exactly like an ordinary engineered feature.
      2. Scaler (overrides the Polars-ds scaler when set).
      3. Binarizer OR KBinsDiscretizer (mutually exclusive).
      4. PolynomialFeatures (guarded by ``memory_safety_max_features``).
      5. Non-linear feature map (RBFSampler / Nystroem / ...).
      6. Dim reducer (PCA / UMAP / ...).

    ``categorical_powerset_concat_enabled`` / ``categorical_group_concat_auto_enabled`` are the
    exception to the above: they run in ``_phase_fit_pipeline``, BEFORE categorical encoding, not
    inside ``apply_preprocessing_extensions`` -- a composite string-concat column produced after
    encoding would be dropped by step 1.5's numeric-only gate before any downstream step saw it.
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
    nonlinear_features: Optional[Literal["RBFSampler", "Nystroem", "AdditiveChi2Sampler", "SkewedChi2Sampler"]] = None
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

    # Row-wise summary stats (mean/std/q10/q50/q90 by default) computed ACROSS a row's own numeric
    # columns via ``mlframe.feature_engineering.row_wise_summary.row_wise_summary_stats``. Fully
    # generic (only needs the numeric column subset the sklearn-bridge already computes) -- no
    # dataset-specific column names or entity IDs required, so this defaults to ON per the
    # "flip generically-safe additive steps to default-ON" directive (see
    # ``training/core/DEFAULTS_CHANGELOG.md``). Runs on the SAME numeric columns the rest of the
    # bridge already filtered to; disable when the extra columns aren't wanted (e.g. tight
    # ``memory_safety_max_features`` budgets or a frame with too few numeric columns to summarize
    # meaningfully).
    row_wise_summary_stats_enabled: bool = True
    # ``None`` = row_wise_summary_stats' own default (mean/std/q10/q50/q90). Typed knob so callers
    # can request a narrower/wider stat set without reaching for the raw function.
    row_wise_summary_stats_list: Optional[List[str]] = None

    # Row-wise top-k "which columns are most extreme in this row" via
    # ``mlframe.feature_engineering.row_wise_extremality.row_wise_top_k_extreme_columns``. Only the
    # numeric ``topK_score`` columns are kept (the ``topK_column`` name columns are categorical/object
    # and would break the numeric-only sklearn-bridge contract enforced right before this step runs).
    # Same generic-additive rationale as ``row_wise_summary_stats_enabled`` -- defaults to ON.
    row_wise_extreme_columns_enabled: bool = True
    row_wise_extreme_columns_k: int = Field(default=3, ge=1)

    # Categorical composite FE (``mlframe.feature_engineering.categorical_powerset_concat`` /
    # ``categorical_group_concat``). Unlike the row-wise steps above, these run BEFORE categorical
    # encoding (inside ``_phase_fit_pipeline``, not ``apply_preprocessing_extensions`` -- a composite
    # string-concat column produced AFTER encoding would be dropped by the extensions stage's
    # numeric-only gate before any downstream step could see it). Both default OFF: unlike
    # row_wise_summary_stats/row_wise_extreme_columns (fixed, bounded output size), these can blow up
    # column count (powerset is 2^n - 1 in the source column count) or add MI-scoring compute
    # (auto-group discovery), so they're opt-in rather than generically-safe-by-default.
    categorical_powerset_concat_enabled: bool = False
    # Cap subset order (2 = pairwise composites only) -- keeps the 2^n blowup bounded by default.
    categorical_powerset_concat_max_order: int = 2
    categorical_group_concat_auto_enabled: bool = False
    categorical_group_concat_min_mi_gain: float = 0.0
    categorical_group_concat_max_group_size: Optional[int] = None
    # Shared safety cap for BOTH steps: skip (WARN, don't raise) when the detected raw categorical
    # column count exceeds this, so a wide-schema caller can't accidentally trigger a
    # 2^30-column powerset or an O(n^2) MI-scoring pass across hundreds of categorical columns.
    categorical_composite_max_source_columns: int = Field(default=12, ge=2)

    # Entity/time-keyed FE (``mlframe.feature_engineering.state_duration`` /
    # ``recency_aggregation``). Both need a per-row entity key (``group_ids``, threaded through from
    # the FeaturesAndTargetsExtractor's own group-key resolution) and run at the SAME pre-encoding
    # point as the categorical composite steps above (no fit-time state to persist -- these are pure
    # functions of (values, group_ids, order), so predict-time replay just re-runs the same
    # computation on the predict frame's own group_ids/timestamps, no metadata-persisted groups
    # needed). No-op entirely when ``group_ids`` is unavailable (e.g. no FeaturesAndTargetsExtractor,
    # or one that doesn't resolve a group key). Column lists are explicit (mirrors ``tfidf_columns``)
    # since duration/recency semantics only make sense for a CALLER-known boolean/numeric column,
    # not something auto-detectable from dtype alone.
    state_duration_columns: List[str] = Field(default_factory=list)
    state_duration_include_activation_count: bool = False
    recency_aggregation_columns: List[str] = Field(default_factory=list)
    recency_aggregation_scheme: Literal["poly", "exp", "power"] = "poly"
    recency_aggregation_param: float = 1.0
    recency_aggregation_agg: Literal["mean", "sum", "min", "max", "std", "var"] = "mean"

    # Cross-sectional-neighbor FE (``mlframe.feature_engineering.cross_sectional_neighbors``).
    # Unlike the entity/time steps above, the key here is a COLUMN already in the frame (e.g. a
    # ``time_id`` snapshot key), not a side-array -- purely column-declaration-driven, no
    # group_ids/timestamps needed. Same pre-encoding point, no fit-time state (predict-time replay
    # just re-runs the computation). ``None`` snapshot_col (default) is a genuine no-op.
    cross_sectional_neighbors_snapshot_col: Optional[str] = None
    cross_sectional_neighbors_feature_cols: List[str] = Field(default_factory=list)
    cross_sectional_neighbors_k: int = Field(default=10, ge=1)
    cross_sectional_neighbors_agg_stats: List[str] = Field(default_factory=lambda: ["mean", "std"])

    # Two-step recency-weighted target encoding (``mlframe.feature_engineering.two_step_target_encode``).
    # Needs group_ids (entity key) AND y_train -- unlike the entity/time steps above, this has REAL
    # fit-time state (a per-entity encoded-value lookup table, persisted onto metadata) since predict
    # time has no target to encode from. Train rows get the leak-free causal=True expanding-window
    # encoding; val/test/predict rows look up their entity's train-only terminal encoding, falling
    # back to a smoothed global prior for entities unseen in train. ``None`` (default) is a no-op.
    two_step_target_encode_columns: List[str] = Field(default_factory=list)
    two_step_target_encode_decay_half_life: float = 30.0
    two_step_target_encode_smoothing: float = 1.0

    # MA-crossover FE (``mlframe.feature_engineering.ma_crossover``). The suite has no rolling-MA
    # step of its own -- moving averages are computed here (per-entity via group_ids if available,
    # else a single global series ordered by timestamps/row order) before being fed to the
    # underlying pairwise-crossover function. No fit-time state; predict-time replay re-runs the
    # same computation. Needs >=2 distinct windows to produce any pair.
    ma_crossover_columns: List[str] = Field(default_factory=list)
    ma_crossover_windows: List[int] = Field(default_factory=lambda: [3, 5, 10])
    ma_crossover_long_window_weight_power: float = 0.0

    # SVD latent-interaction embeddings (``mlframe.feature_engineering.latent_interaction_svd``).
    # Unlike every other composite-FE step, this one needs a SEPARATE auxiliary reference table --
    # ``train_mlframe_models_suite(auxiliary_events_df=...)`` -- an entity x item interaction log
    # (e.g. customer x product purchases), not columns on train/val/test. The SVD basis is fit ONCE
    # on the auxiliary table; row (entity) embeddings join onto train/val/test via ``group_ids``.
    # REAL fit-time state (the fitted TF-IDF/SVD basis) is persisted directly onto metadata
    # (pickled with the model bundle, same precedent as ``extensions_pipeline``) for predict-time
    # replay, which needs its OWN fresh ``auxiliary_events_df`` at predict time. ``None`` row/col
    # entity (default) is a genuine no-op.
    latent_interaction_svd_row_entity: Optional[str] = None
    latent_interaction_svd_col_entity: Optional[str] = None
    latent_interaction_svd_weight_col: Optional[str] = None
    latent_interaction_svd_time_col: Optional[str] = None
    latent_interaction_svd_decay_half_life: Optional[float] = None
    latent_interaction_svd_use_tfidf: bool = True
    latent_interaction_svd_n_components: int = Field(default=10, ge=1)

    # Nearest-past as-of join (``mlframe.feature_engineering.nearest_past_join``). Like
    # latent_interaction_svd, consumes the SAME top-level ``auxiliary_events_df`` (a separate
    # reference table), but has no fit-time state -- the backward as-of match
    # (``right_df[on] <= left_df[on]``) is inherently leak-safe by construction, so predict-time
    # replay just re-runs the same join against a fresh ``auxiliary_events_df``. ``on``/``by`` must
    # be real columns present in BOTH train/val/test AND ``auxiliary_events_df``.
    nearest_past_join_on: Optional[str] = None
    nearest_past_join_by: List[str] = Field(default_factory=list)
    nearest_past_join_value_cols: Optional[List[str]] = None
    nearest_past_join_fallback_by_chain: Optional[List[Optional[List[str]]]] = None
    nearest_past_join_min_group_size: int = Field(default=1, ge=1)

    # Event-proximity decay FE (``mlframe.feature_engineering.event_proximity_decay``). Unlike every
    # other composite-FE step, the "reference" here isn't derivable from the training frame OR an
    # auxiliary table at all -- it's caller-supplied domain knowledge (e.g. a known holiday
    # calendar), so the event dates are a literal config value. Uses ``timestamps`` (the
    # FeaturesAndTargetsExtractor's ts_field resolution, same source as the entity/time composite-FE
    # steps) rather than a train/val/test column: the suite decomposes datetime COLUMNS into numeric
    # day/month/weekday parts in an earlier phase, before this step's insertion point would see them.
    # No fit-time state; predict-time replay re-runs the same computation using the SAME persisted
    # event-date list against the predict frame's own timestamps.
    event_proximity_decay_event_dates: List[Any] = Field(default_factory=list)
    event_proximity_decay_cap: int = 30
    event_proximity_decay_cap_before: Optional[int] = None
    event_proximity_decay_cap_after: Optional[int] = None
    event_proximity_decay_column_prefix: str = "event_proximity"

    memory_safety_max_features: int = 100_000
    # iter-69 byte-aware guard: PolynomialFeatures' projected column count
    # alone (memory_safety_max_features) doesn't capture the actual
    # byte-cost of the dense output array (= n_samples * projected * 8 for
    # float64). On wide post-onehot frames at modest degree=2 the column
    # count stays under 100_000 but the dense output is a 1+ GiB ndarray
    # that the env can't allocate. ``memory_safety_max_bytes`` adds a
    # bytes-aware soft cap: at fit-time, if n_samples * projected * 8
    # exceeds this cap, ``apply_preprocessing_extensions`` auto-tunes the
    # polynomial step downward (flip interaction_only, decrement degree,
    # skip the step) instead of letting the allocation crash. Default
    # 500 MB targets a generous CI envelope; lower for tight environments
    # or raise for fat boxes. ``None`` disables the byte guard entirely.
    memory_safety_max_bytes: Optional[int] = 500_000_000
    verbose_logging: bool = True
    # PySR symbolic regression -- discovers human-readable equations
    # from the data and adds their output as new numeric features.
    # Requires Julia + SymbolicRegression.jl (installed automatically
    # via PySR / juliacall). Off by default; enable with
    # pysr_enabled=True plus a pysr_params dict for budget control.
    pysr_enabled: bool = False
    # Seed threaded into PySR's internal RNG (sample subsampling, GA initial population).
    # Without this, _apply_pysr_fe used to fall back to 42 unconditionally because the
    # field was absent (suite-level random_seed lives on TrainingSplitConfig, a different
    # object, so getattr(config, "random_seed", 42) always hit the default).
    random_seed: int = 42
    pysr_params: Optional[Dict] = Field(
        default=None,
        description="passed to PySRRegressor() as constructor kwargs (escape hatch for power users; "
        "the typed pysr_* fields below cover the common speed/quality knobs)",
    )

    # Typed PySR knobs (None = use the in-suite default applied in pipeline.py:_apply_pysr_fe).
    # Merge order: pipeline.py defaults < these typed fields < pysr_params dict. Power users keep the
    # raw pysr_params escape hatch; typical callers only ever touch these fields.
    pysr_sample_size: Optional[int] = Field(
        default=None,
        description="Rows fed into PySR (random subsample when train > N). Default 200_000. With "
        "pysr_batching=True this is the pool size PySR samples batch_size rows from per GA iter.",
    )
    pysr_niterations: Optional[int] = Field(
        default=None,
        description="GA iterations. Default 400 (4x PySR's own default 40, 2x bruteforce.py legacy). "
        "Each iter is cheap under batching=True; more iters give better coverage of the pool.",
    )
    pysr_batching: Optional[bool] = Field(
        default=None,
        description="Whether each GA iter samples batch_size rows from the pool (default True). "
        "Off = legacy 'all rows per iter' path. Set False only when sample_size is already small.",
    )
    pysr_batch_size: Optional[int] = Field(
        default=None,
        description="Rows per GA iter when batching=True. Default 10000 (PySR's documented GA knee).",
    )
    pysr_precision: Optional[int] = Field(
        default=None,
        description="Float precision of GA eval: 16 / 32 / 64. Default 32 (~2x faster than 64 on SIMD; "
        "symbolic regression discovers equation FORM, not parameter precision).",
    )
    pysr_top_k: Optional[int] = Field(
        default=None,
        description="Top-K equations (by score) to materialise as new feature columns. Default "
        "min(5, population_size // 2); higher means more columns but diminishing-quality picks.",
    )
    pysr_operator_preset: Optional[str] = Field(
        default=None,
        description="Operator preset for the PySR GA: 'minimal' (legacy log+inv, safe-log fix), "
        "'standard' (default for tabular FE -- adds safe_sqrt, sign, square, tanh, exp, max, min), "
        "or 'physics' (trig + power for oscillatory/wave targets). None means use the in-suite "
        "default ('standard'). See mlframe.feature_engineering.pysr_operators.VALID_PRESETS.",
    )
    pysr_warm_start: Optional[bool] = Field(
        default=None,
        description="When True, PySR persists GA-population state across calls within the same "
        "Python process so a subsequent train_mlframe_models_suite call on similar data resumes "
        "from the prior fit instead of restarting from scratch. Useful for hyperparameter "
        "sweeps + multi-target loops sharing one feature matrix. Breaks if `maxsize` or operator "
        "preset changes between calls -- PySR raises in that case rather than silently dropping "
        "state. Default None = warm_start=False (cold start every call).",
    )

    @field_validator("pysr_precision")
    @classmethod
    def _validate_pysr_precision(cls, v: Optional[int]) -> Optional[int]:
        """Restrict ``pysr_precision`` to the float widths PySR actually supports (16/32/64-bit)."""
        if v is not None and v not in (16, 32, 64):
            raise ValueError(f"pysr_precision must be 16, 32, or 64 (got {v!r})")
        return v

    @field_validator("pysr_operator_preset")
    @classmethod
    def _validate_pysr_operator_preset(cls, v: Optional[str]) -> Optional[str]:
        """Validate ``pysr_operator_preset`` against the registered preset names, deferring the import to avoid a hard dependency on PySR at config-module load time."""
        if v is None:
            return v
        from mlframe.feature_engineering.pysr_operators import VALID_PRESETS
        if v not in VALID_PRESETS:
            raise ValueError(f"pysr_operator_preset must be one of {VALID_PRESETS} (got {v!r})")
        return v

    @model_validator(mode="after")
    def _check_mutual_exclusion(self) -> "PreprocessingExtensionsConfig":
        """Cross-field validation: binarization and k-bins discretization are mutually exclusive, and polynomial_degree/kbins must be >= their minimum meaningful value when set."""
        if self.binarization_threshold is not None and self.kbins is not None:
            raise ValueError("binarization_threshold and kbins are mutually exclusive; set at most one.")
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
    cat_text_cardinality_threshold_pct : float
        Data-size-aware companion to the absolute ``cat_text_cardinality_threshold``. Expressed as a fraction of ``n_rows``; default 0.001 (0.1%). The effective promotion threshold used during auto-detection is
        ``min(cat_text_cardinality_threshold, max(50, int(n_rows * cat_text_cardinality_threshold_pct)))``. Rationale: a flat 300-uniq floor is wrong at both ends of the data-size spectrum - on a 100-row toy dataset every string column stays "cat", on a 10M-row dataset 300 is still tiny relative to the population. The pct knob makes the floor scale with sample size while the absolute ``cat_text_cardinality_threshold`` keeps a hard cap so very large datasets don't accidentally route 100k-uniq columns into the text path. The hard floor of 50 prevents pathologically tiny effective thresholds on micro-datasets. Set to 0 to disable the size-aware floor and recover legacy behaviour (effective = absolute threshold).

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
    # Data-size-aware companion to the absolute cap above. Effective threshold = min(abs_threshold, max(50, int(n_rows * pct))); see docstring for full rationale. 0.0 disables the floor (legacy behaviour: effective == abs_threshold). Default 0.001 = 0.1% of rows.
    cat_text_cardinality_threshold_pct: float = Field(default=0.001, ge=0.0, le=1.0)
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

    @model_validator(mode="after")
    def _check_master_toggle_vs_explicit_lists(self):
        """Raise when use_text_features=False but an explicit text_features list is set
        (same for embedding). Pre-2026-05-20 the master-off silently dropped the
        explicit list per the docstring at lines 670-677; operator who composed a preset
        stack (e.g. tfidf_only -> text_features=[...], then lite_mode flipping
        use_text_features=False) lost their text columns to the cat path silently --
        CatBoost burned minutes on a degenerate ordinal encoding of a 10k-unique
        text column.
        """
        if self.text_features and not self.use_text_features:
            raise ValueError(
                "FeatureTypesConfig: text_features=%r supplied but use_text_features=False. "
                "The list would be silently dropped + columns routed to the cat path. "
                "Either drop the text_features list OR set use_text_features=True." % (self.text_features,)
            )
        return self
