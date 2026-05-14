"""
Model pipeline strategies for mlframe.

Implements the Strategy pattern to handle model-specific preprocessing pipelines.
Each model type may require different preprocessing (scaling, encoding, imputation).
"""

import importlib.util
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional, List, Any, Dict, FrozenSet, Tuple, TYPE_CHECKING
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Pre-compiled slug pattern (MEMORY.md: pre-compile regex at module level).
# Only allow alnum, dash, underscore; everything else collapses to a single "_".
_SLUG_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")

if TYPE_CHECKING:
    import polars as pl

# =============================================================================
# Unified categorical type constants
# =============================================================================
# Used across pipeline.py, trainer.py, utils.py, core.py to detect categoricals.
# Import these instead of hardcoding type lists.

PANDAS_CATEGORICAL_DTYPES: FrozenSet[str] = frozenset({
    "category", "object", "string", "string[pyarrow]", "large_string[pyarrow]",
})

def _polars_categorical_dtypes():
    """Lazy import to avoid importing polars at module level."""
    import polars as pl
    return (pl.Categorical, pl.Utf8, pl.String)

def is_polars_categorical(dtype) -> bool:
    """Check whether a Polars dtype is categorical/string.

    Also accepts ``pl.Enum`` -- a fixed-domain categorical type whose
    instance-level dtype object doesn't compare equal to the class-level
    entry in ``_polars_categorical_dtypes()``. Round-9 probe (2026-04-19)
    found this gap: HGBStrategy's cardinality cast branch was keyed off
    ``dtype == pl.Categorical`` and silently skipped Enum columns,
    treating them as numeric and breaking categorical semantics
    downstream. Same class of bug we already fixed once in
    ``_auto_detect_feature_types`` (round 4). Fix it at the source here
    so every Strategy subclass inherits the correct detection.
    """
    if dtype in _polars_categorical_dtypes():
        return True
    import polars as pl
    if hasattr(pl, "Enum") and isinstance(dtype, pl.Enum):
        return True
    return False

def get_polars_cat_columns(df: "pl.DataFrame") -> list:
    """Detect categorical/string columns from a Polars DataFrame schema."""
    return [name for name, dtype in df.schema.items() if is_polars_categorical(dtype)]


class ModelPipelineStrategy(ABC):
    """
    Abstract base class for model-specific pipeline strategies.

    Different model types have different requirements:
    - Tree models (CB, LGB, XGB): Handle NaN natively, no scaling needed
    - HGB: Needs category encoding but no scaling
    - Neural nets (MLP, NGBoost): Need full preprocessing (encoding + imputation + scaling)
    - Linear models: Need full preprocessing
    """

    @property
    @abstractmethod
    def cache_key(self) -> str:
        """Unique key for caching transformed DataFrames."""
        ...

    @property
    @abstractmethod
    def requires_scaling(self) -> bool:
        """Whether this model type requires feature scaling."""
        ...

    @property
    @abstractmethod
    def requires_encoding(self) -> bool:
        """Whether this model type requires category encoding."""
        ...

    @property
    @abstractmethod
    def requires_imputation(self) -> bool:
        """Whether this model type requires missing value imputation."""
        ...

    @property
    def supports_polars(self) -> bool:
        """Whether this model type can accept Polars DataFrames directly for training."""
        return False

    @property
    def supports_text_features(self) -> bool:
        """Whether this model supports text features (free-text string columns)."""
        return False

    @property
    def supports_embedding_features(self) -> bool:
        """Whether this model supports embedding features (list-of-float vector columns)."""
        return False

    # ---- Multi-output (multiclass + multilabel) capability flags ----
    # Strategies override these to opt INTO native dispatch. Default False
    # means the dispatcher falls back to wrapper-based handling
    # (MultiOutputClassifier for multilabel, default sklearn for
    # multiclass -- most libraries already support multiclass natively
    # via library-specific objective kwargs in helpers._classif_objective_kwargs).

    @property
    def supports_native_multiclass(self) -> bool:
        """Whether this strategy supports K>2 single-label classification
        natively (via library objective kwargs).

        True for CB/XGB/LGB/HGB/Linear (all 5 mlframe strategies have
        native paths). NeuralNet / Recurrent default False (their multi-
        output handling is its own track).
        """
        return False

    @property
    def supports_native_multilabel(self) -> bool:
        """Whether this strategy supports K binary independent labels
        natively (single fitted model returns (N, K) probabilities,
        no MultiOutputClassifier wrapper needed).

        Today only CatBoost (loss_function='MultiLogloss'). Override to
        True in CatBoostStrategy.
        """
        return False

    @property
    def supports_native_ranking(self) -> bool:
        """Whether this strategy supports learning-to-rank natively.

        True for CatBoostStrategy / XGBoostStrategy / LightGBMStrategy
        (all three ship native rankers: ``CatBoostRanker``, ``XGBRanker``,
        ``LGBMRanker``). False for HGB / Linear / Neural / Recurrent --
        no ranker exists in those backends, and the suite skips them
        with NotImplementedError when ``target_type.is_ranking``.
        """
        return False

    @property
    def supports_native_quantile(self) -> bool:
        """Whether this strategy supports single-fit multi-quantile
        regression natively.

        True for CatBoost (``loss_function=MultiQuantile:alpha=...``)
        and XGBoost (``objective=reg:quantileerror, quantile_alpha=[...]``).
        False for everyone else -- LGB, HGB, Linear, MLP, Recurrent
        all need K independent fits stacked via
        ``_QuantileMultiOutputWrapper``.
        """
        return False

    def get_quantile_objective_kwargs(self, qr_config) -> dict:
        """Per-strategy kwargs for quantile-regression objective.

        Returns the dict to merge into the regressor constructor kwargs
        when ``target_type.is_quantile`` is in scope. Default returns
        ``{}`` (subclasses without native quantile support route through
        the wrapper instead -- see ``wrap_quantile``).
        """
        return {}

    def wrap_quantile(self, estimator, qr_config):
        """Wrap a base regressor for quantile-regression dispatch.

        - Strategies with ``supports_native_quantile=True`` (CB / XGB)
          return ``estimator`` unchanged -- the native objective kwargs
          (injected via ``get_quantile_objective_kwargs``) make the
          single fit produce (N, K) predictions.
        - Strategies with ``supports_native_quantile=False`` wrap the
          estimator in ``_QuantileMultiOutputWrapper(base, alphas)`` so
          K independent fits are stacked into an (N, K) prediction.
        """
        if self.supports_native_quantile:
            return estimator
        from .quantile_wrapper import _QuantileMultiOutputWrapper

        return _QuantileMultiOutputWrapper(
            base_estimator=estimator,
            alphas=qr_config.alphas,
            crossing_fix=qr_config.crossing_fix,
            n_jobs=qr_config.wrapper_n_jobs,
        )

    def get_ranker_objective_kwargs(
        self,
        ranking_config=None,
        y_max: Optional[float] = None,
    ) -> dict:
        """Per-strategy ranker kwargs (loss_function / objective + auxiliaries).

        Returns the dict to merge into the ranker constructor kwargs.
        Default implementation returns ``{}`` (subclasses without native
        ranker should never reach this -- ``supports_native_ranking=False``
        gates it). Override in CB/XGB/LGB strategies.

        Parameters
        ----------
        ranking_config : LearningToRankConfig, optional
            User-pinned ranking objectives + ensemble method. Defaults
            applied when None.
        y_max : float, optional
            Maximum value of ``y_train`` -- used by XGB to auto-fall-back
            from ``rank:map`` (binary-only) to ``rank:ndcg`` when graded
            relevance is detected (``y_max > 1``).
        """
        return {}

    def get_classif_objective_kwargs(self, target_type, n_classes: int) -> dict:
        """Per-strategy classifier kwargs for the target type.

        Default implementation delegates to the freestanding
        ``helpers._classif_objective_kwargs`` dispatcher. Strategies can
        override to customise (e.g. force a specific eval_metric on
        multilabel).
        """
        from .helpers import _classif_objective_kwargs

        flavor_map = {
            "CatBoostStrategy": "catboost",
            "XGBoostStrategy": "xgboost",
            "TreeModelStrategy": "lightgbm",  # default tree model is LGB
            "HGBStrategy": "hgb",
            "LinearModelStrategy": "linear",
        }
        flavor = flavor_map.get(type(self).__name__, "")
        return _classif_objective_kwargs(flavor, target_type, n_classes)

    def wrap_multilabel(self, estimator, target_type, multilabel_config=None,
                       n_labels: Optional[int] = None):
        """Multilabel dispatch: native vs wrapper vs chain ensemble.

        Default delegates to ``helpers._maybe_wrap_multilabel`` with the
        strategy's ``supports_native_multilabel`` flag. CatBoostStrategy
        overrides this only if it needs CB-specific behaviour beyond the
        flag check (currently doesn't).
        """
        from .helpers import _maybe_wrap_multilabel

        return _maybe_wrap_multilabel(
            estimator,
            target_type,
            multilabel_config=multilabel_config,
            strategy_supports_native_multilabel=self.supports_native_multilabel,
            n_labels=n_labels,
        )

    def feature_tier(self) -> tuple:
        """Hashable key for grouping models by feature support level.

        Models with the same tier can share trimmed DataFrames (text/embedding
        columns dropped once per tier). Higher tiers support more feature types
        and should train first.
        """
        return (self.supports_text_features, self.supports_embedding_features)

    def prepare_polars_dataframe(self, df: "pl.DataFrame", cat_features: List[str]) -> "pl.DataFrame":
        """Prepare a Polars DataFrame for models that support native Polars input.

        Called in the Polars fastpath before training. Override in subclasses
        to apply model-specific transformations (e.g., casting string columns
        to pl.Categorical for HGB).

        Default implementation returns the DataFrame unchanged (suitable for
        CatBoost which handles all dtypes natively).
        """
        return df

    def build_pipeline(
        self,
        base_pipeline: Optional[Pipeline],
        cat_features: List[str],
        category_encoder: Optional[Any] = None,
        imputer: Optional[Any] = None,
        scaler: Optional[Any] = None,
        output_format: str = "pandas",
    ) -> Optional[Pipeline]:
        """
        Build the preprocessing pipeline for this model type.

        Args:
            base_pipeline: Base feature selection pipeline (e.g., RFECV, MRMR) or
                custom transformer (e.g., IncrementalPCA)
            cat_features: List of categorical feature names
            category_encoder: Encoder for categorical features
            imputer: Imputer for missing values
            scaler: Scaler for feature normalization
            output_format: ``"pandas"`` (default) or ``"polars"``. Routed to the
                Pipeline's ``set_output(transform=...)`` so DataFrame dtypes
                survive the chain. Choose ``"polars"`` when the downstream
                consumer is Polars-native (CB / XGB Polars fastpath, HGB) and
                you want to skip the arrow->pandas bridge; ``"pandas"`` for LGB
                and other pandas-only consumers. Requires sklearn >= 1.4 for
                ``"polars"`` support.

        Returns:
            Configured sklearn Pipeline or None if no preprocessing needed

        Note:
            Feature selectors (MRMR, RFECV, SelectorMixin) run FIRST (before preprocessing).
            Custom transformers (PCA, etc.) run LAST (after preprocessing).
        """
        from sklearn.feature_selection import SelectorMixin
        from mlframe.feature_selection.filters import MRMR

        steps = []

        # Determine if base_pipeline is a feature selector (runs FIRST) or transformer (runs LAST)
        is_feature_selector = False
        if base_pipeline is not None:
            is_feature_selector = (
                isinstance(base_pipeline, SelectorMixin)
                or isinstance(base_pipeline, MRMR)
                or isinstance(base_pipeline, Pipeline)  # nested sklearn Pipeline is treated as pre-step
                or hasattr(base_pipeline, 'get_support')  # RFECV and similar
            )

        # Feature selectors go FIRST (before preprocessing)
        if base_pipeline is not None and is_feature_selector:
            steps.append(("pre", base_pipeline))

        # Add category encoding if required and categorical features exist.
        # Observability guard (2026-04-19 round-9 probe): if the strategy
        # declares ``requires_encoding=True`` AND there are cat_features
        # in the data BUT the caller passed ``category_encoder=None``,
        # silently skipping the step meant unbounded categorical string
        # values fed to a model that expected numeric -- sklearn then
        # raised opaquely inside ``LinearRegression.fit`` / etc. Now: WARN
        # so operators see the missing dependency at the source. We don't
        # raise because some tests/callers legitimately pre-encode cats
        # upstream and pass encoder=None; the WARN is enough signal.
        if self.requires_encoding and cat_features:
            if category_encoder is not None:
                steps.append(("ce", category_encoder))
            else:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "%s.build_pipeline: requires_encoding=True and %d "
                    "categorical feature(s) present, but category_encoder "
                    "is None. Encoding step skipped -- downstream model.fit "
                    "may raise on raw string categoricals. Supply a "
                    "category_encoder (e.g. sklearn.preprocessing."
                    "OrdinalEncoder) or pre-encode cats upstream.",
                    type(self).__name__, len(cat_features),
                )

        # Add imputation if required
        if self.requires_imputation and imputer is not None:
            steps.append(("imp", imputer))

        # Add scaling if required
        if self.requires_scaling and scaler is not None:
            steps.append(("scaler", scaler))

        # Custom transformers go LAST (after preprocessing)
        if base_pipeline is not None and not is_feature_selector:
            steps.append(("transform", base_pipeline))

        if not steps:
            return base_pipeline

        # Avoid wrapping base_pipeline in redundant Pipeline when it's the only step
        if len(steps) == 1 and steps[0][0] == "pre":
            return base_pipeline
        if len(steps) == 1 and steps[0][0] == "transform":
            return base_pipeline

        pipeline = Pipeline(steps=steps)
        # Ensure DataFrame dtypes (pd.Categorical, object, pl.Enum) survive the chain.
        # sklearn's default returns numpy, which destroys categoricals -- LGB/CB/XGB
        # then receive numpy with string values and crash on Dataset construction
        # (e.g. "could not convert string to float: 'HOURLY'"). set_output keeps
        # the frame as the requested type so downstream isinstance(X, pd_DataFrame)
        # / pl.DataFrame branches take the native fastpath. Best-effort: some
        # nested transformers (custom, third-party) don't declare
        # get_feature_names_out and sklearn refuses to configure; swallow and
        # continue. "polars" requires sklearn >= 1.4 -- older versions raise.
        try:
            pipeline = pipeline.set_output(transform=output_format)
        except Exception:
            if output_format != "pandas":
                # Fall back to pandas (works on sklearn >= 1.2) if polars is
                # rejected by the installed sklearn or by an inner transformer.
                try:
                    pipeline = pipeline.set_output(transform="pandas")
                except Exception:
                    pass
        return pipeline


class TreeModelStrategy(ModelPipelineStrategy):
    """
    Strategy for tree-based models (CatBoost, LightGBM, XGBoost).

    These models:
    - Handle NaN values natively
    - Don't require feature scaling
    - CatBoost handles categorical features natively
    - LightGBM/XGBoost can handle categoricals with proper setup
    """

    cache_key = "tree"
    requires_scaling = False
    requires_encoding = False
    requires_imputation = False
    # All tree models (CB/LGB/XGB) support multiclass natively via library
    # objective kwargs. Multilabel native is CB-only -- overridden in
    # CatBoostStrategy. LGB has no native multilabel (issue #524 since 2017),
    # XGB 3.x experimental but unstable.
    supports_native_multiclass = True
    # LGB has native LGBMRanker; CB/XGB override below with their own
    # objective dispatch. Setting True at TreeModelStrategy level means
    # the default (LGB) path is correctly enabled.
    supports_native_ranking = True

    def get_ranker_objective_kwargs(self, ranking_config=None, y_max=None):
        """LGBMRanker objective. ``lambdarank`` (default) handles both
        binary and graded relevance. ``rank_xendcg`` is an alternative.
        """
        objective = "lambdarank"
        if ranking_config is not None:
            objective = getattr(ranking_config, "lgb_objective", None) or objective
        return {
            "objective": objective,
            # eval_metric defaults to ndcg for ranker; expose explicitly.
            "metric": "ndcg",
        }

    def build_pipeline(
        self,
        base_pipeline: Optional[Pipeline],
        cat_features: List[str],
        category_encoder: Optional[Any] = None,
        imputer: Optional[Any] = None,
        scaler: Optional[Any] = None,
    ) -> Optional[Pipeline]:
        """Tree models just use the base pipeline (feature selection) if any."""
        return base_pipeline


class CatBoostStrategy(TreeModelStrategy):
    """
    Strategy for CatBoost models.

    Inherits tree model behavior and additionally supports native Polars DataFrames,
    allowing training without pandas conversion (CatBoost >= 1.2.7).
    Also supports text_features and embedding_features natively.
    """

    supports_polars = True
    supports_text_features = True
    supports_embedding_features = True
    # 2026-04-24: native multi-output support via loss_function='MultiClass'
    # for K>2 single-label and 'MultiLogloss' for K independent binary
    # outputs. The dispatch wires these via
    # ModelPipelineStrategy.get_classif_objective_kwargs +
    # _maybe_wrap_multilabel (which short-circuits the wrapper for
    # supports_native_multilabel=True strategies).
    supports_native_multiclass = True
    supports_native_multilabel = True
    supports_native_ranking = True
    # 2026-05-08 QR: CatBoost MultiQuantile loss handles K alphas in one
    # fit; predict returns (N, K) directly.
    supports_native_quantile = True
    # Inherits cache_key = "tree" from TreeModelStrategy so CB/LGB/XGB share
    # transformed-DF cache (they have identical preprocessing requirements).

    def get_quantile_objective_kwargs(self, qr_config) -> dict:
        """CatBoost ``MultiQuantile`` loss_function with comma-joined alphas.

        Format: ``"MultiQuantile:alpha=0.1,0.5,0.9"`` (no brackets, no
        spaces). predict() then returns shape (N, K).
        """
        alphas_str = ",".join(str(a) for a in qr_config.alphas)
        return {"loss_function": f"MultiQuantile:alpha={alphas_str}"}

    def get_ranker_objective_kwargs(self, ranking_config=None, y_max=None):
        """CatBoostRanker loss_function + sensible eval_metric.

        Defaults to ``YetiRankPairwise`` (listwise pairwise — robust on
        both graded and binary labels). Override via
        ``LearningToRankConfig.cb_loss_fn``.

        ``y_max`` is unused by CB (its ranker losses accept both binary
        and graded labels uniformly).
        """
        loss_fn = "YetiRankPairwise"
        if ranking_config is not None:
            loss_fn = getattr(ranking_config, "cb_loss_fn", None) or loss_fn
        return {
            "loss_function": loss_fn,
            # CB ranker exposes NDCG / MAP / MRR via PFound-family eval
            # metrics; use NDCG as the default for early-stopping. Users
            # can override via hyperparams.
            "eval_metric": "NDCG",
        }


class XGBoostStrategy(TreeModelStrategy):
    """
    Strategy for XGBoost models (>= 3.1).

    Inherits tree model behavior and additionally supports native Polars DataFrames.
    XGBoost auto-detects pl.Categorical columns when enable_categorical=True,
    but pl.String columns must be cast to pl.Categorical first.
    No cardinality limit (unlike HGB).
    """

    supports_polars = True
    # XGB has native multiclass via objective='multi:softprob'+num_class.
    # XGB 3.x has experimental multi_strategy='multi_output_tree' for
    # multilabel -- opt-in via MultilabelDispatchConfig.force_native_xgb_multilabel
    # (default False, uses MultiOutputClassifier wrapper). Marked WIP by
    # upstream until v3.1 stable; opting in earlier accepts the upstream
    # stability risk for vector-output trees (smaller model, integrated
    # GPU/SHAP support, faster inference).
    supports_native_multiclass = True
    supports_native_ranking = True
    # 2026-05-08 QR: XGBoost >=2.0 supports single-fit multi-quantile via
    # ``objective="reg:quantileerror", quantile_alpha=[0.1,0.5,0.9]``;
    # predict() returns (N, K).
    supports_native_quantile = True
    # supports_native_multilabel: declared False at class level (matches the
    # ABC default + tells callers "wrapper by default"). The actual native-
    # multilabel decision is dynamic -- see wrap_multilabel + get_classif_
    # objective_kwargs overrides below, which BOTH consult the runtime
    # MultilabelDispatchConfig.force_native_xgb_multilabel flag.
    # Inherits cache_key = "tree" from TreeModelStrategy.

    def get_quantile_objective_kwargs(self, qr_config) -> dict:
        """XGBoost native multi-quantile via ``reg:quantileerror`` +
        ``quantile_alpha=[a1, a2, ...]`` (XGBoost >= 2.0).

        predict() then returns (N, K) -- one column per alpha.
        """
        return {
            "objective": "reg:quantileerror",
            "quantile_alpha": list(qr_config.alphas),
        }

    def get_ranker_objective_kwargs(self, ranking_config=None, y_max=None):
        """XGBRanker objective + auto-fallback for graded labels.

        Default ``rank:ndcg`` (works on graded relevance). When user
        pinned ``rank:map`` but ``y_max > 1`` is detected (graded
        labels), auto-fall-back to ``rank:ndcg`` with WARN -- XGBoost's
        C++ ``is_binary`` check would otherwise crash with a cryptic
        message.

        ``rank:pairwise`` accepted for both binary and graded.
        """
        objective = "rank:ndcg"
        if ranking_config is not None:
            objective = getattr(ranking_config, "xgb_objective", None) or objective

        if (
            objective == "rank:map"
            and y_max is not None
            and y_max > 1
            and ranking_config is not None
            and getattr(ranking_config, "autodetect_label_format", True)
        ):
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "XGB rank:map requires binary labels (y in {0,1}); detected "
                "y_max=%s > 1 (graded relevance). Auto-falling back to "
                "rank:ndcg. Set ranking_config.autodetect_label_format=False "
                "to disable this safety check.",
                y_max,
            )
            objective = "rank:ndcg"

        return {
            "objective": objective,
            # XGB's eval_metric for ranker also defaults to ndcg; expose
            # explicitly so log/early-stop reads the right thing.
            "eval_metric": "ndcg",
        }

    def get_classif_objective_kwargs(self, target_type, n_classes: int,
                                      multilabel_config=None) -> dict:
        """Override base to support opt-in native XGB multilabel.

        When ``target_type == MULTILABEL_CLASSIFICATION`` AND
        ``multilabel_config.force_native_xgb_multilabel`` is True, return
        the native vector-output-tree kwargs:
          {'objective': 'binary:logistic',
           'multi_strategy': 'multi_output_tree',
           'tree_method': 'hist'}

        Otherwise falls back to the base dispatcher (which returns ``{}``
        for multilabel -- wrapper path takes over).
        """
        from .configs import TargetTypes as _TT, MultilabelDispatchConfig
        if (
            target_type == _TT.MULTILABEL_CLASSIFICATION
            and multilabel_config is not None
            and getattr(multilabel_config, "force_native_xgb_multilabel", False)
        ):
            # XGB 3.x experimental native multilabel. tree_method='hist'
            # is required; binary:logistic per-output objective.
            return {
                "objective": "binary:logistic",
                "multi_strategy": "multi_output_tree",
                "tree_method": "hist",
            }
        return super().get_classif_objective_kwargs(target_type, n_classes)

    def wrap_multilabel(self, estimator, target_type, multilabel_config=None,
                       n_labels: Optional[int] = None):
        """Override base to opt into native XGB multilabel when configured.

        When ``force_native_xgb_multilabel=True``, return ``estimator``
        unchanged (no MultiOutputClassifier wrapper) -- the kwargs from
        ``get_classif_objective_kwargs`` already configured the native
        multi-output tree path.
        """
        from .configs import TargetTypes as _TT
        if (
            target_type == _TT.MULTILABEL_CLASSIFICATION
            and multilabel_config is not None
            and getattr(multilabel_config, "force_native_xgb_multilabel", False)
        ):
            return estimator
        # Fall back to base (MultiOutputClassifier / chain / etc.)
        return super().wrap_multilabel(
            estimator, target_type,
            multilabel_config=multilabel_config, n_labels=n_labels,
        )

    def prepare_polars_dataframe(
        self,
        df: "pl.DataFrame",
        cat_features: List[str],
        category_map: Optional[Dict[str, "pl.Enum"]] = None,
    ) -> "pl.DataFrame":
        """Cast string columns to a categorical dtype for XGBoost auto-detection.

        XGBoost detects polars categorical/Enum dtypes natively when
        ``enable_categorical=True`` but does not handle raw ``pl.String``.

        Why ``pl.Enum`` over ``pl.Categorical`` (2026-04-28):
        Polars 1.x has the global string cache enabled by default and
        ``pl.disable_string_cache()`` is a no-op. Every ``pl.Categorical``
        Series in the process therefore shares one monotonically growing
        dictionary. Across pytest runs this lets categories from earlier
        tests bleed into a later test's column dtype: train_df might be
        fitted when the global dict is small, val_df at predict time has
        the same column with a larger dtype.cats list, and XGBoost's
        ``cat_container`` raises "Found a category not in the training
        set" for the ghost levels even when the actual values column is
        clean. ``pl.Enum`` is per-Series (no shared cache), so the dtype
        is fully determined by the levels we pass in.

        ``category_map`` (preferred): a {col -> pl.Enum} dict the caller
        builds from the union of train+val unique values (test must be
        excluded to avoid label leakage). When supplied, all string /
        Categorical / existing-Enum columns named in the map are cast
        to that exact Enum, giving train/val/test (and predict-time
        frames) identical, leak-free dtypes.

        Fallback (no map): cast each column to a per-DF Enum built from
        its own unique values. This still avoids the global-cache leak,
        but train/val/test produced from independent calls will have
        different Enum dtypes -- fine when the caller only ever passes
        one frame, but at predict time XGBoost will reject any val/test
        category not present in the train Enum. Use the explicit
        ``category_map`` route for predict-time correctness.
        """
        import polars as pl

        # When a leak-free pl.Enum map is supplied, cast every named
        # column (cat or already-categorical) to that exact Enum so
        # train/val/test dtypes match across calls.
        if category_map is not None:
            exprs: List[Any] = []
            for c, enum_dtype in category_map.items():
                if c in df.columns:
                    # Map is built train+val UNION (test excluded for leak-freeness),
                    # so test rows may carry OOV values. strict=False nulls them out
                    # rather than crashing -- consistent with core.py:2992 behaviour.
                    exprs.append(pl.col(c).cast(pl.String).cast(enum_dtype, strict=False).alias(c))
            if exprs:
                df = df.with_columns(exprs)
            # Still need to cover any pl.String / pl.Utf8 not present in
            # the map (e.g. brand-new columns surfacing only after the
            # map was built) -- fall back to per-DF Enum.
            remaining = {
                name for name, dtype in df.schema.items()
                if (dtype in (pl.Utf8, pl.String)) and name not in category_map
            }
            if remaining:
                fallback_exprs = []
                for c in remaining:
                    vals = df[c].drop_nulls().unique().cast(pl.String).to_list()
                    enum_dtype = pl.Enum(sorted(set(vals)))
                    fallback_exprs.append(pl.col(c).cast(pl.String).cast(enum_dtype).alias(c))
                df = df.with_columns(fallback_exprs)
            return df

        # No map supplied: legacy behaviour, but using pl.Enum (not
        # pl.Categorical) so the cast doesn't pollute polars 1.x's
        # default global string cache. ``pl.String`` / ``pl.Utf8``
        # columns get an Enum built from their own unique values.
        schema_cats = {
            name for name, dtype in df.schema.items()
            if dtype in (pl.Utf8, pl.String)
        }
        cols_to_cast = schema_cats | {
            c for c in (cat_features or [])
            if c in df.columns and df[c].dtype in (pl.Utf8, pl.String)
        }
        if not cols_to_cast:
            return df
        exprs = []
        for c in cols_to_cast:
            vals = df[c].drop_nulls().unique().cast(pl.String).to_list()
            enum_dtype = pl.Enum(sorted(set(vals)))
            exprs.append(pl.col(c).cast(pl.String).cast(enum_dtype).alias(c))
        return df.with_columns(exprs)

    def build_polars_enum_map(
        self,
        train_df: "pl.DataFrame",
        val_df: "Optional[pl.DataFrame]",
        cat_features: List[str],
    ) -> "Dict[str, pl.Enum]":
        """Build per-column ``pl.Enum`` dtypes from the union of train+val
        unique values. Test data is intentionally excluded -- letting test
        levels widen the Enum would leak label-time information back into
        the model's accepted-category set.

        Returns ``{col_name: pl.Enum([...])}`` for every string /
        Categorical / Enum column present in ``train_df``. Columns absent
        from ``val_df`` contribute only their train levels (still safe).
        """
        import polars as pl

        cat_features = cat_features or []
        candidate_cols = [
            name for name, dtype in train_df.schema.items()
            if dtype in (pl.Utf8, pl.String)
            or dtype == pl.Categorical
            or isinstance(dtype, pl.Enum)
            or name in cat_features
        ]
        candidate_cols = [c for c in candidate_cols if c in train_df.columns]

        # 2026-05-08 perf: batch per-column unique extraction into one
        # collect() per frame (train + val). The previous loop did
        # ``df[col].unique()`` per cat col -- on c0031 (15 cat cols x
        # 2 frames = 30 collects per build) that cost ~300ms across
        # the suite via PyLazyFrame.collect. Batched via implode() it's
        # 2 collects total per call. Same pattern as session 1 win #2
        # (get_trainset_features_stats_polars). Falls back to per-col
        # loop on any error so one bad cast doesn't poison the frame.
        out: Dict[str, pl.Enum] = {}

        def _batched_unique(df: "pl.DataFrame") -> "Dict[str, list]":
            cols_present = [c for c in candidate_cols if c in df.columns]
            if not cols_present:
                return {}
            try:
                lf = df.lazy().select([
                    pl.col(c).cast(pl.String).drop_nulls().unique().implode().alias(c)
                    for c in cols_present
                ])
                row = lf.collect()
                return {c: row[c][0].to_list() for c in cols_present}
            except Exception:
                d: Dict[str, list] = {}
                for c in cols_present:
                    try:
                        d[c] = df[c].drop_nulls().unique().cast(pl.String).to_list()
                    except Exception:
                        d[c] = []
                return d

        train_levels = _batched_unique(train_df)
        val_levels = _batched_unique(val_df) if val_df is not None else {}
        for col in candidate_cols:
            levels: set = set()
            levels.update(train_levels.get(col, []))
            levels.update(val_levels.get(col, []))
            out[col] = pl.Enum(sorted(levels))
        return out


class HGBStrategy(ModelPipelineStrategy):
    """
    Strategy for HistGradientBoosting models.

    These models:
    - Handle NaN values natively
    - Don't require feature scaling
    - Support Polars DataFrames natively (numeric + pl.Categorical)
    - Require category encoding only on the pandas fallback path
    - Hard limit: categorical cardinality must be <= 255 (max_bins constraint)
    """

    cache_key = "hgb"
    requires_scaling = False
    requires_encoding = True  # pandas fallback path still needs encoding
    requires_imputation = False
    supports_polars = True
    # sklearn HistGradientBoostingClassifier auto-detects multiclass from y dtype
    # (no library kwarg needed). No native multilabel; uses MultiOutputClassifier.
    supports_native_multiclass = True

    # HGB max_bins is capped at 255 in sklearn
    _MAX_CATEGORICAL_CARDINALITY = 255

    def prepare_polars_dataframe(
        self,
        df: "pl.DataFrame",
        cat_features: List[str],
        category_map: Optional[Dict[str, "pl.Enum"]] = None,
    ) -> "pl.DataFrame":
        """Cast categorical columns for HGB compatibility, using leak-free
        ``pl.Enum`` (not ``pl.Categorical``) for the same reason XGB does:
        polars 1.x's default global string cache makes every
        ``pl.Categorical`` Series in the process share one growing
        dictionary, so the column's physical codes drift across runs.
        sklearn HGB reads the underlying integer codes directly when
        the dtype reports as categorical, so cross-run code drift is a
        latent pickle-reload hazard.

        - Cardinality <= 255: cast to ``pl.Enum`` (HGB auto-detects via from_dtype)
        - Cardinality > 255: ordinal-encode to ``pl.UInt32`` (treated as continuous)

        ``category_map`` (preferred): a {col -> pl.Enum} dict the caller
        builds from the union of train+val unique values via
        ``build_polars_enum_map``. When supplied, train/val/test cast to
        the SAME Enum so codes are consistent across splits.
        """
        import polars as pl

        from .utils import filter_existing

        schema_cats = set(get_polars_cat_columns(df))
        all_cats = schema_cats | set(cat_features or [])
        existing = filter_existing(df, all_cats)
        if not existing:
            return df

        casts = []
        for col in existing:
            dtype = df[col].dtype
            n_unique = df[col].n_unique()
            high_card = n_unique > self._MAX_CATEGORICAL_CARDINALITY
            if category_map is not None and col in category_map:
                enum_dt = category_map[col]
                # category_map is built from train+val UNION (test EXCLUDED, leak-free).
                # Test rows therefore can carry values absent from the Enum's domain.
                # Use strict=False so OOV values fall through to null rather than
                # crash the lazy collect, matching the dict-alignment routine at
                # core.py:2992 which also passes strict=False on the test split.
                if high_card:
                    casts.append(
                        pl.col(col).cast(pl.String).cast(enum_dt, strict=False).to_physical().cast(pl.UInt32).alias(col)
                    )
                else:
                    casts.append(pl.col(col).cast(pl.String).cast(enum_dt, strict=False).alias(col))
                continue
            # No supplied map: build a per-DF Enum from this frame's own values.
            try:
                vals = df[col].drop_nulls().unique().cast(pl.String).to_list()
            except Exception:
                vals = []
            local_enum = pl.Enum(sorted(set(vals))) if vals else None
            if local_enum is None:
                continue
            if high_card:
                casts.append(
                    pl.col(col).cast(pl.String).cast(local_enum).to_physical().cast(pl.UInt32).alias(col)
                )
            else:
                casts.append(pl.col(col).cast(pl.String).cast(local_enum).alias(col))

        if casts:
            df = df.with_columns(casts)
        return df

    def build_polars_enum_map(
        self,
        train_df: "pl.DataFrame",
        val_df: "Optional[pl.DataFrame]",
        cat_features: List[str],
    ) -> "Dict[str, pl.Enum]":
        """Mirror of ``XGBoostStrategy.build_polars_enum_map``: leak-free
        per-column Enum from train+val UNION (test EXCLUDED). HGB's
        cardinality split into Enum vs UInt32 happens at
        ``prepare_polars_dataframe`` time using the same map - the map
        always carries the FULL value set, the cardinality decision is
        applied per frame.
        """
        import polars as pl

        cat_features = cat_features or []
        candidate_cols = [
            name for name, dtype in train_df.schema.items()
            if dtype in (pl.Utf8, pl.String)
            or dtype == pl.Categorical
            or isinstance(dtype, pl.Enum)
            or name in cat_features
        ]
        candidate_cols = [c for c in candidate_cols if c in train_df.columns]

        # 2026-05-08 perf: batch per-column unique extraction into one
        # collect() per frame (train + val). The previous loop did
        # ``df[col].unique()`` per cat col -- on c0031 (15 cat cols x
        # 2 frames = 30 collects per build) that cost ~300ms across
        # the suite via PyLazyFrame.collect. Batched via implode() it's
        # 2 collects total per call. Same pattern as session 1 win #2
        # (get_trainset_features_stats_polars). Falls back to per-col
        # loop on any error so one bad cast doesn't poison the frame.
        out: Dict[str, pl.Enum] = {}

        def _batched_unique(df: "pl.DataFrame") -> "Dict[str, list]":
            cols_present = [c for c in candidate_cols if c in df.columns]
            if not cols_present:
                return {}
            try:
                lf = df.lazy().select([
                    pl.col(c).cast(pl.String).drop_nulls().unique().implode().alias(c)
                    for c in cols_present
                ])
                row = lf.collect()
                return {c: row[c][0].to_list() for c in cols_present}
            except Exception:
                d: Dict[str, list] = {}
                for c in cols_present:
                    try:
                        d[c] = df[c].drop_nulls().unique().cast(pl.String).to_list()
                    except Exception:
                        d[c] = []
                return d

        train_levels = _batched_unique(train_df)
        val_levels = _batched_unique(val_df) if val_df is not None else {}
        for col in candidate_cols:
            levels: set = set()
            levels.update(train_levels.get(col, []))
            levels.update(val_levels.get(col, []))
            out[col] = pl.Enum(sorted(levels))
        return out


class NeuralNetStrategy(ModelPipelineStrategy):
    """
    Strategy for neural network models (MLP, NGBoost).

    These models:
    - Cannot handle NaN values - need imputation
    - Benefit significantly from feature scaling
    - Require category encoding

    Multi-output dispatch (2026-05-07):
    - **multiclass**: native via ``F.cross_entropy`` (default loss_fn) +
      softmax in ``MLPTorchModel.predict_step`` for K>1 outputs. Already
      works at the model level; the flag below makes the dispatch
      consistent across strategies.
    - **multilabel**: native via per-label ``F.binary_cross_entropy_with_logits``
      + sigmoid output (separate path; see ``get_classif_objective_kwargs``).
    - **learning_to_rank**: native via RankNet/ListNet pairwise loss in
      ``mlframe.training.neural.ranker.MLPRanker``.
    """

    cache_key = "neural"
    requires_scaling = True
    requires_encoding = True
    requires_imputation = True
    supports_native_multiclass = True
    supports_native_multilabel = True
    supports_native_ranking = True

    def get_classif_objective_kwargs(self, target_type, n_classes: int,
                                      multilabel_config=None) -> dict:
        """Per-target loss_fn dispatch for the MLP estimator.

        Returned dict is consumed by ``_configure_mlp_params`` (trainer.py)
        which threads it into ``mlp_kwargs.model_params.loss_fn`` +
        ``mlp_kwargs.datamodule_params.labels_dtype``. Returns the empty
        dict for binary (default ``F.cross_entropy`` already correct).
        """
        from .configs import TargetTypes as _TT

        # Lazy import torch so a non-MLP run doesn't pay for PL/torch import.
        import torch
        import torch.nn.functional as F

        if target_type is None or target_type == _TT.BINARY_CLASSIFICATION:
            return {}  # default cross_entropy is correct for binary
        if target_type == _TT.MULTICLASS_CLASSIFICATION:
            # Default ``F.cross_entropy`` + ``int64`` labels already
            # handle K>2 -- explicit return for symmetry with other strategies.
            return {"loss_fn": F.cross_entropy, "labels_dtype": torch.int64}
        if target_type == _TT.MULTILABEL_CLASSIFICATION:
            # Per-label sigmoid: BCE with logits is numerically stable
            # and accepts (N, K) float32 labels.
            return {
                "loss_fn": F.binary_cross_entropy_with_logits,
                "labels_dtype": torch.float32,
                # Predict-time sigmoid signal so MLPTorchModel.predict_step
                # uses sigmoid (not softmax) for K>1 outputs.
                "task_type": "multilabel",
            }
        return {}

    def get_ranker_objective_kwargs(self, ranking_config=None, y_max=None):
        """MLPRanker loss_fn dispatch. Default ``ranknet`` (pairwise BCE
        on score differences); alternative ``listnet`` (listwise softmax
        cross-entropy). Both accept binary or graded relevance.

        ``y_max`` unused -- both losses handle the full label range.
        ``ranking_config.lgb_objective`` doesn't apply to MLP; MLPRanker
        consumes loss_fn directly via the ``loss_fn`` key.
        """
        loss_fn = "ranknet"
        if ranking_config is not None:
            # Optional override via a dedicated MLP key. Keeps the per-
            # library config clean (cb_loss_fn / xgb_objective / lgb_objective
            # for those three; mlp_loss_fn for MLP).
            loss_fn = getattr(ranking_config, "mlp_loss_fn", None) or loss_fn
        return {"loss_fn": loss_fn}


class LinearModelStrategy(ModelPipelineStrategy):
    """
    Strategy for linear models (Linear, Ridge, Lasso, ElasticNet, etc.).

    These models:
    - Cannot handle NaN values - need imputation
    - Require feature scaling for proper regularization
    - Require category encoding

    Multi-output dispatch:

    - **multiclass**: ``LogisticRegression`` auto-detects K since
      sklearn 1.5; ``multi_class`` kwarg removed in 1.8 (defaults to
      multinomial when liblinear isn't the solver). ``RidgeClassifier``
      / ``SGDClassifier`` use OvR by default. Strategy-level flag = True.
    - **multilabel**: known sklearn quirk that ``RidgeClassifier`` /
      ``RidgeClassifierCV`` accept 2-D y natively (treats as multi-output
      ridge regression + threshold; ``predict`` returns ``(N, K)``).
      However, the metric-reporter pipeline assumes per-class probability
      output (N, K) AND breaks on RidgeClassifier's lack of
      ``predict_proba``. Until the eval path is generalised, all linear
      multilabel goes through ``MultiOutputClassifier`` wrap (correct
      but suboptimal -- one extra fit per label). Tracked as known
      limitation; wrapper path is correct.
    """

    cache_key = "linear"
    requires_scaling = True
    requires_encoding = True
    requires_imputation = True
    # sklearn LogisticRegression supports multiclass natively (auto since
    # 1.5; ``multi_class`` kwarg removed in 1.8).
    supports_native_multiclass = True


class RecurrentModelStrategy(ModelPipelineStrategy):
    """
    Strategy for recurrent models (LSTM, GRU, RNN, Transformer).

    These models:
    - Process sequences internally (handled by RecurrentDataModule)
    - In HYBRID mode, tabular features require preprocessing
    - Need imputation and scaling for tabular features
    - Require category encoding for tabular features

    Multi-output dispatch (2026-05-07):
    - **multiclass**: native via ``num_classes>1`` + CrossEntropyLoss
      + softmax in ``predict_step``. Already wired at the model level
      (RecurrentLightningModule); the flag below makes the dispatch
      consistent across strategies.
    - **multilabel**: native via ``task_type='multilabel'`` ->
      BCEWithLogitsLoss + sigmoid output. Output layer stays at K units,
      activation switches at predict time.
    - **learning_to_rank**: NOT native -- group-aware sequence batching
      (one query's docs per batch, where each doc has its own sequence)
      is non-trivial for recurrent architectures. Deferred; suite
      filters out 'recurrent' models when target_type=LEARNING_TO_RANK.
    """

    cache_key = "recurrent"
    requires_scaling = True
    requires_encoding = True
    requires_imputation = True
    supports_native_multiclass = True
    supports_native_multilabel = True
    # supports_native_ranking stays False -- group-batching for sequences
    # would require a custom sampler that yields one query's sequences
    # per batch; non-trivial integration with RecurrentDataModule.

    def get_classif_objective_kwargs(self, target_type, n_classes: int,
                                      multilabel_config=None) -> dict:
        """Per-target task_type for ``RecurrentLightningModule``.

        Returns a dict with the ``task_type`` kwarg consumed by the
        Lightning module to switch loss + activation. For multiclass
        the default (None / 'multiclass') already uses CrossEntropy +
        softmax -- empty return suffices.
        """
        from .configs import TargetTypes as _TT

        if target_type == _TT.MULTILABEL_CLASSIFICATION:
            return {"task_type": "multilabel"}
        # binary / multiclass / None -> defaults are correct
        return {}


# =============================================================================
# Strategy Registry
# =============================================================================

# Pre-instantiated strategy instances
_TREE_STRATEGY = TreeModelStrategy()
_CATBOOST_STRATEGY = CatBoostStrategy()
_XGBOOST_STRATEGY = XGBoostStrategy()
_HGB_STRATEGY = HGBStrategy()
_NEURAL_STRATEGY = NeuralNetStrategy()
_LINEAR_STRATEGY = LinearModelStrategy()
_RECURRENT_STRATEGY = RecurrentModelStrategy()

# Model name to strategy mapping
MODEL_STRATEGIES: Dict[str, ModelPipelineStrategy] = {
    # Tree models
    "cb": _CATBOOST_STRATEGY,
    "lgb": _TREE_STRATEGY,
    "xgb": _XGBOOST_STRATEGY,
    # HistGradientBoosting
    "hgb": _HGB_STRATEGY,
    # Neural networks
    "mlp": _NEURAL_STRATEGY,
    "ngb": _NEURAL_STRATEGY,
    # Linear models
    "linear": _LINEAR_STRATEGY,
    "ridge": _LINEAR_STRATEGY,
    "lasso": _LINEAR_STRATEGY,
    "elasticnet": _LINEAR_STRATEGY,
    "huber": _LINEAR_STRATEGY,
    "ransac": _LINEAR_STRATEGY,
    "sgd": _LINEAR_STRATEGY,
    "logistic": _LINEAR_STRATEGY,
    # Recurrent models
    "lstm": _RECURRENT_STRATEGY,
    "gru": _RECURRENT_STRATEGY,
    "rnn": _RECURRENT_STRATEGY,
    "transformer": _RECURRENT_STRATEGY,
}


def get_strategy(model_name) -> ModelPipelineStrategy:
    """
    Get the appropriate pipeline strategy for a model type.

    Accepts:
      * string alias (e.g. ``"cb"``, ``"lgb"``, ``"mlp"``, ``"linear"``)
      * sklearn-compatible estimator instance
      * ``(name, estimator)`` tuple

    For non-string inputs dispatch is delegated to
    :func:`_strategy_for_estimator` (MRO-based).

    Returns:
        ModelPipelineStrategy instance for the model type.
        Defaults to TreeModelStrategy for unknown string aliases (with warning)
        and LinearModelStrategy for unregistered estimator classes.
    """
    import warnings

    # Accept tuple form (name, estimator) and fall through to estimator handling.
    if isinstance(model_name, tuple) and len(model_name) == 2:
        return _strategy_for_estimator(model_name[1])

    if isinstance(model_name, str):
        strategy = MODEL_STRATEGIES.get(model_name.lower())
        if strategy is None:
            warnings.warn(f"Unknown model '{model_name}', defaulting to TreeModelStrategy")
            return _TREE_STRATEGY
        return strategy

    # Anything else is treated as an estimator instance.
    return _strategy_for_estimator(model_name)


# ---------------------------------------------------------------------------
# Estimator-instance dispatch (lazy-guarded imports)
# ---------------------------------------------------------------------------

def _catboost_classes():
    if importlib.util.find_spec("catboost") is None:
        return ()
    from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore
    return (CatBoostClassifier, CatBoostRegressor)


def _lightgbm_classes():
    if importlib.util.find_spec("lightgbm") is None:
        return ()
    from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore
    return (LGBMClassifier, LGBMRegressor)


def _xgboost_classes():
    if importlib.util.find_spec("xgboost") is None:
        return ()
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
    return (XGBClassifier, XGBRegressor)


def _hgb_classes():
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )
    return (HistGradientBoostingClassifier, HistGradientBoostingRegressor)


def _strategy_for_estimator(estimator: Any) -> ModelPipelineStrategy:
    """MRO-based dispatch from an estimator instance to a Strategy.

    Unknown classes fall back to :class:`LinearModelStrategy` (scaler-requiring)
    with a WARNING log line.
    """
    cb = _catboost_classes()
    if cb and isinstance(estimator, cb):
        return _CATBOOST_STRATEGY
    lgb = _lightgbm_classes()
    if lgb and isinstance(estimator, lgb):
        return _TREE_STRATEGY
    xgb = _xgboost_classes()
    if xgb and isinstance(estimator, xgb):
        return _XGBOOST_STRATEGY
    if isinstance(estimator, _hgb_classes()):
        return _HGB_STRATEGY

    logger.warning(
        "No registered strategy for %s; defaulting to LinearModelStrategy",
        type(estimator).__name__,
    )
    return _LINEAR_STRATEGY


def _slugify(name: str) -> str:
    """Slugify a user-provided model key to alnum + ``-`` + ``_`` only."""
    slug = _SLUG_PATTERN.sub("_", name).strip("_-")
    return slug or "model"


def _dedupe_key(key: str, used: set) -> str:
    """Return ``key`` if unused, else ``key_2``, ``key_3`` etc."""
    if key not in used:
        used.add(key)
        return key
    i = 2
    while f"{key}_{i}" in used:
        i += 1
    new = f"{key}_{i}"
    used.add(new)
    return new


def _resolve_model_spec(
    entry: Any,
    used_keys: Optional[set] = None,
) -> Tuple[str, Optional[Any], ModelPipelineStrategy]:
    """Resolve one ``mlframe_models`` entry to ``(key, estimator, strategy)``.

    Parameters
    ----------
    entry:
        One of:
          * string alias (``"cb"``): ``estimator`` returned as ``None``.
          * estimator instance: ``key`` is ``type(entry).__name__``.
          * ``(name, estimator)`` tuple: ``key`` is ``_slugify(name)``.
    used_keys:
        Optional mutable set of already-issued keys. Duplicates are disambiguated
        by appending ``_2``, ``_3`` suffixes.

    Returns
    -------
    Tuple of metadata key, estimator instance (or ``None`` for string aliases),
    and resolved strategy.
    """
    if used_keys is None:
        used_keys = set()

    # Tuple form: (name, estimator)
    if isinstance(entry, tuple) and len(entry) == 2:
        name, est = entry
        if not isinstance(name, str):
            raise TypeError(
                f"Tuple model spec requires (str, estimator); got name of type {type(name).__name__}"
            )
        key = _dedupe_key(_slugify(name), used_keys)
        strat = _strategy_for_estimator(est)
        return key, est, strat

    # String alias
    if isinstance(entry, str):
        key = _dedupe_key(entry, used_keys)
        return key, None, get_strategy(entry)

    # Otherwise treat as estimator instance
    key = _dedupe_key(type(entry).__name__, used_keys)
    strat = _strategy_for_estimator(entry)
    return key, entry, strat


def get_cache_key(model_name: str, pre_pipeline_name: str = "") -> str:
    """
    Get the cache key for a model's transformed DataFrames.

    Models with the same cache key can share transformed DataFrames.
    The pre_pipeline_name is included to differentiate between different
    feature selectors (e.g., MRMR vs RFECV) that would otherwise share cache.

    Args:
        model_name: Name of the model
        pre_pipeline_name: Name/identifier of the pre-pipeline (e.g., "mrmr", "rfecv")

    Returns:
        Cache key string (e.g., "tree", "tree_mrmr", "neural_rfecv")
    """
    base_key = get_strategy(model_name).cache_key
    if pre_pipeline_name:
        return f"{base_key}_{pre_pipeline_name}"
    return base_key


# =============================================================================
# Pipeline Cache Helper
# =============================================================================


class PipelineCache:
    """
    Cache for transformed DataFrames to avoid redundant preprocessing.

    Different model types that require the same preprocessing can share
    cached DataFrames, improving efficiency when training multiple models.

    Note:
        This class is NOT thread-safe. It is designed for sequential use within
        a single training run. If parallel model training is implemented in the
        future, this class should be extended with proper locking mechanisms.
    """

    def __init__(self):
        self._cache: Dict[str, Tuple[Any, Any, Any]] = {}

    def get(self, cache_key: str) -> Optional[Tuple[Any, Any, Any]]:
        """
        Get cached DataFrames for a cache key.

        Args:
            cache_key: The cache key (from strategy.cache_key)

        Returns:
            Tuple of (train_df, val_df, test_df) or None if not cached
        """
        return self._cache.get(cache_key)

    def set(self, cache_key: str, train_df: Any, val_df: Any, test_df: Any) -> None:
        """
        Cache transformed DataFrames.

        Args:
            cache_key: The cache key (from strategy.cache_key)
            train_df: Transformed training DataFrame
            val_df: Transformed validation DataFrame
            test_df: Transformed test DataFrame
        """
        self._cache[cache_key] = (train_df, val_df, test_df)

    def has(self, cache_key: str) -> bool:
        """Check if a cache key exists."""
        return cache_key in self._cache

    def clear(self) -> None:
        """Clear all cached DataFrames."""
        self._cache.clear()


__all__ = [
    "PANDAS_CATEGORICAL_DTYPES",
    "is_polars_categorical",
    "get_polars_cat_columns",
    "ModelPipelineStrategy",
    "TreeModelStrategy",
    "CatBoostStrategy",
    "XGBoostStrategy",
    "HGBStrategy",
    "NeuralNetStrategy",
    "LinearModelStrategy",
    "RecurrentModelStrategy",
    "MODEL_STRATEGIES",
    "get_strategy",
    "_resolve_model_spec",
    "get_cache_key",
    "PipelineCache",
]
