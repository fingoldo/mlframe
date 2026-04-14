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
    """Check whether a Polars dtype is categorical/string."""
    return dtype in _polars_categorical_dtypes()

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

        # Add category encoding if required and categorical features exist
        if self.requires_encoding and cat_features and category_encoder is not None:
            steps.append(("ce", category_encoder))

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

        return Pipeline(steps=steps)


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
    # Inherits cache_key = "tree" from TreeModelStrategy so CB/LGB/XGB share
    # transformed-DF cache (they have identical preprocessing requirements).


class XGBoostStrategy(TreeModelStrategy):
    """
    Strategy for XGBoost models (>= 3.1).

    Inherits tree model behavior and additionally supports native Polars DataFrames.
    XGBoost auto-detects pl.Categorical columns when enable_categorical=True,
    but pl.String columns must be cast to pl.Categorical first.
    No cardinality limit (unlike HGB).
    """

    supports_polars = True
    # Inherits cache_key = "tree" from TreeModelStrategy (see CatBoostStrategy note).

    def prepare_polars_dataframe(self, df: "pl.DataFrame", cat_features: List[str]) -> "pl.DataFrame":
        """Cast string columns to pl.Categorical for XGBoost auto-detection.

        XGBoost detects pl.Categorical natively when enable_categorical=True
        but does NOT handle raw pl.String columns.
        """
        import polars as pl

        schema_cats = {
            name for name, dtype in df.schema.items()
            if dtype in (pl.Utf8, pl.String)
        }
        cols_to_cast = schema_cats | {c for c in (cat_features or []) if c in df.columns and df[c].dtype in (pl.Utf8, pl.String)}
        if not cols_to_cast:
            return df
        return df.with_columns([pl.col(c).cast(pl.Categorical) for c in cols_to_cast])


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

    # HGB max_bins is capped at 255 in sklearn
    _MAX_CATEGORICAL_CARDINALITY = 255

    def prepare_polars_dataframe(self, df: "pl.DataFrame", cat_features: List[str]) -> "pl.DataFrame":
        """Cast categorical columns for HGB compatibility.

        - Cardinality <= 255: cast pl.String → pl.Categorical (HGB auto-detects via from_dtype)
        - Cardinality > 255: ordinal-encode to pl.UInt32 (HGB treats as continuous, bins into histogram)

        Detects columns from both the provided cat_features list AND the DF schema
        (pl.String/pl.Utf8/pl.Categorical columns), so it works even when the pipeline
        has already converted some columns.
        """
        import polars as pl

        # Detect all string/categorical columns from schema (may include columns
        # not in cat_features if pipeline didn't detect them)
        from .utils import filter_existing

        schema_cats = set(get_polars_cat_columns(df))
        all_cats = schema_cats | set(cat_features or [])
        existing = filter_existing(df, all_cats)
        if not existing:
            return df

        casts = []
        for col in existing:
            dtype = df[col].dtype
            if dtype == pl.Categorical:
                n_unique = df[col].n_unique()
                if n_unique > self._MAX_CATEGORICAL_CARDINALITY:
                    # Too many categories — ordinal encode to integer
                    casts.append(pl.col(col).cast(pl.String).cast(pl.Categorical).to_physical().cast(pl.UInt32).alias(col))
                # else: already pl.Categorical with acceptable cardinality, keep as-is
            elif dtype == pl.String or dtype == pl.Utf8:
                n_unique = df[col].n_unique()
                if n_unique <= self._MAX_CATEGORICAL_CARDINALITY:
                    casts.append(pl.col(col).cast(pl.Categorical).alias(col))
                else:
                    # Ordinal encode high-cardinality strings
                    casts.append(pl.col(col).cast(pl.Categorical).to_physical().cast(pl.UInt32).alias(col))

        if casts:
            df = df.with_columns(casts)
        return df


class NeuralNetStrategy(ModelPipelineStrategy):
    """
    Strategy for neural network models (MLP, NGBoost).

    These models:
    - Cannot handle NaN values - need imputation
    - Benefit significantly from feature scaling
    - Require category encoding
    """

    cache_key = "neural"
    requires_scaling = True
    requires_encoding = True
    requires_imputation = True


class LinearModelStrategy(ModelPipelineStrategy):
    """
    Strategy for linear models (Linear, Ridge, Lasso, ElasticNet, etc.).

    These models:
    - Cannot handle NaN values - need imputation
    - Require feature scaling for proper regularization
    - Require category encoding
    """

    cache_key = "linear"
    requires_scaling = True
    requires_encoding = True
    requires_imputation = True


class RecurrentModelStrategy(ModelPipelineStrategy):
    """
    Strategy for recurrent models (LSTM, GRU, RNN, Transformer).

    These models:
    - Process sequences internally (handled by RecurrentDataModule)
    - In HYBRID mode, tabular features require preprocessing
    - Need imputation and scaling for tabular features
    - Require category encoding for tabular features
    """

    cache_key = "recurrent"
    requires_scaling = True
    requires_encoding = True
    requires_imputation = True


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
