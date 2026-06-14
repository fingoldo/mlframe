"""
Model pipeline strategies for mlframe.

Implements the Strategy pattern to handle model-specific preprocessing pipelines.
Each model type may require different preprocessing (scaling, encoding, imputation).
"""

from __future__ import annotations


import importlib.util
import logging
import re
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Pre-compiled slug pattern (MEMORY.md: pre-compile regex at module level).
# Only allow alnum, dash, underscore; everything else collapses to a single "_".
_SLUG_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")

if TYPE_CHECKING:
    import polars as pl

# =============================================================================
# Unified categorical type constants
# =============================================================================
# Used across pipeline.py, trainer.py, utils.py, core.py to detect categoricals; import these instead of hardcoding type lists.
# ``PANDAS_CATEGORICAL_DTYPES`` is the single source of truth in ``base``; re-exported here (and in ``__all__``) so callers don't duplicate the set.
from .base import (  # noqa: F401
    PANDAS_CATEGORICAL_DTYPES,
    PANDAS_CATEGORICAL_SELECT_DTYPES,
    ModelPipelineStrategy,
)


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


from .tree_cb import TreeModelStrategy, CatBoostStrategy  # noqa: E402, F401
from .xgboost import XGBoostStrategy  # noqa: E402, F401
from .hgb import HGBStrategy  # noqa: E402, F401
from .neural import (  # noqa: E402, F401
    NeuralNetStrategy,
    LinearModelStrategy,
    RecurrentModelStrategy,
)


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
    "catboost": _CATBOOST_STRATEGY,  # alias
    "lgb": _TREE_STRATEGY,
    "lightgbm": _TREE_STRATEGY,  # alias
    "xgb": _XGBOOST_STRATEGY,
    "xgboost": _XGBOOST_STRATEGY,  # alias
    # HistGradientBoosting
    "hgb": _HGB_STRATEGY,
    "histgradientboosting": _HGB_STRATEGY,  # alias
    # Neural networks
    "mlp": _NEURAL_STRATEGY,
    "ngb": _NEURAL_STRATEGY,
    # Linear models
    "linear": _LINEAR_STRATEGY,
    "lr": _LINEAR_STRATEGY,  # common shorthand; previously fell through to TreeModelStrategy + UserWarning
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
            warnings.warn(f"Unknown model '{model_name}', defaulting to TreeModelStrategy", stacklevel=2)
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


def is_catboost_model(entry: Any) -> bool:
    """True if ``entry`` (alias str, estimator instance, or ``(name, est)`` tuple) is a CatBoost model.

    Routes via the strategy registry so an estimator INSTANCE (``CatBoostClassifier()``)
    is classified the same as the ``"cb"`` / ``"catboost"`` aliases. A bare ``str(m).lower()``
    membership test mis-routes the instance (it stringifies to ``"<catboost...object at 0x..>"``).
    """
    if isinstance(entry, str):
        return entry.lower() in ("cb", "catboost")
    return isinstance(get_strategy(entry), CatBoostStrategy)


def is_neural_model(entry: Any) -> bool:
    """True if ``entry`` is a neural / recurrent model (mlp / lstm / gru / rnn / transformer / ngb).

    Strategy-routed so a torch MLP / recurrent estimator INSTANCE classifies the same as the
    string aliases; a bare name-tuple membership test silently misses the instance.
    """
    if isinstance(entry, str):
        return entry.lower() in ("mlp", "recurrent", "ngb", "lstm", "gru", "rnn", "transformer")
    return isinstance(get_strategy(entry), (NeuralNetStrategy, RecurrentModelStrategy))


from .pipeline_cache import (  # noqa: E402, F401
    PipelineCache,
    _resolve_pipeline_cache_budget,
    _estimate_slot_nbytes,
    _estimate_entry_nbytes,
    _DEFAULT_PIPELINE_CACHE_BYTES_LIMIT,
)


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
    "is_catboost_model",
    "is_neural_model",
    "_resolve_model_spec",
    "PipelineCache",
]
