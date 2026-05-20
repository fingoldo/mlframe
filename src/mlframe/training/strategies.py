"""
Model pipeline strategies for mlframe.

Implements the Strategy pattern to handle model-specific preprocessing pipelines.
Each model type may require different preprocessing (scaling, encoding, imputation).
"""

from __future__ import annotations


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


# Wave 104 (2026-05-21): ModelPipelineStrategy ABC moved to _strategies_base.py.
from ._strategies_base import ModelPipelineStrategy  # noqa: F401, E402

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

        Defaults to ``YetiRankPairwise`` (listwise pairwise вЂ” robust on
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


# Wave 104 (2026-05-21): XGBoostStrategy moved to _strategies_xgboost.py.
from ._strategies_xgboost import XGBoostStrategy  # noqa: F401, E402

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
        # Wave 72 (2026-05-21): track which cols use strict=False (test-side
        # OOV-tolerant cast) so we can quantify cast-failure rate post-with_columns.
        _strict_false_cols: list[str] = []
        for col in existing:
            n_unique = df[col].n_unique()
            high_card = n_unique > self._MAX_CATEGORICAL_CARDINALITY
            if category_map is not None and col in category_map:
                enum_dt = category_map[col]
                # category_map is built from train+val UNION (test EXCLUDED, leak-free).
                # Test rows therefore can carry values absent from the Enum's domain.
                # Use strict=False so OOV values fall through to null rather than
                # crash the lazy collect, matching the dict-alignment routine at
                # core.py:2992 which also passes strict=False on the test split.
                _strict_false_cols.append(col)
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
            # Wave 72 (2026-05-21): pre-cast null counts for strict=False columns;
            # post-cast delta surfaces silent OOV-nulling.
            _null_pre = {c: int(df[c].null_count()) for c in _strict_false_cols if c in df.columns}
            df = df.with_columns(casts)
            if _null_pre:
                _null_deltas = {
                    c: int(df[c].null_count()) - _null_pre[c]
                    for c in _null_pre
                }
                _nonzero = {c: d for c, d in _null_deltas.items() if d > 0}
                if _nonzero:
                    import logging as _lg
                    _lg.getLogger(__name__).info(
                        "[hgb cat-cast] %d col(s) had OOV nulls cast-failed: %s",
                        len(_nonzero), _nonzero,
                    )
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
        from .configs import TargetTypes

        # Lazy import torch so a non-MLP run doesn't pay for PL/torch import.
        import torch
        import torch.nn.functional as F

        if target_type is None or target_type == TargetTypes.BINARY_CLASSIFICATION:
            return {}  # default cross_entropy is correct for binary
        if target_type == TargetTypes.MULTICLASS_CLASSIFICATION:
            # Default ``F.cross_entropy`` + ``int64`` labels already
            # handle K>2 -- explicit return for symmetry with other strategies.
            return {"loss_fn": F.cross_entropy, "labels_dtype": torch.int64}
        if target_type == TargetTypes.MULTILABEL_CLASSIFICATION:
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
        from .configs import TargetTypes

        if target_type == TargetTypes.MULTILABEL_CLASSIFICATION:
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


# CACHE-P1-2: get_cache_key removed. The helper was exported in __all__ but
# carried no internal callers - PipelineCache (below) does its own cache-key
# composition via ``get_strategy(model_name).cache_key`` directly. Tests that
# exercised it were removed in the same change. External callers that still
# reference ``strategies.get_cache_key`` should switch to
# ``get_strategy(model_name).cache_key`` (same value, fewer indirections).


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

    def __init__(self, verbose: bool = True):
        """Construct a pre-pipeline cache.

        ``verbose=True`` is the new default: HIT/MISS lines are emitted
        at ``logger.info`` and routinely needed when triaging
        "why-did-this-suite-re-fit" tickets. The lines are throttled by
        the per-call HIT vs MISS branch (one log per get) and add no
        measurable overhead vs the dict lookup itself, so the cost of
        leaving them on by default is negligible against the diagnostic
        value of having them already on when the operator wants them.
        Pass ``verbose=False`` to silence in tight unit-test loops.
        """
        self._cache: Dict[str, Tuple[Any, Any, Any]] = {}
        # Observability counters. Cheap (two integer bumps per call); the
        # bench in tests asserts microsecond-scale overhead so they stay on
        # by default.
        self.n_hits: int = 0
        self.n_misses: int = 0
        self.verbose: bool = bool(verbose)

    def get(self, cache_key: str) -> Optional[Tuple[Any, Any, Any]]:
        """
        Get cached DataFrames for a cache key.

        Args:
            cache_key: The cache key (from strategy.cache_key)

        Returns:
            Tuple of (train_df, val_df, test_df) or None if not cached
        """
        val = self._cache.get(cache_key)
        if val is None:
            self.n_misses += 1
            if self.verbose:
                logger.info("PipelineCache MISS key=%s (hits=%d misses=%d size=%d)", cache_key, self.n_hits, self.n_misses, len(self._cache))
        else:
            self.n_hits += 1
            if self.verbose:
                logger.info("PipelineCache HIT  key=%s (hits=%d misses=%d size=%d)", cache_key, self.n_hits, self.n_misses, len(self._cache))
        return val

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
        if self.verbose:
            logger.info("PipelineCache SET  key=%s (size=%d)", cache_key, len(self._cache))

    def has(self, cache_key: str) -> bool:
        """Check if a cache key exists."""
        return cache_key in self._cache

    def clear(self) -> None:
        """Clear all cached DataFrames."""
        self._cache.clear()

    def cache_size_bytes(self) -> int:
        """Best-effort ``sys.getsizeof`` sum across every cached frame slot.

        ``sys.getsizeof`` on a pandas/polars frame reports the Python
        container overhead, not the underlying Arrow / numpy buffer size,
        so this is a LOWER BOUND - useful as a "did the cache grow?"
        smoke signal rather than a precise memory accounting.
        """
        import sys
        total = sys.getsizeof(self._cache)
        for entry in self._cache.values():
            try:
                total += sys.getsizeof(entry)
                for slot in entry:
                    if slot is None:
                        continue
                    try:
                        total += int(sys.getsizeof(slot))
                    except Exception:
                        pass
            except Exception:
                continue
        return int(total)

    def __repr__(self) -> str:
        return f"PipelineCache(keys={len(self._cache)}, hits={self.n_hits}, misses={self.n_misses})"


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
    # CACHE-P1-2: ``get_cache_key`` removed (dead helper). See module body.
    "PipelineCache",
]
