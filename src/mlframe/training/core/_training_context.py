"""TrainingContext - shared mutable state for the training orchestrator phases."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin
    from sklearn.pipeline import Pipeline


# ``slots=True`` on @dataclass landed in Python 3.10. On 3.9 we lose the
# performance + typo-catching benefits but the class still works (attribute
# access goes through __dict__ as usual). Conditional kwarg keeps a single
# dataclass declaration without splitting the class definition per version.
_DATACLASS_KW = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(**_DATACLASS_KW)
class TrainingContext:
    """Mutable shared state for train_mlframe_models_suite phases.

    ``slots=True`` (Python 3.10+) skips per-instance ``__dict__`` allocation: attribute access reads via fixed-offset slot
    descriptors instead of a hash-table lookup (~3-5x faster per ``ctx.X``) and the instance footprint shrinks to ~one
    pointer per field. Adding undeclared attributes raises ``AttributeError`` at runtime, which catches typos that
    bare-dict instances silently accepted. On Python 3.9 we fall back to the standard __dict__ form.
    """

    model_name: str = ""
    target_name: str = ""

    preprocessing_config: Any = None
    pipeline_config: Any = None
    feature_types_config: Any = None
    split_config: Any = None
    hyperparams_config: Any = None
    behavior_config: Any = None
    reporting_config: Any = None
    output_config: Any = None
    outlier_detection_config: Any = None
    feature_selection_config: Any = None
    confidence_analysis_config: Any = None
    baseline_diagnostics_config: Any = None
    dummy_baselines_config: Any = None
    quantile_regression_config: Any = None
    composite_target_discovery_config: Any = None
    linear_model_config: Any = None
    multilabel_dispatch_config: Any = None
    ranking_config: Any = None
    recurrent_config: Any = None
    recurrent_models: list[str] | None = None

    verbose: int = 1
    data_dir: str = ""
    models_dir: str = ""
    save_charts: bool = False
    outlier_detector: Any = None
    od_val_set: bool = False
    use_mrmr_fs: bool = False
    use_ordinary_models: bool = True
    use_mlframe_ensembles: bool = False
    mrmr_kwargs: dict | None = None
    rfecv_models: list[str] | None = None
    custom_pre_pipelines: Any | None = None
    common_params_dict: dict = field(default_factory=dict)
    mlframe_models: list[str] = field(default_factory=list)
    # Computed once per suite (strategies depend only on mlframe_models, which is suite-constant).
    # Previously rebuilt inside the per-(target x pre_pipeline) inner loop, paying O(t * pp * m)
    # get_strategy() calls; lifting here makes it O(m).
    strategy_by_model: dict = field(default_factory=dict)
    sorted_mlframe_models: list[str] = field(default_factory=list)

    df: pl.DataFrame | pd.DataFrame | None = None  # del'd after split
    target_by_type: dict = field(default_factory=dict)
    group_ids_raw: np.ndarray | pd.Series | None = None
    group_ids: np.ndarray | pd.Series | None = None
    timestamps: np.ndarray | None = None
    artifacts: dict = field(default_factory=dict)
    additional_columns_to_drop: list[str] = field(default_factory=list)
    sample_weights: dict | None = None
    baseline_rss_mb: float = 0.0
    df_size_mb: float = 0.0
    sequences: list[np.ndarray] | None = None

    train_idx: np.ndarray | None = None
    val_idx: np.ndarray | None = None
    test_idx: np.ndarray | None = None
    train_details: Any | None = None
    val_details: Any | None = None
    test_details: Any | None = None
    fairness_subgroups: dict | None = None
    fairness_features: list[str] | None = None

    train_sequences: list[np.ndarray] | None = None
    val_sequences: list[np.ndarray] | None = None
    test_sequences: list[np.ndarray] | None = None

    train_df: pl.DataFrame | pd.DataFrame | None = None
    val_df: pl.DataFrame | pd.DataFrame | None = None
    test_df: pl.DataFrame | pd.DataFrame | None = None

    pipeline: Pipeline | None = None
    extensions_pipeline: Pipeline | None = None
    was_polars_input: bool = False
    all_models_polars_native: bool = False
    polars_pipeline_applied: bool = False
    train_df_polars_pre: pl.DataFrame | None = None
    val_df_polars_pre: pl.DataFrame | None = None
    test_df_polars_pre: pl.DataFrame | None = None
    # Pandas-side mutation-immune metadata snapshot of the PRE-pipeline train_df. Replaces the legacy
    # ``train_df_pandas_pre`` shallow-copy that shared the source block-manager and leaked in-place numpy
    # pokes (``df[col].values[i] = x``) into the downstream auto-detect phase. The dict captures column
    # names / dtype strings / cardinality / non-null counts / embedding-shape sniff result at snapshot
    # time so any subsequent mutation on train_df cannot corrupt the auto-detect view. The val/test
    # pandas-pre slots are no longer carried -- auto-detect only ever reads train.
    train_df_pandas_pre_meta: dict | None = None
    preprocessing_extensions: Any = None

    cat_features: list[str] = field(default_factory=list)
    cat_features_polars: list[str] = field(default_factory=list)
    text_features: list[str] = field(default_factory=list)
    embedding_features: list[str] = field(default_factory=list)
    text_emb_set: set = field(default_factory=set)
    _dropped_high_card_data: dict = field(default_factory=dict)

    train_df_pd: pd.DataFrame | None = None
    val_df_pd: pd.DataFrame | None = None
    test_df_pd: pd.DataFrame | None = None
    train_df_polars: pl.DataFrame | None = None
    val_df_polars: pl.DataFrame | None = None
    test_df_polars: pl.DataFrame | None = None

    filtered_train_df: pl.DataFrame | pd.DataFrame | None = None
    filtered_val_df: pl.DataFrame | pd.DataFrame | None = None
    filtered_train_idx: np.ndarray | None = None
    filtered_val_idx: np.ndarray | None = None
    train_od_idx: np.ndarray | None = None
    val_od_idx: np.ndarray | None = None
    outlier_detection_result: Any = None

    category_encoder: TransformerMixin | None = None
    imputer: TransformerMixin | None = None
    scaler: TransformerMixin | None = None
    trainset_features_stats: Any = None
    defer_pandas_conv: bool = False
    train_df_size_bytes_cached: int | None = None
    val_df_size_bytes_cached: int | None = None
    _all_target_audits: dict = field(default_factory=dict)
    _non_neural_train_times: list[float] = field(default_factory=list)
    # CODE-P1-10: per-suite cache of compute_model_input_fingerprint results keyed by
    # (target_type, target_name, id(strategy), tier_suffix, kind_suffix, pre_pipeline_name) so the
    # weight-schema inner loop reuses the (schema_hash, input_schema) tuple instead of recomputing
    # it once per weight iteration.
    _model_input_fingerprint_cache: dict = field(default_factory=dict)
    # CONV-MED-5: per-suite cache of pandas views of polars DFs keyed by id(polars_df) so two
    # non-Polars-native strategies that share the same source polars frame only pay one
    # ``get_pandas_view_of_polars_df`` conversion total within a single _train_one_target call.
    _pandas_view_cache: dict = field(default_factory=dict)
    # Suite-scoped cache observability counters. ``_train_one_target`` writes hit/miss
    # bumps here (``setdefault("<cache>", {"hits": 0, "misses": 0})``); ``finalize_suite``
    # aggregates the result into ``metadata["cache_stats"]``. Slot is REQUIRED because
    # the dataclass uses slots=True - without it, ``ctx._cache_stats = {}`` raises
    # AttributeError on every suite call. Surfaced by fuzz iter#126.
    _cache_stats: dict = field(default_factory=dict)

    models: dict = field(default_factory=lambda: {})
    # Per-target ensemble outputs from ``score_ensemble`` (use_mlframe_ensembles=True). Keyed
    # ``ensembles[target_type][target_name] = dict[ensemble_method -> ens_result]``. Mirrored
    # into ``models`` under the same target slot so finalize_suite + downstream consumers see
    # them without needing a new code path.
    ensembles: dict = field(default_factory=lambda: {})
    metadata: dict = field(default_factory=dict)
    slug_to_original_target_type: dict = field(default_factory=dict)
    slug_to_original_target_name: dict = field(default_factory=dict)
