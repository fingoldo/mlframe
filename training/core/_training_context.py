"""
TrainingContext — shared state for the training orchestrator.

A single dataclass that carries all mutable state through the training phases,
eliminating the ~50-parameter function signatures that would otherwise be needed
to extract the per-target training loop and model training logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl


@dataclass
class TrainingContext:
    """Mutable shared state for train_mlframe_models_suite phases."""

    # ── Identifiers ──────────────────────────────────────────────
    model_name: str = ""
    target_name: str = ""

    # ── Configs (Pydantic or dict) ───────────────────────────────
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

    # ── Scalar settings ─────────────────────────────────────────
    verbose: int = 1
    data_dir: str = ""
    models_dir: str = ""
    save_charts: bool = False
    outlier_detector: Any = None
    od_val_set: bool = False
    use_mrmr_fs: bool = False
    mrmr_kwargs: Optional[Dict] = None
    rfecv_models: Optional[List[str]] = None
    custom_pre_pipelines: Optional[Any] = None
    common_params_dict: Dict = field(default_factory=dict)
    mlframe_models: List[str] = field(default_factory=list)

    # ── Data & targets ──────────────────────────────────────────
    df: Any = None  # pl.DataFrame | pd.DataFrame | None (del'd after split)
    target_by_type: Dict = field(default_factory=dict)
    group_ids_raw: Optional[Any] = None
    group_ids: Optional[Any] = None
    timestamps: Optional[np.ndarray] = None
    artifacts: Dict = field(default_factory=dict)
    additional_columns_to_drop: List[str] = field(default_factory=list)
    sample_weights: Optional[Dict] = None
    baseline_rss_mb: float = 0.0
    df_size_mb: float = 0.0
    sequences: Optional[List[np.ndarray]] = None

    # ── Split indices ───────────────────────────────────────────
    train_idx: Optional[np.ndarray] = None
    val_idx: Optional[np.ndarray] = None
    test_idx: Optional[np.ndarray] = None
    train_details: Optional[Any] = None
    val_details: Optional[Any] = None
    test_details: Optional[Any] = None
    fairness_subgroups: Optional[Dict] = None
    fairness_features: Optional[List[str]] = None

    # ── Sequence splits ─────────────────────────────────────────
    train_sequences: Optional[List[np.ndarray]] = None
    val_sequences: Optional[List[np.ndarray]] = None
    test_sequences: Optional[List[np.ndarray]] = None

    # ── DataFrames (post-split, pre-pipeline) ───────────────────
    train_df: Any = None
    val_df: Any = None
    test_df: Any = None

    # ── Pipeline ────────────────────────────────────────────────
    pipeline: Any = None
    extensions_pipeline: Any = None
    was_polars_input: bool = False
    all_models_polars_native: bool = False
    polars_pipeline_applied: bool = False
    train_df_polars_pre: Any = None
    val_df_polars_pre: Any = None
    test_df_polars_pre: Any = None
    preprocessing_extensions: Any = None

    # ── Feature metadata ────────────────────────────────────────
    cat_features: List[str] = field(default_factory=list)
    cat_features_polars: List[str] = field(default_factory=list)
    text_features: List[str] = field(default_factory=list)
    embedding_features: List[str] = field(default_factory=list)
    text_emb_set: set = field(default_factory=set)
    _dropped_high_card_data: Dict = field(default_factory=dict)

    # ── Post-pipeline DataFrames ────────────────────────────────
    train_df_pd: Any = None
    val_df_pd: Any = None
    test_df_pd: Any = None
    train_df_polars: Any = None
    val_df_polars: Any = None
    test_df_polars: Any = None

    # ── OD-filtered frames & indices ────────────────────────────
    filtered_train_df: Any = None
    filtered_val_df: Any = None
    filtered_train_idx: Optional[np.ndarray] = None
    filtered_val_idx: Optional[np.ndarray] = None
    train_od_idx: Optional[np.ndarray] = None
    val_od_idx: Optional[np.ndarray] = None
    outlier_detection_result: Any = None

    # ── Training intermediates ──────────────────────────────────
    category_encoder: Any = None
    imputer: Any = None
    scaler: Any = None
    trainset_features_stats: Any = None
    can_skip_pandas_conv: bool = False
    train_df_size_bytes_cached: Optional[int] = None
    val_df_size_bytes_cached: Optional[int] = None
    _all_target_audits: Dict = field(default_factory=dict)
    _non_neural_train_times: List[float] = field(default_factory=list)

    # ── Models & metadata ───────────────────────────────────────
    models: Dict = field(default_factory=lambda: {})
    metadata: Dict = field(default_factory=dict)
    slug_to_original_target_type: Dict = field(default_factory=dict)
    slug_to_original_target_name: Dict = field(default_factory=dict)
