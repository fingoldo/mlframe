"""Leaf-level utility helpers extracted from core.py.

27 helper functions covering: logging visibility, metric extraction, column
augmentation/restoration across splits, frame-shape pretty-print, dataset-reuse
capability detection, input-column-vs-metadata validation, feature-type
auto-detection + exclusivity guard, outlier-detection orchestrator,
directory setup, common-params builder, pre-pipeline builder, process-model
kwargs assembly, Polars->pandas conversion, pipeline-component extraction,
fairness-subgroup compute, CB-metamodel skip heuristic, metadata
initialisation + finalisation.

Each function is a leaf (zero internal-helper dependencies on its peers in
this module). Pulled out of core.py so the giant train_mlframe_models_suite
orchestrator lives in isolation; core.py re-exports every symbol below at
its bottom for full back-compat.
"""
from __future__ import annotations

import glob
import logging
import os
import sys
from collections import defaultdict
from copy import deepcopy
from os.path import exists, join
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import joblib
import numpy as np
import pandas as pd
import polars as pl
import psutil
import scipy.stats as stats
from pyutilz.strings import slugify
from pyutilz.system import (
    clean_ram, ensure_dir_exists, tqdmu, tqdmu_lazy_start,
)
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import category_encoders as ce

from ..configs import (
    BaselineDiagnosticsConfig,
    CompositeTargetDiscoveryConfig,
    ConfidenceAnalysisConfig,
    DummyBaselinesConfig,
    FeatureSelectionConfig,
    FeatureTypesConfig,
    LinearModelConfig,
    ModelHyperparamsConfig,
    MultilabelDispatchConfig,
    OutlierDetectionConfig,
    OutputConfig,
    PreprocessingBackendConfig,
    PreprocessingConfig,
    PreprocessingExtensionsConfig,
    QuantileRegressionConfig,
    ReportingConfig,
    TargetTypes,
    TrainingBehaviorConfig,
    TrainingConfig,
    TrainingSplitConfig,
)
from ..extractors import FeaturesAndTargetsExtractor
from ..helpers import (
    get_trainset_features_stats,
    get_trainset_features_stats_polars,
)
from ..io import load_mlframe_model
from ..models import LINEAR_MODEL_TYPES, is_linear_model
from ..phases import format_phase_summary, phase, reset_phase_registry
from ..pipeline import (
    apply_preprocessing_extensions,
    fit_and_transform_pipeline,
    prepare_df_for_catboost,
)
from ..preprocessing import (
    create_split_dataframes,
    load_and_prepare_dataframe,
    preprocess_dataframe,
    save_split_artifacts,
)
from ..splitting import make_train_test_split
from ..strategies import (
    PipelineCache,
    get_polars_cat_columns,
    get_strategy,
)
from ..utils import (
    compute_model_input_fingerprint,
    drop_columns_from_dataframe,
    estimate_df_size_mb,
    filter_existing,
    get_pandas_view_of_polars_df,
    get_process_rss_mb,
    log_phase,
    log_ram_usage,
    maybe_clean_ram_and_gpu,
)
from mlframe.feature_selection.filters import MRMR
from mlframe.metrics import create_fairness_subgroups

logger = logging.getLogger(__name__)


# Module-level constants

DEFAULT_PROBABILITY_THRESHOLD: float = 0.5


                continue
            if isinstance(df_, pl.DataFrame):
                cols_to_keep = [c for c in df_.columns if c not in cols_to_exclude]
                tier_dfs[key] = df_.select(cols_to_keep)
            else:
                cols_to_drop = filter_existing(df_, cols_to_exclude)
                tier_dfs[key] = df_.drop(columns=cols_to_drop) if cols_to_drop else df_

    tier_cache[tier] = tier_dfs
    return tier_dfs


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------


import os
import psutil
import joblib
import pandas as pd
import polars as pl
from typing import Union, Optional, List, Dict, Any, Tuple, TypeVar
from copy import deepcopy
import numpy as np
import scipy.stats as stats
from collections import defaultdict
from os.path import join, exists
import glob
from pyutilz.system import clean_ram, tqdmu, tqdmu_lazy_start
from pyutilz.strings import slugify
from sklearn.base import clone
from sklearn.pipeline import Pipeline
import category_encoders as ce

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from pyutilz.system import ensure_dir_exists

from ..configs import (
    PreprocessingConfig,
    TrainingSplitConfig,
    PreprocessingBackendConfig,
    FeatureTypesConfig,
    FeatureSelectionConfig,
    ModelHyperparamsConfig,
    TrainingBehaviorConfig,
    TrainingConfig,
    TargetTypes,
    LinearModelConfig,
    PreprocessingExtensionsConfig,
    MultilabelDispatchConfig,
    ReportingConfig,
    OutputConfig,
    OutlierDetectionConfig,
    ConfidenceAnalysisConfig,
    BaselineDiagnosticsConfig,
    CompositeTargetDiscoveryConfig,
    DummyBaselinesConfig,
    QuantileRegressionConfig,
)
from ..preprocessing import (
    load_and_prepare_dataframe,
    preprocess_dataframe,
    save_split_artifacts,
    create_split_dataframes,
)
from ..pipeline import fit_and_transform_pipeline, prepare_df_for_catboost, apply_preprocessing_extensions
from mlframe.feature_selection.filters import MRMR
from ..utils import (
    log_ram_usage,
    log_phase,
    drop_columns_from_dataframe,
    get_pandas_view_of_polars_df,
    estimate_df_size_mb,
    get_process_rss_mb,
    maybe_clean_ram_and_gpu,
    filter_existing,
    compute_model_input_fingerprint,
)
from ..helpers import get_trainset_features_stats_polars, get_trainset_features_stats
from ..models import is_linear_model, LINEAR_MODEL_TYPES
from ..strategies import get_strategy, get_polars_cat_columns, PipelineCache
from ..io import load_mlframe_model
from ..splitting import make_train_test_split
from ..phases import phase, reset_phase_registry, format_phase_summary

# Extractors from new module
from ..extractors import FeaturesAndTargetsExtractor

# score_ensemble is in ensembling module
from ...ensembling import score_ensemble

# Training execution functions from train_eval module
from ..train_eval import process_model, select_target
from ..drift_report import compute_label_distribution_drift, format_drift_report
from ..baseline_diagnostics import BaselineDiagnostics, format_baseline_diagnostics_report
from ._phase_helpers import (  # noqa: E402,F401
from ._misc_helpers import (
    _ensure_logging_visible,
    _entry_metric,
    _augment_with_dropped_high_card_cols,
    _build_full_column_from_splits,
    _drop_cols_df,
    _validate_trusted_path,
    _df_shape_str,
    _elapsed_str,
    _detect_dataset_reuse_capabilities,
    _validate_input_columns_against_metadata,
    _filter_polars_cat_features_by_dtype,
    _auto_detect_feature_types,
    _validate_feature_type_exclusivity,
    _build_tier_dfs,
)  # noqa: E402,F401
    _apply_plot_style_overrides,
    _defensive_copy_and_expand_multilabel_regression,
    _init_composite_discovery_metadata,
    _phase_global_outlier_detection,
    _phase_pandas_conversion_and_cat_prep,
    _log_cardinality_and_drift_snapshot,
    _phase_auto_detect_feature_types,
    _phase_fit_pipeline,
    _phase_train_val_test_split,
    _phase_load_and_preprocess,
    _build_suite_common_params_dict,
    _maybe_dispatch_to_ltr_ranker_suite,
)
from ._phase_helpers import (
    _apply_plot_style_overrides,
    _defensive_copy_and_expand_multilabel_regression,
    _init_composite_discovery_metadata,
    _phase_global_outlier_detection,
    _phase_pandas_conversion_and_cat_prep,
    _log_cardinality_and_drift_snapshot,
    _phase_auto_detect_feature_types,
    _phase_fit_pipeline,
    _phase_train_val_test_split,
    _phase_load_and_preprocess,
    _build_suite_common_params_dict,
    _maybe_dispatch_to_ltr_ranker_suite,
)  # noqa: E402,F401
from ._setup_helpers import (  # noqa: E402,F401
    DEFAULT_PROBABILITY_THRESHOLD,
    _ensure_config,
    _apply_outlier_detection_global,
    _setup_model_directories,
    _build_common_params_for_target,
    _build_pre_pipelines,
    _build_process_model_kwargs,
    _convert_dfs_to_pandas,
    _get_pipeline_components,
    _compute_fairness_subgroups,
    _should_skip_catboost_metamodel,
    _create_initial_metadata,
    _initialize_training_defaults,
    _finalize_and_save_metadata,
)
