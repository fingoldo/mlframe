"""Core training functions for mlframe."""

from __future__ import annotations


import logging

logger = logging.getLogger(__name__)

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from pyutilz.strings import slugify
from pyutilz.system import tqdmu_lazy_start

from ..configs import (
    BaselineDiagnosticsConfig,
    CompositeTargetDiscoveryConfig,
    ConfidenceAnalysisConfig,
    DummyBaselinesConfig,
    FeatureSelectionConfig,
    FeatureTypesConfig,
    LearningToRankConfig,
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
    TrainingSplitConfig,
)
from pathlib import Path as _P  # PATHLIB-IMPORT-PER-CALL: hoist to module scope (was paid per suite call)
from ..extractors import FeaturesAndTargetsExtractor
from ..feature_handling.fingerprint import reset_session as reset_fh_session
from ..helpers import (
    TrainMlframeSuitePrecomputed,
    get_trainset_features_stats,
    get_trainset_features_stats_polars,
)
from ..phases import phase, reset_phase_registry
from ..utils import (
    log_phase,
    log_ram_usage,
)

from .utils import (
    _ensure_logging_visible,
    _finalize_and_save_metadata,
    _get_pipeline_components,
    _initialize_training_defaults,
    _log_cardinality_and_drift_snapshot,
    _maybe_dispatch_to_ltr_ranker_suite,
    _phase_auto_detect_feature_types,
    _phase_fit_pipeline,
    _phase_global_outlier_detection,
    _phase_load_and_preprocess,
    _phase_pandas_conversion_and_cat_prep,
    _phase_train_val_test_split,
)
from ._training_context import TrainingContext
# CODE-P1-8: single consolidated import for all per-phase entry points (was 8 separate ``from
# ._phase_X import Y`` lines). Call e.g. ``pr.apply_polars_categorical_fixes(...)``.
from . import _phase_runners as pr


from ._misc_helpers import _bulk_setattr_to_ctx, _split_preds_probs, _prep_polars_df  # noqa: F401


# The prelude patch handles (apply_loky_cpu_count_override /
# apply_third_party_patches_once) live on ``_main_train_suite`` -- the module
# that actually holds the ``train_mlframe_models_suite`` body and whose globals
# the live prelude resolves against. This facade only re-exports the callable,
# so no module-level seam is kept here.


# Re-export predict / load entry points for back-compat.
from .predict import (  # noqa: E402,F401
    predict_mlframe_models_suite,
    predict_from_models,
    load_mlframe_suite,
)


# ----------------------------------------------------------------------
# Sibling-module re-export. The 1008-LOC ``train_mlframe_models_suite``
# body lives in ``_main_train_suite.py`` so this file stays below the
# 1k-LOC monolith threshold.
# ----------------------------------------------------------------------
from ._main_train_suite import train_mlframe_models_suite  # noqa: E402,F401
