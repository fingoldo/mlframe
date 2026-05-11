"""``mlframe.training.core`` -- training-suite top-level package.

Re-exports the public API so existing callers continue to do:

    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.core import predict_mlframe_models_suite, load_mlframe_suite

without caring that the implementation lives across three sibling modules.

Layout
------
- :mod:`mlframe.training.core.main` -- ``train_mlframe_models_suite`` orchestrator
- :mod:`mlframe.training.core.utils` -- 27 leaf utility helpers + ``DEFAULT_PROBABILITY_THRESHOLD``
- :mod:`mlframe.training.core.predict` -- ``predict_*`` / ``load_mlframe_suite`` entry points
"""
from __future__ import annotations

# Public training entry points.
from .main import train_mlframe_models_suite  # noqa: F401

# Predict / load entry points.
from .predict import (  # noqa: F401
    predict_mlframe_models_suite,
    predict_from_models,
    load_mlframe_suite,
)

# Leaf-utility helpers + constants (legacy back-compat: tests + downstream
# composite modules import some of these directly from ``mlframe.training.core``).
from .utils import (  # noqa: F401
    DEFAULT_PROBABILITY_THRESHOLD,
    _ensure_logging_visible,
    _entry_metric,
    _augment_with_dropped_high_card_cols,
    _build_full_column_from_splits,
    _build_suite_common_params_dict,
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
    _maybe_dispatch_to_ltr_ranker_suite,
    _log_cardinality_and_drift_snapshot,
    _phase_auto_detect_feature_types,
    _phase_fit_pipeline,
    _phase_load_and_preprocess,
    _phase_pandas_conversion_and_cat_prep,
    _phase_train_val_test_split,
)
