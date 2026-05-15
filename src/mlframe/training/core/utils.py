"""Leaf-level utility helpers for mlframe.training.core, re-exported from sub-modules for backward compatibility."""
from __future__ import annotations

DEFAULT_PROBABILITY_THRESHOLD: float = 0.5

from ._setup_helpers import (  # noqa: E402,F401
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

from ._phase_helpers import (  # noqa: E402,F401
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

from ._misc_helpers import (  # noqa: E402,F401
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
)
