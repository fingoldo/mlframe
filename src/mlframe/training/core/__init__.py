"""``mlframe.training.core`` training-suite top-level package: re-exports the public API.

Underscore-module convention
----------------------------
Every ``_phase_*.py`` / ``_predict_main_*.py`` / ``_setup_helpers.py`` / ``_misc_helpers.py``
sibling under this package is **internal**. The leading underscore is the project's portable
"do not import from outside the sibling cluster" signal:

* Production code (anything that ships as part of the installed wheel) must NOT import any
  ``mlframe.training.core._*`` module from elsewhere in the package tree -- the public surface is
  exclusively the names re-exported from this ``__init__.py``.
* Tests, ``_benchmarks/``, ``_profile_*`` harnesses, and ``audit/`` agents MAY import internal
  modules directly for white-box coverage / instrumentation -- a tight test boundary.
* Renames / signature changes inside a ``_*`` module DO NOT require a deprecation cycle. If
  external code reaches into an underscore module it accepts the breakage risk.

This is enforced as a meta-test: see ``tests/test_meta/test_no_production_underscore_imports.py``.
"""
from __future__ import annotations

# Back-compat for tests that read these via ``mlframe.training.core.X`` before monkeypatching them at submodule level.
from sklearn.base import clone  # noqa: F401
from ..pipeline import fit_and_transform_pipeline  # noqa: F401

from .main import train_mlframe_models_suite  # noqa: F401

from .predict import (  # noqa: F401
    predict_mlframe_models_suite,
    predict_from_models,
    load_mlframe_suite,
)

# Legacy back-compat: tests + downstream composite modules import some of these directly from ``mlframe.training.core``.
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
    _apply_plot_style_overrides,
    _init_composite_discovery_metadata,
    _setup_model_directories,
    _build_common_params_for_target,
    _build_pre_pipelines,
    _build_process_model_kwargs,
    _convert_dfs_to_pandas,
    _get_pipeline_components,
    _compute_fairness_subgroups,
    _should_skip_catboost_metamodel,
    _create_initial_metadata,
    _defensive_copy_and_expand_multilabel_regression,
    _initialize_training_defaults,
    _finalize_and_save_metadata,
    _maybe_dispatch_to_ltr_ranker_suite,
    _log_cardinality_and_drift_snapshot,
    _phase_auto_detect_feature_types,
    _phase_fit_pipeline,
    _phase_global_outlier_detection,
    _phase_load_and_preprocess,
    _phase_pandas_conversion_and_cat_prep,
    _phase_train_val_test_split,
)
