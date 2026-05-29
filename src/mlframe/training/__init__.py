"""
mlframe.training - Modular training pipeline for machine learning models.

This package provides a clean, RAM-efficient training pipeline with support for:
- Linear models (linear, ridge, lasso, elasticnet, huber, ransac, sgd)
- Tree-based models (CatBoost, LightGBM, XGBoost, HistGradientBoosting)
- Neural networks (MLP with PyTorch Lightning)
- AutoML (AutoGluon, LightAutoML)

## Main Functions

- `train_mlframe_models_suite`: Train regular ML models with full pipeline
- `train_automl_models_suite`: Train AutoML models (separate function for RAM efficiency)

## Configuration Classes

All configuration uses Pydantic models but supports dict-like instantiation:

- `PreprocessingConfig`: Data preprocessing settings
- `TrainingSplitConfig`: Train/val/test splitting settings
- `PreprocessingBackendConfig`: Polars-ds pipeline settings
- `LinearModelConfig`: Linear model hyperparameters
- `TreeModelConfig`: Tree model hyperparameters
- `MLPConfig`: Neural network settings
- `AutoMLConfig`: AutoML settings
- `TrainingConfig`: Aggregated configuration

## Example Usage

```python
from mlframe.training import train_mlframe_models_suite, PreprocessingConfig, TrainingSplitConfig

# Train a suite of models including new linear models
models, metadata = train_mlframe_models_suite(
    df="data.parquet",
    target_name="target",
    model_name="experiment_1",
    preprocessor=my_preprocessor,
    mlframe_models=["linear", "ridge", "lasso", "cb", "lgb", "xgb"],
    preprocessing_config=PreprocessingConfig(fillna_value=0.0),
    split_config=TrainingSplitConfig(test_size=0.1, val_size=0.1),
)
```

## AutoML Usage

```python
from mlframe.training import train_automl_models_suite, AutoMLConfig

# Train AutoML models separately (more RAM-efficient)
automl_models = train_automl_models_suite(
    train_df=train_df,  # Must include target column
    test_df=test_df,    # Must include target column
    target_name="target",
    config=AutoMLConfig(
        use_autogluon=True,
        use_lama=False,
        autogluon_fit_params=dict(time_limit=3600),
    ),
)
```
"""

from __future__ import annotations

# joblib / loky shell out to wmic on Windows to count physical cores. The
# wmic subprocess costs ~1s per fresh process and fires inside sklearn
# StratifiedShuffleSplit, LightGBM training params, and any other joblib
# consumer that lets ``n_jobs`` default to None.
#
# Two-pronged mitigation:
#   1. LOKY_MAX_CPU_COUNT env var short-circuits the logical-cores probe
#      path. Honour user-provided overrides (e.g. CI pin to 2).
#   2. Monkey-patch ``loky.backend.context._count_physical_cores`` to return
#      ``os.cpu_count()`` directly. The env var alone DOES NOT cover the
#      ``only_physical_cores=True`` branch, which sklearn's
#      StratifiedShuffleSplit (and other joblib consumers) actually hits.
#      The physical-vs-logical distinction is rarely load-bearing in our
#      paths; treating logical-count as physical-count is a benign
#      approximation that avoids the wmic round trip entirely.
import os as _os

# Audit 2026-05-17 (Wave 1.5): the loky physical-core count override
# was originally applied at import time as a global side effect, which
# affected any joblib consumer in the same Python process even when the
# user only imported mlframe for a single helper. We now expose it as an
# opt-in idempotent function ``apply_loky_cpu_count_override()`` and the
# suite entrypoint (``train_mlframe_models_suite``) calls it once on the
# first invocation. ``LOKY_MAX_CPU_COUNT`` env var is still set at
# import time -- it's a single env-var write, easy to reverse, and the
# wmic-spawn-on-Windows perf hit only triggers when something actually
# probes physical cores.
if not _os.environ.get("LOKY_MAX_CPU_COUNT"):
    _os.environ["LOKY_MAX_CPU_COUNT"] = str(_os.cpu_count() or 1)

_loky_override_applied = False


def apply_loky_cpu_count_override() -> None:
    """Patch ``loky.backend.context._count_physical_cores`` to return
    the result of ``os.cpu_count()`` (logical = physical approximation).

    Idempotent. Originally applied at import time -- per the 2026-05-17
    audit we made this opt-in: the suite entrypoint invokes it, but
    bare ``import mlframe.training`` no longer mutates joblib state.
    """
    global _loky_override_applied
    if _loky_override_applied:
        return
    try:
        from joblib.externals.loky.backend import context as _loky_ctx
        _count = _os.cpu_count() or 1
        _loky_ctx._count_physical_cores = lambda: (_count, _count)
        _loky_override_applied = True
    except Exception:
        pass


# Lazy imports via __getattr__ — deferred until actually accessed (kept for
# fast-import / optional-dep isolation; no longer guarding a training_old cycle).

_LAZY_IMPORTS = {
    # Core functions
    'train_mlframe_models_suite': ('.core', 'train_mlframe_models_suite'),
    'train_automl_models_suite': ('.automl', 'train_automl_models_suite'),

    # Configuration classes
    'PreprocessingConfig': ('.configs', 'PreprocessingConfig'),
    'TrainingSplitConfig': ('.configs', 'TrainingSplitConfig'),
    'PreprocessingBackendConfig': ('.configs', 'PreprocessingBackendConfig'),
    'PreprocessingExtensionsConfig': ('.configs', 'PreprocessingExtensionsConfig'),
    'FeatureSelectionConfig': ('.configs', 'FeatureSelectionConfig'),
    'FeatureTypesConfig': ('.configs', 'FeatureTypesConfig'),
    'LinearModelConfig': ('.configs', 'LinearModelConfig'),
    'TreeModelConfig': ('.configs', 'TreeModelConfig'),
    'MLPConfig': ('.configs', 'MLPConfig'),
    'NGBConfig': ('.configs', 'NGBConfig'),
    'AutoMLConfig': ('.configs', 'AutoMLConfig'),
    'ModelHyperparamsConfig': ('.configs', 'ModelHyperparamsConfig'),
    'TrainingBehaviorConfig': ('.configs', 'TrainingBehaviorConfig'),
    'TrainingConfig': ('.configs', 'TrainingConfig'),
    'config_from_dict': ('.configs', 'config_from_dict'),
    'TargetTypes': ('.configs', 'TargetTypes'),
    # 2026-04-24: multi-output configs (multiclass / multilabel dispatch)
    'MultilabelDispatchConfig': ('.configs', 'MultilabelDispatchConfig'),
    'EnsemblingConfig': ('.configs', 'EnsemblingConfig'),
    # 2026-05-04: learning-to-rank dispatch
    'LearningToRankConfig': ('.configs', 'LearningToRankConfig'),
    # 2026-05-08: quantile-regression dispatch
    'QuantileRegressionConfig': ('.configs', 'QuantileRegressionConfig'),
    'train_mlframe_ranker_suite': ('.ranker_suite', 'train_mlframe_ranker_suite'),
    'fit_ranker': ('.ranking', 'fit_ranker'),
    'predict_ranker_scores': ('.ranking', 'predict_ranker_scores'),
    'ensemble_ranker_scores': ('.ranking', 'ensemble_ranker_scores'),
    # Probability-surface helpers — re-exported without underscore prefix.
    # Users writing custom estimators need these to canonicalise predict_proba
    # output and apply the standard decision rule per target_type.
    'canonical_predict_proba_shape': ('.helpers', '_canonical_predict_proba_shape'),
    'predict_from_probs': ('.helpers', '_predict_from_probs'),

    # Selection-bias / drift tools (Session 7)
    'compute_label_distribution_drift': ('.drift_report', 'compute_label_distribution_drift'),
    'format_drift_report': ('.drift_report', 'format_drift_report'),
    'PULearningWrapper': ('.pu_learning', 'PULearningWrapper'),
    'estimate_c_from_unbiased_positives': ('.pu_learning', 'estimate_c_from_unbiased_positives'),
    'audit_target_over_time': ('.target_temporal_audit', 'audit_target_over_time'),
    'audit_targets_over_time': ('.target_temporal_audit', 'audit_targets_over_time'),
    'audit_residuals': ('.regression_residual_audit', 'audit_residuals'),
    'format_residual_audit_report': ('.regression_residual_audit', 'format_residual_audit_report'),
    'plot_residual_diagnostics': ('.regression_residual_audit', 'plot_residual_diagnostics'),
    'ResidualAudit': ('.regression_residual_audit', 'ResidualAudit'),
    'plot_target_over_time': ('.target_temporal_audit', 'plot_target_over_time'),
    'format_temporal_audit_report': ('.target_temporal_audit', 'format_temporal_audit_report'),
    'find_change_points': ('.target_temporal_audit', 'find_change_points'),
    'find_change_points_pelt': ('.target_temporal_audit', 'find_change_points_pelt'),
    'find_change_points_zscore': ('.target_temporal_audit', 'find_change_points_zscore'),
    'TemporalAuditResult': ('.target_temporal_audit', 'TemporalAuditResult'),

    # Model utilities
    'create_linear_model': ('.models', 'create_linear_model'),
    'is_linear_model': ('.models', 'is_linear_model'),
    'is_tree_model': ('.models', 'is_tree_model'),
    'is_neural_model': ('.models', 'is_neural_model'),
    'LINEAR_MODEL_TYPES': ('.models', 'LINEAR_MODEL_TYPES'),

    # Preprocessing utilities
    'load_and_prepare_dataframe': ('.preprocessing', 'load_and_prepare_dataframe'),
    'preprocess_dataframe': ('.preprocessing', 'preprocess_dataframe'),
    'save_split_artifacts': ('.preprocessing', 'save_split_artifacts'),
    'create_split_dataframes': ('.preprocessing', 'create_split_dataframes'),

    # Pipeline utilities
    'create_polarsds_pipeline': ('.pipeline', 'create_polarsds_pipeline'),
    'fit_and_transform_pipeline': ('.pipeline', 'fit_and_transform_pipeline'),
    'prepare_df_for_catboost': ('.pipeline', 'prepare_df_for_catboost'),
    'prepare_dfs_for_catboost_joint': ('.pipeline', 'prepare_dfs_for_catboost_joint'),

    # Utility functions
    'log_ram_usage': ('.utils', 'log_ram_usage'),
    'log_phase': ('.utils', 'log_phase'),
    'drop_columns_from_dataframe': ('.utils', 'drop_columns_from_dataframe'),
    'get_pandas_view_of_polars_df': ('.utils', 'get_pandas_view_of_polars_df'),
    'save_series_or_df': ('.utils', 'save_series_or_df'),
    'process_nans': ('.utils', 'process_nans'),
    'process_nulls': ('.utils', 'process_nulls'),
    'process_infinities': ('.utils', 'process_infinities'),
    'remove_constant_columns': ('.utils', 'remove_constant_columns'),
    'clean_ram_and_gpu': ('.utils', 'clean_ram_and_gpu'),
    'estimate_df_size_mb': ('.utils', 'estimate_df_size_mb'),
    'get_process_rss_mb': ('.utils', 'get_process_rss_mb'),
    'should_clean_ram': ('.utils', 'should_clean_ram'),
    'maybe_clean_ram_and_gpu': ('.utils', 'maybe_clean_ram_and_gpu'),
    'get_numeric_columns': ('.utils', 'get_numeric_columns'),
    'get_categorical_columns': ('.utils', 'get_categorical_columns'),
    'filter_existing': ('.utils', 'filter_existing'),

    # IO utilities
    'save_mlframe_model': ('.io', 'save_mlframe_model'),
    'load_mlframe_model': ('.io', 'load_mlframe_model'),

    # Splitting utilities
    'make_train_test_split': ('.splitting', 'make_train_test_split'),

    # Evaluation utilities
    'evaluate_model': ('.evaluation', 'evaluate_model'),

    # Training execution functions
    'select_target': ('.train_eval', 'select_target'),
    'process_model': ('.train_eval', 'process_model'),

    # Core trainer functions
    'train_and_evaluate_model': ('.trainer', 'train_and_evaluate_model'),
    'configure_training_params': ('.trainer', 'configure_training_params'),
    'DataConfig': ('.configs', 'DataConfig'),
    'TrainingControlConfig': ('.configs', 'TrainingControlConfig'),
    'MetricsConfig': ('.configs', 'MetricsConfig'),
    'ReportingConfig': ('.configs', 'ReportingConfig'),
    'FeatureImportanceConfig': ('.configs', 'FeatureImportanceConfig'),
    'OutputConfig': ('.configs', 'OutputConfig'),
    'OutlierDetectionConfig': ('.configs', 'OutlierDetectionConfig'),
    'NamingConfig': ('.configs', 'NamingConfig'),
    'ConfidenceAnalysisConfig': ('.configs', 'ConfidenceAnalysisConfig'),
    'PredictionsContainer': ('.configs', 'PredictionsContainer'),
    'FairnessConfig': ('.configs', 'FairnessConfig'),

    # Helper functions
    'get_trainset_features_stats': ('.helpers', 'get_trainset_features_stats'),
    'get_trainset_features_stats_polars': ('.helpers', 'get_trainset_features_stats_polars'),
    'get_training_configs': ('.helpers', 'get_training_configs'),
    'parse_catboost_devices': ('.helpers', 'parse_catboost_devices'),
    'LightGBMCallback': ('.helpers', 'LightGBMCallback'),
    'CatBoostCallback': ('.helpers', 'CatBoostCallback'),
    'XGBoostCallback': ('.helpers', 'XGBoostCallback'),

    # Grid runner
    'run_grid': ('.grid', 'run_grid'),

    # GPU availability probes (public re-exports of ``_gpu_probe`` constants so cross-package consumers do not have to reach into a private module).
    'CUDA_IS_AVAILABLE': ('._gpu_probe', 'CUDA_IS_AVAILABLE'),
    'XGB_GPU_AVAILABLE': ('._gpu_probe', 'XGB_GPU_AVAILABLE'),
    'LGB_GPU_AVAILABLE': ('._gpu_probe', 'LGB_GPU_AVAILABLE'),

    # Model-tag formatting helpers (public surface; underscore source remains the implementation).
    'short_model_tag': ('._format', 'short_model_tag'),
    'strip_shim_suffix': ('._format', 'strip_shim_suffix'),
}

_cache = {}


def __getattr__(name):
    """Lazy import handler for module attributes."""
    if name in _LAZY_IMPORTS:
        if name not in _cache:
            module_name, attr_name = _LAZY_IMPORTS[name]
            import importlib
            # Import the submodule directly, not through the package
            full_module = f"mlframe.training.{module_name[1:]}"  # Remove leading dot and add separator
            module = importlib.import_module(full_module)
            _cache[name] = getattr(module, attr_name)
        return _cache[name]
    raise AttributeError(f"module 'mlframe.training' has no attribute {name!r}")


# train_and_evaluate_model is now in trainer.py and loaded via _LAZY_IMPORTS


from ._partial_fit_es_wrapper import PartialFitESWrapper  # noqa: E402, F401


__all__ = [
    # Core functions
    'train_mlframe_models_suite',
    'train_automl_models_suite',
    'make_train_test_split',
    'train_and_evaluate_model',
    'configure_training_params',

    # Early stopping helpers
    'PartialFitESWrapper',

    # Configuration
    'PreprocessingConfig',
    'TrainingSplitConfig',
    'PreprocessingBackendConfig',
    'PreprocessingExtensionsConfig',
    'FeatureSelectionConfig',
    'FeatureTypesConfig',
    'LinearModelConfig',
    'TreeModelConfig',
    'MLPConfig',
    'NGBConfig',
    'AutoMLConfig',
    'ModelHyperparamsConfig',
    'TrainingBehaviorConfig',
    'TrainingConfig',
    'config_from_dict',
    'TargetTypes',
    'MultilabelDispatchConfig',
    'EnsemblingConfig',
    'LearningToRankConfig',
    'QuantileRegressionConfig',
    'train_mlframe_ranker_suite',
    'fit_ranker',
    'predict_ranker_scores',
    'ensemble_ranker_scores',
    'canonical_predict_proba_shape',
    'predict_from_probs',
    'compute_label_distribution_drift',
    'format_drift_report',
    'PULearningWrapper',
    'estimate_c_from_unbiased_positives',
    'audit_target_over_time',
    'audit_targets_over_time',
    'audit_residuals',
    'format_residual_audit_report',
    'plot_residual_diagnostics',
    'ResidualAudit',
    'plot_target_over_time',
    'format_temporal_audit_report',
    'find_change_points',
    'find_change_points_pelt',
    'find_change_points_zscore',
    'TemporalAuditResult',
    'DataConfig',
    'TrainingControlConfig',
    'MetricsConfig',
    'ReportingConfig',
    'FeatureImportanceConfig',
    'OutputConfig',
    'OutlierDetectionConfig',
    'NamingConfig',
    'ConfidenceAnalysisConfig',
    'PredictionsContainer',
    'FairnessConfig',

    # Models
    'create_linear_model',
    'is_linear_model',
    'is_tree_model',
    'is_neural_model',
    'LINEAR_MODEL_TYPES',

    # Preprocessing
    'load_and_prepare_dataframe',
    'preprocess_dataframe',
    'save_split_artifacts',
    'create_split_dataframes',

    # Pipeline
    'create_polarsds_pipeline',
    'fit_and_transform_pipeline',
    'prepare_df_for_catboost',
    'prepare_dfs_for_catboost_joint',

    # Utils
    'log_ram_usage',
    'log_phase',
    'drop_columns_from_dataframe',
    'save_mlframe_model',
    'load_mlframe_model',
    'get_pandas_view_of_polars_df',
    'save_series_or_df',
    'process_nans',
    'process_nulls',
    'process_infinities',
    'remove_constant_columns',
    # Previously importable via `from mlframe.training import ...` but missing from __all__
    'NGBConfig',
    'clean_ram_and_gpu',
    'estimate_df_size_mb',
    'get_process_rss_mb',
    'should_clean_ram',
    'maybe_clean_ram_and_gpu',
    'get_numeric_columns',
    'get_categorical_columns',

    # Evaluation
    'evaluate_model',

    # Training execution
    'select_target',
    'process_model',

    # Helper functions (from helpers.py)
    'get_trainset_features_stats',
    'get_trainset_features_stats_polars',
    'get_training_configs',
    'parse_catboost_devices',
    'LightGBMCallback',
    'CatBoostCallback',
    'XGBoostCallback',

    # Grid runner
    'run_grid',

    # GPU availability probes
    'CUDA_IS_AVAILABLE',
    'XGB_GPU_AVAILABLE',
    'LGB_GPU_AVAILABLE',

    # Model-tag formatting helpers
    'short_model_tag',
    'strip_shim_suffix',
]


__version__ = '2.0.0'  # Major refactoring version
