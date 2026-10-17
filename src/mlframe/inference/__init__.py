"""Prediction, explainability, and post-training analysis.

Submodules:
    predict         - batch and streaming inference (was inference.py).
    explainability  - SHAP and permutation-based explanation wrappers.
    postanalysis    - post-training analysis utilities.
"""

from __future__ import annotations


from mlframe.inference.predict import *
from mlframe.inference.explainability import *
from mlframe.inference.native_gpu_shap import native_gpu_shap_available, native_xgboost_gpu_shap_contribs
from mlframe.inference.postanalysis import *
from mlframe.inference.logical_constraints import apply_logical_constraints, discover_logical_constraints
from mlframe.inference.group_zero_sum_constraint import apply_group_zero_sum_constraint
from mlframe.inference.time_budget_ensemble import TimeBudgetEnsemble
from mlframe.inference.recursive_forecast import recursive_multi_step_forecast, diagnose_error_accumulation
from mlframe.inference.entity_prediction_collapse import collapse_predictions_by_group

# Curate the star-import surface explicitly (mirrors mlframe.metrics.__init__'s pattern).
__all__ = [
    "Any",
    "Iterable",
    "Optional",
    "TimeBudgetEnsemble",
    "Tuple",
    "analyze_xgboost_model",
    "annotations",
    "apply_group_zero_sum_constraint",
    "apply_logical_constraints",
    "collapse_predictions_by_group",
    "compute_shap_on_cv",
    "diagnose_error_accumulation",
    "discover_logical_constraints",
    "entity_prediction_collapse",
    "explainability",
    "format_classification_report",
    "get_models_raw_predictions",
    "group_zero_sum_constraint",
    "init_model_instance",
    "isdir",
    "isfile",
    "join",
    "log_throttle",
    "logger",
    "logging",
    "logical_constraints",
    "native_gpu_shap",
    "native_gpu_shap_available",
    "native_xgboost_gpu_shap_contribs",
    "np",
    "os",
    "pd",
    "postanalysis",
    "predict",
    "read_trained_models",
    "recursive_forecast",
    "recursive_multi_step_forecast",
    "safe_joblib_load",
    "show_custom_calibration_plot",
    "splitext",
    "sys",
    "time_budget_ensemble",
    "tqdmu",
    "warnings",
]
