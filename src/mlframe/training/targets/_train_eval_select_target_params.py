"""Stages of ``select_target``."""

from __future__ import annotations

# --- imports (managed) ---
from mlframe.training.configs import ModelHyperparamsConfig
from mlframe.training.configs import TrainingBehaviorConfig
# --- end imports ---


def _resolve_effective_params(hyperparams_config, behavior_config):
    """Stage of ``select_target``."""
    if hyperparams_config is None:
        hyperparams_config = ModelHyperparamsConfig()
    if behavior_config is None:
        behavior_config = TrainingBehaviorConfig()

    effective_config_params = hyperparams_config.model_dump(exclude_none=True)
    # ``exclude_none=True`` drops fields the caller EXPLICITLY set to None, conflating "unset" with
    # "explicitly None". For ``early_stopping_rounds`` None is a meaningful sentinel that get_training_configs
    # reads as "disable ES entirely" (0 = auto = iterations//3, int = as-is). Dropping it makes the user's
    # explicit disable silently fall back to get_training_configs's auto-ES default, so the booster early-stops
    # regardless -- a caller passing early_stopping_rounds=None never gets the full-iteration run they asked for.
    # Re-add any field the caller explicitly set to None so explicit intent survives the dump.
    for _explicit in hyperparams_config.model_fields_set:
        if _explicit not in effective_config_params and getattr(hyperparams_config, _explicit, "") is None:
            effective_config_params[_explicit] = None
    defined_behavior_fields = set(TrainingBehaviorConfig.model_fields.keys())
    _SUITE_LEVEL_FLAGS = {
        "enable_crash_reporting",
        "continue_on_model_failure",
        "align_polars_categorical_dicts",
        "model_file_hash_suffix",
        "target_temporal_audit_column",
        "target_temporal_audit_granularity",
        "target_temporal_audit_save_plot",
        "report_residual_audit",
        "confidence_ensemble_quantile",
        "feature_drift_auto_apply_neural_overrides",
    }
    effective_behavior_params = {
        k: v for k, v in behavior_config.model_dump(exclude_none=True).items() if k in defined_behavior_fields and k not in _SUITE_LEVEL_FLAGS
    }
    precomputed_fairness = behavior_config.precomputed_fairness_subgroups
    return effective_behavior_params, effective_config_params, precomputed_fairness
