"""Hyperparameter candidate sampling/filtering utilities plus a CatBoost-specific ML-guided trial suggestion system.

Provides a rule-based constraint DSL (``check_condition``/``check_rules``) for filtering out invalid hyperparameter
combinations produced by sklearn's ``ParameterSampler`` (``generate_valid_candidates``), and ``ParamsOptimizer`` /
``CatboostParamsOptimizer``, which learn from past trial results stored in a DB (via ``pyutilz.db``) to bias future
candidate sampling toward promising regions of CatBoost's hyperparameter space using a CatBoost surrogate model.

Thin re-export facade (monolith split, CLAUDE.md "sibling re-export" convention): the actual
implementations live in ``_tuning_types.py`` (``MLTaskType``/``HashableDict``), ``tuning_rules.py``
(the rule DSL + ``ParamsOptimizer``), and ``tuning_catboost.py`` (``CatboostParamsOptimizer``) --
split three ways specifically to avoid an import cycle: ``CatboostParamsOptimizer`` needs
``ParamsOptimizer`` as a base class, and ``ParamsOptimizer`` needs the rule-DSL functions, so neither
sibling can import back through this facade without cycling.
"""

from __future__ import annotations

from ._tuning_types import HashableDict, MLTaskType
from .tuning_catboost import CatboostParamsOptimizer
from .tuning_rules import (
    ParamsOptimizer,
    check_condition,
    check_rules,
    create_ctr_params,
    double_check_dist_params,
    favorize_unexplored,
    generate_valid_candidates,
    get_model,
    justify_estimator,
    normalize_probs,
    objective_to_sampling_weights,
    prepare_trials_dataset,
    preprocess_df,
    trained_models,
    value_by_key,
)

__all__ = [
    "MLTaskType",
    "HashableDict",
    "trained_models",
    "check_condition",
    "value_by_key",
    "check_rules",
    "double_check_dist_params",
    "generate_valid_candidates",
    "preprocess_df",
    "prepare_trials_dataset",
    "normalize_probs",
    "objective_to_sampling_weights",
    "favorize_unexplored",
    "get_model",
    "justify_estimator",
    "create_ctr_params",
    "ParamsOptimizer",
    "CatboostParamsOptimizer",
]
