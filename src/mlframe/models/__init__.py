"""Model operations: ensembling, hyperparameter tuning, splitting.

Submodules:
    ensembling    - stacking, blending, voting, weighted ensembles.
    optimization  - hyperparameter optimization (Optuna, GridSearch wrappers).
    tuning        - tuning utilities and CV-aware fit-predict loops.
    selection     - train / test / cv-fold splitting (was model_selection.py).
    rf_proximity  - Breiman random-forest proximity matrix + outlier measure.
"""

from __future__ import annotations


from mlframe.models.ensembling import *
from mlframe.models.optimization import *
from mlframe.models.tuning import *
from mlframe.models.selection import *
from mlframe.models.rf_proximity import *
from mlframe.models.lgbm_defaults import default_lgbm_params
from mlframe.models.additive_interaction_diagnostic import additive_interaction_diagnostic
from mlframe.models.masked_multilabel_objective import flatten_masked_multilabel, masked_multilabel_logloss_objective

# Curate the star-import surface explicitly (mirrors mlframe.metrics.__init__'s pattern).
__all__ = [
    "Any",
    "BIG_VALUE",
    "BackwardEliminationResult",
    "Callable",
    "CandidateSamplingMethod",
    "CaruanaSelectionResult",
    "CatboostParamsOptimizer",
    "EnsembleLeaderboard",
    "GroupTimeSeriesSplit",
    "HashableDict",
    "MBHOptimizer",
    "MLTaskType",
    "NOT_READY",
    "OptimizationDirection",
    "OptimizationProgressPlotting",
    "Optional",
    "ParamsOptimizer",
    "RANK_FUSION_METHODS",
    "SIMPLE_ENSEMBLING_METHODS",
    "SMALL_VALUE",
    "Sequence",
    "StepwiseSelectionResult",
    "StreamingAccumulator",
    "Union",
    "additive_interaction_diagnostic",
    "annotations",
    "base",
    "batch_numaggs",
    "build_predictive_kwargs",
    "caruana_greedy_selection",
    "check_condition",
    "check_rules",
    "combine_float_predictions",
    "combine_probs",
    "compare_ensembles",
    "compute_candidates_exploration_scores",
    "compute_high_correlation_pairs",
    "compute_member_quality_gate",
    "create_ctr_params",
    "default_lgbm_params",
    "double_check_dist_params",
    "enrich_ensemble_preds_with_numaggs",
    "ensemble_probabilistic_predictions",
    "ensemble_probabilistic_predictions_streaming",
    "ensembling",
    "favorize_unexplored",
    "flatten_masked_multilabel",
    "flavour_policy",
    "float_aggregation",
    "generate_fibonacci",
    "generate_valid_candidates",
    "get_model",
    "greedy_backward_ensemble_elimination",
    "indexable",
    "justify_estimator",
    "lgbm_defaults",
    "logger",
    "logging",
    "masked_multilabel_logloss_objective",
    "masked_multilabel_objective",
    "member_metrics",
    "normalize_probs",
    "np",
    "objective_to_sampling_weights",
    "optimization",
    "optimize_finite_onedimensional_search_space",
    "pd",
    "plot_search_state",
    "predict",
    "prepare_trials_dataset",
    "preprocess_df",
    "process_method",
    "proximity_to_distance",
    "quality_gate",
    "rank_average_blend",
    "rf_outlier_measure",
    "rf_proximity",
    "rf_proximity_matrix",
    "robust_float_ensemble",
    "rrf_ensemble",
    "score",
    "score_ensemble",
    "score_flavours",
    "score_gate",
    "score_levels",
    "score_validate",
    "selection",
    "stepwise_ensemble_selection",
    "timer",
    "trained_models",
    "tuning",
    "tuning_catboost",
    "tuning_rules",
    "value_by_key",
]
