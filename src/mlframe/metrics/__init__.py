"""Probabilistic, classification, ranking, and quantile metrics.

Submodules:
    core      - core metric definitions (ICE, ECE, Brier REL/RES/UNC, CMAEW, ...).
    quantile  - quantile-specific metrics (pinball, coverage).
    ranking   - ranking-task metrics (NDCG, MRR, ...).
    scoring   - sklearn-compatible scoring wrappers.

Import convention: this package's ``__all__`` (below) is the curated EXTERNAL-facing surface -- the
one a caller outside ``metrics/`` who doesn't know the internal module layout should reach for. It is
NOT a mandate that every internal call site rewrite ``from mlframe.metrics.core import X`` to
``from mlframe.metrics import X``: ``core.py`` (unlike the underscore-prefixed internal submodules) is
itself an intentionally public, directly-importable module, and importing straight from it avoids
eagerly loading this package's other submodules (``quantile``/``ranking``/``scoring``/``calibration``/
``classification``/``regression``/``iteration_metrics``) just to reach one ``core`` symbol. Both forms
are supported and bit-identical (the facade re-exports ``core`` in full via ``from .core import *``);
pick whichever import-cost tradeoff fits the call site.
"""

from __future__ import annotations


from mlframe.metrics.core import *
from mlframe.metrics.quantile import *
from mlframe.metrics.ranking import *
from mlframe.metrics.scoring import *
from mlframe.metrics.rank_correlation import (
    set_spearmanr_dispatch_threshold,
    spearmanr_batched,
    spearmanr_batched_dispatch,
)
from mlframe.metrics._core_auc_brier import bootstrap_auc_distribution_parallel

# Public re-export so cross-package consumers (importance.py, _eval_helpers.py) can avoid reaching into ``mlframe.metrics.calibration`` internals directly. The underscore-prefixed source remains the implementation; the public name is the documented surface.
# Public re-export of the Brier kernel so reporting consumers (model_card) import it from the package surface instead of the ``_core_auc_brier`` implementation module.
from mlframe.metrics._core_auc_brier import fast_brier_score_loss

# Public re-export of the NumPy-version-agnostic trapezoidal integral so reporting/charts consumers (binary.py, model_comparison.py, risk_coverage.py) import it from the package surface instead of the ``_trapezoid_shared`` implementation module.
from mlframe.metrics._trapezoid_shared import trapezoid

# Lean full-suite per-target-type aggregator used by per-iteration metric capture (meta-learning / HPO-from-early-observation).
from mlframe.metrics.iteration_metrics import compute_all_metrics

# Per-target calibration / classification panel. Documented in README as ``from mlframe.metrics import fast_calibration_report``; make it a first-class export instead of relying on the transitive ``from .core import *`` side-effect. ``CalibrationReport`` is its NamedTuple return type.
from mlframe.metrics.classification import (
    CalibrationReport,
    fast_calibration_report,
)

# Public re-export of the scalar classification / regression metrics so callers use the documented
# ``from mlframe.metrics import quadratic_weighted_kappa`` surface instead of reaching into the concrete submodule (which
# is exactly what the metric registry itself imports). Only ``core`` is star-imported above, so these submodule names
# were otherwise reachable only as ``mlframe.metrics.classification.*`` / ``.regression.*``. Explicit (not star) to avoid
# shadowing anything the ``core`` star already brought in.
from mlframe.metrics.classification import (
    apply_cutpoints,
    cumulative_gains_curve,
    exploss,
    gains_table,
    lift_curve,
    optimal_ordinal_cutpoints,
    optimal_threshold,
    quadratic_weighted_kappa,
    weighted_kappa,
)
from mlframe.metrics.regression import (
    fast_epsilon_band_accuracy,
    fast_logcosh_loss,
    fast_mrae,
    fast_percent_better,
    fast_rel_mae,
    fast_rmspe,
)

# Preserve the historical ``from mlframe.metrics import *`` surface (every public name the star-imports above brought in)
# while GUARANTEEING the documented ``fast_calibration_report`` / ``CalibrationReport`` are part of it as a first-class contract.
from ._numba_params import NUMBA_NJIT_PARAMS

__all__ = [
    "CalibrationReport",
    "Callable",
    "DEFAULT_TITLE_METRICS_TOKENS",
    "ICE",
    "LogUniform",
    "NUMBA_NJIT_PARAMS",
    "ProbaScoreProxy",
    "TITLE_METRIC_TOKENS",
    "accuracy_ratio",
    "annotations",
    "apply_cutpoints",
    "average_precision_score",
    "balanced_accuracy_binary",
    "bootstrap_auc_distribution_parallel",
    "brier_and_precision_score",
    "brier_score_loss",
    "brier_skill_score",
    "calibration",
    "calibration_metrics_from_freqs",
    "cb_logits_to_probs_binary",
    "cb_logits_to_probs_multiclass",
    "classification",
    "cohen_kappa_binary",
    "compute_all_metrics",
    "compute_batch_aucs",
    "compute_batch_rmse",
    "compute_brier_decomposition_debiased",
    "compute_ece_and_brier_decomposition",
    "compute_ece_brier_full_and_debiased",
    "compute_ece_debiased",
    "compute_fairness_metrics",
    "compute_grouped_group_aucs",
    "compute_mean_aucs_per_group",
    "compute_pr_recall_f1_metrics",
    "compute_probabilistic_multiclass_error",
    "compute_ranking_summary",
    "compute_robustness_metrics",
    "core",
    "coverage",
    "coverage_error",
    "create_fairness_subgroups",
    "create_fairness_subgroups_indices",
    "create_robustness_standard_bins",
    "create_robustness_subgroups",
    "create_robustness_subgroups_indices",
    "crps_from_quantiles",
    "cumulative_gains_curve",
    "dcg_at_k",
    "expected_reciprocal_rank",
    "exploss",
    "f_beta_score",
    "fast_adjusted_r2_score",
    "fast_aucs",
    "fast_aucs_per_group",
    "fast_aucs_per_group_optimized",
    "fast_binary_confusion_metrics_block",
    "fast_binary_probability_metrics_block",
    "fast_brier_score_loss",
    "fast_calibration_binning",
    "fast_calibration_metrics",
    "fast_calibration_report",
    "fast_classification_report",
    "fast_concordance_index",
    "fast_cv_rmse",
    "fast_epsilon_band_accuracy",
    "fast_explained_variance",
    "fast_gamma_deviance",
    "fast_huber_loss",
    "fast_ice_only",
    "fast_kendall_tau",
    "fast_log_loss",
    "fast_log_loss_binary",
    "fast_logcosh_loss",
    "fast_mape_mean",
    "fast_mase",
    "fast_max_error",
    "fast_mdape",
    "fast_mean_absolute_error",
    "fast_mean_bias_error",
    "fast_mean_squared_error",
    "fast_mrae",
    "fast_multiclass_confusion_metrics_block",
    "fast_multilabel_classification_metrics_block",
    "fast_nash_sutcliffe",
    "fast_numba_auc_nonw",
    "fast_numba_auc_weighted",
    "fast_numba_aucs",
    "fast_numba_aucs_simple",
    "fast_numba_aucs_with_ks",
    "fast_pearson_corr",
    "fast_percent_better",
    "fast_poisson_deviance",
    "fast_precision",
    "fast_r2_score",
    "fast_regression_metrics_block",
    "fast_regression_metrics_block_extended",
    "fast_rel_mae",
    "fast_rmse",
    "fast_rmsle",
    "fast_rmspe",
    "fast_roc_auc",
    "fast_roc_auc_unstable",
    "fast_roc_curve",
    "fast_root_mean_squared_error",
    "fast_smape",
    "fast_spearman_corr",
    "fast_tweedie_deviance",
    "fast_wmape",
    "format_classification_report",
    "g_mean_binary",
    "gains_table",
    "gini_from_auc",
    "gpu_multiple_pr_auc_scores",
    "gpu_multiple_rmse_scores",
    "gpu_multiple_roc_auc_scores",
    "hamming_loss",
    "hit_at_k",
    "hosmer_lemeshow_test",
    "integral_calibration_error_from_metrics",
    "is_gpu_metrics_available",
    "iteration_metrics",
    "jaccard_score_multilabel",
    "js_divergence",
    "kl_divergence",
    "ks_distribution_distance",
    "ks_statistic",
    "label_ranking_average_precision",
    "label_ranking_loss",
    "lift_at_k",
    "lift_curve",
    "logger",
    "logging",
    "make_bootstrap_auc_resampler",
    "make_brier_precision_scorer",
    "make_scorer",
    "map_at_k",
    "matthews_corrcoef_binary",
    "matthews_corrcoef_multiclass",
    "maximum_absolute_percentage_error",
    "mean_interval_width",
    "mrr",
    "multilabel_auc_macro",
    "multilabel_auc_per_label",
    "multilabel_auc_weighted",
    "multilabel_f1_macro",
    "multilabel_f1_micro",
    "multilabel_f1_weighted",
    "ndcg_at_k",
    "np",
    "numba",
    "numba_warmup",
    "one_error",
    "optimal_ordinal_cutpoints",
    "optimal_threshold",
    "pinball_loss",
    "pinball_loss_per_alpha",
    "pit_values",
    "population_stability_index",
    "precision_at_k",
    "predictions_time_instability",
    "prewarm_numba_cache",
    "probability_separation_score",
    "quadratic_weighted_kappa",
    "quantile",
    "quantile_summary",
    "rank_correlation",
    "ranked_probability_score",
    "ranking",
    "regression",
    "render_title_metric_token",
    "rmse_loss",
    "rmse_score",
    "rmsle_loss",
    "rmsle_score",
    "robust_mlperf_metric",
    "scoring",
    "set_gpu_thresholds",
    "set_spearmanr_dispatch_threshold",
    "show_calibration_plot",
    "show_plots_unless_agg",
    "spearmanr_batched",
    "spearmanr_batched_dispatch",
    "specificity_npv_fpr_fnr",
    "spiegelhalter_z",
    "subset_accuracy",
    "top_k_accuracy",
    "trapezoid",
    "uniform",
    "wasserstein_1d",
    "weighted_kappa",
    "winkler_score",
]
