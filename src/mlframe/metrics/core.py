"""Shared helpers for the metrics package: optional plotly import gate for interactive plot paths."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _require_plotly():
    """Return the plotly modules used by the interactive plotting paths, or raise an actionable error.

    plotly ships only in the ``viz`` extra, while ``mlframe.metrics.core`` is a public deep-import that must
    work on a bare ``pip install mlframe``. Plotly-backed plotting is therefore loaded lazily here so importing
    this module never pulls plotly, and a caller that actually requests a plotly chart gets a clear install hint.
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.io import write_image
    except ImportError as exc:
        raise ImportError("plotly is required for interactive plotting; install it with `pip install mlframe[viz]`.") from exc
    return px, go, write_image

# NUMBA_NJIT_PARAMS lives in ``._numba_params`` so split-out sibling modules
# (``_calibration_plot.py``, etc.) can import the same object without
# duplicating the dict. Re-exported here to preserve every historical
# ``from mlframe.metrics.core import NUMBA_NJIT_PARAMS`` import site.
from ._numba_params import (  # noqa: F401
    NUMBA_NJIT_PARAMS,
    _PARALLEL_REDUCTION_THRESHOLD,
    _PARALLEL_MULTILABEL_THRESHOLD,
)

# CB logits->probabilities (binary + multiclass) live in ``_core_cb_logits.py``;
# imported at module-top so subsequent in-module references resolve before the
# bottom-of-file re-export block fires.
from ._core_cb_logits import (  # noqa: F401
    _cb_logits_to_probs_binary_seq, _cb_logits_to_probs_binary_par,
    cb_logits_to_probs_binary,
    _cb_logits_to_probs_multiclass_seq, _cb_logits_to_probs_multiclass_par,
    cb_logits_to_probs_multiclass,
)

# AUC + Brier kernels and combined Brier+precision scorer live in
# ``_core_auc_brier.py``; imported here so historical names
# (``fast_roc_auc``, ``fast_numba_auc_nonw``, ``fast_aucs``, ``fast_numba_aucs``,
# ``fast_brier_score_loss``, ``brier_score_loss``, ``brier_and_precision_score``,
# ``make_brier_precision_scorer``) resolve at module scope.
from ._core_auc_brier import (  # noqa: F401
    fast_roc_auc, fast_roc_auc_unstable, fast_numba_auc_nonw, fast_numba_auc_weighted,
    fast_roc_curve, average_precision_score,
    make_bootstrap_auc_resampler,
    fast_aucs, fast_numba_aucs, fast_numba_aucs_with_ks,
    _fast_brier_score_loss_seq, _fast_brier_score_loss_par,
    fast_brier_score_loss, brier_score_loss,
    brier_and_precision_score, make_brier_precision_scorer,
)

# GPU dispatch + batch metrics live in ``_gpu_metrics.py``; re-exported here so
# historical ``from mlframe.metrics.core import compute_batch_aucs`` (and the other
# moved names) imports continue to resolve. See sibling for SSOT.
from ._gpu_metrics import (  # noqa: F401
    _GPU_BATCH_THRESHOLD_N, _GPU_BATCH_THRESHOLD_M,
    _GPU_AVAILABLE, _NUMBA_CUDA_AVAILABLE,
    _CUPY_SSE_PER_COL, _NUMBA_RMSE_KERNEL,
    set_gpu_thresholds, is_gpu_metrics_available,
    _is_numba_cuda_available, _require_cupy,
    _get_cupy_sse_kernel, _get_numba_rmse_kernel,
    gpu_multiple_rmse_scores, gpu_multiple_roc_auc_scores,
    gpu_multiple_pr_auc_scores,
    _normalize_scores_2d, compute_batch_rmse, compute_batch_aucs,
    _resolve_backend,
)

# Precision + classification report + MAPE kernels live in
# ``_core_precision_mape.py``; imported here so existing call sites resolve.
from ._core_precision_mape import (  # noqa: F401
    fast_precision, fast_classification_report,
    _max_abs_pct_error_kernel, _max_abs_pct_error_kernel_par,
    _MAPE_ZERO_WARN_SEEN, maximum_absolute_percentage_error,
)

# Multilabel metrics (hamming_loss, subset_accuracy, jaccard_score_multilabel +
# private numba kernels + bitmap-popcount fastpath) live in
# ``_multilabel_metrics.py``; re-exported below so historical
# ``from mlframe.metrics.core import hamming_loss`` imports keep resolving.
from ._multilabel_metrics import (  # noqa: F401
    _fast_hamming_loss_seq, _fast_hamming_loss_par,
    _fast_subset_accuracy_seq, _fast_subset_accuracy_par,
    _fast_jaccard_score_seq, _fast_jaccard_score_par,
    _popcount64, _fast_jaccard_bitmap_seq,
    _can_use_bitmap_jaccard, _pack_for_bitmap, _pack_for_bitmap_numpy,
    _pack_for_bitmap_kernel_seq, _pack_for_bitmap_kernel_par,
    _coerce_multilabel_array, _validate_multilabel_pair,
    hamming_loss, subset_accuracy, jaccard_score_multilabel,
)

# Closed set of title-metrics tokens recognised by render_title_metrics() and
# validated by ReportingConfig at construction time. Order in DEFAULT matches
# the historical title layout (ICE first, then BR with decomposition, ECE between
# BR and CMAEW per spec, then LL, ROC_AUC, PR_AUC). Adding a new token requires:
# 1) extending TITLE_METRIC_TOKENS, 2) adding a render_* branch in
# render_title_metric_token, 3) updating ReportingConfig validator allowed-set.
TITLE_METRIC_TOKENS: frozenset = frozenset({
    "ICE", "BR", "BR_DECOMP", "ECE", "CMAEW",
    "COV", "LL", "ROC_AUC", "PR_AUC", "DENS",
    # 2026-05-28 audit batch additions. Each has a render_* branch
    # in ``_calibration_plot.render_title_metric_token``. Adding new
    # tokens here REQUIRES adding to the renderer; ReportingConfig
    # validates against this set at config construction time.
    # GINI intentionally NOT a title token: =2*AUC-1, redundant with
    # ROC_AUC for a chart-title operator; still computed into metrics
    # dict (under "Gini") for downstream callers that report on it.
    "KS", "MCC", "BSS",
})


# calibration plot rendering (render_title_metric_token, fast_calibration_binning,
# _close_unless_interactive, show_calibration_plot, DEFAULT_TITLE_METRICS_TOKENS)
# moved to sibling _calibration_plot.py; re-exported below.
from .calibration import (  # noqa: F401
    DEFAULT_TITLE_METRICS_TOKENS,
    render_title_metric_token,
    fast_calibration_binning,
    _close_unless_interactive,
    show_calibration_plot,
)

# Calibration metric kernels (CMAEW, ECE, Murphy Brier-decomp,
# fast_calibration_metrics) live in ``_calibration_metrics.py``; re-exported
# below so historical
# ``from mlframe.metrics.core import calibration_metrics_from_freqs`` (and the
# other moved names) imports continue to resolve.
from .calibration import (  # noqa: F401
    calibration_metrics_from_freqs,
    compute_brier_decomposition_debiased,
    compute_ece_and_brier_decomposition,
    compute_ece_brier_full_and_debiased,
    compute_ece_debiased,
    fast_calibration_metrics,
)

# Per-group AUC helpers live in ``_auc_per_group.py``; re-exported below so
# historical ``from mlframe.metrics.core import fast_aucs_per_group`` (and the
# other moved names) imports continue to resolve. See sibling for SSOT.
from ._auc_per_group import (  # noqa: F401
    fast_aucs_per_group, fast_aucs_per_group_optimized,
    compute_grouped_group_aucs, fast_numba_aucs_simple,
    compute_mean_aucs_per_group,
)

# Classification + calibration report block moved to _classification_report.py.
from .classification._classification_report import (  # noqa: F401
    format_classification_report,
    _compute_pr_recall_f1_metrics_seq,
    _compute_pr_recall_f1_metrics_par,
    compute_pr_recall_f1_metrics,
    CalibrationReport,
    fast_calibration_report,
    _batch_per_class_ice_kernel,
    fast_ice_only,
    predictions_time_instability,
)

# ICE metric + ``compute_probabilistic_multiclass_error`` live in
# ``_ice_metric.py``; re-exported below so historical
# ``from mlframe.metrics.core import ICE`` / ``compute_probabilistic_multiclass_error``
# imports continue to resolve. See sibling for SSOT.
from ._ice_metric import (  # noqa: F401
    compute_probabilistic_multiclass_error,
    ICE,
    _install_catboost_sklearn_clone_patch,
)

# ICE-from-base-metrics aggregator lives in ``_calibration_metrics.py`` (it
# composes the calibration outputs into the single ICE score). Re-exported
# below so historical ``from mlframe.metrics.core import integral_calibration_error_from_metrics``
# imports keep resolving.
from .calibration import integral_calibration_error_from_metrics  # noqa: F401

# Regression metrics live in ``_regression_metrics.py``; re-exported below to keep
# ``from mlframe.metrics.core import fast_*`` imports stable. See sibling for SSOT.
from .regression import (  # noqa: F401
    _fast_mae_seq, _fast_mae_par, _fast_mse_seq, _fast_mse_par,
    _fast_max_error_seq, _fast_r2_score_seq, _fast_r2_score_par,
    _fast_r2_variance_seq,
    _fast_mae_weighted_seq, _fast_mae_weighted_par,
    _fast_mse_weighted_seq, _fast_mse_weighted_par,
    _fast_r2_score_weighted_seq, _fast_r2_score_weighted_par,
    _aggregate_multioutput, _to_2d,
    fast_mean_absolute_error, fast_mean_squared_error,
    fast_root_mean_squared_error, fast_max_error, fast_r2_score,
    _fused_regression_pass1_seq, _fused_regression_pass1_par,
    _fused_regression_pass2_seq, _fused_regression_pass2_par,
    fast_regression_metrics_block,
)

# Binary log-loss + probability-separation kernels live in
# ``_log_loss_and_separation.py``; re-exported below so historical
# ``from mlframe.metrics.core import fast_log_loss`` imports keep resolving.
from ._log_loss_and_separation import (  # noqa: F401
    _fast_log_loss_binary_seq, _fast_log_loss_binary_par,
    fast_log_loss_binary, fast_log_loss,
    _probability_separation_score_seq, _probability_separation_score_par,
    probability_separation_score,
)

# Fairness / robustness subgrouping + metrics live in ``_fairness_metrics.py``;
# re-exported below so historical
# ``from mlframe.metrics.core import create_fairness_subgroups`` (and the other
# moved names) imports continue to resolve. See sibling for SSOT.
from ._fairness_metrics import (  # noqa: F401
    create_fairness_subgroups, create_fairness_subgroups_indices,
    create_robustness_standard_bins, compute_fairness_metrics,
    create_robustness_subgroups, create_robustness_subgroups_indices,
    compute_robustness_metrics, robust_mlperf_metric,
)

# Numba warmup helpers (``numba_warmup``, ``_assert_numba_nogil_active``,
# ``prewarm_numba_cache``, ``_prewarm_numba_cache_body``) live in
# ``_core_numba_warmup.py``. Re-exported at bottom so the warmup module's
# function-body imports of ``mlframe.metrics.core`` resolve against the fully
# populated facade (avoids partial-module-import error on lazy callers).
from ._core_numba_warmup import (  # noqa: F401
    numba_warmup,
    _assert_numba_nogil_active,
    prewarm_numba_cache,
    _prewarm_numba_cache_body,
)

# Additional classification metrics (binary + multiclass) - 2026-05-28
# audit batch. Re-exported here so historical
# ``from mlframe.metrics.core import ks_statistic`` style imports resolve
# at module scope.
from .classification._classification_extras import (  # noqa: F401
    ks_statistic,
    matthews_corrcoef_binary,
    cohen_kappa_binary,
    balanced_accuracy_binary,
    g_mean_binary,
    brier_skill_score,
    gini_from_auc,
    specificity_npv_fpr_fnr,
    f_beta_score,
    spiegelhalter_z,
    lift_at_k,
    top_k_accuracy,
    matthews_corrcoef_multiclass,
    ranked_probability_score,
    fast_binary_confusion_metrics_block,
    fast_binary_probability_metrics_block,
    fast_multiclass_confusion_metrics_block,
    # Tier 2 additions:
    hosmer_lemeshow_test,
    accuracy_ratio,
)

# Additional regression metrics (RMSLE / MAPE-mean / SMAPE / ...) - 2026-05-28.
from .regression import (  # noqa: F401
    fast_rmsle,
    fast_mape_mean,
    fast_smape,
    fast_mdape,
    fast_wmape,
    fast_mase,
    fast_mean_bias_error,
    fast_cv_rmse,
    fast_nash_sutcliffe,
    fast_explained_variance,
    fast_adjusted_r2_score,
    fast_huber_loss,
    fast_pearson_corr,
    fast_spearman_corr,
    fast_kendall_tau,
    fast_concordance_index,
    fast_regression_metrics_block_extended,
    # Tier 2 additions:
    fast_poisson_deviance,
    fast_gamma_deviance,
    fast_tweedie_deviance,
)

# Additional multilabel metrics (LRAP / coverage / ranking-loss / one-error
# / macro+micro+weighted F1) - 2026-05-28.
from ._multilabel_extras import (  # noqa: F401
    label_ranking_average_precision,
    coverage_error,
    label_ranking_loss,
    one_error,
    multilabel_f1_macro,
    multilabel_f1_micro,
    multilabel_f1_weighted,
    fast_multilabel_classification_metrics_block,
    # 2026-05-28 follow-up: per-label AUC + macro/weighted
    multilabel_auc_per_label,
    multilabel_auc_macro,
    multilabel_auc_weighted,
)

# Additional ranking (LTR) metrics (DCG / ERR / Hit@k / Precision@k) - 2026-05-28.
from ._ranking_extras import (  # noqa: F401
    dcg_at_k,
    expected_reciprocal_rank,
    hit_at_k,
    precision_at_k,
)

# Distributional / drift metrics (PSI / KL / JS / Wasserstein / KS) - 2026-05-28.
from ._drift import (  # noqa: F401
    population_stability_index,
    kl_divergence,
    js_divergence,
    wasserstein_1d,
    ks_distribution_distance,
)
