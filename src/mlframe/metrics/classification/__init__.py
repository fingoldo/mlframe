"""Classification metrics: scalar metrics, the classification/calibration report block, and goodness-of-fit tests.

Submodules (internal):
    _classification_report      - format/fast classification report + PR-recall-F1 + per-class ICE kernels + calibration report.
    _classification_extras      - scalar classification metrics (KS, MCC, Cohen-kappa, RPS, lift@k, confusion/probability blocks, ...).
    _classification_calibration - Hosmer-Lemeshow goodness-of-fit + accuracy-ratio (re-exported through _classification_extras).

The public surface below mirrors exactly the names ``mlframe.metrics.core`` re-exports from these modules, so cross-package consumers can import from the public ``mlframe.metrics.classification`` path instead of reaching into the underscore-prefixed implementation modules.
"""

from __future__ import annotations

from ._classification_report import (
    CalibrationReport,
    format_classification_report,
    _compute_pr_recall_f1_metrics_seq,
    _compute_pr_recall_f1_metrics_par,
    compute_pr_recall_f1_metrics,
    fast_calibration_report,
    _batch_per_class_ice_kernel,
    fast_ice_only,
    predictions_time_instability,
)

from ._classification_extras import (
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
    _multiclass_confusion_kernel,
    hosmer_lemeshow_test,
    accuracy_ratio,
)

from ._threshold_optimization import (
    optimal_threshold,
    THRESHOLD_METRICS,
)

from ._gains_lift import (
    cumulative_gains_curve,
    lift_curve,
    gains_table,
    exploss,
)

from ._weighted_kappa import (
    quadratic_weighted_kappa,
    weighted_kappa,
    KAPPA_WEIGHTS,
)

from ._ordinal_cutpoints import (
    optimal_ordinal_cutpoints,
    apply_cutpoints,
    CUTPOINT_METRICS,
)
