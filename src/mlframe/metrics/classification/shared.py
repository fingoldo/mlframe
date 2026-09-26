"""Cross-package API of ``mlframe.metrics.classification``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.metrics.classification`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from . import (  # noqa: F401
    _multiclass_confusion_kernel as multiclass_confusion_kernel,
)
from ._classification_calibration import (  # noqa: F401
    accuracy_ratio,
    hosmer_lemeshow_test,
)
from ._classification_extras import (  # noqa: F401
    balanced_accuracy_binary,
    brier_skill_score,
    cohen_kappa_binary,
    f_beta_score,
    g_mean_binary,
    gini_from_auc,
    ks_statistic,
    lift_at_k,
    matthews_corrcoef_binary,
    matthews_corrcoef_multiclass,
    ranked_probability_score,
    specificity_npv_fpr_fnr,
    spiegelhalter_z,
    top_k_accuracy,
)
from ._classification_extras_blocks import (  # noqa: F401
    fast_binary_confusion_metrics_block,
    fast_binary_probability_metrics_block,
    fast_multiclass_confusion_metrics_block,
)
from ._classification_report import (  # noqa: F401
    CalibrationReport,
    _batch_per_class_ice_kernel as batch_per_class_ice_kernel,
    _batch_per_class_ice_kernel_serial as batch_per_class_ice_kernel_serial,
    _compute_pr_recall_f1_metrics_par as compute_pr_recall_f1_metrics_par,
    _compute_pr_recall_f1_metrics_seq as compute_pr_recall_f1_metrics_seq,
    compute_pr_recall_f1_metrics,
    fast_calibration_report,
    fast_ice_only,
    format_classification_report,
    predictions_time_instability,
)
from ._ice_kernel import (  # noqa: F401
    _ice_kernel_dispatch as ice_kernel_dispatch,
)
from ._ice_kernel_weighted import (  # noqa: F401
    batch_per_class_ice_weighted,
    bin_index,
)

__all__ = [
    "CalibrationReport",
    "accuracy_ratio",
    "balanced_accuracy_binary",
    "batch_per_class_ice_kernel",
    "batch_per_class_ice_kernel_serial",
    "batch_per_class_ice_weighted",
    "bin_index",
    "brier_skill_score",
    "cohen_kappa_binary",
    "compute_pr_recall_f1_metrics",
    "compute_pr_recall_f1_metrics_par",
    "compute_pr_recall_f1_metrics_seq",
    "f_beta_score",
    "fast_binary_confusion_metrics_block",
    "fast_binary_probability_metrics_block",
    "fast_calibration_report",
    "fast_ice_only",
    "fast_multiclass_confusion_metrics_block",
    "format_classification_report",
    "g_mean_binary",
    "gini_from_auc",
    "hosmer_lemeshow_test",
    "ice_kernel_dispatch",
    "ks_statistic",
    "lift_at_k",
    "matthews_corrcoef_binary",
    "matthews_corrcoef_multiclass",
    "multiclass_confusion_kernel",
    "predictions_time_instability",
    "ranked_probability_score",
    "specificity_npv_fpr_fnr",
    "spiegelhalter_z",
    "top_k_accuracy",
]
