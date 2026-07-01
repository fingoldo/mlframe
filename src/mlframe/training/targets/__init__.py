"""Target construction, selection, transform, and audit subsystem.

Groups the target-side helpers used by ``train_mlframe_models_suite``:

- ``target_temporal_audit`` -- audit a target's distribution drift over
  time (change-point detection + per-bin summaries + plotting).
- ``regression_residual_audit`` -- post-fit residual diagnostics
  (heteroskedasticity / normality / autocorrelation) + plotting.
- ``_target_distribution_analyzer`` -- pre-fit target / feature
  distribution analysis (modality, tail, transform recommendation).
- ``_train_eval_select_target`` -- ``select_target`` (resolves the
  per-target model spec, including composite ``MTRESID=`` stamping).
- ``_ttr_eval_set_scaling`` -- the eval-set-aware
  ``TransformedTargetRegressor`` wrapper.

The public surface is re-exported here so existing
``from mlframe.training.targets import X`` import sites resolve from the
documented package path.
"""
from __future__ import annotations

from .target_temporal_audit import (  # noqa: F401
    audit_targets_over_time,
    audit_target_over_time,
    format_temporal_audit_report,
    coerce_timestamps_for_audit,
    plot_target_over_time,
    TemporalAuditResult,
    TimeBin,
)
from .regression_residual_audit import (  # noqa: F401
    audit_residuals,
    format_residual_audit_report,
    plot_residual_diagnostics,
    ResidualAudit,
)
from ._target_distribution_analyzer import (  # noqa: F401
    analyze_target_distribution,
    analyze_feature_distribution,
)
from ._train_eval_select_target import select_target  # noqa: F401
from ._ttr_eval_set_scaling import _TTRWithEvalSetScaling  # noqa: F401
from ._target_distribution_analyzer_stats import _lag1_autocorr_grouped  # noqa: F401

__all__ = [
    "audit_targets_over_time",
    "audit_target_over_time",
    "format_temporal_audit_report",
    "coerce_timestamps_for_audit",
    "plot_target_over_time",
    "TemporalAuditResult",
    "TimeBin",
    "audit_residuals",
    "format_residual_audit_report",
    "plot_residual_diagnostics",
    "ResidualAudit",
    "analyze_target_distribution",
    "analyze_feature_distribution",
    "select_target",
]
