"""Model-performance reporting subsystem.

Groups the report builders used by ``evaluation.report_model_perf``:

- ``_reporting`` -- the top-level ``report_model_perf`` dispatch (regression
  vs classification) plus shared constants / styling / multilabel helpers.
- ``_reporting_regression`` -- ``report_regression_model_perf`` (MAE / RMSE /
  R2 / scatter + residual-audit plotting).
- ``_reporting_probabilistic`` -- ``report_probabilistic_model_perf``
  (classification probability metrics, calibration, multilabel/multiclass).

The public surface is re-exported here so existing
``from mlframe.training.reporting import X`` import sites resolve from the
documented package path; the reporting config classes stay in the flat
``_reporting_configs`` module (owned by the ``configs`` subsystem).
"""
from __future__ import annotations

from ._reporting import (  # noqa: F401
    report_model_perf,
    report_regression_model_perf,
    report_probabilistic_model_perf,
    _canonical_multilabel_y,
    _style_with_caption,
    _maybe_display,
    DEFAULT_PLOT_SAMPLE_SIZE,
    DEFAULT_REPORT_NDIGITS,
    DEFAULT_CALIB_REPORT_NDIGITS,
    DEFAULT_NBINS,
    DEFAULT_FIGSIZE,
    DEFAULT_RANDOM_SEED,
)

__all__ = [
    "report_model_perf",
    "report_regression_model_perf",
    "report_probabilistic_model_perf",
    "_canonical_multilabel_y",
    "_style_with_caption",
]
