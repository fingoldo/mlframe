"""Calibration metrics and calibration-plot rendering.

Submodules (internal):
    _calibration_plot    - calibration binning + reliability-plot rendering + title-token helpers.
    _calibration_metrics - CMAEW / ECE / Murphy Brier-decomposition kernels + the ICE-from-metrics aggregator.

The public surface below mirrors exactly the names ``mlframe.metrics.core`` (and the package ``__init__``) re-export from these modules, so cross-package consumers can import from the public ``mlframe.metrics.calibration`` path instead of reaching into the underscore-prefixed implementation modules.
"""

from __future__ import annotations

from ._calibration_plot import (  # noqa: F401
    DEFAULT_TITLE_METRICS_TOKENS,
    render_title_metric_token,
    fast_calibration_binning,
    _close_unless_interactive,
    show_calibration_plot,
    _show_plots_unless_agg,
)

from ._calibration_metrics import (  # noqa: F401
    calibration_metrics_from_freqs,
    compute_ece_and_brier_decomposition,
    fast_calibration_metrics,
    integral_calibration_error_from_metrics,
)
