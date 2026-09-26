"""Cross-package API of ``mlframe.metrics.calibration``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.metrics.calibration`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from ._calibration_metrics import (
    calibration_metrics_from_freqs,
    compute_brier_decomposition_debiased,
    compute_ece_and_brier_decomposition,
    compute_ece_brier_full_and_debiased,
    compute_ece_debiased,
    integral_calibration_error_from_metrics,
)
from ._calibration_plot import (
    DEFAULT_TITLE_METRICS_TOKENS,
    _close_unless_interactive as close_unless_interactive,
    _fast_calibration_binning_prange as fast_calibration_binning_prange,
    _show_plots_unless_agg as show_plots_unless_agg,
    calibration_binning,
    render_title_metric_token,
    show_calibration_plot,
)

__all__ = [
    "DEFAULT_TITLE_METRICS_TOKENS",
    "calibration_binning",
    "calibration_metrics_from_freqs",
    "close_unless_interactive",
    "compute_brier_decomposition_debiased",
    "compute_ece_and_brier_decomposition",
    "compute_ece_brier_full_and_debiased",
    "compute_ece_debiased",
    "fast_calibration_binning_prange",
    "integral_calibration_error_from_metrics",
    "render_title_metric_token",
    "show_calibration_plot",
    "show_plots_unless_agg",
]
