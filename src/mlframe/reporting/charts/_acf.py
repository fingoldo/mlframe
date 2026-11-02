"""FFT-based autocorrelation / partial-autocorrelation kernels for diagnostic panels.

The implementation lives in :mod:`mlframe.utils.acf_kernels` (pure numpy) so that feature engineering can use it without importing the plotting stack
through ``mlframe.reporting``; this module keeps the old import path for the chart code.
"""

from __future__ import annotations

from mlframe.utils.acf_kernels import (
    MAX_ACF_LAGS,
    MAX_ACF_SERIES,
    acf_fft,
    gap_fraction,
    lag_tick_labels,
    pacf_levinson,
    significance_band,
)

__all__ = [
    "gap_fraction",
    "MAX_ACF_SERIES",
    "MAX_ACF_LAGS",
    "acf_fft",
    "pacf_levinson",
    "significance_band",
    "lag_tick_labels",
]
