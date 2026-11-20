"""Fourier/Pade fits ignore non-finite rows; the sigmoid slope is not skewed by an epsilon pad on a tiny-scale column."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import bases


def test_fourier_fit_params_are_finite_with_nan_rows():
    """A NaN row must not freeze NaN lo/span into the replay parameters."""
    _, params = bases._fourier_fit(np.array([1.0, 2.0, np.nan, 4.0]))
    assert params["lo"] == 1.0 and params["span"] == 3.0


def test_pade_fit_params_are_finite_with_nan_rows():
    """A NaN row must not freeze NaN mean/std into the replay parameters."""
    _, params = bases._pade_fit(np.array([1.0, 2.0, np.nan, 4.0, 3.0]))
    assert np.isfinite(params["mean"]) and np.isfinite(params["std"]) and params["std"] > 0


def test_sigmoid_slope_has_no_epsilon_pad_on_a_tiny_scale_column():
    """The slope is 4 / (iqr / 9) exactly when the interquantile range is far above the floor."""
    x = np.linspace(0.0, 1e-10, 1000)
    _, params = bases._sigmoid_fit(x)
    iqr = float(np.quantile(x, 0.9) - np.quantile(x, 0.1))
    assert abs(params["slope"] - 4.0 / (iqr / 9.0)) <= 1e-9 * params["slope"]
