"""The Pade z-score must be the same array at fit time and at replay time."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import bases
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_pade_apply_reproduces_the_fit_axis_exactly(dtype: type) -> None:
    """Fit and replay share the criterion dtype, so replay reproduces the fit axis exactly."""
    x = np.random.default_rng(0).normal(3.0, 2.0, 1000).astype(dtype)
    z_fit, params = bases._pade_fit(x)
    z_replay = bases._pade_apply(x, params)
    assert z_fit.dtype == z_replay.dtype == _crit_np_dtype()
    np.testing.assert_array_equal(z_fit, z_replay)


def test_pade_degenerate_column_maps_to_zero_in_float64() -> None:
    """A constant column is the all-zero axis on both sides, in the criterion dtype."""
    x = np.full(10, 4.0)
    z_fit, params = bases._pade_fit(x)
    assert params["degenerate"] is True
    assert z_fit.dtype == bases._pade_apply(x, params).dtype == _crit_np_dtype()
