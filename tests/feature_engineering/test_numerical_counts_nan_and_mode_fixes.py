"""NaN is not a value in the count / mode / crossing statistics, and the numba mode is the mode of the whole array."""

from __future__ import annotations

import numpy as np

from mlframe.feature_engineering._numerical_counts import compute_nunique_mode_quantiles_numba, compute_nunique_modes_quantiles_numpy


def test_numba_mode_is_not_dropped_when_the_last_run_is_a_single_value():
    """[1, 1, 2]: 1 repeats, so it is the mode; the last run (the lone 2) must not erase it."""
    n_unique, mode, _q = compute_nunique_mode_quantiles_numba(np.array([1.0, 1.0, 2.0]))
    assert n_unique == 2 and mode == 1.0


def test_numba_mode_is_nan_when_nothing_repeats():
    """No repeated value: no mode."""
    _n, mode, _q = compute_nunique_mode_quantiles_numba(np.array([1.0, 2.0, 3.0]))
    assert np.isnan(mode)


def test_nan_is_not_counted_as_a_distinct_value_or_a_mode():
    """[nan, nan, nan, 1, 2]: two distinct finite values and no repeated one, instead of three values with NaN as the mode."""
    out = compute_nunique_modes_quantiles_numpy(np.array([np.nan, np.nan, np.nan, 1.0, 2.0]), q=[0.0, 0.5, 1.0])
    assert out[0] == 2
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_crossing_that_spans_a_nan_is_counted():
    """[-1, nan, 1] crosses the zero mark once; the NaN between the two is skipped, not a reset."""
    out = compute_nunique_modes_quantiles_numpy(np.array([-1.0, np.nan, 1.0]), q=[0.5])
    assert out[-1] == 1


def test_all_nan_input_is_undefined_not_a_crash():
    """Only NaN: zero distinct values and NaN modes."""
    out = compute_nunique_modes_quantiles_numpy(np.full(4, np.nan), q=[0.5])
    assert out[0] == 0 and np.isnan(out[1])
