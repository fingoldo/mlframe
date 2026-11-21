"""Cluster 04 regressions: scipy shape parameters precede loc and scale in the SAPP draws, and non-numeric values are counted when coerced (audit 2026-11-20, findings 04-F23, 04-F25)."""

import pandas as pd

from mlframe.data.datasets.frames import to_numpy
from mlframe.data.datasets._sapp import get_sapp_dataset


def test_sapp_draws_use_shape_parameters_before_loc_and_scale():
    """A gamma with shape 2 must have mean ~ shape*scale + loc; swapped arguments gave a different law (04-F25)."""
    df, _target = get_sapp_dataset(loc=1.0, scale=2.0, distr_name="gamma", distr_params=(2.0,), N=20_000, random_state=0)
    expected_mean = 1.0 + 2.0 * 2.0
    assert abs(float(df["X1"].mean()) - expected_mean) < 0.2


def test_non_numeric_values_are_counted_in_the_notes():
    """A string that cannot be read as a number used to become NaN with no trace (04-F23)."""
    frame = pd.DataFrame({"a": [1.0, 2.0, "oops"]})
    _matrix, _names, notes = to_numpy(frame)
    assert any("non-numeric" in n for n in notes)
