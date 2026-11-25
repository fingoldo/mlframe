"""``_x_plan`` hashes its matrix for a cache key, so it must accept a datetime64 matrix without raising."""

import numpy as np

from mlframe.feature_selection.filters._mah import _x_plan


def test_x_plan_accepts_a_datetime64_matrix_and_matches_its_int64_view():
    """A datetime64 matrix used to raise ``cannot include dtype 'M' in a buffer`` while building the key.

    Its bytes equal the int64 view's bytes, so the plan must match the int64 call exactly.
    """
    ints = np.arange(12, dtype=np.int64).reshape(3, 4)
    dates = ints.view("datetime64[ns]")

    assert _x_plan(dates, 2) == _x_plan(ints, 2)
