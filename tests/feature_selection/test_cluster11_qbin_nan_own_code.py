"""Cluster 11 regression: qbin does not put NaN rows into a real bin (audit 2026-11-20, finding 11-F4).

NaN used to be ranked with the finite values and so took the top bins. NaN rows now get the dedicated code k, and finite rows are binned on their own
ranks. Without NaN the codes are the same equal-frequency bins as before.
"""

import numpy as np

from mlframe.feature_selection._benchmarks.fe_operator_factory.common.binning import qbin


def test_nan_rows_get_their_own_code():
    """NaN rows carry the code k, outside every real bin (11-F4)."""
    x = np.array([1.0, np.nan, 2.0, 3.0, np.nan, 4.0])
    codes = qbin(x, k=4)
    assert codes[1] == 4 and codes[4] == 4
    assert set(codes[[0, 2, 3, 5]]).issubset(set(range(4)))


def test_finite_only_input_is_unchanged():
    """Control: with no NaN the equal-frequency codes are the same as the finite ranks (11-F4)."""
    x = np.array([5.0, 1.0, 3.0, 2.0, 4.0, 6.0, 0.0, 7.0])
    codes = qbin(x, k=4)
    assert codes.tolist() == [2, 0, 1, 1, 2, 3, 0, 3]
