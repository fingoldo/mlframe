"""The PIT chi-square ignores NaN consistently: its statistic equals the one on the finite values alone (audit 2026-11-20, finding 22-F13)."""

import numpy as np

from mlframe.calibration.quality import chi_square_statistic


def test_nan_pit_values_do_not_change_the_statistic():
    """NaN was dropped from the histogram but counted in the expected total, inflating the statistic."""
    rng = np.random.default_rng(0)
    finite = rng.uniform(size=400)
    with_nan = np.concatenate([finite, np.full(50, np.nan)])
    assert np.isclose(chi_square_statistic(with_nan), chi_square_statistic(finite))
