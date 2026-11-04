"""The calibration kernels return NaN, not a worst-case-looking number, for an empty slice."""
import numpy as np

from mlframe.metrics.calibration._calibration_metrics import (
    compute_brier_decomposition_debiased,
    compute_ece_and_brier_decomposition,
    compute_ece_brier_full_and_debiased,
    compute_ece_debiased,
)


def test_empty_input_is_nan_everywhere():
    """Empty bootstrap or window slices used to score ECE 1.0 and be averaged in as measurements."""
    y = np.array([], dtype=np.float64)
    p = np.array([], dtype=np.float64)
    assert np.isnan(compute_ece_debiased(y, p, 10))
    for result in (compute_ece_and_brier_decomposition(y, p, 10), compute_brier_decomposition_debiased(y, p, 10), compute_ece_brier_full_and_debiased(y, p, 10)):
        assert all(np.isnan(v) for v in result), result
