"""Cluster 17 regression: out-of-[0,1] scores give NaN from the fixed-grid ECE/Brier kernels (audit 2026-11-20, finding 17-F10).

The kernels clamped an out-of-range score into an end bin but kept its raw value in the bin mean, so the result looked finite and was wrong, while
fast_brier_score_loss and the fused Brier block already return NaN for such input. A score a hair above 1.0 from rounding is still accepted.
"""

import numpy as np
import pytest

from mlframe.metrics.calibration._calibration_metrics import (
    compute_brier_decomposition_debiased,
    compute_ece_and_brier_decomposition,
    compute_ece_brier_full_and_debiased,
    compute_ece_debiased,
)

KERNELS = [
    compute_ece_and_brier_decomposition,
    compute_ece_debiased,
    compute_brier_decomposition_debiased,
    compute_ece_brier_full_and_debiased,
]


def _all_nan(result) -> bool:
    """True when every element of a scalar or tuple result is NaN."""
    return bool(np.all(np.isnan(np.atleast_1d(np.asarray(result, dtype=np.float64)))))


@pytest.mark.parametrize("kernel", KERNELS)
def test_out_of_range_score_gives_nan(kernel):
    """y_pred = [1.2, 0.1] is rejected with NaN instead of a finite value built on the raw 1.2 (17-F10)."""
    y = np.array([1.0, 0.0])
    assert _all_nan(kernel(y, np.array([1.2, 0.1]), 10))
    assert _all_nan(kernel(y, np.array([0.9, -0.2]), 10))


@pytest.mark.parametrize("kernel", KERNELS)
def test_in_range_and_rounding_edge_scores_stay_finite(kernel):
    """Scores inside [0, 1], and one a hair above 1.0 from rounding, still give finite results (17-F10)."""
    y = np.array([1.0, 0.0, 1.0, 0.0])
    assert not _all_nan(kernel(y, np.array([0.9, 0.1, 0.8, 0.2]), 5))
    assert not _all_nan(kernel(y, np.array([1.0 + 1e-12, 0.1, 0.8, 0.2]), 5))
