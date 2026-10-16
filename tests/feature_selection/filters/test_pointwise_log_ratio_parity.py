"""The pointwise log-ratio helper is the term the I_ccs kernel computes inline; a Python re-sum with the helper matches the kernel."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._pid_decomposition import _i_ccs_redundancy_njit, _pointwise_log_ratio


def _i_ccs_via_helper(joint: np.ndarray) -> float:
    """Reference I_ccs built from the helper, term by term, with the kernel's sign and min-magnitude rule."""
    p = joint / joint.sum()
    p_x1, p_x2, p_y = p.sum(axis=(1, 2)), p.sum(axis=(0, 2)), p.sum(axis=(0, 1))
    p_x1y, p_x2y = p.sum(axis=1), p.sum(axis=0)
    total = 0.0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            for k in range(joint.shape[2]):
                if p_x1[i] <= 0 or p_x2[j] <= 0 or p_y[k] <= 0 or p_x1y[i, k] <= 0 or p_x2y[j, k] <= 0:
                    continue
                dh1 = _pointwise_log_ratio(p_x1y[i, k], p_x1[i], p_y[k])
                dh2 = _pointwise_log_ratio(p_x2y[j, k], p_x2[j], p_y[k])
                if dh1 * dh2 <= 0.0:
                    continue
                common = min(abs(dh1), abs(dh2))
                total += p[i, j, k] * (common if dh1 > 0.0 else -common)
    return max(0.0, total)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_helper_reproduces_the_kernel_redundancy(seed: int) -> None:
    """On random joint count tensors, the helper-built sum equals the njit kernel to float round-off."""
    rng = np.random.default_rng(seed)
    joint = rng.integers(1, 40, size=(3, 4, 2)).astype(np.float64)
    assert _i_ccs_via_helper(joint) == pytest.approx(_i_ccs_redundancy_njit(joint), abs=1e-12)


def test_helper_zero_outside_the_support() -> None:
    """A non-positive probability gives zero, the same guard the kernel applies."""
    assert _pointwise_log_ratio(0.0, 0.5, 0.5) == 0.0
    assert _pointwise_log_ratio(0.2, 0.0, 0.5) == 0.0
