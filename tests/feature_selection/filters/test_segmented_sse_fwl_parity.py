"""_segmented_sse (full lstsq per cut) must equal the Frisch-Waugh rank-1 SSE the hinge precheck and scan use."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._hinge_basis_fe import _segmented_sse


def _fwl_sse(x: np.ndarray, y: np.ndarray, tau: float) -> float:
    """SSE of y ~ [1, x, relu(x - tau)] via the rank-1 update on the QR of B=[1, x]."""
    B = np.column_stack([np.ones_like(x), x])
    Q, _ = np.linalg.qr(B)
    r_y = y - Q @ (Q.T @ y)
    relu = np.maximum(x - tau, 0.0)
    r_relu = relu - Q @ (Q.T @ relu)
    return float(r_y @ r_y - (r_relu @ r_y) ** 2 / (r_relu @ r_relu))


@pytest.mark.parametrize("tau", [-0.8, 0.0, 0.5, 1.2])
def test_full_lstsq_sse_equals_fwl_update(tau):
    """Same SSE up to float reduction order on a kinked signal."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=600)
    y = 1.0 + 0.5 * x + 2.0 * np.maximum(x - 0.4, 0.0) + rng.normal(scale=0.3, size=600)
    assert _segmented_sse(x, y, tau) == pytest.approx(_fwl_sse(x, y, tau), rel=1e-9)


def test_sse_separates_the_true_knot_from_a_wrong_one():
    """Negative control: the parity is not vacuous, the true knot has a much smaller SSE than a distant one."""
    rng = np.random.default_rng(6)
    x = rng.normal(size=800)
    y = 3.0 * np.maximum(x - 0.5, 0.0) + rng.normal(scale=0.1, size=800)
    assert _segmented_sse(x, y, 0.5) < 0.2 * _segmented_sse(x, y, -1.5)
