"""Parity of the superseded ``_segmented_sse`` reference with the FWL rank-1 scan that replaced it in the hinge detector.

``_detect_hinge_breakpoints`` scores each candidate knot through ``_hinge_cut_scan_njit``, which is the FWL update on a
fixed linear QR. ``_segmented_sse`` refits the same two-segment design by lstsq per knot and is kept only as the
reference, so for each knot the scan's SSE must equal the lstsq SSE.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._hinge_basis_fe import _hinge_cut_scan_njit, _linear_qr_fit, _segmented_sse


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_segmented_sse_matches_fwl_scan_per_knot(seed):
    """Each knot's SSE from the FWL scan equals the lstsq reference within FP-reorder tolerance."""
    rng = np.random.default_rng(seed)
    n = 800
    x = rng.uniform(-2.0, 2.0, n)
    y = np.where(x > 0.5, 2.0 * (x - 0.5), 0.0) + 0.3 * x + rng.normal(0.0, 0.2, n)
    Q, r_y, sse_B = _linear_qr_fit(x, y)
    for tau in (-1.0, -0.2, 0.5, 1.3):
        ref = _segmented_sse(x, y, tau)
        _tau, got = _hinge_cut_scan_njit(Q, r_y, sse_B, x, np.array([tau]), 2, np.empty(0, dtype=np.float64))
        assert np.isclose(got, ref, rtol=1e-9, atol=1e-9), (tau, got, ref)
