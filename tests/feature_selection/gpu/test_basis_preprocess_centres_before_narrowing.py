"""The fused basis preprocess centres a large-offset column before narrowing it to the criterion dtype.

A column near 1e10 with unit spread lost its spread in float32 (the step there is about 1e3), so the standardised output collapsed toward zero
(audit PARG3-2, wave 3). The float32 default is kept; only the centring moved to float64.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover
    _HAS_DEVICE = False

pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")


def test_hermite_preprocess_keeps_the_spread_of_a_large_offset_column():
    """The standardised output of 1e10 + N(0,1) matches the float64 host standardisation, not a collapsed column."""
    from mlframe.feature_selection.filters import _gpu_resident_basis as gb

    rng = np.random.default_rng(0)
    M = 1e10 + rng.normal(size=(2000, 2))
    out = cp.asnumpy(gb._gpu_basis_preprocess_nonrobust_fused(cp, cp.asarray(M), "hermite"))
    expected = (M - M.mean(axis=0)) / M.std(axis=0)
    assert out.std(axis=0) == pytest.approx([1.0, 1.0], rel=1e-2)
    np.testing.assert_allclose(out, expected, atol=2e-2)


def test_robust_preprocess_is_shift_invariant_for_a_large_offset():
    """Shifting a column by 1e10 leaves the robust-preprocess output unchanged, as it does on the host."""
    from mlframe.feature_selection.filters import _gpu_resident_basis as gb

    rng = np.random.default_rng(1)
    base = rng.normal(size=(2000, 2))
    out_base = cp.asnumpy(gb._gpu_basis_preprocess_robust_fused(cp, cp.asarray(base), "hermite"))
    out_shift = cp.asnumpy(gb._gpu_basis_preprocess_robust_fused(cp, cp.asarray(base + 1e10), "hermite"))
    np.testing.assert_allclose(out_shift, out_base, atol=2e-2)
