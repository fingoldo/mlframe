"""The fused float32 round-up of the edges equals the cast / compare / nextafter / where form on ordinary, boundary, subnormal, NaN and infinite edges."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._fe_batched_mi import _edges_rounded_up_f32


def _reference(edges):
    """The unfused form (what the function was before it became one kernel)."""
    e32 = edges.astype(cp.float32)
    low = e32.astype(cp.float64) < edges
    return cp.ascontiguousarray(cp.where(low, cp.nextafter(e32, cp.float32(cp.inf)), e32))


def test_fused_equals_unfused():
    """Bit-for-bit equal output (compared as uint32) on a mix of edge kinds, 2-D layout kept."""
    r = np.random.default_rng(0)
    base = r.standard_normal((9, 40)) * 10.0 ** r.integers(-6, 6, (9, 40))
    f32_exact = r.standard_normal((9, 40)).astype(np.float32).astype(np.float64)  # already representable: stays
    just_above = np.nextafter(f32_exact, np.inf)  # a double a hair above a float32 value: rounds up to the next float32
    special = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 1e-45, 3.4e38, 1e-300, 1e300]).reshape(9, 1) * np.ones((1, 40))
    for arr in (base, f32_exact, just_above, special):
        e = cp.asarray(np.ascontiguousarray(arr))
        got, want = _edges_rounded_up_f32(cp, e), _reference(e)
        assert got.shape == want.shape and got.dtype == want.dtype == cp.float32
        np.testing.assert_array_equal(cp.asnumpy(got).view(np.uint32), cp.asnumpy(want).view(np.uint32))
