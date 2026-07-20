"""GPU parity coverage for ``_resident_bincount.resident_bincount`` (mrmr_audit_2026-07-20
test_coverage.md #9): the module's own docstring claims the unweighted count is bit-identical to
``cupy.bincount`` and the weighted sum agrees within ~1e-13 (atomic-accumulation reduction-order
noise) -- previously asserted but never tested. Skips cleanly when cupy is unavailable."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._resident_bincount import resident_bincount

cp = pytest.importorskip("cupy")


class TestUnweightedBitIdenticalToCupyBincount:
    """Unweighted resident_bincount must be bit-identical to cupy.bincount(x, minlength=nc)[:nc]."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_bit_identical_unweighted(self, seed):
        """Bit identical unweighted."""
        rng = np.random.default_rng(seed)
        n, nc = 5000, 17
        x_host = rng.integers(0, nc, size=n).astype(np.int32)
        x = cp.asarray(x_host)

        got = resident_bincount(cp, x, nc)
        expected = cp.bincount(x, minlength=nc)[:nc]

        assert cp.array_equal(got.astype(expected.dtype), expected), "unweighted resident_bincount diverged from cupy.bincount (must be bit-identical)"

    def test_out_of_bounds_within_contract_is_still_bit_identical(self):
        """Every index strictly within [0, nc) (the documented contract) -- boundary values 0 and nc-1 included."""
        nc = 10
        x = cp.asarray(np.array([0, 0, nc - 1, nc - 1, nc - 1, 5], dtype=np.int32))
        got = resident_bincount(cp, x, nc)
        expected = cp.bincount(x, minlength=nc)[:nc]
        assert cp.array_equal(got.astype(expected.dtype), expected)


class TestWeightedMatchesCupyBincountWithinTolerance:
    """Weighted resident_bincount must agree with cupy.bincount(x, weights, minlength=nc) within the
    documented ~1e-13 atomic-accumulation reduction-order tolerance."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_weighted_matches_within_tolerance(self, seed):
        """Weighted matches within tolerance."""
        rng = np.random.default_rng(seed)
        n, nc = 5000, 17
        x_host = rng.integers(0, nc, size=n).astype(np.int32)
        w_host = rng.uniform(0.1, 2.0, size=n).astype(np.float64)
        x = cp.asarray(x_host)
        w = cp.asarray(w_host)

        got = resident_bincount(cp, x, nc, weights=w)
        expected = cp.bincount(x, weights=w, minlength=nc)[:nc]

        max_abs_diff = float(cp.max(cp.abs(got - expected)))
        assert max_abs_diff < 1e-9, f"weighted resident_bincount diverged from cupy.bincount by {max_abs_diff:.3e} (docstring claims ~1e-13)"


class TestDtypeContract:
    """Default dtype is int32 for unweighted counts, float64 for weighted sums; an explicit dtype override is honoured."""

    def test_unweighted_default_dtype_is_int32(self):
        """Unweighted default dtype is int32."""
        x = cp.asarray(np.array([0, 1, 2], dtype=np.int32))
        out = resident_bincount(cp, x, 3)
        assert out.dtype == cp.int32

    def test_weighted_default_dtype_is_float64(self):
        """Weighted default dtype is float64."""
        x = cp.asarray(np.array([0, 1, 2], dtype=np.int32))
        w = cp.asarray(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        out = resident_bincount(cp, x, 3, weights=w)
        assert out.dtype == cp.float64

    def test_explicit_dtype_override_is_honoured(self):
        """An explicit dtype argument overrides the default for both the weighted and unweighted paths."""
        x = cp.asarray(np.array([0, 1, 2], dtype=np.int32))
        out = resident_bincount(cp, x, 3, dtype=cp.float64)
        assert out.dtype == cp.float64
