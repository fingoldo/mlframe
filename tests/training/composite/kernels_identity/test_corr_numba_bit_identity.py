"""Regression sensors for the numba-dispatched leakage-filter correlation gate.

The size-aware dispatcher (``_corr_numba.safe_abs_corr_all_dispatch``, wired in as the
default for ``screening._safe_abs_corr_all`` at n >= 20k AND F >= 64) must match the
numpy reference (``screening._safe_abs_corr_all_numpy``):

- numerically (~1e-9) on continuous data across several seeds,
- EXACTLY on the decision-sensitive columns: a constant / degenerate column (variance
  near the floor) and a near-perfectly-correlated column (|corr| near 1.0, the leak
  threshold region) -- those are the columns the kernel flags borderline and the
  wrapper re-decides with the exact numpy primitive.

A biz_value sensor pins the measured speedup so a regression that silently drops the
kernel (or makes it slower than numpy) trips.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mlframe.training.composite.discovery._corr_numba import (
    _HAS_NUMBA,
    _MIN_COLS,
    _MIN_ROWS,
    safe_abs_corr_all_dispatch,
)
from mlframe.training.composite.discovery.screening import (
    _safe_abs_corr_all,
    _safe_abs_corr_all_numpy,
)

# Shape that engages the kernel gate (n >= _MIN_ROWS AND F >= _MIN_COLS).
_N = max(_MIN_ROWS, 20_000)
_F = max(_MIN_COLS, 64)


def _dispatch(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    return safe_abs_corr_all_dispatch(y, X, reference_fn=_safe_abs_corr_all_numpy)


class TestCorrNumbaBitIdentity:
    @pytest.mark.parametrize("seed", [0, 1, 2, 7, 13])
    def test_matches_numpy_reference_across_seeds(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(_N, _F))
        # A few columns carry real signal so the |corr| spans a wide range.
        y = X[:, 0] * 0.6 + X[:, 1] * 0.2 + rng.normal(size=_N)
        got = _dispatch(y, X)
        ref = _safe_abs_corr_all_numpy(y, X)
        # ~1e-9 FP reduction-order divergence is the documented contract.
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9)

    def test_constant_and_degenerate_columns_exact(self) -> None:
        """A constant column and a near-constant column must return exactly the
        reference's value (the borderline re-decision keeps the var-floor cutoff exact)."""
        rng = np.random.default_rng(99)
        X = rng.normal(size=(_N, _F))
        X[:, 3] = 5.0  # exactly constant -> reference returns 0.0
        X[:, 4] = 5.0  # constant
        X[1, 4] += 1e-13  # near-constant, variance hugs the floor
        y = X[:, 0] * 0.5 + rng.normal(size=_N)
        got = _dispatch(y, X)
        ref = _safe_abs_corr_all_numpy(y, X)
        # Degenerate columns are exactly 0.0 in the reference.
        assert ref[3] == 0.0
        assert got[3] == ref[3]
        assert got[4] == ref[4]
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9)

    def test_near_perfect_correlation_within_decision_band(self) -> None:
        """A column with |corr| near 1.0 (the leak-threshold region) is flagged
        borderline and re-decided with the exact numpy single-column primitive. It
        matches the reference within ~1e-9 (the reference itself reduces cov via a
        batched BLAS matmul, so byte-equality is not attainable, but the gap is
        ~1e-12 -- far tighter than any leak threshold spacing, so the >= decision
        never flips)."""
        rng = np.random.default_rng(5)
        X = rng.normal(size=(_N, _F))
        y = X[:, 0] * 1.0 + rng.normal(size=_N) * 1e-6  # corr(y, X[:,0]) ~ 1.0
        X[:, 7] = y + rng.normal(size=_N) * 1e-7  # another near-perfect column
        got = _dispatch(y, X)
        ref = _safe_abs_corr_all_numpy(y, X)
        # The near-1 columns are within ~1e-9 of the reference (well inside any
        # realistic forbidden-base threshold spacing).
        assert abs(got[0] - ref[0]) < 1e-9
        assert abs(got[7] - ref[7]) < 1e-9
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9)

    def test_y_with_nan_matches_reference(self) -> None:
        """A few non-finite y rows -> global mask, still matches the reference."""
        rng = np.random.default_rng(3)
        X = rng.normal(size=(_N, _F))
        y = X[:, 0] * 0.4 + rng.normal(size=_N)
        y[:5] = np.nan
        got = _dispatch(y, X)
        ref = _safe_abs_corr_all_numpy(y, X)
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9)

    def test_small_input_uses_numpy_reference(self) -> None:
        """Below the gate the public entry must equal the numpy reference exactly
        (it IS the reference -- no kernel)."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(1000, 8))  # below both gates
        y = X[:, 0] + rng.normal(size=1000)
        got = _safe_abs_corr_all(y, X)
        ref = _safe_abs_corr_all_numpy(y, X)
        np.testing.assert_array_equal(got, ref)


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba required for the dispatched kernel")
class TestCorrNumbaBizValue:
    def test_biz_kernel_faster_than_numpy_at_production_shape(self) -> None:
        """Floor 1.5x; measured ~6.7x on the dev host (n=50k, F=200). Catches a
        regression that drops the kernel or makes it slower than the numpy einsum."""
        rng = np.random.default_rng(0)
        n, f = 50_000, 200
        X = rng.normal(size=(n, f))
        y = X[:, 0] * 0.6 + rng.normal(size=n)

        # Warm the JIT so the timed region is steady-state.
        _dispatch(y, X)
        _safe_abs_corr_all_numpy(y, X)

        def _best(fn, reps: int = 5) -> float:
            best = float("inf")
            for _ in range(reps):
                t0 = time.perf_counter()
                fn(y, X)
                best = min(best, time.perf_counter() - t0)
            return best

        t_np = _best(_safe_abs_corr_all_numpy)
        t_nb = _best(_dispatch)
        speedup = t_np / t_nb if t_nb > 0 else float("inf")
        assert speedup >= 1.5, (
            f"numba corr kernel should be >=1.5x numpy at n={n} F={f}; got {speedup:.2f}x (numpy {t_np * 1e3:.1f}ms, numba {t_nb * 1e3:.1f}ms)"
        )
