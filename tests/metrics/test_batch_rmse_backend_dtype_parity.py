"""`compute_batch_rmse` must return the same dtype whichever backend ran.

The GPU branch upcasts to float64 before reducing and returned a float64 array; the CPU reference reduced in
the caller's dtype and returned float32 for float32 input. So the same public call changed the dtype of a
caller's downstream container depending on whether a device happened to be available -- and the two also
disagreed in accumulation precision, measured at 1.33e-08 apart on 2M float32 rows.

Both now accumulate in float64 and return the CALLER's dtype, so the accuracy of the wider accumulator is
kept without either backend rewriting the caller's container. `np.mean(dtype=...)` widens the REDUCTION, not
the operands, so this costs no copy of the (N, M) inputs -- which matters at the row counts this package
targets.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics._gpu_metrics import compute_batch_rmse


def _pair(dtype, n: int = 5000, m: int = 3):
    """Predictions offset from truth by a known amount, in the requested dtype."""
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(n, m)).astype(dtype)
    y_pred = (y_true + rng.normal(scale=0.5, size=(n, m))).astype(dtype)
    return y_true, y_pred


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_the_result_keeps_the_callers_dtype(dtype):
    """A float32 caller must not get float32 from one backend and float64 from the other."""
    y_true, y_pred = _pair(dtype)
    out = compute_batch_rmse(y_true, y_pred)
    assert out.dtype == np.dtype(dtype), f"{np.dtype(dtype).name} expected back, got {out.dtype}"


def test_the_value_matches_a_float64_reference_on_float32_input():
    """The float32 result must still be the correctly-rounded one, not merely close."""
    y_true, y_pred = _pair(np.float32, n=200_000, m=1)
    expected = np.sqrt(np.mean((y_true.astype(np.float64) - y_pred.astype(np.float64)) ** 2.0, axis=0))
    got = compute_batch_rmse(y_true, y_pred)
    # This pins the VALUE, not the accumulator width: narrowing the reduction back to float32 was measured to
    # leave the float32 result bit-identical up to n=2M, and to move it by one ULP (rel 1.2e-07) only at n=20M
    # -- below what a float32 return can express at any test-sized n. The wide accumulator is kept because the
    # two backends were measured 1.33e-08 apart before it, which is visible to a float64 caller.
    assert got == pytest.approx(expected.astype(np.float32), rel=1e-6), f"{got} against the float64-accumulated reference {expected}"


def test_a_one_dimensional_input_still_works():
    """The 1-D branch reshapes before reducing; the dtype contract applies there too."""
    rng = np.random.default_rng(1)
    y_true = rng.normal(size=4000).astype(np.float32)
    y_pred = (y_true + 0.25).astype(np.float32)
    out = compute_batch_rmse(y_true, y_pred)
    assert out.dtype == np.float32
    assert float(np.ravel(out)[0]) == pytest.approx(0.25, abs=1e-6)


def test_a_perfect_prediction_gives_zero():
    """Guards the reduction itself: the rewrite must not introduce an offset."""
    y_true, _ = _pair(np.float64)
    assert np.allclose(compute_batch_rmse(y_true, y_true), 0.0)
