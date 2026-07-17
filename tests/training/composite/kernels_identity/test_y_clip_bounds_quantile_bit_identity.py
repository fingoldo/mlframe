"""Bit-identity + perf-sentinel for the single-np.quantile fast path in _y_train_clip_bounds.

The helper now uses one np.quantile(y, (0.001, 0.999)) (single sort, ~2x) instead of two separate calls.
np.quantile with a vector of probs returns the SAME values as separate scalar calls -> bit-identical clip bounds.
"""

import time

import numpy as np
import pytest

from mlframe.training.composite.estimator import _y_train_clip_bounds


def _reference_bounds(finite):
    """Pre-optimization two-call recompute (reference for bit-identity)."""
    q_low = float(np.quantile(finite, 0.001))
    q_high = float(np.quantile(finite, 0.999))
    span = q_high - q_low
    if span <= 0:
        med = float(np.median(finite))
        return med - 0.1 * abs(med) - 1e-6, med + 0.1 * abs(med) + 1e-6
    # mirror module fracs via the helper's own output on the non-degenerate branch:
    return q_low, q_high  # only used to pin the quantiles themselves


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("n", [1000, 50_000])
def test_quantiles_bit_identical_to_two_calls(seed, n):
    """Quantiles bit identical to two calls."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n).astype(np.float64)
    q_low_ref = float(np.quantile(y, 0.001))
    q_high_ref = float(np.quantile(y, 0.999))
    q_low_new, q_high_new = (float(v) for v in np.quantile(y, (0.001, 0.999)))
    assert q_low_new == q_low_ref
    assert q_high_new == q_high_ref


def test_clip_bounds_stable_on_fixture():
    """Clip bounds stable on fixture."""
    rng = np.random.default_rng(42)
    y = rng.standard_normal(20_000).astype(np.float64)
    lo, hi = _y_train_clip_bounds(y)
    # Pin: derived from the single-call quantiles + the module's extension fracs.
    q_low = float(np.quantile(y, 0.001))
    q_high = float(np.quantile(y, 0.999))
    span = q_high - q_low
    assert lo == pytest.approx(q_low - 0.9 * span, abs=0.0)
    assert hi == pytest.approx(q_high + 9.0 * span, abs=0.0)
    assert lo < hi


def test_perf_sentinel_one_call_not_slower():
    """Perf sentinel one call not slower."""
    rng = np.random.default_rng(3)
    y = rng.standard_normal(100_000).astype(np.float64)

    def two():
        """Two."""
        return float(np.quantile(y, 0.001)), float(np.quantile(y, 0.999))

    def one():
        """One."""
        return tuple(float(v) for v in np.quantile(y, (0.001, 0.999)))

    one()
    two()
    iters = 200
    t0 = time.perf_counter()
    for _ in range(iters):
        two()
    t_two = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(iters):
        one()
    t_one = time.perf_counter() - t0
    # Single-sort path must not be slower than the two-sort path (expect ~2x faster).
    assert t_one <= t_two * 1.10, f"one-call {t_one:.4f}s should be <= two-call {t_two:.4f}s"
