"""Bit-identity + unit tests for the batched per-fold linear_residual OLS solver.

``_linear_residual_fit_batched`` solves K independent single-base OLS systems in
one vectorised pass (closed-form normal equations); it MUST be bit-identical to
applying the scalar reference ``_linear_residual_fit_closed`` per fold -- INCLUDING
the degenerate zero-variance fold, the ``n<2`` fold, and the empty fold.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.transforms.linear import (
    _linear_residual_fit_batched,
    _linear_residual_fit_closed,
)


def _make_segments(seed: int = 0):
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for s in (5, 20, 200, 2000, 20000):
        x = rng.normal(size=s)
        y = 1.3 * x - 0.4 + rng.normal(size=s) * 0.3
        xs.append(x)
        ys.append(y)
    # Degenerate folds the scalar guard must handle identically.
    xs.append(np.full(50, 3.0))           # zero-variance base -> (0, mean(y)).
    ys.append(rng.normal(size=50))
    xs.append(np.array([7.0]))            # n<2 -> (0, mean(y)).
    ys.append(np.array([2.5]))
    return xs, ys


def test_batched_bit_identical_to_sequential_closed_form():
    xs, ys = _make_segments(seed=1)
    alphas, betas = _linear_residual_fit_batched(xs, ys)
    assert alphas.shape == (len(xs),)
    assert betas.shape == (len(xs),)
    for k in range(len(xs)):
        a_seq, b_seq = _linear_residual_fit_closed(xs[k], ys[k])
        # Exact equality (same float64 reductions, same arithmetic).
        assert a_seq == alphas[k], f"fold {k}: alpha {a_seq} != {alphas[k]}"
        assert b_seq == betas[k], f"fold {k}: beta {b_seq} != {betas[k]}"


def test_batched_zero_variance_fold_exact():
    """A constant-base fold returns (0, mean(y)) -- bit-identical, no NaN/inf."""
    rng = np.random.default_rng(3)
    x_const = np.full(64, 5.0)
    y = rng.normal(size=64)
    a_seq, b_seq = _linear_residual_fit_closed(x_const, y)
    assert a_seq == 0.0
    alphas, betas = _linear_residual_fit_batched([x_const], [y])
    assert alphas[0] == 0.0 == a_seq
    assert betas[0] == b_seq == float(np.mean(y))
    assert np.isfinite(betas[0])


def test_batched_empty_and_singleton_folds():
    xs = [np.array([]), np.array([4.0]), np.array([1.0, 2.0, 3.0])]
    ys = [np.array([]), np.array([9.0]), np.array([2.0, 4.0, 6.0])]
    alphas, betas = _linear_residual_fit_batched(xs, ys)
    assert alphas[0] == 0.0 and betas[0] == 0.0          # empty.
    assert alphas[1] == 0.0 and betas[1] == 9.0          # n<2.
    for k in range(3):
        a_seq, b_seq = _linear_residual_fit_closed(xs[k], ys[k])
        assert a_seq == alphas[k] and b_seq == betas[k]


def test_batched_empty_input_returns_empty():
    alphas, betas = _linear_residual_fit_batched([], [])
    assert alphas.shape == (0,) and betas.shape == (0,)


def test_batched_length_mismatch_raises():
    with pytest.raises(ValueError, match="x-segments"):
        _linear_residual_fit_batched([np.zeros(3)], [np.zeros(3), np.zeros(3)])


def test_closed_form_recovers_known_line():
    """Exact line (no noise): closed-form recovers slope/intercept to machine eps."""
    x = np.linspace(-1.0, 1.0, 500)
    y = 2.5 * x + 0.75
    a, b = _linear_residual_fit_closed(x, y)
    assert a == pytest.approx(2.5, abs=1e-10)
    assert b == pytest.approx(0.75, abs=1e-10)


def test_batched_matches_closed_form_on_float32_inputs():
    """float32 base/y (the discovery auto-base pool dtype) still bit-identical."""
    rng = np.random.default_rng(11)
    xs = [rng.normal(size=n).astype(np.float32) for n in (123, 777, 3000)]
    ys = [(0.9 * x + rng.normal(size=x.size) * 0.2).astype(np.float32) for x in xs]
    alphas, betas = _linear_residual_fit_batched(xs, ys)
    for k in range(len(xs)):
        a_seq, b_seq = _linear_residual_fit_closed(xs[k], ys[k])
        assert a_seq == alphas[k]
        assert b_seq == betas[k]
