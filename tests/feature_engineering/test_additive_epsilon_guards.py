"""Guarded-denominator replacements for the old ``x / (den + eps)`` pads: identical away from the guard, finite at it."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mlframe.feature_engineering.transformer.class_conditional_anchor import _softmax_similarity
from mlframe.feature_engineering.transformer.class_conditional_anchor import _squared_dists
from mlframe.feature_engineering.transformer.rf_proximity import _softmax_aggregate
from mlframe.training.neural.flat import Snake


def _old_softmax_similarity(X, anchors, temp):
    """The pre-fix additive-pad formula."""
    logits = -_squared_dists(X, anchors) / (temp + 1e-9)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_softmax_similarity_matches_old_pad_at_ordinary_temperature(seed):
    """For a temperature far above the pad the two formulas agree to float32 resolution."""
    rng = np.random.default_rng(seed)
    X, anchors = rng.normal(size=(50, 4)), rng.normal(size=(7, 4))
    for temp in (0.1, 1.0, 5.0):
        np.testing.assert_allclose(_softmax_similarity(X, anchors, temp), _old_softmax_similarity(X, anchors, temp), rtol=1e-5, atol=1e-7)


def test_softmax_similarity_zero_temperature_is_finite_hard_assignment():
    """temp == 0 hits the guard: the result is the finite one-hot nearest-anchor assignment, rows sum to 1."""
    rng = np.random.default_rng(3)
    X, anchors = rng.normal(size=(30, 3)), rng.normal(size=(5, 3))
    out = _softmax_similarity(X, anchors, 0.0)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-5)
    np.testing.assert_array_equal(out.argmax(axis=1), _squared_dists(X, anchors).argmin(axis=1))


def test_softmax_aggregate_matches_old_pad_and_survives_zero_temperature():
    """rf_proximity softmax: ordinary temp agrees with the old pad; temp 0 stays finite."""
    rng = np.random.default_rng(4)
    sims = rng.uniform(size=(20, 6))
    ids = rng.integers(0, 40, size=(20, 6))
    y = rng.normal(size=40)
    got_mean, _ = _softmax_aggregate(ids, sims.copy(), y, 0.5)
    logits = sims / (0.5 + 1e-9)
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(got_mean, (w * y[ids]).sum(axis=1).astype(np.float32), rtol=1e-5, atol=1e-6)
    m0, s0 = _softmax_aggregate(ids, sims.copy(), y, 0.0)
    assert np.isfinite(m0).all() and np.isfinite(s0).all()


@pytest.mark.parametrize("alpha", [1.0, 0.3, -0.7, -2.0])
def test_snake_matches_closed_form_including_negative_alpha(alpha):
    """Snake(x) = x + (1 - cos(2 a x)) / (2 a) for alpha of either sign (the guard is by magnitude, so a negative alpha is not clamped to the floor)."""
    x = torch.linspace(-3, 3, 41)
    out = Snake(alpha=alpha)(x)
    expected = x + (1 - torch.cos(2 * alpha * x)) / (2 * alpha)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_snake_zero_alpha_is_finite():
    """alpha == 0 hits the guard and yields a finite tensor (old pad produced the same finite value)."""
    out = Snake(alpha=0.0)(torch.linspace(-1, 1, 9))
    assert torch.isfinite(out).all()
