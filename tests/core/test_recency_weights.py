"""Unit tests for parametric recency weight vectors."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.core.recency_weights import recency_weights


def test_empty_history_returns_empty():
    """Empty history returns empty."""
    w = recency_weights(0, "poly", 1.0)
    assert w.shape == (0,)


def test_identity_params_give_uniform():
    """Identity params give uniform."""
    for scheme, ident in (("poly", 0.0), ("exp", 1.0), ("power", 0.0)):
        w = recency_weights(6, scheme, ident)
        assert np.allclose(w, 1.0 / 6), f"{scheme} identity should be uniform"


def test_normalized_sums_to_one():
    """Normalized sums to one."""
    for scheme, p in (("poly", 1.5), ("exp", 0.7), ("power", 2.0)):
        w = recency_weights(10, scheme, p)
        assert np.isclose(w.sum(), 1.0)


def test_newest_is_heaviest_and_monotone():
    # Convention: out[0] oldest, out[-1] newest; recency weighting -> non-decreasing.
    """Newest is heaviest and monotone."""
    for scheme, p in (("poly", 2.0), ("exp", 0.5), ("power", 1.0)):
        w = recency_weights(8, scheme, p)
        assert np.all(np.diff(w) >= -1e-12), f"{scheme} weights must be non-decreasing oldest->newest"
        assert w[-1] > w[0], f"{scheme} newest must outweigh oldest"


def test_unnormalized_matches_normalized_up_to_scale():
    """Unnormalized matches normalized up to scale."""
    w_norm = recency_weights(7, "exp", 0.6, normalize=True)
    w_raw = recency_weights(7, "exp", 0.6, normalize=False)
    assert np.allclose(w_raw / w_raw.sum(), w_norm)


def test_poly_matches_closed_form():
    """Poly matches closed form."""
    d, delta = 5, 2.0
    w = recency_weights(d, "poly", delta, normalize=False)
    # out[pos] corresponds to i = d - pos; poly raw = ((d-i+1)/d)**delta.
    expected = np.array([((d - (d - pos) + 1) / d) ** delta for pos in range(d)])
    assert np.allclose(w, expected)


@pytest.mark.parametrize(
    "scheme,param",
    [("poly", -0.1), ("exp", 0.0), ("exp", 1.5), ("power", -1.0)],
)
def test_invalid_params_raise(scheme, param):
    """Invalid params raise."""
    with pytest.raises(ValueError):
        recency_weights(4, scheme, param)


def test_unknown_scheme_raises():
    """Unknown scheme raises."""
    with pytest.raises(ValueError):
        recency_weights(4, "nope", 1.0)


def test_negative_d_raises():
    """Negative d raises."""
    with pytest.raises(ValueError):
        recency_weights(-1, "poly", 1.0)
