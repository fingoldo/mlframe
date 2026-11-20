"""brute_force_top_n feeds numba kernels that run with bounds checking off, so its inputs are validated up front."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import brute_force_top_n


def _data(n=50, f=4):
    """Small regression problem."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, f)), np.zeros(n), rng.normal(size=n)


def test_brute_force_rejects_a_target_of_the_wrong_length():
    """A short y was read past its end inside the kernel."""
    phi, base, y = _data()
    with pytest.raises(ValueError, match="y has shape"):
        brute_force_top_n(phi, base, y[:-5], classification=False, metric="mse", top_n=3, parallel=False)


def test_brute_force_rejects_a_base_of_the_wrong_length():
    """A short per-row base was read past its end inside the kernel."""
    phi, base, y = _data()
    with pytest.raises(ValueError):
        brute_force_top_n(phi, base[:-5], y, classification=False, metric="mse", top_n=3, parallel=False)


def test_brute_force_accepts_a_scalar_base():
    """A scalar base is broadcast to every row."""
    phi, base, y = _data()
    a = brute_force_top_n(phi, 0.0, y, classification=False, metric="mse", top_n=3, parallel=False)
    b = brute_force_top_n(phi, base, y, classification=False, metric="mse", top_n=3, parallel=False)
    assert a == b


def test_brute_force_rejects_non_positive_top_n():
    """top_n=0 wrote to element -1 of an empty buffer."""
    phi, base, y = _data()
    with pytest.raises(ValueError, match="top_n"):
        brute_force_top_n(phi, base, y, classification=False, metric="mse", top_n=0, parallel=False)


def test_brute_force_min_card_zero_behaves_as_one():
    """min_card=0 would index comb[-1] of an empty row; it is clamped to 1."""
    phi, base, y = _data()
    a = brute_force_top_n(phi, base, y, classification=False, metric="mse", min_card=0, top_n=5, parallel=False)
    b = brute_force_top_n(phi, base, y, classification=False, metric="mse", min_card=1, top_n=5, parallel=False)
    assert a, "the search returned no combinations"
    assert a == b
    assert all(len(c) >= 1 for _, c in a)
