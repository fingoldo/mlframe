"""The opt-in stable-order cache of the extra-basis detector returns the order np.argsort gives, from a fresh sort or the cache."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_extra_basis_fe_helpers as H


def test_cached_order_equals_argsort(monkeypatch):
    """Ties and a repeated request, with the cache on and off."""
    x = np.round(np.random.default_rng(0).standard_normal(5000) * 3).astype(np.float32)
    want = np.argsort(x, kind="stable")
    for flag in ("0", "1"):
        monkeypatch.setenv("MLFRAME_EXTRA_BASIS_ORDER_CACHE", flag)
        assert np.array_equal(H.cached_stable_argsort(x), want)
        assert np.array_equal(H.cached_stable_argsort(x.copy()), want)
