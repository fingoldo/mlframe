"""Refitting an MRMR on the same X and y with different ``groups`` must not replay the previous selection."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._fit_cache import _fit_cache_key, _fit_signature


def _stub():
    """Minimal stand-in for the fitted-state-free parts of an MRMR."""
    return SimpleNamespace(get_params=lambda deep=True: {"a": 1}, group_aware_mi=False)


def _data():
    """Small frame and target."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"x": rng.normal(size=40), "z": rng.normal(size=40)}), pd.Series(rng.integers(0, 2, 40), name="y")


def test_in_object_skip_signature_changes_with_groups():
    """The signature behind the same-inputs skip differs for two different groupings and matches for equal ones."""
    X, y = _data()
    g1, g2 = np.arange(40) % 4, np.arange(40) % 5

    def sig(g):
        """Signature for one grouping."""
        return _fit_signature(_stub(), y, X, g)[2]

    assert sig(g1) != sig(g2)
    assert sig(g1) == sig(g1.copy())


def test_process_cache_key_changes_with_groups_even_without_group_aware_mi():
    """Groups reach the accuracy gate regardless of group_aware_mi, so the shared cache key must see them."""
    X, y = _data()
    h = _fit_signature(_stub(), y, X)[1]
    k1 = _fit_cache_key(_stub(), X, y, h, np.arange(40) % 4)
    k2 = _fit_cache_key(_stub(), X, y, h, np.arange(40) % 5)
    assert k1 is not None and k1 != k2
