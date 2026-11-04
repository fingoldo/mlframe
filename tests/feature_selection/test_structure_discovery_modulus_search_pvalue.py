"""The p-value of a discovered modular relation accounts for the modulus search."""

from __future__ import annotations

import types

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.shared import escalate_modulus
from mlframe.feature_selection.structure_discovery import _column_perm_pvalue, _modulus_search_pvalue


def _bed():
    """Two integer columns and a structureless binary target, with a stand-in coarse hit at modulus 7."""
    rng = np.random.default_rng(3)
    n = 800
    a, b = rng.integers(0, 50, n), rng.integers(0, 50, n)
    X = pd.DataFrame({"a": a, "b": b})
    y = (rng.random(n) < 0.5).astype(int)
    hit = types.SimpleNamespace(modulus=7, cols=("a", "b"), op="sum", c_arr=(a + b).astype(np.float64))
    return X, y, hit


def test_searched_modulus_is_less_significant_than_a_fixed_one_on_noise():
    """On noise the best of ~15 searched moduli looks better than any one; the single-modulus null calls that significant, the search-aware null does not."""
    X, y, hit = _bed()
    _best_m, best_mi, residue = escalate_modulus(X, y, hit, nbins=12)
    p_fixed = _column_perm_pvalue(residue, y, 12, 100, 0)
    p_search = _modulus_search_pvalue(X, y, hit, best_mi, 12, 100, 0)
    assert p_search > p_fixed + 0.1, f"search-aware p={p_search:.3f} should exceed the fixed-modulus p={p_fixed:.3f}"


def test_disabled_gives_nan():
    """Zero permutations disables the test."""
    X, y, hit = _bed()
    assert np.isnan(_modulus_search_pvalue(X, y, hit, 0.1, 12, 0, 0))
