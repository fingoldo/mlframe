"""CPX3 identity regression: apply_target_encoding vectorized map == per-row dict.get.

Pins that the vectorized ``pd.Series.map(lookup).fillna(global_mean)`` path
produces bit-identical output to the reference per-row ``dict.get`` loop,
including the unseen-category global-mean fallback.
"""
import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._target_encoding_fe import (
    apply_target_encoding,
    _column_to_str,
)


def _reference_per_row(X_test, col, recipe):
    """Pre-optimization per-row dict.get loop (the behaviour we must preserve)."""
    cats = _column_to_str(X_test[col])
    lookup = recipe["lookup"]
    global_mean = float(recipe["global_mean"])
    out = np.empty(len(cats), dtype=np.float64)
    for i, c in enumerate(cats):
        out[i] = lookup.get(c, global_mean)
    return out


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_apply_target_encoding_matches_per_row_loop(seed):
    rng = np.random.default_rng(seed)
    n = 20_000
    card = 200
    cats = rng.integers(0, card + 25, size=n)  # +25 -> some unseen categories
    X_test = pd.DataFrame({"cat": cats})
    lookup = {str(k): float(rng.normal(0.3, 0.1)) for k in range(card)}
    recipe = {"lookup": lookup, "global_mean": 0.2718}

    new = apply_target_encoding(X_test, "cat", recipe)
    ref = _reference_per_row(X_test, "cat", recipe)
    assert np.array_equal(new, ref), f"max|diff|={np.max(np.abs(new - ref))}"


def test_apply_target_encoding_all_unseen_uses_global():
    """When NO category is in the lookup, every row falls back to global_mean."""
    X_test = pd.DataFrame({"cat": np.array([99, 100, 101])})
    recipe = {"lookup": {"0": 1.0, "1": 2.0}, "global_mean": 0.5}
    out = apply_target_encoding(X_test, "cat", recipe)
    assert np.array_equal(out, np.array([0.5, 0.5, 0.5]))


def test_apply_target_encoding_object_column_identity():
    """String/object categorical keys round-trip identically."""
    X_test = pd.DataFrame({"cat": ["a", "b", "c", "a", "z"]})
    recipe = {"lookup": {"a": 1.5, "b": 2.5, "c": 3.5}, "global_mean": 0.0}
    new = apply_target_encoding(X_test, "cat", recipe)
    ref = _reference_per_row(X_test, "cat", recipe)
    assert np.array_equal(new, ref)
    assert np.array_equal(new, np.array([1.5, 2.5, 3.5, 1.5, 0.0]))
