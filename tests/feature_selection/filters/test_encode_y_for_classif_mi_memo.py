"""encode_y_for_classif_mi bins a given continuous target once, not once per FE family.

About seventeen cascade stages each discretise the same fit target through this helper. For a continuous target every call paid a full
``np.unique`` sort plus a ``pd.qcut`` sort and quantile pass, all producing the identical codes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _y_encoding


def _count_qcut(monkeypatch):
    """Patch ``pandas.qcut`` with a counting passthrough and return the counter."""
    calls = {"n": 0}
    real = pd.qcut

    def _spy(*args, **kwargs):
        """Count the call and delegate."""
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(pd, "qcut", _spy)
    return calls


def _continuous(n=20_000, seed=0):
    """A continuous float target with far more than 32 distinct values."""
    return np.random.default_rng(seed).normal(size=n)


def test_same_continuous_target_is_binned_once(monkeypatch):
    """Two calls on equal content (distinct array objects) run qcut once and return equal codes."""
    _y_encoding._clear_encode_cache()
    calls = _count_qcut(monkeypatch)
    y = _continuous()
    first = _y_encoding.encode_y_for_classif_mi(y)
    second = _y_encoding.encode_y_for_classif_mi(y.copy())
    assert calls["n"] == 1, f"qcut ran {calls['n']} times for the same target"
    np.testing.assert_array_equal(first, second)


def test_changed_content_is_rebinned(monkeypatch):
    """An in-place change to the target's values must not be served the stale codes."""
    _y_encoding._clear_encode_cache()
    calls = _count_qcut(monkeypatch)
    y = _continuous()
    before = _y_encoding.encode_y_for_classif_mi(y)
    y[: len(y) // 2] = np.sort(y[: len(y) // 2])[::-1] * 3.0
    after = _y_encoding.encode_y_for_classif_mi(y)
    assert calls["n"] == 2
    uncached = _y_encoding._encode_y_uncached(y)
    np.testing.assert_array_equal(after, uncached)
    assert not np.array_equal(before, after)


def test_caller_mutating_the_result_does_not_corrupt_later_calls(monkeypatch):
    """Codes handed back are the caller's to modify; the next call on the same target still returns the correct codes."""
    _y_encoding._clear_encode_cache()
    y = _continuous()
    expected = _y_encoding._encode_y_uncached(y)
    got = _y_encoding.encode_y_for_classif_mi(y)
    got[:] = -1
    np.testing.assert_array_equal(_y_encoding.encode_y_for_classif_mi(y), expected)


def test_memoised_codes_match_uncached_on_every_target_shape():
    """Integer, low-level float, half-step float and continuous targets encode identically with and without the memo."""
    rng = np.random.default_rng(3)
    targets = [
        rng.integers(0, 5, size=5000),
        rng.choice([0.0, 0.5, 1.0, 1.5], size=5000),
        rng.choice([1.0, 2.0, 5.0], size=5000),
        rng.normal(size=5000),
        rng.exponential(size=5000).astype(np.float32),
    ]
    for t in targets:
        _y_encoding._clear_encode_cache()
        np.testing.assert_array_equal(_y_encoding.encode_y_for_classif_mi(t), _y_encoding._encode_y_uncached(t))
        np.testing.assert_array_equal(_y_encoding.encode_y_for_classif_mi(t), _y_encoding._encode_y_uncached(t))
