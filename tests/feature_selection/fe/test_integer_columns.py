"""The shared integer-column helpers: one definition, an early exit that does not change it, and a per-frame list that is reused and not stale."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _integer_columns as ic


def _reference_exact(x) -> bool:
    """The original exact-integer test: a full pass over the finite values."""
    a = np.asarray(x)
    if np.issubdtype(a.dtype, np.integer):
        return True
    if not np.issubdtype(a.dtype, np.floating):
        return False
    finite = a[np.isfinite(a)]
    return bool(finite.size > 0 and np.all(finite == np.floor(finite)))


def test_exact_test_matches_the_full_pass_including_a_late_fractional_value():
    """The early exit on the first rows gives the same answer as the full pass, also when the first rows are whole numbers and a fractional value comes later."""
    rng = np.random.default_rng(0)
    n = 10_000
    cases = {
        "int": rng.integers(0, 9, n),
        "whole-float": rng.integers(0, 9, n).astype(float),
        "late-fraction": np.concatenate([rng.integers(0, 9, n - 5).astype(float), [0.5, 1.25, 2.0, 3.0, 4.0]]),
        "early-fraction": np.concatenate([[0.5], rng.integers(0, 9, n - 1).astype(float)]),
        "normal": rng.normal(size=n),
        "with-nan": np.where(rng.random(n) < 0.1, np.nan, rng.integers(0, 9, n).astype(float)),
        "all-nan": np.full(n, np.nan),
        "string": np.array(["a", "b"] * (n // 2)),
    }
    for name, col in cases.items():
        assert ic.is_integer_col(col) == _reference_exact(col), name
    assert ic.is_integer_col(cases["late-fraction"]) is False


def test_noisy_integers_pass_and_the_decision_is_stable_under_row_strides():
    """A noisy integer column passes and a continuous one does not, whatever the length: above the decision rows the test reads an evenly spaced subset."""
    rng = np.random.default_rng(1)
    for n in (200, 5_000, 120_000):
        noisy = rng.integers(0, 30, n) + rng.normal(0, 0.08, n)
        assert ic.is_integer_like_col(noisy), n
        assert not ic.is_integer_like_col(rng.normal(size=n) * 5), n


def test_eligible_columns_are_remembered_per_frame(monkeypatch):
    """The second call for the same frame reads the remembered list; a different column subset only evaluates what is new."""
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"a": rng.integers(0, 9, 500), "b": rng.integers(0, 9, 500) + rng.normal(0, 0.05, 500), "c": rng.normal(size=500)})
    calls = []
    real = ic.is_integer_like_col
    monkeypatch.setattr(ic, "is_integer_like_col", lambda x: (calls.append(1), real(x))[1])
    ic._cache.clear()
    assert ic.eligible_integer_columns(X) == ["a", "b"]
    assert len(calls) == 3
    assert ic.eligible_integer_columns(X) == ["a", "b"]
    assert ic.eligible_integer_columns(X, ["b", "c"]) == ["b"]
    assert len(calls) == 3, "the remembered answer was not reused"


def test_a_changed_frame_is_not_served_a_stale_answer():
    """The same frame object with different values gets a fresh answer; a different frame with the same shape does too."""
    rng = np.random.default_rng(3)
    ic._cache.clear()
    X = pd.DataFrame({"a": rng.integers(0, 9, 500).astype(float), "b": rng.normal(size=500)})
    assert ic.eligible_integer_columns(X) == ["a"]
    X["a"] = rng.normal(size=500)  # the column stops being an integer column
    X["b"] = rng.integers(0, 9, 500).astype(float)
    assert ic.eligible_integer_columns(X) == ["b"]
    Y = pd.DataFrame({"a": rng.normal(size=500), "b": rng.integers(0, 9, 500).astype(float)})
    assert ic.eligible_integer_columns(Y) == ["b"]


def test_to_int64_rounds_floats_and_keeps_exact_integers():
    """Floats are rounded to the nearest integer, exact integers and integer dtypes are unchanged, negative values round to the nearest as well."""
    assert ic.to_int64(np.array([4.98, 5.02, -3.02, -2.6])).tolist() == [5, 5, -3, -3]
    assert ic.to_int64(np.array([0.0, 7.0, -4.0])).tolist() == [0, 7, -4]
    assert ic.to_int64(np.array([1, 2, 3], dtype=np.int32)).tolist() == [1, 2, 3]
