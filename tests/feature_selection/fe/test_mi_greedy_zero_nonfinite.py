"""``_zero_nonfinite`` of the greedy-MI candidates returns what ``nan_to_num(nan=0, posinf=0, neginf=0)`` returned."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _mi_greedy_fe as greedy


def test_equals_nan_to_num_on_both_float_widths():
    """NaN and both infinities become 0, every other value and the dtype are unchanged; an all-finite array comes back as it is."""
    for dtype in (np.float32, np.float64):
        v = np.array([1.5, np.nan, -np.inf, np.inf, -0.0, 3e30], dtype=dtype)
        got = greedy._zero_nonfinite(v.copy())
        want = np.nan_to_num(v.copy(), nan=0.0, posinf=0.0, neginf=0.0)
        assert got.dtype == want.dtype and np.array_equal(got, want) and np.signbit(got[4])
        clean = np.arange(5, dtype=dtype)
        assert greedy._zero_nonfinite(clean) is clean


def test_every_registered_transform_materialises_the_same_column():
    """Each unary, trigonometric and binary transform on a column with a NaN and a zero gives the column the ``nan_to_num`` form gave."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.standard_normal(4000).astype(np.float32), "b": rng.standard_normal(4000).astype(np.float32)})
    X.loc[5, "a"] = np.nan
    X.loc[7, "b"] = 0.0
    cache = {c: X[c].to_numpy() for c in X}
    unary = {**greedy.UNARY_TRANSFORMS, **greedy.TRIG_BOUNDED_TRANSFORMS}
    for name, fn in unary.items():
        got = greedy._materialise_candidate(X, ("a",), name, cache)
        with np.errstate(all="ignore"):
            want = np.nan_to_num(np.asarray(fn(cache["a"]), dtype=got.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        assert np.array_equal(got, want), name
    for name, fn in greedy.BINARY_TRANSFORMS.items():
        got = greedy._materialise_candidate(X, ("a", "b"), name, cache)
        with np.errstate(all="ignore"):
            want = np.nan_to_num(np.asarray(fn(cache["a"], cache["b"]), dtype=got.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        assert np.array_equal(got, want), name


def test_no_spread_decision_equals_the_std_rule():
    """Constant, near-constant, tiny-range and ordinary columns of both float widths: the range shortcut and the standard-deviation rule give the same verdict."""
    rng = np.random.default_rng(2)
    cases = [np.zeros(1000), np.full(1000, 3.5), np.full(1000, 3.5) + 1e-14 * rng.standard_normal(1000), 1e-9 * rng.standard_normal(1000), 1e-4 * rng.standard_normal(1000)]
    cases += [rng.standard_normal(1000), np.r_[np.zeros(999), 1.0], np.r_[np.zeros(99999), 2e-6], 1e8 + rng.standard_normal(1000)]
    for base in cases:
        for dtype in (np.float32, np.float64):
            arr = base.astype(dtype)
            assert greedy._has_no_spread(arr) == bool(float(arr.std()) <= 1e-12), (dtype, float(arr.std()))
