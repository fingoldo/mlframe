"""The modular sweep keeps integer columns that were measured with a little noise: they look like floats, so an exact-integer test drops them.

A column whose values cluster at integers (mean distance to the nearest integer at most 0.12) is rounded, never truncated, in the scan, the ranking and the replay; a
continuous column (mean distance 0.25) and a column that rounds to a constant stay out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _pairwise_modular_fe as pm


def _planted(sigma: float, p: int = 12, n: int = 4000, seed: int = 3):
    """Integer columns plus Gaussian noise of ``sigma`` and a (x0 + x5) mod 3 target."""
    rng = np.random.default_rng(seed)
    ints = rng.integers(0, 30, size=(n, p))
    X = pd.DataFrame(ints.astype(float) + (rng.normal(0, sigma, size=(n, p)) if sigma else 0.0), columns=[f"x{i}" for i in range(p)])
    y = ((ints[:, 0] + ints[:, 5]) % 3).astype(int)
    return X, y, ints


@pytest.mark.parametrize("sigma", [0.02, 0.1])
def test_scan_finds_the_planted_pair_through_noise(sigma):
    """The planted ``(x0 + x5) mod 3`` pair is responded although every column is a float with noise."""
    X, y, _ints = _planted(sigma)
    assert all(not pm._is_integer_col(np.asarray(X[c])) for c in X.columns)  # the exact-integer test would have dropped all of them
    hits = pm.cheap_modular_scan(X, y)
    assert any(h.op == "sum" and set(h.cols) == {"x0", "x5"} and h.responded for h in hits), "planted pair not responded through noise"


def test_continuous_columns_are_not_eligible():
    """Standard-normal columns (mean distance to an integer about 0.25) are not treated as noisy integers, and a scan over them finds nothing to sweep."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(2000, 6)) * 5, columns=list("abcdef"))
    assert not any(pm._is_integer_like_col(np.asarray(X[c])) for c in X.columns)
    assert pm.cheap_modular_scan(X, rng.integers(0, 2, 2000)) == []


def test_near_constant_and_heavy_noise_columns_are_not_eligible():
    """A column that rounds to one value, and a column whose noise spreads it evenly between the integers, stay out."""
    rng = np.random.default_rng(1)
    assert not pm._is_integer_like_col(5.0 + rng.normal(0, 0.01, 1000))
    assert not pm._is_integer_like_col(rng.integers(0, 30, 1000) + rng.normal(0, 0.4, 1000))


def test_rounding_not_truncation_and_exact_integers_are_unchanged():
    """``_to_int64`` rounds floats to the nearest integer (4.98 -> 5, -3.02 -> -3) and leaves exact integers, integer-valued floats and int dtypes as they were."""
    assert pm._to_int64(np.array([4.98, 5.02, -3.02, 2.51, 2.49])).tolist() == [5, 5, -3, 3, 2]
    exact = np.array([0.0, 7.0, -4.0, 29.0])
    assert pm._to_int64(exact).tolist() == exact.astype(np.int64).tolist()
    assert pm._to_int64(np.array([1, 2, 3], dtype=np.int32)).tolist() == [1, 2, 3]


def test_replay_uses_the_same_rounding_as_the_scan():
    """The hybrid emits a recipe for the noisy pair and ``apply_pairwise_modular`` reproduces the residue of the ROUNDED operands on a fresh noisy frame."""
    X, y, ints = _planted(0.1)
    _names, recipes = pm.hybrid_pairwise_modular_fe_with_recipes(X, y)
    assert recipes, "no recipe emitted for the noisy planted pair"
    r = next(r for r in recipes if set(r.src_names) == {"x0", "x5"} and r.extra["op"] == "sum")
    rng = np.random.default_rng(99)
    fresh = pd.DataFrame(ints.astype(float) + rng.normal(0, 0.1, size=ints.shape), columns=X.columns)  # same integers, new noise
    got = pm.apply_pairwise_modular(fresh, r.extra["op"], r.src_names, r.extra["modulus"])
    expected = np.mod(ints[:, 0] + ints[:, 5], int(r.extra["modulus"])).astype(float)
    assert np.array_equal(got, expected)
