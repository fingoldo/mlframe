"""Property-based invariants for info-theory + discretisation primitives.

Covers ten algebraic / structural properties of ``entropy``, ``mi`` and
``discretize_array`` from ``mlframe.feature_selection.filters``:

1.  ``test_property_mi_symmetry``                 - ``mi(x, y) == mi(y, x)``.
2.  ``test_property_mi_self_equals_entropy``      - ``mi(x, x) == H(x)``.
3.  ``test_property_mi_with_constant_is_zero``    - MI with a constant is 0.
4.  ``test_property_mi_non_negative``             - ``mi(x, y) >= 0``.
5.  ``test_property_entropy_non_negative``        - ``H(x) >= 0``.
6.  ``test_property_entropy_bounded_by_log_k``    - ``H(x) <= log(k)`` for k unique vals.
7.  ``test_property_entropy_uniform_max``         - uniform(k) approaches ``log(k)`` on large n.
8.  ``test_property_discretize_array_in_range``   - bin codes land in ``[0, nbins)``.
9.  ``test_property_discretize_length_preserved`` - length is preserved.
10. ``test_property_discretize_quantile_monotone``- sorted input -> non-decreasing bin codes.

The underlying public ``mi`` / ``entropy`` take a ``factors_data`` matrix + frequency
arrays respectively, so two tiny adapters here (``_mi_arrs`` / ``_entropy_arr``) lift
them to the 1-D integer-array signature these properties are most naturally stated in.
The adapters do not implement the math, they only marshal arguments.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings, strategies as st, HealthCheck
from hypothesis.extra.numpy import arrays

from mlframe.feature_selection.filters import (
    discretize_array,
    entropy,
    mi,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies (shared across tests).
# ---------------------------------------------------------------------------
_NBINS = st.integers(min_value=2, max_value=20)
_FLOAT_ARRAY = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=20, max_value=200),
    elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
)
_INT_ARRAY = arrays(
    dtype=np.int32,
    shape=st.integers(min_value=20, max_value=200),
    elements=st.integers(min_value=0, max_value=5),
)
_PAIRED_INT_ARRAYS = st.integers(min_value=20, max_value=200).flatmap(
    lambda n: st.tuples(
        arrays(
            dtype=np.int32,
            shape=n,
            elements=st.integers(min_value=0, max_value=5),
        ),
        arrays(
            dtype=np.int32,
            shape=n,
            elements=st.integers(min_value=0, max_value=5),
        ),
    )
)

_SETTINGS = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


# ---------------------------------------------------------------------------
# Adapters: lift the public APIs to a 1-D-array-only signature.
# ---------------------------------------------------------------------------
def _mi_arrs(x: np.ndarray, y: np.ndarray) -> float:
    """``mi`` on two pre-binned 1-D integer arrays. Marshals into the factors_data
    layout the public ``mi`` expects (one column per variable)."""
    x = np.ascontiguousarray(x, dtype=np.int32)
    y = np.ascontiguousarray(y, dtype=np.int32)
    factors_data = np.column_stack([x, y]).astype(np.int32)
    nbins = np.array(
        [int(x.max()) + 1 if x.size else 1, int(y.max()) + 1 if y.size else 1],
        dtype=np.int64,
    )
    return float(mi(factors_data, np.array([0], dtype=np.int64), np.array([1], dtype=np.int64), nbins))


def _entropy_arr(x: np.ndarray) -> float:
    """Shannon entropy of a 1-D integer array, computed via the public ``entropy``
    helper that consumes a frequency vector."""
    if x.size == 0:
        return 0.0
    _, counts = np.unique(x, return_counts=True)
    freqs = counts.astype(np.float64) / counts.sum()
    return float(entropy(freqs))


# ---------------------------------------------------------------------------
# Property 1: MI symmetry.
# ---------------------------------------------------------------------------
@pytest.mark.fast
@given(_PAIRED_INT_ARRAYS)
@_SETTINGS
def test_property_mi_symmetry(pair):
    x, y = pair
    assert abs(_mi_arrs(x, y) - _mi_arrs(y, x)) < 1e-9


# ---------------------------------------------------------------------------
# Property 2: MI(X, X) == H(X).
# ---------------------------------------------------------------------------
@given(_INT_ARRAY)
@_SETTINGS
def test_property_mi_self_equals_entropy(x):
    assert abs(_mi_arrs(x, x) - _entropy_arr(x)) < 1e-9


# ---------------------------------------------------------------------------
# Property 3: MI with a constant is exactly 0.
# ---------------------------------------------------------------------------
@given(_INT_ARRAY)
@_SETTINGS
def test_property_mi_with_constant_is_zero(x):
    const = np.zeros_like(x, dtype=np.int32)
    assert _mi_arrs(x, const) < 1e-12


# ---------------------------------------------------------------------------
# Property 4: MI non-negativity.
# ---------------------------------------------------------------------------
@given(_PAIRED_INT_ARRAYS)
@_SETTINGS
def test_property_mi_non_negative(pair):
    x, y = pair
    assert _mi_arrs(x, y) >= -1e-12


# ---------------------------------------------------------------------------
# Property 5: Entropy non-negativity.
# ---------------------------------------------------------------------------
@given(_INT_ARRAY)
@_SETTINGS
def test_property_entropy_non_negative(x):
    assert _entropy_arr(x) >= 0.0


# ---------------------------------------------------------------------------
# Property 6: H(X) <= log(k), where k = #unique values.
# ---------------------------------------------------------------------------
@given(_INT_ARRAY)
@_SETTINGS
def test_property_entropy_bounded_by_log_k(x):
    k = int(np.unique(x).size)
    if k <= 1:
        # log(1) = 0 ; entropy is exactly 0 here, trivially bounded.
        assert _entropy_arr(x) <= 1e-12
        return
    assert _entropy_arr(x) <= np.log(k) + 1e-9


# ---------------------------------------------------------------------------
# Property 7: uniform(k) entropy approaches log(k) for large n.
# ---------------------------------------------------------------------------
@given(st.integers(min_value=2, max_value=8), st.integers(min_value=0, max_value=2**32 - 1))
@_SETTINGS
def test_property_entropy_uniform_max(k, seed):
    n = 1000
    rng = np.random.default_rng(seed)
    # Build a balanced sample then shuffle, so the empirical distribution is
    # genuinely uniform up to integer division (n // k * k of n samples).
    base = np.tile(np.arange(k, dtype=np.int32), n // k)
    rng.shuffle(base)
    h = _entropy_arr(base)
    assert abs(h - np.log(k)) < 0.1


# ---------------------------------------------------------------------------
# Property 8: discretize_array output stays in [0, nbins).
# ---------------------------------------------------------------------------
@given(_FLOAT_ARRAY, _NBINS)
@_SETTINGS
def test_property_discretize_array_in_range(x, nbins):
    out = discretize_array(x, n_bins=nbins)
    assert out.min() >= 0
    assert out.max() < nbins


# ---------------------------------------------------------------------------
# Property 9: discretize_array preserves length.
# ---------------------------------------------------------------------------
@given(_FLOAT_ARRAY, _NBINS)
@_SETTINGS
def test_property_discretize_length_preserved(x, nbins):
    assert len(discretize_array(x, n_bins=nbins)) == len(x)


# ---------------------------------------------------------------------------
# Property 10: sorted input -> non-decreasing bin codes (quantile method).
# ---------------------------------------------------------------------------
@given(_FLOAT_ARRAY, _NBINS)
@_SETTINGS
def test_property_discretize_quantile_monotone(x, nbins):
    xs = np.sort(x)
    out = discretize_array(xs, n_bins=nbins, method="quantile")
    # Cast to int64 so np.diff on int8 codes cannot overflow.
    diffs = np.diff(out.astype(np.int64))
    assert (diffs >= 0).all()
