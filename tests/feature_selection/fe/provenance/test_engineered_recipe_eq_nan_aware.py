"""Wave 9.1 loop-iter-45 regression: ``EngineeredRecipe.__eq__`` MUST
be NaN-aware and must not raise on nested containers.

Pre-fix at ``engineered_recipes.py:42, 45``:

1. ``np.array_equal(va, vb)`` returns False on NaN-containing arrays
   because NaN != NaN. Persisted recipes whose lookups / diagnostics
   contained NaN (factorize / target_encoding lookups,
   cluster_aggregate's ``pca_var_ratio`` when PCA degenerates) failed
   pickle round-trip equality and ``sklearn.clone`` == fitted checks.

2. Scalar ``float('nan')`` in extra hit the else branch, where ``va !=
   vb`` is True for nan vs nan, breaking deepcopy/pickle round-trip
   equality on that path too.

3. Nested list-of-arrays in extra raised ``ValueError: truth value
   ambiguous`` from ``va != vb`` instead of returning bool - leaking
   an exception out of ``__eq__``.

Severity: P1. Persisted MRMR with engineered recipes failing
round-trip equality breaks ``clone(fitted) == fitted`` and dedup
logic that relies on ``__eq__``.

Fix at engineered_recipes.py:32 (~20 LOC):
- ``np.array_equal(va, vb, equal_nan=True)`` for arrays (with
  fallback for older numpy).
- Both-NaN scalar floats -> equal.
- ``ValueError`` from ambiguous truth-value -> return False
  conservatively rather than raising.
"""

from __future__ import annotations

import copy
import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

import numpy as np


def test_pickle_roundtrip_with_nan_in_ndarray_extra():
    """Persisted recipe with NaN-containing array extra must equal
    itself after pickle round-trip.
    """
    from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe

    r = EngineeredRecipe(
        name="te_x",
        kind="target_encoding",
        src_names=("a",),
        extra={"lookup": np.array([0.1, np.nan, 0.5])},
    )
    assert r == pickle.loads(pickle.dumps(r))  # nosec B301 -- round-trip of a locally-created, trusted object


def test_deepcopy_with_scalar_nan_extra():
    """Recipe with scalar ``float('nan')`` in extra must equal itself
    after deepcopy.
    """
    from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe

    r = EngineeredRecipe(
        name="c",
        kind="cluster_aggregate",
        src_names=("a", "b"),
        extra={
            "method": "mean_z",
            "member_mean": np.array([1.0, 2.0]),
            "member_std": np.array([1.0, 1.0]),
            "signs": np.array([1.0, 1.0]),
            "pca_var_ratio": float("nan"),
        },
    )
    assert r == copy.deepcopy(r)


def test_eq_does_not_raise_on_nested_list_of_arrays():
    """List-of-arrays in extra must NOT raise ``ValueError: truth
    value ambiguous`` from inside __eq__ - return bool instead.
    """
    from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe

    r = EngineeredRecipe(
        name="t",
        kind="target_encoding",
        src_names=("a",),
        extra={"per_fold": [np.array([1, 2]), np.array([3, 4])]},
    )
    # Just verify it returns a bool, not raises.
    result = r == copy.deepcopy(r)
    assert isinstance(result, bool)


def test_clean_array_equality_still_works():
    """Negative control: equality on clean integer arrays unchanged."""
    from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe

    r = EngineeredRecipe(
        name="x",
        kind="factorize",
        src_names=("a",),
        extra={"lookup": np.array([0, 1, 2])},
    )
    assert r == copy.deepcopy(r)


def test_different_scalar_values_still_unequal():
    """Negative control: different scalar values remain unequal."""
    from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe

    r_a = EngineeredRecipe(
        name="x",
        kind="factorize",
        src_names=("a",),
        extra={"val": 1.0},
    )
    r_b = EngineeredRecipe(
        name="x",
        kind="factorize",
        src_names=("a",),
        extra={"val": 2.0},
    )
    assert r_a != r_b


def test_different_array_shapes_unequal():
    """Sanity: arrays of different shapes must be unequal."""
    from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe

    r_a = EngineeredRecipe(
        name="x",
        kind="factorize",
        src_names=("a",),
        extra={"lookup": np.array([0, 1, 2])},
    )
    r_b = EngineeredRecipe(
        name="x",
        kind="factorize",
        src_names=("a",),
        extra={"lookup": np.array([0, 1, 2, 3])},
    )
    assert r_a != r_b
