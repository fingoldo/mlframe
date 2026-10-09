"""infer_classification decides from a prefix when it can, and agrees with the full-cardinality definition everywhere."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _fe_accuracy_gate as gate


def _full(y):
    """The definition: distinct finite values against the dtype's cardinality limit."""
    y = np.asarray(y).ravel()
    if y.dtype.kind in ("b", "O", "U", "S"):
        return True
    finite = y[np.isfinite(y)] if y.dtype.kind == "f" else y
    if finite.size == 0:
        return True
    frac = 0.05 if y.dtype.kind in ("i", "u") else 0.02
    return np.unique(finite).size <= max(20, int(frac * finite.size))


@pytest.mark.parametrize("make", [
    lambda r: r.normal(size=200_000),
    lambda r: r.integers(0, 3, size=200_000).astype(np.float64),
    lambda r: r.integers(0, 3, size=200_000),
    lambda r: r.integers(0, 50_000, size=200_000),
    lambda r: np.round(r.normal(size=200_000), 2),
    lambda r: np.concatenate([np.zeros(100_000), r.normal(size=100_000)]),
    lambda r: np.where(r.random(1_000) < 0.2, np.nan, r.integers(0, 4, size=1_000).astype(np.float64)),
    lambda r: np.full(10, np.nan),
])
def test_matches_the_full_cardinality_definition(make):
    """Each shape (continuous, few levels, int ids, rounded, constant head then continuous, NaNs, all NaN) gives the definition's answer."""
    gate._INFER_CLS_MEMO.clear()
    y = make(np.random.default_rng(4))
    assert gate.infer_classification(y) == _full(y)
