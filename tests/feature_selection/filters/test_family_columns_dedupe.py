"""Repeated code columns are dropped before the family-wise tests; the verdicts do not change."""

import numpy as np

from mlframe.feature_selection.filters import _family_maxt_gate as gate
from mlframe.feature_selection.filters._family_columns_dedupe import unique_columns


def test_unique_columns_maps_every_row_to_its_first_equal_row():
    """``cols[keep][inverse]`` rebuilds ``cols``; ``keep`` holds the first row of each group; an all-distinct matrix returns None."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 6, (5, 300)).astype(np.int8)
    cols = base[[0, 1, 0, 2, 1, 3, 0, 4]]
    keep, inverse = unique_columns(cols)
    assert keep.tolist() == [0, 1, 3, 5, 7]
    assert np.array_equal(cols[keep][inverse], cols)
    assert unique_columns(base) is None
    assert unique_columns(base[:1]) is None


def _family(seed):
    """One strong column, many noise columns, and exact repeats of both."""
    rng = np.random.default_rng(seed)
    n = 4000
    y = rng.integers(0, 2, n).astype(np.int64)
    strong = np.where(rng.random(n) < 0.75, y, 1 - y).astype(np.int8)
    noise = rng.integers(0, 4, (12, n)).astype(np.int8)
    unique = np.vstack([strong[None], noise])
    reps = unique[rng.integers(0, unique.shape[0], 60)]
    return y, np.vstack([unique, reps])


def test_family_verdicts_are_the_same_with_and_without_the_distinct_column_reduction(monkeypatch):
    """Per-column batches of a family with repeats: the strong column passes and the noise columns fail, with or without the reduction."""
    y, cols = _family(1)
    batches = [cols[j][:, None] for j in range(cols.shape[0])]
    with_dedupe = gate.family_pass_batches(batches, y, n_perm=399, seed=3)
    monkeypatch.setattr(gate, "unique_columns", lambda c: None)
    without = gate.family_pass_batches(batches, y, n_perm=399, seed=3)
    assert with_dedupe == without
    assert with_dedupe[0] and not any(with_dedupe[1:13])
