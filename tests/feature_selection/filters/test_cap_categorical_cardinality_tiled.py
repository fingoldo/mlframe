"""``cap_categorical_cardinality`` through transposed tiles equals the plain per-column definition on ragged shapes."""

import numpy as np
import pytest

from mlframe.feature_selection.filters.discretization._categorical_encoding import cap_categorical_cardinality


def _reference(codes, cap):
    """The per-column definition: the cap-1 most frequent codes keep ids 0..cap-2 (stable), the rest fold into cap-1, -1 stays."""
    out = codes.copy()
    for j in range(codes.shape[1]):
        col = codes[:, j]
        mask = col >= 0
        if not mask.any():
            continue
        k = int(col[mask].max()) + 1
        if k <= cap:
            continue
        counts = np.bincount(col[mask].astype(np.int64), minlength=k)
        keep = np.argsort(-counts, kind="stable")[: cap - 1]
        remap = np.full(k, cap - 1, dtype=col.dtype)
        remap[keep] = np.arange(cap - 1)
        out[mask, j] = remap[col[mask].astype(np.int64)]
    return out


@pytest.mark.parametrize("n,p", [(1, 1), (7, 3), (1025, 17), (3000, 33), (2049, 16)])
def test_matches_the_per_column_definition(n, p):
    """Mixed columns (above the cap, under it, all missing) in ragged shapes give the reference matrix."""
    rng = np.random.default_rng(n + p)
    codes = rng.integers(0, 80, (n, p)).astype(np.float64)
    codes[rng.random((n, p)) < 0.05] = -1
    if p > 2:
        codes[:, 1] = rng.integers(0, 5, n)  # under the cap
        codes[:, 2] = -1  # all missing
    got = cap_categorical_cardinality(codes, 16)
    assert np.array_equal(got, _reference(codes, 16))
    assert got.dtype == codes.dtype


def test_a_matrix_under_the_cap_is_returned_unchanged():
    """No column above the cap: the very same object comes back (no copy)."""
    codes = np.random.default_rng(0).integers(0, 5, (500, 20)).astype(np.float64)
    assert cap_categorical_cardinality(codes, 32) is codes
