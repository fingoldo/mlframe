"""Identity regression for the njit single-pass scatter in
``_build_factorize_lookup`` (``_cat_kway_materialize``).

Pins that the njit population kernel ``_scatter_factorize_lookup`` produces a
lookup table BIT-IDENTICAL to the prior numpy fancy-index form
(``lookup[a + b*nbins_a] = post_prune_class``, last-write-wins in row order)
across uniform / skewed / unequal-cardinality / unseen-code data. A future
"just inline numpy again" or a row-order change in the kernel would flip the
last-write-wins tie on duplicate codes and fail here.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._cat_kway_materialize import (
    _build_factorize_lookup,
    _scatter_factorize_lookup,
)
from mlframe.feature_selection.filters.info_theory import merge_vars


def _numpy_reference(factors_data, idx_a, idx_b, nbins_a, nbins_b, classes_pair_post):
    """The exact pre-fix numpy population, kept here as the A-side oracle."""
    lookup = np.full(int(nbins_a) * int(nbins_b), -1, dtype=np.int64)
    vals_a = factors_data[:, idx_a].astype(np.int64)
    vals_b = factors_data[:, idx_b].astype(np.int64)
    pre_prune_codes = vals_a + vals_b * int(nbins_a)
    lookup[pre_prune_codes] = classes_pair_post.astype(np.int64)
    return lookup


def _make(n, nba, nbb, seed):
    rng = np.random.default_rng(seed)
    fd = np.empty((n, 2), dtype=np.int32)
    fd[:, 0] = rng.integers(0, nba, n)
    fd[:, 1] = rng.integers(0, nbb, n)
    classes, _, _ = merge_vars(
        factors_data=fd,
        vars_indices=np.array([0, 1], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=np.array([nba, nbb], dtype=np.int64),
        dtype=np.int32,
    )
    return fd, classes


@pytest.mark.parametrize("n", [37, 600, 2407, 10000])
@pytest.mark.parametrize("nba,nbb", [(10, 10), (20, 16), (3, 7), (1, 5)])
def test_scatter_bit_identical_to_numpy(n, nba, nbb):
    fd, cls = _make(n, nba, nbb, seed=n * 97 + nba * 13 + nbb)
    ref = _numpy_reference(fd, 0, 1, nba, nbb, cls)
    got = _scatter_factorize_lookup(fd, 0, 1, nba, nbb, cls)
    assert np.array_equal(ref, got)


def test_build_factorize_lookup_unseen_codes_preserved():
    # Data covering only a subset of codes -> -1 sentinels must survive raise mode.
    fd, cls = _make(50, 10, 10, seed=1)
    lookup, n_eff = _build_factorize_lookup(
        factors_data=fd,
        idx_a=0,
        idx_b=1,
        nbins_a=10,
        nbins_b=10,
        classes_pair_post=cls,
        unknown_strategy="raise",
    )
    ref = _numpy_reference(fd, 0, 1, 10, 10, cls)
    # raise mode leaves unseen at -1, matching the raw scatter
    assert np.array_equal(lookup, ref)


def test_scatter_last_write_wins_on_duplicate_codes():
    # Two rows share code (a=2,b=3) but carry different post-prune classes;
    # numpy fancy-index and the njit loop both keep the LAST row's value.
    fd = np.array([[2, 3], [2, 3], [4, 1]], dtype=np.int32)
    cls = np.array([5, 9, 1], dtype=np.int32)  # row1 (=9) must win for code 2+3*10
    ref = _numpy_reference(fd, 0, 1, 10, 10, cls)
    got = _scatter_factorize_lookup(fd, 0, 1, 10, 10, cls)
    assert np.array_equal(ref, got)
    assert got[2 + 3 * 10] == 9
