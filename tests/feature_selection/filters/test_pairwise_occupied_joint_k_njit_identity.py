"""Regression: njit ``_pairwise_occupied_joint_k`` is bit-identical to the prior
pure-Python set-per-pair reference across mixed cardinalities, ties, and dtypes.

Pins the optimization in ``_permutation_null.py`` (Python set-per-pair loop -> njit
boolean-seen kernel, ~90-240x). Identity is by construction (same distinct joint codes
counted); this asserts it on shapes that would expose an off-by-one in the flat code
index (asymmetric nbins, fully-tied columns, sparse occupancy)."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from mlframe.feature_selection.filters._permutation_null import _pairwise_occupied_joint_k


def _reference(factors_data, pair_a, pair_b, nbins):
    """Verbatim prior pure-Python body (the OLD side of the A/B)."""
    n = int(factors_data.shape[0])
    n_pairs = int(pair_a.shape[0])
    out = np.empty(n_pairs, dtype=np.int64)
    for p in range(n_pairs):
        a = int(pair_a[p])
        b = int(pair_b[p])
        nb_b = int(nbins[b])
        seen = set()
        for i in range(n):
            seen.add(int(factors_data[i, a]) * nb_b + int(factors_data[i, b]))
        out[p] = len(seen)
    return out


def _all_pairs(p):
    pairs = list(combinations(range(p), 2))
    pa = np.fromiter((x[0] for x in pairs), dtype=np.int64, count=len(pairs))
    pb = np.fromiter((x[1] for x in pairs), dtype=np.int64, count=len(pairs))
    return pa, pb


def test_identity_uniform_cardinality():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 10, size=(500, 8)).astype(np.int32)
    nbins = np.full(8, 10, dtype=np.int64)
    pa, pb = _all_pairs(8)
    assert np.array_equal(
        _pairwise_occupied_joint_k(data, pa, pb, nbins),
        _reference(data, pa, pb, nbins),
    )


def test_identity_asymmetric_cardinality():
    """Per-column nbins differ -> exercises the a*nbins_b+b code arithmetic both ways."""
    rng = np.random.default_rng(7)
    nbins = np.array([2, 3, 5, 7, 11, 4], dtype=np.int64)
    cols = [rng.integers(0, k, size=600) for k in nbins]
    data = np.stack(cols, axis=1).astype(np.int32)
    pa, pb = _all_pairs(len(nbins))
    assert np.array_equal(
        _pairwise_occupied_joint_k(data, pa, pb, nbins),
        _reference(data, pa, pb, nbins),
    )


def test_identity_tied_and_constant_columns():
    """Fully-tied (constant) and low-occupancy columns: distinct-count must still match."""
    n = 400
    data = np.zeros((n, 4), dtype=np.int32)
    data[:, 0] = 3  # constant
    data[:, 1] = np.arange(n) % 2  # 2 levels
    data[:, 2] = np.arange(n) % 5  # 5 levels
    data[:, 3] = 1  # constant
    nbins = np.array([8, 2, 5, 8], dtype=np.int64)
    pa, pb = _all_pairs(4)
    assert np.array_equal(
        _pairwise_occupied_joint_k(data, pa, pb, nbins),
        _reference(data, pa, pb, nbins),
    )


def test_identity_int64_input_dtype():
    """Native int64 factors_data must give the identical count too."""
    rng = np.random.default_rng(11)
    data = rng.integers(0, 6, size=(300, 5)).astype(np.int64)
    nbins = np.full(5, 6, dtype=np.int64)
    pa, pb = _all_pairs(5)
    assert np.array_equal(
        _pairwise_occupied_joint_k(data, pa, pb, nbins),
        _reference(data, pa, pb, nbins),
    )
