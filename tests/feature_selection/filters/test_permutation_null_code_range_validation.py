"""The maxT floors reject codes and indices outside their declared ranges instead of feeding them to njit kernels that index tables without bounds checks."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._permutation_null import (
    pairwise_mm_joint_bias,
    pooled_pair_permutation_null_joint_mi_floor,
    pooled_permutation_null_gain_floor,
    pooled_triple_permutation_null_joint_mi_floor,
)


def _pool(n=200, p=4, seed=0):
    """A small coded pool, its per-column nbins, and a binary per-row target with its class frequencies."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 3, size=(n, p)).astype(np.int32)
    nb = np.full(p, 3, dtype=np.int64)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    fy = np.bincount(y, minlength=2) / n
    return data, nb, y, fy


def test_order1_floor_rejects_a_code_beyond_its_nbins():
    """A candidate code of 5 under nbins=3 would write outside the joint table, so the floor raises."""
    data, nb, y, _ = _pool()
    full = np.column_stack([data, y]).astype(np.int32)
    nbins = np.append(nb, 2)
    full[0, 1] = 5
    with pytest.raises(ValueError, match="outside"):
        pooled_permutation_null_gain_floor(full, nbins, np.arange(4), 4, n_permutations=5, random_seed=0)


def test_order1_floor_rejects_a_target_code_beyond_its_nbins():
    """A target code of 4 under nbins=2 is refused."""
    data, nb, y, _ = _pool()
    full = np.column_stack([data, y]).astype(np.int32)
    nbins = np.append(nb, 2)
    full[3, 4] = 4
    with pytest.raises(ValueError, match="target codes"):
        pooled_permutation_null_gain_floor(full, nbins, np.arange(4), 4, n_permutations=5, random_seed=0)


def test_pair_floor_rejects_a_code_beyond_its_nbins():
    """The pair floor refuses a column whose codes exceed its nbins entry."""
    data, nb, y, fy = _pool()
    data[0, 0] = 7
    with pytest.raises(ValueError, match="outside"):
        pooled_pair_permutation_null_joint_mi_floor(data, nb, np.array([0, 1]), np.array([2, 3]), y, fy, n_permutations=3, random_seed=0)


def test_triple_floor_rejects_an_out_of_range_triple_index():
    """A triple index past the last column would read outside factors_data."""
    data, nb, y, fy = _pool()
    with pytest.raises(ValueError, match="pair indices"):
        pooled_triple_permutation_null_joint_mi_floor(data, nb, np.array([0, 1]), np.array([1, 2]), np.array([2, 9]), y, fy, n_permutations=3, random_seed=0)


def test_triple_floor_rejects_a_per_class_target_instead_of_per_row_codes():
    """A target vector of the wrong length is refused instead of being indexed past its end."""
    data, nb, y, fy = _pool()
    with pytest.raises(ValueError, match="one target code per row"):
        pooled_triple_permutation_null_joint_mi_floor(data, nb, np.array([0, 1]), np.array([1, 2]), np.array([2, 3]), y[:10], fy, n_permutations=3, random_seed=0)


def test_triple_floor_rejects_mismatched_triple_lengths():
    """triple_c shorter than triple_a is refused."""
    data, nb, y, fy = _pool()
    with pytest.raises(ValueError, match="same length"):
        pooled_triple_permutation_null_joint_mi_floor(data, nb, np.array([0, 1]), np.array([1, 2]), np.array([2]), y, fy, n_permutations=3, random_seed=0)


def test_triple_floor_rejects_a_code_beyond_its_nbins():
    """A code beyond nbins in a triple's third column is refused."""
    data, nb, y, fy = _pool()
    data[5, 3] = 8
    with pytest.raises(ValueError, match="outside"):
        pooled_triple_permutation_null_joint_mi_floor(data, nb, np.array([0, 1]), np.array([1, 2]), np.array([3, 3]), y, fy, n_permutations=3, random_seed=0)


def test_pairwise_mm_bias_rejects_out_of_range_pair_index():
    """The Miller-Madow bias kernel refuses a pair index past the last column."""
    data, nb, _, _ = _pool()
    with pytest.raises(ValueError, match="pair indices"):
        pairwise_mm_joint_bias(data, np.array([0, 1]), np.array([2, 10]), nb, 2)


def test_valid_inputs_still_produce_a_floor():
    """A well-formed pool is unaffected by the new checks."""
    data, nb, y, fy = _pool()
    floor = pooled_triple_permutation_null_joint_mi_floor(data, nb, np.array([0, 1]), np.array([1, 2]), np.array([2, 3]), y, fy, n_permutations=5, random_seed=0)
    assert np.isfinite(floor) and floor >= 0.0
