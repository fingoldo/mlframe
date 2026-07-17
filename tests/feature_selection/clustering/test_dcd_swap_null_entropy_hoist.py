"""Regression: DCD swap permutation-null entropy hoist must stay bit-identical.

The swap nulls (``_run_member_null`` + the aggregate null in ``_dcd_swap.py``)
hoist the permutation-invariant ``H(Z)`` / ``H(Y,Z)`` out of the B-loop and pass
them via ``conditional_mi(entropy_z=, entropy_yz=)``. This is bit-identical by
construction (only the shuffled X column changes across draws). These tests pin:

1. ``conditional_mi`` returns the IDENTICAL value when H(Z)/H(Y,Z) are supplied
   pre-computed vs recomputed internally (the contract the hoist relies on).
2. The hoist matches the from-scratch recompute across many shuffles of X (the
   exact loop-body invariant the swap nulls exploit).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory import (
    conditional_mi,
    entropy,
    merge_vars,
)


def _data(n=600, n_cols=10, n_bins=8, seed=11):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, n_bins, size=(n, n_cols)).astype(np.int32)
    nbins = np.full(n_cols, n_bins, dtype=np.int64)
    return data, nbins


@pytest.mark.parametrize("n_z", [1, 3, 6])
def test_conditional_mi_precomputed_entropy_bit_identical(n_z):
    """Supplying hoisted H(Z)/H(Y,Z) must not change conditional_mi's output."""
    data, nbins = _data()
    x = np.array([0], dtype=np.int64)
    y = np.array([9], dtype=np.int64)
    z = np.arange(1, 1 + n_z, dtype=np.int64)

    base = conditional_mi(
        factors_data=data,
        x=x,
        y=y,
        z=z,
        var_is_nominal=None,
        factors_nbins=nbins,
        entropy_cache=None,
        can_use_x_cache=False,
        can_use_y_cache=False,
    )

    # Hoist exactly as the swap nulls do.
    _, fz, _ = merge_vars(data, np.sort(z), None, nbins)
    h_z = float(entropy(fz))
    _, fyz, _ = merge_vars(data, np.sort(np.concatenate([y, z])), None, nbins)
    h_yz = float(entropy(fyz))

    hoisted = conditional_mi(
        factors_data=data,
        x=x,
        y=y,
        z=z,
        var_is_nominal=None,
        factors_nbins=nbins,
        entropy_z=h_z,
        entropy_yz=h_yz,
        entropy_cache=None,
        can_use_x_cache=False,
        can_use_y_cache=False,
    )
    assert hoisted == base, f"hoist diverged: {hoisted!r} != {base!r}"


def test_hoist_matches_recompute_across_shuffles():
    """The swap-null loop invariant: across B shuffles of X, the hoisted
    H(Z)/H(Y,Z) reproduce the same CMI sequence as the per-draw recompute."""
    data, nbins = _data(n=800, seed=3)
    x_col, y_col = 0, 9
    z = np.arange(1, 7, dtype=np.int64)
    x = np.array([x_col], dtype=np.int64)
    y = np.array([y_col], dtype=np.int64)

    _, fz, _ = merge_vars(data, np.sort(z), None, nbins)
    h_z = float(entropy(fz))
    _, fyz, _ = merge_vars(data, np.sort(np.concatenate([y, z])), None, nbins)
    h_yz = float(entropy(fyz))

    rng = np.random.default_rng(99)
    perm = data.copy()
    col = perm[:, x_col].copy()
    for _ in range(30):
        sh = col.copy()
        rng.shuffle(sh)
        perm[:, x_col] = sh
        old = conditional_mi(
            factors_data=perm,
            x=x,
            y=y,
            z=z,
            var_is_nominal=None,
            factors_nbins=nbins,
            entropy_cache=None,
            can_use_x_cache=False,
            can_use_y_cache=False,
        )
        new = conditional_mi(
            factors_data=perm,
            x=x,
            y=y,
            z=z,
            var_is_nominal=None,
            factors_nbins=nbins,
            entropy_z=h_z,
            entropy_yz=h_yz,
            entropy_cache=None,
            can_use_x_cache=False,
            can_use_y_cache=False,
        )
        assert new == old
