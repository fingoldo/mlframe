"""Regression pins for the fused single-var relevance-MI fast path in mi_direct (2026-07-05).

``permutation.py:_relevance_mi_1var_fused`` replaces the analytic-null branch's
``merge_vars(x)`` (length-n classes_x build + remap, then discarded) + ``compute_mi_from_classes``
with a single fused O(n) pass. These tests pin BIT-IDENTITY of the fused MI + occupied-bin count
against the legacy ``merge_vars`` + ``compute_mi_from_classes`` path, and that ``mi_direct``'s
analytic branch returns exactly that value. FAILS on pre-fix code (the fused kernel does not exist).
"""

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory import merge_vars, compute_mi_from_classes
from mlframe.feature_selection.filters.permutation import _relevance_mi_1var_fused, mi_direct


def _legacy_mi_bx(fd, ix, factors_nbins, classes_y, freqs_y, dtype=np.int32):
    """Legacy mi bx."""
    ax_classes, ax_freqs, _ = merge_vars(
        factors_data=fd,
        vars_indices=np.array([ix], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=factors_nbins,
        dtype=dtype,
    )
    mi = compute_mi_from_classes(ax_classes, ax_freqs, classes_y, freqs_y, dtype=dtype)
    return mi, int(ax_freqs.shape[0])


def _make(n, nb_x, nb_y, seed):
    """Helper that make."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, nb_x, size=n).astype(np.int32)
    y = ((x + rng.integers(0, nb_y, size=n)) % nb_y).astype(np.int32)
    fd = np.column_stack([x, y]).astype(np.int32)
    factors_nbins = np.array([nb_x, nb_y], dtype=np.int64)
    cy, fy, _ = merge_vars(
        factors_data=fd,
        vars_indices=np.array([1], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=factors_nbins,
        dtype=np.int32,
    )
    return fd, factors_nbins, cy, fy


@pytest.mark.parametrize(
    "n,nb_x,nb_y,seed",
    [
        (30_000, 16, 10, 1),
        (30_000, 20, 12, 7),
        (50_000, 8, 6, 3),  # some empty x-bins likely (dense y)
        (40_000, 32, 4, 5),  # unequal cardinalities
    ],
)
def test_fused_1var_bit_identical_to_merge_vars(n, nb_x, nb_y, seed):
    """Fused 1var bit identical to merge vars."""
    fd, fnb, cy, fy = _make(n, nb_x, nb_y, seed)
    mi_old, bx_old = _legacy_mi_bx(fd, 0, fnb, cy, fy)
    mi_new, bx_new = _relevance_mi_1var_fused(fd, 0, nb_x, cy, fy)
    assert mi_new == mi_old, f"MI not bit-identical: {mi_new!r} vs {mi_old!r}"
    assert bx_new == bx_old, f"occupied-bin count differs: {bx_new} vs {bx_old}"


def test_fused_handles_empty_x_bins():
    """An x with several NEVER-occupied bins must yield the SAME MI + occupied count as merge_vars
    (which prunes them). Guards the skip-empty-row branch of the fused kernel."""
    rng = np.random.default_rng(11)
    n = 30_000
    # Only use bins {0, 3, 7} of a declared nb_x=10 -> 7 empty bins pruned by merge_vars.
    x = rng.choice(np.array([0, 3, 7], dtype=np.int32), size=n)
    y = rng.integers(0, 8, size=n).astype(np.int32)
    fd = np.column_stack([x, y]).astype(np.int32)
    fnb = np.array([10, 8], dtype=np.int64)
    cy, fy, _ = merge_vars(factors_data=fd, vars_indices=np.array([1], dtype=np.int64), var_is_nominal=None, factors_nbins=fnb, dtype=np.int32)
    mi_old, bx_old = _legacy_mi_bx(fd, 0, fnb, cy, fy)
    mi_new, bx_new = _relevance_mi_1var_fused(fd, 0, 10, cy, fy)
    assert mi_new == mi_old
    assert bx_new == bx_old == 3


def test_mi_direct_analytic_matches_legacy_value():
    """mi_direct (analytic branch, default n>=25k raw MI) must return exactly the fused MI, which is
    exactly the legacy merge_vars+compute_mi_from_classes MI."""
    n, nb_x, nb_y = 30_000, 20, 10
    fd, fnb, cy, fy = _make(n, nb_x, nb_y, 99)
    mi_ref, _ = _legacy_mi_bx(fd, 0, fnb, cy, fy)
    mi_out, _conf = mi_direct(
        fd,
        x=(0,),
        y=(1,),
        factors_nbins=fnb,
        classes_y=cy,
        freqs_y=fy,
        npermutations=10,
        prefer_gpu=False,
    )
    assert mi_out == mi_ref, f"mi_direct analytic value diverged: {mi_out!r} vs {mi_ref!r}"
