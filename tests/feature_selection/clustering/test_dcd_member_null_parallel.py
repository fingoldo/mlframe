"""Regression: the DCD member permutation-null must give the SAME swap decision when parallelized.

``_dcd_swap._run_member_null`` delegates to ``_dcd_swap_null.run_member_null``, which pre-generates all B
shuffles of ONLY the member column with the SAME rng sequence the serial path used, then ``prange``s the
per-draw conditional MI over a thread-local shuffled column (no frame copy, no mutate-restore on the parallel
path). These tests pin:

1. ``conditional_mi_col`` (standalone x-column variant) matches the shared ``conditional_mi`` on a shuffled
   column: H(X,Y,Z) is bit-identical, H(X,Z) differs only by FP reduction order (<=1e-9).
2. The parallel member-null p-value equals the serial p-value (bit-identical in practice; at most one
   1/(B+1) tie-flip from the ~1e-14 H(X,Z) FP delta on a pure-noise member) AND the accept/reject decision at
   ``swap_alpha`` is unchanged -- the selection contract.
3. ``run_member_null`` leaves ``state.factors_data`` byte-unchanged (the mutate-restore contract holds; the
   parallel path never mutates it at all).

Pre-fix (no ``_dcd_swap_null`` module) the imports fail -> these tests fail, as required.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory import conditional_mi, entropy, merge_vars, mi
from mlframe.feature_selection.filters._numba_utils import unpack_and_sort
from mlframe.feature_selection.filters._dynamic_cluster_discovery._dcd_swap_null import (
    conditional_mi_col,
    mi_col,
    run_member_null,
    _PARALLEL_MIN_B,
)


def _synth(n, ncols, nb, seed):
    """Helper that synth."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, nb, size=(n, ncols)).astype(np.int32)
    nbins = np.full(ncols, nb, dtype=np.int64)
    return data, nbins


def _make_state(data, nbins):
    """Make state."""
    return types.SimpleNamespace(factors_data=data, factors_nbins=nbins, _perm_seed=0)


def _serial_member_pvalue(data, nbins, member_idx, y_idx, z_idx, member_rel, B, seed):
    """Reference serial null: exact mutate-restore loop calling the shared conditional_mi / mi."""
    rng = np.random.default_rng(seed)
    y = np.array([y_idx], dtype=np.int64)
    has_z = len(z_idx) > 0
    if has_z:
        z = np.sort(np.array(z_idx, dtype=np.int64))
        _, fz, _ = merge_vars(data, z, None, nbins)
        h_z = float(entropy(fz))
        _, fyz, _ = merge_vars(data, unpack_and_sort(y, z), None, nbins)
        h_yz = float(entropy(fyz))
    col = data[:, member_idx].copy()
    n_exceed = 0
    for _ in range(B):
        sh = col.copy()
        rng.shuffle(sh)
        data[:, member_idx] = sh
        if has_z:
            v = float(
                conditional_mi(
                    factors_data=data,
                    x=np.array([member_idx]),
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
            )
        else:
            v = float(mi(data, np.array([member_idx]), y, nbins))
        if v >= member_rel:
            n_exceed += 1
    data[:, member_idx] = col
    return (n_exceed + 1) / (B + 1)


@pytest.mark.parametrize("n_z", [1, 2, 4])
def test_conditional_mi_col_matches_conditional_mi(n_z):
    """The standalone x-column CMI matches the factors_data-index CMI on shuffled columns (<=1e-9)."""
    data, nbins = _synth(3000, 14, 8, seed=5)
    member_idx, y_idx = 0, 13
    z = np.sort(np.arange(1, 1 + n_z, dtype=np.int64))
    y = np.array([y_idx], dtype=np.int64)
    zc, fz, znc = merge_vars(data, z, None, nbins)
    h_z = float(entropy(fz))
    yzc, fyz, yznc = merge_vars(data, unpack_and_sort(y, z), None, nbins)
    h_yz = float(entropy(fyz))
    nb_x = int(nbins[member_idx])
    rng = np.random.default_rng(17)
    col = data[:, member_idx].copy()
    for _ in range(20):
        sh = col.copy()
        rng.shuffle(sh)
        data[:, member_idx] = sh
        ref = float(
            conditional_mi(
                factors_data=data,
                x=np.array([member_idx]),
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
        )
        got = conditional_mi_col(sh.astype(np.int64), nb_x, zc.astype(np.int64), int(znc), yzc.astype(np.int64), int(yznc), h_z, h_yz)
        assert abs(ref - got) <= 1e-9, f"CMI mismatch n_z={n_z}: {ref!r} vs {got!r}"
    data[:, member_idx] = col


def test_mi_col_matches_mi_no_z():
    """The no-Z path: standalone mi_col matches the shared mi on shuffled columns (<=1e-9)."""
    data, nbins = _synth(3000, 10, 8, seed=6)
    member_idx, y_idx = 0, 9
    y = np.array([y_idx], dtype=np.int64)
    _, fx, _ = merge_vars(data, np.array([member_idx]), None, nbins)
    h_x = float(entropy(fx))
    yc, fy, ync = merge_vars(data, np.sort(y), None, nbins)
    h_y = float(entropy(fy))
    nb_x = int(nbins[member_idx])
    rng = np.random.default_rng(23)
    col = data[:, member_idx].copy()
    for _ in range(20):
        sh = col.copy()
        rng.shuffle(sh)
        data[:, member_idx] = sh
        ref = float(mi(data, np.array([member_idx]), y, nbins))
        got = mi_col(sh.astype(np.int64), nb_x, yc.astype(np.int64), int(ync), h_x, h_y)
        assert abs(ref - got) <= 1e-9
    data[:, member_idx] = col


@pytest.mark.parametrize(
    "n,ncols,nb,n_z,B",
    [
        (2000, 12, 8, 1, 199),
        (2000, 12, 8, 4, 199),
        (5000, 15, 6, 3, 99),
        (4000, 12, 8, 0, 199),  # no-Z path
    ],
)
def test_member_null_parallel_matches_serial_decision(n, ncols, nb, n_z, B):
    """Parallel member-null p-value == serial (up to one 1/(B+1) FP tie-flip) AND identical accept/reject."""
    data, nbins = _synth(n, ncols, nb, seed=7)
    member_idx, y_idx = 0, ncols - 1
    z_idx = list(range(1, 1 + n_z))
    y = np.array([y_idx], dtype=np.int64)
    if n_z > 0:
        z = np.sort(np.array(z_idx, dtype=np.int64))
        member_rel = float(
            conditional_mi(
                factors_data=data,
                x=np.array([member_idx]),
                y=y,
                z=z,
                var_is_nominal=None,
                factors_nbins=nbins,
                entropy_cache=None,
                can_use_x_cache=False,
                can_use_y_cache=False,
            )
        )
    else:
        member_rel = float(mi(data, np.array([member_idx]), y, nbins))

    p_serial = _serial_member_pvalue(data.copy(), nbins, member_idx, y_idx, z_idx, member_rel, B, seed=999)

    state = _make_state(data.copy(), nbins)
    state._perm_seed = 999 - member_idx  # so run_member_null's rng == default_rng(999) for anchor=0
    p_par = run_member_null(state=state, member_idx=member_idx, member_rel=member_rel, B_=B, anchor=0, target=y, S_minus_anchor=list(z_idx))

    alpha = 0.05
    assert (p_serial < alpha) == (p_par < alpha), f"decision flipped: p_serial={p_serial} p_par={p_par}"
    assert abs(p_serial - p_par) <= 1.0 / (B + 1) + 1e-12, f"p diverged >1/(B+1): {p_serial} vs {p_par}"


def test_run_member_null_does_not_mutate_factors_data():
    """Mutate-restore contract: factors_data is byte-unchanged after the null (parallel path never mutates)."""
    data, nbins = _synth(20000, 16, 8, seed=8)  # n large enough to take the parallel prange path
    member_idx = 0
    y = np.array([15], dtype=np.int64)
    z_idx = [1, 2, 3]
    before = data.copy()
    state = _make_state(data, nbins)
    B = max(_PARALLEL_MIN_B, 50)
    p = run_member_null(state=state, member_idx=member_idx, member_rel=0.0, B_=B, anchor=2, target=y, S_minus_anchor=z_idx)
    assert 0.0 < p <= 1.0
    assert np.array_equal(before, data), "run_member_null must leave factors_data byte-identical"


def test_tiny_B_serial_fallback_still_correct():
    """B below _PARALLEL_MIN_B uses the serial mutate-restore fallback and stays decision-correct."""
    data, nbins = _synth(2000, 12, 8, seed=9)
    member_idx, y_idx = 0, 11
    z_idx = [1, 2]
    y = np.array([y_idx], dtype=np.int64)
    z = np.sort(np.array(z_idx, dtype=np.int64))
    member_rel = float(
        conditional_mi(
            factors_data=data,
            x=np.array([member_idx]),
            y=y,
            z=z,
            var_is_nominal=None,
            factors_nbins=nbins,
            entropy_cache=None,
            can_use_x_cache=False,
            can_use_y_cache=False,
        )
    )
    B = _PARALLEL_MIN_B - 1
    assert B >= 1
    p_serial = _serial_member_pvalue(data.copy(), nbins, member_idx, y_idx, z_idx, member_rel, B, seed=999)
    state = _make_state(data.copy(), nbins)
    state._perm_seed = 999 - member_idx
    p = run_member_null(state=state, member_idx=member_idx, member_rel=member_rel, B_=B, anchor=0, target=y, S_minus_anchor=z_idx)
    # Serial fallback uses the SAME rng path -> bit-identical p-value.
    assert p == p_serial, f"serial-fallback p diverged: {p} vs {p_serial}"
