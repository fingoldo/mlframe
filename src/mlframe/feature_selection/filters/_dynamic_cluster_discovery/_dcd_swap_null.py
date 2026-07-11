"""Parallel permutation-null kernels for the DCD member-swap gate (carved from ``_dcd_swap.py``).

The member permutation null in ``evaluate_swap_candidate`` draws ``B`` shuffles of ONE cluster-member
column and, for each, recomputes ``I(member ; y | Selected - anchor)`` under the null. Pre-carve this ran
SERIALLY (~1 core): each draw mutated ``state.factors_data[:, member_idx]`` in place, called ``conditional_mi``,
and restored the column in a ``finally`` (the mutate-and-restore pattern that avoids copying the 100 GB frame).
The B-loop is the bottleneck at production ``B`` (default ``swap_npermutations=199``).

This module parallelizes the B-loop across cores WITHOUT copying the frame:

  1. **Serial, cheap:** pre-generate all ``B`` shuffles of ONLY the member column into a ``(B, n)`` int64
     array using the SAME ``rng_m`` sequence the serial path used -- so the permutation multiset (and thus
     the p-value) is bit-identical. ``B * n * 8`` bytes (e.g. 199 x 30k x 8 = 47 MB), NOT a frame copy.
  2. **Parallel, expensive:** ``prange`` over the ``B`` shuffles, each iteration computing the per-draw
     conditional MI from a THREAD-LOCAL shuffled column against the precomputed ``(Z)`` and ``(Y,Z)`` class
     labelings. No iteration touches ``state.factors_data`` (Z and Y are read from it read-only, the shuffled
     X is passed directly), so the parallel path needs no mutate-restore at all.

The per-draw CMI is ``H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z)``. ``H(Z)`` and ``H(Y,Z)`` are permutation-invariant
(only X is shuffled) and hoisted once. ``H(X,Y,Z)`` melts the shuffled X onto the precomputed ``(Y,Z)`` class
labels -- BIT-IDENTICAL to the serial ``_entropy_x_onto_classes`` path. ``H(X,Z)`` melts the shuffled X onto
the precomputed ``Z`` class labels; the serial path instead re-merges the sorted ``X u Z`` union, so the two
sum the SAME nonzero-frequency multiset in a different order -> a ~1e-15 FP reduction-order delta that cannot
move the ``null_rel >= member_rel`` count (the p-value is unchanged; validated in the regression test).

Selection contract (CLAUDE.md FE/MRMR bar): the member-null p-value ``(n_exceed + 1) / (B + 1)`` and hence the
swap accept/reject decision at ``swap_alpha`` are unchanged. Because the shuffles are pre-generated with the
same rng, the p-value is bit-identical in practice, exceeding the (looser) selection-equivalence requirement.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from numba import njit, prange

from ..info_theory import conditional_mi, entropy, merge_vars, mi
from .._numba_utils import unpack_and_sort

if TYPE_CHECKING:
    from . import DCDState

# Below this B the njit(parallel) spawn (~50us + per-thread JIT dispatch) is not worth it; run serial.
_PARALLEL_MIN_B = 8


@njit(cache=True)
def _entropy_col_onto_classes(x_col, nb_x, base_classes, base_nclasses, n_rows) -> float:
    """H of the joint (X, base) where ``base`` is a dense 0..base_nclasses-1 labeling of the conditioning
    variables. Histograms ``base_classes[row] + x_col[row] * base_nclasses`` once, prunes empty bins, and
    reduces via the shared ``entropy`` (numpy ``.sum`` reduction) -- BIT-IDENTICAL to the serial
    ``_entropy_x_onto_classes`` when ``base`` is the (Y,Z) labeling."""
    expected = base_nclasses * nb_x
    freqs = np.zeros(expected, dtype=np.int64)
    for row in range(n_rows):
        freqs[base_classes[row] + x_col[row] * base_nclasses] += 1
    nz = freqs[freqs > 0]
    return float(entropy(nz / n_rows))


@njit(cache=True)
def conditional_mi_col(
    x_col: np.ndarray,
    nb_x: int,
    z_classes: np.ndarray,
    z_nclasses: int,
    yz_classes: np.ndarray,
    yz_nclasses: int,
    entropy_z: float,
    entropy_yz: float,
) -> float:
    """``I(X; Y | Z)`` for a STANDALONE candidate column ``x_col`` (not a factors_data index), given the
    precomputed dense class labelings of ``Z`` and ``(Y,Z)`` and their (permutation-invariant) entropies.

    Substitutes ``x_col`` for the single X column the shared ``conditional_mi`` would read from
    ``factors_data[:, x_idx]``; the Y and Z columns are already folded into ``z_classes`` / ``yz_classes``.
    ``H(X,Y,Z)`` is bit-identical to the serial path; ``H(X,Z)`` differs only by FP reduction order (see the
    module docstring)."""
    n = x_col.shape[0]
    h_xz = _entropy_col_onto_classes(x_col, nb_x, z_classes, z_nclasses, n)
    h_xyz = _entropy_col_onto_classes(x_col, nb_x, yz_classes, yz_nclasses, n)
    res = h_xz + entropy_yz - entropy_z - h_xyz
    if res < 0.0:
        res = 0.0
    return float(res)


@njit(cache=True)
def mi_col(x_col: np.ndarray, nb_x: int, y_classes: np.ndarray, y_nclasses: int, entropy_x: float, entropy_y: float) -> float:
    """``I(X; Y)`` for a standalone ``x_col`` with precomputed ``Y`` labeling. ``H(X)`` (permutation-invariant)
    and ``H(Y)`` are hoisted; only ``H(X,Y)`` is recomputed per draw."""
    n = x_col.shape[0]
    h_xy = _entropy_col_onto_classes(x_col, nb_x, y_classes, y_nclasses, n)
    res = entropy_x + entropy_y - h_xy
    if res < 0.0:
        res = 0.0
    return float(res)


@njit(parallel=True, cache=True)
def _member_null_cmi_prange(shuffles, nb_x, z_classes, z_nclasses, yz_classes, yz_nclasses, entropy_z, entropy_yz, member_rel) -> int:
    """prange over the ``B`` pre-generated shuffles: count draws whose conditional MI meets/exceeds the
    observed ``member_rel``. Each iteration reads its own row of ``shuffles`` and allocates its own histogram
    (thread-local), so there is no shared mutable state -- no frame copy, no mutate-restore."""
    B = shuffles.shape[0]
    exceed = np.zeros(B, dtype=np.int64)
    for b in prange(B):
        val = conditional_mi_col(shuffles[b], nb_x, z_classes, z_nclasses, yz_classes, yz_nclasses, entropy_z, entropy_yz)
        if val >= member_rel:
            exceed[b] = 1
    return int(exceed.sum())


@njit(parallel=True, cache=True)
def _member_null_mi_prange(shuffles, nb_x, y_classes, y_nclasses, entropy_x, entropy_y, member_rel) -> int:
    """prange no-Z variant: count draws whose (unconditional) MI meets/exceeds ``member_rel``."""
    B = shuffles.shape[0]
    exceed = np.zeros(B, dtype=np.int64)
    for b in prange(B):
        val = mi_col(shuffles[b], nb_x, y_classes, y_nclasses, entropy_x, entropy_y)
        if val >= member_rel:
            exceed[b] = 1
    return int(exceed.sum())


def _run_member_null_serial(*, state, member_idx, member_rel, B_, rng_m, target_arr_m, S_minus_anchor, entropy_z, entropy_yz, member_col_orig, logger) -> float:
    """Exact legacy serial mutate-and-restore path -- fallback for tiny ``B`` where prange spawn is not worth
    it. Preserves the try/except fail-closed (return 1.0) + finally-restore semantics."""
    try:
        n_exceed_m = 0
        for _ in range(B_):
            shuffled = member_col_orig.copy()
            rng_m.shuffle(shuffled)
            state.factors_data[:, member_idx] = shuffled
            if len(S_minus_anchor) > 0:
                null_rel_m = float(
                    conditional_mi(
                        factors_data=state.factors_data,
                        x=np.array([member_idx], dtype=np.int64),
                        y=target_arr_m,
                        z=np.array(S_minus_anchor, dtype=np.int64),
                        var_is_nominal=None,
                        factors_nbins=state.factors_nbins,
                        entropy_z=entropy_z,
                        entropy_yz=entropy_yz,
                        entropy_cache=None,
                        can_use_x_cache=False,
                        can_use_y_cache=False,
                    )
                )
            else:
                null_rel_m = float(
                    mi(
                        state.factors_data,
                        np.array([member_idx], dtype=np.int64),
                        target_arr_m,
                        state.factors_nbins,
                    )
                )
            if null_rel_m >= member_rel:
                n_exceed_m += 1
        return float((n_exceed_m + 1) / (B_ + 1))
    except Exception as exc:
        logger.warning("DCD swap: member permutation null (serial) failed (B=%s): %r", B_, exc)
        return 1.0
    finally:
        state.factors_data[:, member_idx] = member_col_orig


def run_member_null(
    *,
    state: "DCDState",
    member_idx: int,
    member_rel: float,
    B_: int,
    anchor: int,
    target: np.ndarray,
    S_minus_anchor: Sequence[int],
    logger: Optional[logging.Logger] = None,
) -> float:
    """Member permutation-null p-value ``(n_exceed + 1) / (B + 1)``, parallelized across cores for
    production ``B`` and falling back to the exact serial mutate-restore path for tiny ``B``.

    The rng seeding + rolling ``state._perm_seed`` bump are byte-identical to the pre-carve inline path so the
    permutation sequence (and p-value) is reproducible and unchanged. Fail-closed to ``1.0`` on any error.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    if B_ <= 0:
        return 0.0

    member_col_orig = state.factors_data[:, member_idx].copy()
    # Preserve the EXACT legacy seeding + rolling-seed bump (reproducibility contract).
    base_seed = int(getattr(state, "_perm_seed", 0))
    rng_m = np.random.default_rng(base_seed + int(anchor) * 7919 + int(member_idx))
    state._perm_seed = base_seed + B_ + 1
    target_arr_m = np.asarray(target, dtype=np.int64)

    # Hoist the permutation-invariant H(Z) + H(Y,Z) (and the dense class labelings the parallel kernel melts
    # X onto) ONCE. Only the member column is shuffled, so Y/Z stay fixed across all B draws.
    entropy_z = -1.0
    entropy_yz = -1.0
    has_z = len(S_minus_anchor) > 0
    try:
        fnbins = np.asarray(state.factors_nbins, dtype=np.int64)
        if has_z:
            z_arr = np.sort(np.array(S_minus_anchor, dtype=np.int64))
            z_classes, fz, z_nclasses = merge_vars(state.factors_data, z_arr, None, fnbins)
            entropy_z = float(entropy(fz))
            yz_arr = unpack_and_sort(target_arr_m, z_arr)
            yz_classes, fyz, yz_nclasses = merge_vars(state.factors_data, yz_arr, None, fnbins)
            entropy_yz = float(entropy(fyz))
        else:
            # No-Z: I(X;Y). H(X) is permutation-invariant (shuffle preserves the marginal) -> hoist it too.
            _, fx, _ = merge_vars(state.factors_data, np.array([member_idx], dtype=np.int64), None, fnbins)
            entropy_x = float(entropy(fx))
            y_classes, fy, y_nclasses = merge_vars(state.factors_data, np.sort(target_arr_m), None, fnbins)
            entropy_y = float(entropy(fy))
    except Exception as exc:
        logger.warning("DCD swap: member-null hoist failed (B=%s): %r", B_, exc)
        # Restore is unnecessary (no mutation yet) but harmless; fall through to serial which handles its own.
        return _run_member_null_serial(
            state=state, member_idx=member_idx, member_rel=member_rel, B_=B_, rng_m=rng_m,
            target_arr_m=target_arr_m, S_minus_anchor=S_minus_anchor, entropy_z=entropy_z,
            entropy_yz=entropy_yz, member_col_orig=member_col_orig, logger=logger,
        )

    # Tiny B: prange spawn not worth it -> exact serial path (uses the same rng_m, so identical draws).
    if B_ < _PARALLEL_MIN_B:
        return _run_member_null_serial(
            state=state, member_idx=member_idx, member_rel=member_rel, B_=B_, rng_m=rng_m,
            target_arr_m=target_arr_m, S_minus_anchor=S_minus_anchor, entropy_z=entropy_z,
            entropy_yz=entropy_yz, member_col_orig=member_col_orig, logger=logger,
        )

    try:
        n = member_col_orig.shape[0]
        # Pre-generate all B shuffles SERIALLY with the SAME rng_m as the serial path (bit-identical multiset).
        base_col = member_col_orig.astype(np.int64)
        shuffles = np.empty((B_, n), dtype=np.int64)
        for b in range(B_):
            s = base_col.copy()
            rng_m.shuffle(s)
            shuffles[b] = s
        nb_x = int(fnbins[member_idx])
        if has_z:
            n_exceed = _member_null_cmi_prange(
                shuffles, nb_x, z_classes.astype(np.int64), int(z_nclasses),
                yz_classes.astype(np.int64), int(yz_nclasses), entropy_z, entropy_yz, float(member_rel),
            )
        else:
            n_exceed = _member_null_mi_prange(
                shuffles, nb_x, y_classes.astype(np.int64), int(y_nclasses),
                entropy_x, entropy_y, float(member_rel),
            )
        return float((n_exceed + 1) / (B_ + 1))
    except Exception as exc:
        logger.warning("DCD swap: member permutation null (parallel) failed (B=%s): %r", B_, exc)
        return 1.0  # conservative: fail closed
