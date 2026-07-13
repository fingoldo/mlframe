"""Batched relevance kernel for the DCD anchor-refinement cluster-member ranking loop.

``evaluate_swap_candidate``'s anchor-refinement step (``_dcd_swap.py``) scores every cluster member's
``I(member; y | Selected - anchor)`` (or ``I(member; y)`` when the conditioning set is empty, the
first-selected-feature edge case) ONE MEMBER AT A TIME via ``conditional_mi``/``mi``, bounded by
``max_cluster_size`` (default 12). ``info_theory._cmi_cuda._cpu_cmi_loop_parallel`` already exists and is
SAFE to reuse for the Z-conditioned case (it forwards an arbitrary-width ``z`` straight into
``conditional_mi``, unlike the public ``conditional_mi_batched_dispatch``/``_cpu_cmi_loop`` dispatchers,
whose default Y,Z-entropy-hoist fast path assumes Z is a SINGLE column via ``z[0]`` -- verified by reading
``_cmi_yz_fixed_terms``, would silently compute the WRONG H(Y,Z) for a multi-column ``S_minus_anchor``).
This module supplies the missing no-Z sibling (:func:`_mi_loop_parallel`) so both branches of the loop can
batch.

The pre-fix loop also threaded a ``entropy_cache`` dict through ``conditional_mi`` for cross-call H(Z)/H(Y,Z)
memoisation; the ONE real call site (``_fit_impl_core.py``) always passes ``entropy_cache=None`` (verified by
reading the call site), so batching loses nothing here -- there was never a live cache to lose.
"""
from __future__ import annotations

import numba
import numpy as np
from numba import prange

from ..info_theory import mi
from ..info_theory._cmi_cuda import _cpu_cmi_loop_parallel

__all__ = ["batched_member_relevance"]


@numba.njit(cache=True, parallel=True)
def _mi_loop_parallel(factors_data: np.ndarray, cand_indices: np.ndarray, y: np.ndarray, factors_nbins: np.ndarray) -> np.ndarray:
    """``prange`` unconditional MI over candidates -- the no-Z sibling of ``_cpu_cmi_loop_parallel``, same
    shape (one independent ``mi(X_j; Y)`` reduction per candidate, fanned across cores)."""
    p = cand_indices.shape[0]
    out = np.empty(p, dtype=np.float64)
    for i in prange(p):
        xi = np.empty(1, dtype=np.int64)
        xi[0] = cand_indices[i]
        out[i] = mi(factors_data, xi, y, factors_nbins)
    return out


def batched_member_relevance(
    factors_data: np.ndarray, cand_indices: np.ndarray, target_arr: np.ndarray,
    s_minus_anchor: list, factors_nbins: np.ndarray,
) -> np.ndarray:
    """Relevance of every candidate in ``cand_indices`` against ``target_arr``, conditioned on
    ``s_minus_anchor`` when non-empty -- one batched, parallel call replacing the per-member Python loop.
    Returns ``(len(cand_indices),)`` float64, same order as ``cand_indices``.
    """
    cand = np.asarray(cand_indices, dtype=np.int64)
    if s_minus_anchor:
        vin = np.empty(0, dtype=np.int64)
        return np.asarray(_cpu_cmi_loop_parallel(
            factors_data, cand, target_arr, np.array(s_minus_anchor, dtype=np.int64), factors_nbins, vin,
        ))
    return np.asarray(_mi_loop_parallel(factors_data, cand, target_arr, np.asarray(factors_nbins, dtype=np.int64)))
