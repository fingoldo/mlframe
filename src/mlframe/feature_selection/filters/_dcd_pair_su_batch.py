"""Layer 51 (2026-05-31): batched pairwise-SU sibling for ``_dynamic_cluster_discovery``.

Carved out of ``_dynamic_cluster_discovery.py`` (already > 1k LOC, per
memory rule) so the batched-entry path stays a focused module.

WHY
---
DCD computes pairwise SU one pair at a time via ``pair_su(state, a, b)``.
For p features there are O(p^2) potential pairs. With L47 tau-auto
calibration sampling 100 random pairs that's 100 individual function
calls; with the L48 hierarchy at 20 anchors that's 190 calls; the live
in-greedy-loop discover step touches even more. Each call pays:

- Python attribute / cache lookups on the shared ``DCDState`` object.
- An entropy() call per UNIQUE column appearing in the batch (already
  cached by ``state.column_entropy_cache`` from iter587, but the cache
  miss costs a merge_vars + entropy call).
- A merge_vars + entropy on the joint (genuinely per-pair).

``pair_su_batch`` shaves the marginal-entropy cost by sweeping all
UNIQUE column indices that appear in the requested pair list FIRST,
populating ``state.column_entropy_cache`` in one pass, then dispatching
through ``pair_su`` for the joint H(X_a, X_b) work. The per-pair
joint computation is genuinely unique per pair and runs through the
existing single-pair codepath -- which guarantees BIT-EQUIVALENCE
with the single-pair API (same merge_vars, same entropy, same FP
summation order, same SU formula).

Net win: the unique-column sweep avoids the merge_vars + entropy
per-column cost being paid multiple times when a column appears in
many pairs (which is the common case for tau-auto's ~100-pair sample
and the L48 hierarchy's anchors-cross-anchors O(K^2) sweep).

The function ALSO honours ``state.pairwise_su_cache``: pairs already
in the cache return their stored value instead of recomputing. This
lets a caller warm part of the cache via single calls then batch a
larger sweep with cache hits for the warm subset.

Contract:

  - Bit-equivalent to looped ``pair_su(state, a, b)`` for any pair list,
    any ``distance`` ('su' / 'vi' / 'auto' / 'sotoca_pla'), and any
    cache state.
  - Returns a 1-D float64 ndarray of length len(pair_indices).
  - Mutates ``state.pairwise_su_cache`` and ``state.column_entropy_cache``
    in-place (cache-warming is a feature: subsequent single-pair calls
    hit the cache).
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def pair_su_batch(
    state,
    pair_indices: Iterable,
    factors_data: Optional[np.ndarray] = None,
    factors_nbins: Optional[np.ndarray] = None,
    entropy_cache: Optional[dict] = None,
    dtype=np.int32,
) -> np.ndarray:
    """Compute SU(a, b) for every (a, b) in ``pair_indices`` via a
    batched marginal-entropy warmup + per-pair joint dispatch.

    Parameters
    ----------
    state
        ``DCDState`` instance owning the pairwise-SU cache + column
        entropy cache.
    pair_indices
        Iterable of (a, b) integer pairs. Order preserved in the output.
        Duplicate pairs are allowed; each returns the same cached value.
        Self-pairs ``a == b`` return ``1.0`` for ``distance in ('su',
        'vi', 'auto', 'sotoca_pla')`` (matches ``pair_su``).
    factors_data, factors_nbins, dtype
        Optional overrides for the state-owned data / bin counts /
        per-sample class-id dtype. Default to ``state.factors_data`` /
        ``state.factors_nbins`` / ``np.int32`` -- mirrors ``pair_su``.
    entropy_cache
        Forwarded to the single-pair ``pair_su`` dispatch (unused by
        the SU/VI branches that consume the per-column cache on state,
        accepted for symmetric API with ``pair_su``).

    Returns
    -------
    np.ndarray
        ``float64`` array of shape ``(len(pair_indices),)`` holding
        ``SU(a, b)`` for each input pair in the input order.
    """
    # Lazy-import to break the circular dep with the parent module.
    from ._dynamic_cluster_discovery import pair_su

    pairs_list = list(pair_indices)
    n_pairs = len(pairs_list)
    out = np.empty(n_pairs, dtype=np.float64)
    if n_pairs == 0:
        return out

    fd = factors_data if factors_data is not None else state.factors_data
    fn = factors_nbins if factors_nbins is not None else state.factors_nbins

    # Batched marginal-entropy warmup: collect the unique column indices
    # that participate in any pair AND are not already in
    # state.column_entropy_cache, then compute their marginal entropies
    # in one sweep. Subsequent per-pair dispatches hit the warm cache
    # for both marginals, leaving only the genuinely-unique joint
    # H(X_a, X_b) work to run per pair.
    if fd is not None and fn is not None:
        ec = state.column_entropy_cache
        needed: set = set()
        for ab in pairs_list:
            try:
                a, b = int(ab[0]), int(ab[1])
            except (TypeError, ValueError, IndexError):
                continue
            if a == b:
                continue
            if a not in ec:
                needed.add(a)
            if b not in ec:
                needed.add(b)
        if needed:
            # Re-use the same merge_vars / entropy primitives + the
            # state-owned fn_arr / pair_buf scratch buffers that
            # pair_su uses. Bit-equivalent: identical merge_vars call
            # shape (vars_indices=[col], var_is_nominal=None, fn_arr,
            # dtype) -> identical freqs -> identical entropy().
            try:
                from .info_theory import entropy, merge_vars
                fn_arr = state._fn_arr_cached
                if fn_arr is None:
                    fn_arr = np.asarray(fn, dtype=np.int64)
                    state._fn_arr_cached = fn_arr
                pair_buf = state._pair_idx_buf
                if pair_buf is None:
                    pair_buf = np.empty(2, dtype=np.int64)
                    state._pair_idx_buf = pair_buf
                for col_idx in needed:
                    pair_buf[0] = int(col_idx)
                    _, freqs_c, _ = merge_vars(
                        fd, pair_buf[:1], None, fn_arr, dtype=dtype,
                    )
                    ec[int(col_idx)] = float(entropy(freqs_c))
            except Exception:  # nosec B110 - non-trivial body; best-effort/optional path, no module logger
                # Defensive: if the warmup hits an unexpected shape
                # (caller-supplied fd with wrong dtype, etc), drop the
                # warmup and let pair_su handle each pair fresh. The
                # final SU values are still bit-equivalent because the
                # warmup only POPULATES the cache; pair_su would have
                # populated the same entries on miss.
                pass

    # Per-pair joint dispatch. ``pair_su`` honours the same cache the
    # warmup populated, so this is the bit-equivalent fall-through.
    for i, ab in enumerate(pairs_list):
        try:
            a, b = int(ab[0]), int(ab[1])
        except (TypeError, ValueError, IndexError):
            out[i] = 0.0
            continue
        out[i] = float(pair_su(
            state, a, b,
            entropy_cache=entropy_cache,
            factors_data=fd, factors_nbins=fn,
            dtype=dtype,
        ))
    return out


__all__ = ["pair_su_batch"]
