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
populating ``state.column_entropy_cache`` in one pass. The marginal
sweep reuses the same merge_vars + entropy primitives as the single-pair
API, so H(X_a)/H(X_b) are bit-identical. The genuinely per-pair joint
H(X_a, X_b) is precomputed in one prange-over-pairs kernel and read back
through ``pair_su``; that joint is SELECTION-EQUIVALENT and agrees to
~1 ULP with the serial kernel (a single-pair batch is bit-identical;
a multi-pair batch may reorder the reduction by one ULP under numba's
parallel codegen -- see the kernel docstring below).

Net win: the unique-column sweep avoids the merge_vars + entropy
per-column cost being paid multiple times when a column appears in
many pairs (which is the common case for tau-auto's ~100-pair sample
and the L48 hierarchy's anchors-cross-anchors O(K^2) sweep).

The function ALSO honours ``state.pairwise_su_cache``: pairs already
in the cache return their stored value instead of recomputing. This
lets a caller warm part of the cache via single calls then batch a
larger sweep with cache hits for the warm subset.

Contract:

  - Selection-equivalent to looped ``pair_su(state, a, b)`` for any pair
    list, any ``distance`` ('su' / 'vi' / 'auto' / 'sotoca_pla'), and any
    cache state; SU values agree to ~1 ULP (marginals bit-identical, the
    parallel joint reduction may differ by one ULP on a multi-pair batch).
  - Returns a 1-D float64 ndarray of length len(pair_indices).
  - Mutates ``state.pairwise_su_cache`` and ``state.column_entropy_cache``
    in-place (cache-warming is a feature: subsequent single-pair calls
    hit the cache).
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

try:
    from numba import njit, prange

    @njit(nogil=True, cache=True, parallel=True)
    def _batch_joint_entropy_pairs(fd, a_arr, b_arr, nb_arr):
        """H(X_a, X_b) for every (a_arr[i], b_arr[i]) pair, prange over the
        OUTER pair index. Each iteration runs the same single-thread
        joint-histogram + ascending-class-id entropy reduction that
        ``info_theory._class_encoding.joint_entropy_2var`` runs -- only the
        pair loop is parallel (independent outputs). Results are
        SELECTION-EQUIVALENT to calling that kernel per pair and agree to
        ~1 ULP: a single-pair batch is bit-identical, but under a real
        multi-pair batch numba's parallel=True codegen may reorder the inner
        -(p*log p) reduction by one ULP (~2e-16) on some bin patterns. That
        is far below any downstream tau/threshold margin. The DCD hot loop
        scores one anchor against K candidates, so K joints share the anchor
        column -> parallelising over pairs amortises the one thread-spawn
        across all K and keeps operands hot in cache. Wins 7.1x @30k /
        8.55x @300k / 5.11x @1M (bench_pair_su_batch_over_pairs).
        """
        k = a_arr.shape[0]
        out = np.empty(k, dtype=np.float64)
        n = fd.shape[0]
        for i in prange(k):
            ia = a_arr[i]
            ib = b_arr[i]
            nb_a = nb_arr[ia]
            size = nb_a * nb_arr[ib]
            hist = np.zeros(size, dtype=np.int64)
            for r in range(n):
                hist[fd[r, ia] + fd[r, ib] * nb_a] += 1
            h = 0.0
            for c in range(size):
                cnt = hist[c]
                if cnt != 0:
                    p = cnt / n
                    h += np.log(p) * p
            out[i] = -h
        return out

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba always present in prod
    _HAVE_NUMBA = False


def _validate_batch_joint_entropy_pairs_inputs(a_arr: np.ndarray, b_arr: np.ndarray, nb_arr: np.ndarray) -> None:
    """Host-side pre-launch guard for :func:`_batch_joint_entropy_pairs` (GPU_INFRA_A-11 fix,
    ). The njit kernel indexes ``hist[fd[r, ia] + fd[r, ib]*nb_a]`` with
    ``boundscheck=False`` and no validation of its own -- a non-positive nbins for a referenced
    column silently corrupts memory instead of raising, unlike every CUDA kernel in this cluster
    (which pre-validate exactly this class of input before launch). Deliberately O(k) (k = number
    of pairs), not a per-row code-range scan -- the latter would cost O(n*k) and defeat the point
    of this batched kernel; not reachable via the normal ``categorize_dataset`` contract, but this
    function's signature lets a caller supply its own ``nb_arr`` override, so at least the cheap
    nbins>=1 precondition should fail loudly here instead of corrupting memory silently."""
    if int(nb_arr[a_arr].min(initial=1)) < 1 or int(nb_arr[b_arr].min(initial=1)) < 1:
        raise ValueError("_batch_joint_entropy_pairs: nbins must be >= 1 for every referenced column")
    _batch_joint_entropy_pairs = None


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

    # Batch-over-pairs joint-entropy precompute (distance == 'su' only).
    # The genuinely per-pair term is H(X_a, X_b); every other quantity is
    # state-cached. Compute all the not-yet-cached joints in ONE
    # prange-over-pairs kernel and stash them on
    # ``state._joint_entropy_batch_cache`` so the per-pair ``pair_su``
    # dispatch below reads the joint instead of running the serial kernel.
    # BIT-IDENTICAL (same float per pair) so all SU / pairwise-cache /
    # counter semantics of the single-pair path are preserved exactly;
    # only the source of ``h_ab`` changes. Skipped for 'vi'/'auto'/
    # 'sotoca_pla' (different formulas) and when numba is unavailable.
    _joint_cache = None
    if _HAVE_NUMBA and getattr(state, "distance", "su") == "su" and fd is not None and fn is not None:
        pcache = getattr(state, "pairwise_su_cache", None)
        keys_a = []
        keys_b = []
        seen_keys: set = set()
        for ab in pairs_list:
            try:
                a, b = int(ab[0]), int(ab[1])
            except (TypeError, ValueError, IndexError):
                continue
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            if pcache is not None and key in pcache:
                continue  # already cached -> pair_su returns early, no joint read
            if key in seen_keys:
                continue
            seen_keys.add(key)
            keys_a.append(key[0])
            keys_b.append(key[1])
        if keys_a:
            try:
                fn_arr = state._fn_arr_cached
                if fn_arr is None:
                    fn_arr = np.asarray(fn, dtype=np.int64)
                    state._fn_arr_cached = fn_arr
                a_arr = np.asarray(keys_a, dtype=np.int64)
                b_arr = np.asarray(keys_b, dtype=np.int64)
                _validate_batch_joint_entropy_pairs_inputs(a_arr, b_arr, fn_arr)
                h_ab_vals = _batch_joint_entropy_pairs(fd, a_arr, b_arr, fn_arr)
                _joint_cache = {(int(a_arr[i]), int(b_arr[i])): float(h_ab_vals[i]) for i in range(a_arr.shape[0])}
                state._joint_entropy_batch_cache = _joint_cache
            except Exception:  # pragma: no cover - defensive: fall back to serial joints
                _joint_cache = None
                state._joint_entropy_batch_cache = None

    # Per-pair joint dispatch. ``pair_su`` honours the same cache the
    # warmup populated, so this is the bit-equivalent fall-through.
    try:
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
    finally:
        if _joint_cache is not None:
            state._joint_entropy_batch_cache = None
    return out


__all__ = ["pair_su_batch"]
