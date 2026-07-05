"""DCD pairwise metrics + pruning predicates (carved from ``_dynamic_cluster_discovery.py``).

Leaf module: MI-based pairwise symmetric uncertainty / variation of information, the binary
aggregate helper, and the multi-order ``should_be_pruned`` predicate. The info-theory kernels
(``mi``/``entropy``/``merge_vars``/``symmetric_uncertainty``) are lazy-imported in-body as in the
original, so this module imports no DCD parent symbol.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from . import DCDState

logger = logging.getLogger(__name__)


def pair_su(state: DCDState, a: int, b: int, entropy_cache: Optional[dict] = None, factors_data=None, factors_nbins=None, dtype=np.int32) -> float:
    """Cached symmetric-uncertainty between columns ``a`` and ``b``.

    Returns SU(X_a, X_b) in [0, 1] under ``distance='su'`` (default). For
    ``distance='vi'`` returns ``1 - VI/log(2*n_bins)`` to remain bounded;
    for ``'sotoca_pla'`` returns the target-aware distance from the
    Sotoca-Pla 2010 formula. For Layer 46 ``distance='auto'`` returns
    ``max(SU, VI_sim)`` per pair — picks up both linear-friendly
    duplicates (SU strong) and non-linear functional equivalences
    (VI tighter), at the cost of one extra MI computation per pair.

    Cache key: ``(min(a,b), max(a,b))`` tuple (4× cheaper than frozenset
    per Critic2 finding).
    """
    if a == b:
        return 1.0
    key = (a, b) if a < b else (b, a)
    cache = state.pairwise_su_cache
    if key in cache:
        cache.move_to_end(key)
        state.n_cache_hits += 1
        return float(cache[key])
    fd = factors_data if factors_data is not None else state.factors_data
    fn = factors_nbins if factors_nbins is not None else state.factors_nbins
    if fd is None or fn is None:
        return 0.0
    state.n_cache_misses += 1
    state.n_su_calls += 1
    # Layer 46 (2026-05-31): ``"auto"`` runs the SU and VI branches
    # in sequence and reports the tighter redundancy score. The two
    # underlying scores share the per-column entropy cache + fn_arr /
    # pair_buf scratch buffers, so the extra work over a pure SU pair
    # is one extra mi() njit call + the final max(). The VI branch
    # already caches H(X_a), H(X_b); SU caches them too; the only
    # net-new work is the mi(a, b) joint computation.
    if state.distance == "auto":
        _orig_distance = state.distance
        # Roll back the counter increments we did above; the two
        # recursive calls below will re-increment them once each
        # (one pair counts as one auto-call worth of SU/MI work).
        state.n_cache_misses -= 1
        state.n_su_calls -= 1
        try:
            state.distance = "su"
            su_score = pair_su(
                state, a, b,
                entropy_cache=entropy_cache,
                factors_data=fd, factors_nbins=fn, dtype=dtype,
            )
            # Pop the SU score from the cache so the VI computation
            # below does not return the SU score on lookup; the final
            # max() will repopulate the cache under the same key.
            state.pairwise_su_cache.pop(key, None)
            state.distance = "vi"
            vi_score = pair_su(
                state, a, b,
                entropy_cache=entropy_cache,
                factors_data=fd, factors_nbins=fn, dtype=dtype,
            )
            state.pairwise_su_cache.pop(key, None)
        finally:
            state.distance = _orig_distance
        su_combined = float(max(su_score, vi_score))
        cache[key] = su_combined
        state.cache_evict_lru()
        return su_combined
    if state.distance == "su":
        # 2026-06-03 bench-attempt-rejected (bench_miller_madow_su_edges): wiring
        # a Miller-Madow entropy correction into this SU edge gives NO actionable
        # win. At DCD's binning cap (~10 bins) and realistic n, independent
        # high-cardinality pairs already score SU ~0.01-0.03 (far below tau), so
        # there is no plug-in over-clustering to correct; MM only shaves ~0.01
        # off true-redundancy SU. Plug-in entropy stays.
        # iter587: avoid the 3-merge_vars cost of calling
        # ``symmetric_uncertainty(x=[a], y=[b], ...)`` for every pair.
        # H(X_a) and H(X_b) are functions of a single column each, so the
        # marginal merge_vars + entropy work is identical for every pair
        # containing that column. Cache per-column entropies in state
        # (lazy, populated on miss); only the joint H(X_a, X_b) is
        # genuinely pair-unique and runs every call. Net work per pair:
        # 3 merge_vars + 3 entropy --> 1 merge_vars + 1 entropy + 2 dict
        # lookups (cold) or 0 + 0 + 2 (warm), preserving bit-equivalent
        # SU since both formulations compute the same H(X_a) + H(X_b) +
        # H(X_a, X_b) -> 2*(H_a + H_b - H_ab)/(H_a + H_b).
        from ..info_theory import entropy, merge_vars, joint_entropy_2var
        # iter589: cache the int64 view of factors_nbins on state so it
        # is not re-asarray'd every call. ``fn`` is typically already
        # int64 but np.asarray still pays a dispatch.
        fn_arr = state._fn_arr_cached
        if fn_arr is None:
            fn_arr = np.asarray(fn, dtype=np.int64)
            state._fn_arr_cached = fn_arr
        # iter589: reuse a 2-element int64 scratch buffer instead of
        # allocating ``np.array([a, b], dtype=np.int64)`` every call.
        # merge_vars takes vars_indices by-value (its njit body iterates
        # the contents and accumulates classes in a separate output),
        # so mutating the buffer between successive calls is safe.
        pair_buf = state._pair_idx_buf
        if pair_buf is None:
            pair_buf = np.empty(2, dtype=np.int64)
            state._pair_idx_buf = pair_buf
        ec = state.column_entropy_cache
        h_a = ec.get(a)
        if h_a is None:
            pair_buf[0] = a
            _, freqs_a, _ = merge_vars(
                fd, pair_buf[:1], None, fn_arr, dtype=dtype,
            )
            h_a = float(entropy(freqs_a))
            ec[a] = h_a
        h_b = ec.get(b)
        if h_b is None:
            pair_buf[0] = b
            _, freqs_b, _ = merge_vars(
                fd, pair_buf[:1], None, fn_arr, dtype=dtype,
            )
            h_b = float(entropy(freqs_b))
            ec[b] = h_b
        # iter (2026-06-08): the joint H(X_a, X_b) is the ONLY genuinely per-pair term
        # (the marginals are state-cached above), so it runs on every cache-miss pair --
        # ~24k pairs at 600 rows / ~345k at full scene. ``merge_vars`` here builds a
        # length-n ``final_classes`` array + a lookup-table remap pass that this SU path
        # discards (it only consumes the pruned freqs for ``entropy``); the dedicated 2-var
        # ``joint_freqs_2var`` kernel returns the IDENTICAL normalized nonzero freqs without
        # that wasted alloc+remap, ~23x faster per pair at BIT-IDENTICAL numerics (verified
        # max-abs-diff 0.0 vs merge_vars across uniform/sparse/skew/const data, all n;
        # bench D:/Temp/microbench_pairsu2.py).
        # iter (2026-06-08 wasted-work sweep): the per-pair joint freqs array is consumed by
        # NOTHING but the very next ``entropy`` call -> ``joint_entropy_2var`` FUSES the
        # histogram->entropy reduction, dropping the intermediate normalized-freqs array, the
        # ``freqs[freqs > 0]`` mask, and entropy's ``log(freqs) * freqs`` temporary. ~1.24x
        # per pair, BIT-IDENTICAL (max-abs-diff 0.0 vs entropy(joint_freqs_2var(...)) over
        # 960 cases; bench D:/Temp/ww_micro_jointentropy.py, test_joint_entropy_2var.py).
        h_ab = float(joint_entropy_2var(fd, a, b, int(fn_arr[a]), int(fn_arr[b])))
        denom = h_a + h_b
        if denom <= 1e-12:
            su = 0.0
        else:
            su = 2.0 * (h_a + h_b - h_ab) / denom
    elif state.distance == "vi":
        # 2026-05-30 Wave 9.1 fix (loop iter 23): proper Variation of
        # Information distance. Pre-fix this branch was a silent alias
        # of ``"su"``: it called ``symmetric_uncertainty`` and returned
        # SU, ignoring the user's opt-in to VI semantics. The in-source
        # comment even confessed "Here we approximate via SU"; user got
        # zero of the documented VI behaviour.
        #
        # Implementation: VI(X, Y) = H(X) + H(Y) - 2 I(X; Y) = H(X, Y)
        # - I(X; Y). Normalised by ``log(K_x * K_y)`` -> stays in
        # [0, 1] (Meila 2007 normalization). Returned as a similarity
        # score ``1 - VI_norm`` so the membership rule
        # ``score > tau_cluster`` stays semantically aligned with the
        # SU branch ("higher = more redundant -> prune from pool").
        from ..info_theory import mi, entropy, merge_vars
        import math
        # iter590: same buffer-reuse + fn_arr cache as the iter589 SU
        # branch above. Replaces per-call np.array([a]) / np.array([b]) /
        # np.asarray(fn) allocations with the cached state-side buffers.
        fn_arr = state._fn_arr_cached
        if fn_arr is None:
            fn_arr = np.asarray(fn, dtype=np.int64)
            state._fn_arr_cached = fn_arr
        pair_buf = state._pair_idx_buf
        if pair_buf is None:
            pair_buf = np.empty(2, dtype=np.int64)
            state._pair_idx_buf = pair_buf
        # mi(fd, x, y, ...) takes 1-D vars arrays. Use two non-overlapping
        # 1-element views into the same 2-element pair_buf so x=pair_buf[0:1]
        # and y=pair_buf[1:2]; the mi njit body reads both views and never
        # mutates them, so aliasing into one backing buffer is safe.
        pair_buf[0] = a
        pair_buf[1] = b
        mi_ab = float(mi(fd, pair_buf[0:1], pair_buf[1:2], fn_arr, dtype=dtype))
        # iter587: same per-column entropy cache as the SU branch above --
        # H(X_a) / H(X_b) recomputation was the dominant per-call cost.
        ec = state.column_entropy_cache
        h_a = ec.get(a)
        if h_a is None:
            pair_buf[0] = a
            _, freqs_a, _ = merge_vars(fd, pair_buf[:1], None, fn_arr, dtype=dtype)
            h_a = float(entropy(freqs_a))
            ec[a] = h_a
        h_b = ec.get(b)
        if h_b is None:
            pair_buf[0] = b
            _, freqs_b, _ = merge_vars(fd, pair_buf[:1], None, fn_arr, dtype=dtype)
            h_b = float(entropy(freqs_b))
            ec[b] = h_b
        vi = max(0.0, h_a + h_b - 2.0 * mi_ab)
        norm = math.log(max(2, int(fn_arr[a]) * int(fn_arr[b])))
        # 2026-05-30 Wave 9.1 fix (loop iter 32, parity with sotoca_pla):
        # clamp upper bound to 1.0 defensively. Numerical FP noise can
        # push vi/norm below 0, allowing the score to exceed 1.0 even
        # though the analytical bound holds.
        su = max(0.0, min(1.0, 1.0 - vi / norm)) if norm > 0 else 0.0
    elif state.distance == "sotoca_pla":
        # 2026-06-03 (audit integration-defaults-1): LEAKAGE FIREWALL.
        # The Sotoca-Pla 2010 distance is TARGET-AWARE -- it reads I(X_i; Y)
        # and I(X_j; Y):
        #   d(X_i, X_j) = 2 H(X_i, X_j) - I(X_i; X_j) - I(X_i; Y) - I(X_j; Y)
        # But ``pair_su`` is used ONLY to decide UNSUPERVISED cluster membership
        # / pool pruning (discover_cluster_members, tau auto-calibration, the
        # hierarchy analyser). Letting a y-aware score choose which candidates
        # get pruned lets the target decide the feature support -- and a pruned
        # candidate never gets a later y-aware accept/reject re-screen, so this
        # is exactly the leak the rest of the clustering machinery forbids
        # (_cluster_aggregate.py: "Aggregation is UNSUPERVISED; only the
        # accept/reject GATE sees y"). So for membership we ALWAYS fall back to
        # the unsupervised symmetric uncertainty here, regardless of target
        # availability. A legitimate target-aware comparison belongs in the
        # swap accept-gate (evaluate_swap_candidate), which is permitted to see
        # y; it does not route through this distance.
        if not getattr(state, "_warned_sotoca_membership", False):
            logger.warning(
                "DCD distance 'sotoca_pla' is target-aware and cannot drive "
                "unsupervised cluster membership without leaking y into the "
                "feature support; using symmetric uncertainty (SU) for pruning "
                "instead. Set dcd_distance to 'su'/'vi'/'auto' to silence this."
            )
            state._warned_sotoca_membership = True
        from ..info_theory import symmetric_uncertainty
        # iter616 caching pattern (fn_arr + pair_buf scratch) retained.
        fn_arr = state._fn_arr_cached
        if fn_arr is None:
            fn_arr = np.asarray(fn, dtype=np.int64)
            state._fn_arr_cached = fn_arr
        pair_buf = state._pair_idx_buf
        if pair_buf is None:
            pair_buf = np.empty(2, dtype=np.int64)
            state._pair_idx_buf = pair_buf
        pair_buf[0] = a
        pair_buf[1] = b
        su = float(symmetric_uncertainty(
            fd, pair_buf[0:1], pair_buf[1:2], fn_arr, dtype=dtype,
        ))
    else:
        raise ValueError(f"DCD: unknown distance {state.distance!r}")
    cache[key] = su
    state.cache_evict_lru()
    return float(su)


def pair_vi(state: DCDState, a: int, b: int, factors_data=None, factors_nbins=None, dtype=np.int32) -> float:
    """Variation-of-Information distance between columns ``a`` and ``b``
    in nats: ``VI(X_a, X_b) = H(X_a) + H(X_b) - 2 I(X_a; X_b)``.

    VI is a proper metric (Meila 2007): non-negative, symmetric, zero
    iff X_a and X_b are functionally equivalent, and satisfies the
    triangle inequality. The ``"vi"`` and ``"auto"`` branches of
    ``pair_su`` consume the SAME H_a, H_b and I_ab via the per-column
    entropy cache, then normalise to a [0,1] similarity score.

    This helper returns raw VI in nats for users who need the metric
    interpretation (cluster cohesion plots, comparison with SU).
    Reuses ``state.column_entropy_cache`` and the ``_pair_idx_buf`` /
    ``_fn_arr_cached`` scratch buffers populated by ``pair_su``.
    """
    if a == b:
        return 0.0
    fd = factors_data if factors_data is not None else state.factors_data
    fn = factors_nbins if factors_nbins is not None else state.factors_nbins
    if fd is None or fn is None:
        return 0.0
    from ..info_theory import mi, entropy, merge_vars
    fn_arr = state._fn_arr_cached
    if fn_arr is None:
        fn_arr = np.asarray(fn, dtype=np.int64)
        state._fn_arr_cached = fn_arr
    pair_buf = state._pair_idx_buf
    if pair_buf is None:
        pair_buf = np.empty(2, dtype=np.int64)
        state._pair_idx_buf = pair_buf
    ec = state.column_entropy_cache
    h_a = ec.get(a)
    if h_a is None:
        pair_buf[0] = a
        _, freqs_a, _ = merge_vars(fd, pair_buf[:1], None, fn_arr, dtype=dtype)
        h_a = float(entropy(freqs_a))
        ec[a] = h_a
    h_b = ec.get(b)
    if h_b is None:
        pair_buf[0] = b
        _, freqs_b, _ = merge_vars(fd, pair_buf[:1], None, fn_arr, dtype=dtype)
        h_b = float(entropy(freqs_b))
        ec[b] = h_b
    pair_buf[0] = a
    pair_buf[1] = b
    mi_ab = float(mi(fd, pair_buf[0:1], pair_buf[1:2], fn_arr, dtype=dtype))
    return max(0.0, h_a + h_b - 2.0 * mi_ab)

def should_be_pruned(state: DCDState, candidate) -> bool:
    """Return True iff ``candidate`` (int column index OR tuple of indices
    for higher-order interactions) is DCD-pruned.

    For tuples (interactions_order >= 2), returns True iff ALL components
    are pruned (Critic1/B-3 fix). A partially-pruned tuple stays scoreable.
    """
    if state is None or state.pool_pruned_mask is None:
        return False
    mask = state.pool_pruned_mask
    n_cols = mask.shape[0]
    if isinstance(candidate, (int, np.integer)):
        idx = int(candidate)
        return 0 <= idx < n_cols and bool(mask[idx])
    # Iterable / tuple / array of indices.
    try:
        components = list(candidate)
    except TypeError:
        return False
    if not components:
        return False
    for comp in components:
        idx = int(comp)
        if not (0 <= idx < n_cols and bool(mask[idx])):
            return False
    return True


def _binarize_aggregate(values: np.ndarray, *, method: str, n_bins: int, dtype) -> np.ndarray:
    """Discretise a continuous aggregate to integer bin codes matching the
    MRMR quantization style. Conservative fallback: uniform binning when
    quantile produces degenerate edges."""
    values = np.asarray(values, dtype=np.float64).ravel()
    n_bins = max(2, int(n_bins))
    if values.size == 0:
        return np.zeros(0, dtype=dtype)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=dtype)
    if method == "quantile":
        try:
            edges = np.quantile(finite, np.linspace(0, 1, n_bins + 1))
            edges = np.unique(edges)
            if edges.size < 3:
                raise ValueError("degenerate quantile edges")
            binned = np.searchsorted(edges[1:-1], values, side="right")
        except Exception:
            mn, mx = float(finite.min()), float(finite.max())
            if mx <= mn:
                return np.zeros(values.shape, dtype=dtype)
            binned = np.clip(((values - mn) / (mx - mn) * n_bins).astype(np.int64), 0, n_bins - 1)
    else:
        mn, mx = float(finite.min()), float(finite.max())
        if mx <= mn:
            return np.zeros(values.shape, dtype=dtype)
        binned = np.clip(((values - mn) / (mx - mn) * n_bins).astype(np.int64), 0, n_bins - 1)
    return binned.astype(dtype)
