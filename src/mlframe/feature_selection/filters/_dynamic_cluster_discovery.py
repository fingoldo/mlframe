"""Dynamic Cluster Discovery (DCD) — Wave 9 (2026-05-30).

Organic in-greedy-loop cluster discovery for MRMR. After each feature
acceptance, the Pool is pruned by ``SU(x, just_selected) > tau_cluster``
using ONLY MI-based distances (no Pearson — captures non-linear deps).
When a cluster accumulates ``cluster_size_threshold`` members, the raw
anchor can be swapped with a PC1 aggregate if the conditional relevance
``I(rep ; y | Selected − anchor) > anchor's * (1 + swap_gain_threshold)``.

Plan v2 (post-2-critic-review) addresses:
- B-1: pool_pruned_mask instead of mutating candidates list
- B-2: pre-confirmation swap with fresh permutation null on rep
- B-3: multi-order interactions — should_be_pruned(tuple) uses ALL components
- B-4: gate fires outside the ``if full_npermutations:`` branch
- H-4: PC1 swap uses conditional_mi against Selected−anchor (not unconditional)
- H-5: DCDState.X_raw_ref captured for PC1 aggregation
- F: dcd_config passed as kwargs (joblib-safe), not via thread-local state
- Critic2/A: symmetric_uncertainty signature respected (vars_indices arrays)
- Critic2/B: entropy_cache shared with ctx (no local dup)
- Tuple key for pairwise_su_cache (4x cheaper than frozenset)

Pre-impl gate (bench_dcd_pair_su_scaling) PASSED at all scales:
    p=100   B/A = 0.036 (19x better than 0.7 target)
    p=1000  B/A = 0.027 (26x better)
    p=10000 B/A = 0.003 (233x better)

References:
- Brown 2012 JMLR v13 (CMIM/JMI unifying framework)
- Bennasar 2015 ESWA 42(22) (JMIM redundancy)
- Sotoca-Pla 2010 PR 43(6):2068-2081 (CMI hierarchical clustering)
"""
from __future__ import annotations

import logging
import threading as _threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Thread-local toggle (matches Wave 8 SU/JMIM/BUR pattern at info_theory.py:282)
# =============================================================================


_DCD_STATE = _threading.local()


def use_dcd() -> bool:
    """Read-only check for the DCD thread-local activation flag.

    Set by ``MRMR.fit`` when ``dcd_enable=True``; read at the
    ``screen_predictors`` integration point to gate the ``if use_dcd():``
    branch. DCD STATE itself (the DCDState dataclass) is passed via
    ``screen_predictors`` kwargs — NOT through thread-local — so the
    joblib parallel-backend path serialises DCDState via pickling rather
    than relying on shared per-thread state.
    """
    return bool(getattr(_DCD_STATE, "active", False))


def set_dcd_active(active: bool) -> None:
    _DCD_STATE.active = bool(active)


# =============================================================================
# DCDState dataclass — single object holding all per-fit DCD bookkeeping
# =============================================================================


@dataclass
class SwapDecision:
    """Outcome of ``evaluate_swap_candidate`` — atomic information needed to
    perform a commit_swap if accepted."""
    accept: bool
    new_col_idx: int = -1
    aggregate_name: str = ""
    binned_rep: Optional[np.ndarray] = None
    new_nbins: int = 0
    recipe_obj: Any = None
    rep_relevance: float = 0.0
    anchor_relevance_in_ctx: float = 0.0
    perm_p_value: float = 1.0


@dataclass
class DCDState:
    """All DCD bookkeeping for a single MRMR.fit invocation.

    The state lives on ``ScreenContext.dcd`` and is mutated in-place by the
    discover / swap functions. Subscribers (``should_skip_candidate``,
    ``dcd_summary``, downstream FE step) consult it read-only.
    """
    # -- core cluster bookkeeping --
    cluster_anchors: dict = field(default_factory=dict)              # anchor_idx -> set[member_col]
    member_to_anchor: dict = field(default_factory=dict)             # member_col -> anchor_idx
    pairwise_su_cache: "OrderedDict" = field(default_factory=OrderedDict)  # (min,max) -> SU
    # iter587: per-column marginal-entropy cache for the SU / VI branches of
    # pair_su. Pre-fix, every pair_su(a, b) call recomputed H(X_a) and
    # H(X_b) via merge_vars + entropy -- 3 merge_vars per call when only
    # the joint H(X_a, X_b) is genuinely unique per (a, b). For 30 features
    # pairwise = 435 pairs and each feature appears in 29 pairs, so each
    # marginal entropy gets recomputed 29x. The c0066 @100k profile
    # attributed 0.58 s tottime / 243 calls (~2.4 ms each) to
    # ``symmetric_uncertainty``; ~2/3 of that is the redundant marginal
    # merge_vars + entropy. Caching by column index drops the dominant
    # share to a single lookup per (a, b).
    column_entropy_cache: dict = field(default_factory=dict)         # int -> float
    pool_pruned_mask: Optional[np.ndarray] = None                    # bool[p_initial]; True == pruned
    swap_log: list = field(default_factory=list)
    n_su_calls: int = 0
    n_cache_hits: int = 0
    n_cache_misses: int = 0
    # -- gates and runtime flags --
    pruning_allowed: bool = False                                     # outer-loop two-shot gate
    # -- tunables (forwarded from MRMR ctor) --
    tau_cluster: float = 0.7
    distance: str = "su"
    cluster_size_threshold: int = 4
    swap_gain_threshold: float = 0.05
    swap_method: str = "pca_pc1"
    pairwise_cache_max: int = 50_000
    min_cluster_size: int = 2
    max_cluster_size: int = 12
    swap_alpha: float = 0.05                                          # permutation-null p-value threshold for swap accept
    # -- references to host MRMR matrix (mutated on swap) --
    X_raw_ref: Any = None                                             # pd.DataFrame or np.ndarray
    quantization_method: str = "quantile"
    quantization_nbins: int = 10
    quantization_dtype: type = np.int32
    target_indices: Optional[np.ndarray] = None
    factors_data: Optional[np.ndarray] = None
    factors_nbins: Optional[np.ndarray] = None
    cols: Optional[list] = None
    nbins: Optional[np.ndarray] = None
    # -- coexistence policy --
    suppress_legacy_postoc: bool = True

    def cache_evict_lru(self) -> None:
        while len(self.pairwise_su_cache) > int(self.pairwise_cache_max):
            self.pairwise_su_cache.popitem(last=False)


# =============================================================================
# Construction
# =============================================================================


def _kernel_tuning_cache_lookup_tau(factors_data, factors_nbins,
                                     fallback: float = 0.7) -> float:
    """Wave 9.1: route ``dcd_tau_cluster`` through pyutilz kernel_tuning_cache.

    Looks up a calibrated tau by ``(n_samples, n_features, mean_pairwise_su_proxy)``
    fingerprint. Falls back to the constructor-supplied value when the cache
    is cold / unavailable. Per memory rule: hardcoded thresholds should route
    through ``pyutilz.system.kernel_tuning_cache`` so per-host calibration
    persists across runs.
    """
    try:
        from ._kernel_tuning import get_kernel_tuning_cache
        _cache = get_kernel_tuning_cache()
        if _cache is None:
            return float(fallback)
        n_samples = int(factors_data.shape[0]) if factors_data is not None else 0
        n_features = int(factors_data.shape[1]) if factors_data is not None else 0
        # Coarse buckets to keep the cache key stable across small fluctuations.
        n_samples_bucket = int(round(np.log10(max(n_samples, 1)) * 2)) / 2
        n_features_bucket = int(round(np.log10(max(n_features, 1)) * 2)) / 2
        entry = _cache.lookup(
            "dcd_tau_cluster",
            n_samples_log10=float(n_samples_bucket),
            n_features_log10=float(n_features_bucket),
        )
        if entry is not None and "tau_cluster" in entry:
            return float(entry["tau_cluster"])
    except Exception:
        pass
    return float(fallback)


def make_dcd_state(
    *,
    X_raw,
    factors_data,
    factors_nbins,
    cols,
    nbins,
    target_indices,
    quantization_method: str = "quantile",
    quantization_nbins: int = 10,
    quantization_dtype: type = np.int32,
    **dcd_config,
) -> DCDState:
    """Construct a DCDState. ``dcd_config`` accepts: ``tau_cluster``,
    ``distance``, ``cluster_size_threshold``, ``swap_gain_threshold``,
    ``swap_method``, ``pairwise_cache_max``, ``min_cluster_size``,
    ``max_cluster_size``, ``suppress_legacy_postoc``.

    Wave 9.1: ``tau_cluster`` is routed through ``kernel_tuning_cache``
    when the constructor value matches the dev-machine default 0.7 — lets
    per-host calibration override the dev constant without code edits.
    Pass an explicit non-0.7 value to bypass the cache lookup.
    """
    p_initial = int(factors_data.shape[1]) if factors_data is not None else 0
    state = DCDState(
        pool_pruned_mask=np.zeros(p_initial, dtype=bool),
        X_raw_ref=X_raw,
        factors_data=factors_data,
        factors_nbins=np.asarray(factors_nbins),
        cols=list(cols) if cols is not None else [],
        nbins=np.asarray(nbins) if nbins is not None else np.array([], dtype=np.int64),
        target_indices=np.asarray(target_indices) if target_indices is not None
                                                  else np.array([], dtype=np.int64),
        quantization_method=str(quantization_method),
        quantization_nbins=int(quantization_nbins),
        quantization_dtype=quantization_dtype,
    )
    # Forward optional tunables.
    for key in (
        "tau_cluster", "distance", "cluster_size_threshold",
        "swap_gain_threshold", "swap_method", "pairwise_cache_max",
        "min_cluster_size", "max_cluster_size", "swap_alpha",
        "suppress_legacy_postoc",
    ):
        if key in dcd_config:
            setattr(state, key, dcd_config[key])
    # Wave 9.1: kernel_tuning_cache override for ``tau_cluster`` when the
    # constructor-supplied value is the dev-machine default 0.7. Lets
    # per-host calibration kick in without code edits while preserving
    # user-pinned overrides.
    if abs(float(state.tau_cluster) - 0.7) < 1e-9:
        calibrated = _kernel_tuning_cache_lookup_tau(
            factors_data, factors_nbins, fallback=float(state.tau_cluster),
        )
        state.tau_cluster = calibrated
    return state


# =============================================================================
# Pairwise SU with (min,max) tuple cache + shared entropy_cache
# =============================================================================


def pair_su(state: DCDState, a: int, b: int,
            entropy_cache: Optional[dict] = None,
            factors_data=None, factors_nbins=None,
            dtype=np.int32) -> float:
    """Cached symmetric-uncertainty between columns ``a`` and ``b``.

    Returns SU(X_a, X_b) in [0, 1] under ``distance='su'`` (default). For
    ``distance='vi'`` returns ``1 - VI/log(2*n_bins)`` to remain bounded;
    for ``'sotoca_pla'`` returns the target-aware distance from the
    Sotoca-Pla 2010 formula.

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
    if state.distance == "su":
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
        from .info_theory import entropy, merge_vars
        fn_arr = np.asarray(fn, dtype=np.int64)
        ec = state.column_entropy_cache
        h_a = ec.get(a)
        if h_a is None:
            _, freqs_a, _ = merge_vars(
                fd, np.array([a], dtype=np.int64), None, fn_arr, dtype=dtype,
            )
            h_a = float(entropy(freqs_a))
            ec[a] = h_a
        h_b = ec.get(b)
        if h_b is None:
            _, freqs_b, _ = merge_vars(
                fd, np.array([b], dtype=np.int64), None, fn_arr, dtype=dtype,
            )
            h_b = float(entropy(freqs_b))
            ec[b] = h_b
        _, freqs_ab, _ = merge_vars(
            fd, np.array([a, b], dtype=np.int64), None, fn_arr, dtype=dtype,
        )
        h_ab = float(entropy(freqs_ab))
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
        from .info_theory import mi, entropy, merge_vars
        import math
        x_idx = np.array([a], dtype=np.int64)
        y_idx = np.array([b], dtype=np.int64)
        fn_arr = np.asarray(fn, dtype=np.int64)
        mi_ab = float(mi(fd, x_idx, y_idx, fn_arr, dtype=dtype))
        # iter587: same per-column entropy cache as the SU branch above --
        # H(X_a) / H(X_b) recomputation was the dominant per-call cost.
        ec = state.column_entropy_cache
        h_a = ec.get(a)
        if h_a is None:
            _, freqs_a, _ = merge_vars(fd, x_idx, None, fn_arr, dtype=dtype)
            h_a = float(entropy(freqs_a))
            ec[a] = h_a
        h_b = ec.get(b)
        if h_b is None:
            _, freqs_b, _ = merge_vars(fd, y_idx, None, fn_arr, dtype=dtype)
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
        # Sotoca-Pla 2010 target-aware distance
        #   d(X_i, X_j) = 2 H(X_i, X_j) - I(X_i; X_j) - I(X_i; Y) - I(X_j; Y)
        # Normalised by 2 H(X_i, X_j) to stay in [0, 1] range.
        # Fallback to SU if target_indices unavailable.
        if state.target_indices is None or state.target_indices.size == 0:
            from .info_theory import symmetric_uncertainty
            su = float(symmetric_uncertainty(
                fd,
                np.array([a], dtype=np.int64),
                np.array([b], dtype=np.int64),
                np.asarray(fn, dtype=np.int64),
                dtype=dtype,
            ))
        else:
            from .info_theory import mi, entropy, merge_vars
            mi_ab = float(mi(fd, np.array([a], dtype=np.int64),
                              np.array([b], dtype=np.int64),
                              np.asarray(fn, dtype=np.int64), dtype=dtype))
            mi_ay = float(mi(fd, np.array([a], dtype=np.int64),
                              state.target_indices,
                              np.asarray(fn, dtype=np.int64), dtype=dtype))
            mi_by = float(mi(fd, np.array([b], dtype=np.int64),
                              state.target_indices,
                              np.asarray(fn, dtype=np.int64), dtype=dtype))
            # H(X_a, X_b) via concatenated vars_indices.
            _, freqs_ab, _ = merge_vars(
                factors_data=fd,
                vars_indices=np.array([a, b], dtype=np.int64),
                var_is_nominal=None,
                factors_nbins=np.asarray(fn, dtype=np.int64),
                dtype=dtype,
            )
            h_ab = float(entropy(freqs_ab))
            denom = 2.0 * h_ab if h_ab > 1e-12 else 1.0
            d = (2.0 * h_ab - mi_ab - mi_ay - mi_by) / denom
            # Convert distance to similarity-like score (1 - d) so the
            # threshold check ``score > tau_cluster`` semantics stay
            # uniform with SU.
            # 2026-05-30 Wave 9.1 fix (loop iter 32): clamp to [0, 1].
            # Pre-fix the upper bound was unenforced: when target is
            # highly informative about both X_a and X_b
            # (``I(X_a;Y) + I(X_b;Y) > I(X_a;X_b)``, a common and
            # expected case for target-informative features), ``d`` goes
            # negative and the score exceeded 1.0 (live demo: 1.5).
            # The score then triggered the cluster-membership rule
            # ``score > tau_cluster`` for ANY tau in [0, 1] including
            # the default 0.7, falsely pruning the very features
            # sotoca_pla was designed to surface. Comment at line 305
            # claimed "stay in [0,1] range" - true only after this clamp.
            su = max(0.0, min(1.0, 1.0 - d))
    else:
        raise ValueError(f"DCD: unknown distance {state.distance!r}")
    cache[key] = su
    state.cache_evict_lru()
    return float(su)


# =============================================================================
# Should-be-pruned check (used by _confirm_predictor.should_skip_candidate)
# =============================================================================


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


# =============================================================================
# Cluster discovery (no mutation of candidates list)
# =============================================================================


def discover_cluster_members(
    state: DCDState,
    just_selected,
    candidate_pool,
    entropy_cache: Optional[dict] = None,
    factors_data=None,
    factors_nbins=None,
) -> set:
    """For each candidate in ``candidate_pool``, compute SU(c, just_selected).
    If above ``tau_cluster`` and cluster not at ``max_cluster_size``, mark
    ``pool_pruned_mask[c] = True`` and add ``c`` to ``cluster_anchors[anchor]``.

    Does NOT mutate the caller's candidate list (Critic1/B-1 fix). The
    ``should_be_pruned`` check in ``_confirm_predictor.should_skip_candidate``
    is what stops pruned candidates from being re-scored.

    ``just_selected`` can be an int (interactions_order=1) or a tuple of
    ints (higher-order); in the tuple case we use the FIRST component as
    the anchor — DCD's cluster semantics are single-anchor.

    Returns the set of newly added member column indices.
    """
    if state is None or state.pool_pruned_mask is None:
        return set()
    if isinstance(just_selected, (int, np.integer)):
        anchor = int(just_selected)
    else:
        try:
            anchor = int(next(iter(just_selected)))
        except (TypeError, StopIteration):
            return set()
    # 2026-05-30 Wave 9.1 fix (loop iter 44): refuse to anchor on a
    # column that was already swap-pruned. Pre-fix
    # ``state.cluster_anchors.setdefault(anchor, set())`` resurrected a
    # dead anchor as an empty entry, then the discover loop added
    # arbitrary candidates as its "members" - inflating
    # ``MRMR.dcd_["n_anchors"]`` and silently corrupting the published
    # anchor->member graph. ``_screen_predictors._dcd_discover`` only
    # passes freshly-selected ``var`` indices in the live path so this
    # was latent, but the function is in ``__all__`` and the docstring
    # promises set-default semantics - so direct API users tripped it.
    n_cols = state.pool_pruned_mask.shape[0]
    if 0 <= anchor < n_cols and bool(state.pool_pruned_mask[anchor]):
        return set()
    anchors = state.cluster_anchors.setdefault(anchor, set())
    newly_added: set = set()
    for c in candidate_pool:
        try:
            c_int = int(c)
        except (TypeError, ValueError):
            continue
        if c_int == anchor:
            continue
        if c_int < 0 or c_int >= n_cols:
            continue
        if state.pool_pruned_mask[c_int]:
            continue
        if c_int in state.member_to_anchor:
            continue
        if len(anchors) >= int(state.max_cluster_size):
            break
        su = pair_su(state, c_int, anchor,
                     entropy_cache=entropy_cache,
                     factors_data=factors_data, factors_nbins=factors_nbins)
        if su > float(state.tau_cluster):
            anchors.add(c_int)
            state.member_to_anchor[c_int] = anchor
            state.pool_pruned_mask[c_int] = True
            newly_added.add(c_int)
    return newly_added


# =============================================================================
# Pre-confirmation swap evaluation
# =============================================================================


def evaluate_swap_candidate(
    state: DCDState,
    anchor: int,
    selected_vars: list,
    *,
    X_raw=None,
    target_y=None,
    factors_data=None,
    factors_nbins=None,
    cached_MIs: Optional[dict] = None,
    entropy_cache: Optional[dict] = None,
    full_npermutations: int = 0,
) -> SwapDecision:
    """Decide whether to swap ``anchor`` with a PC1 aggregate of its cluster.

    Returns a SwapDecision with ``accept=False`` if either:
      - the cluster is below ``min_cluster_size`` members
      - PC1 fit fails (degenerate / NaN variance)
      - conditional relevance ``I(rep ; y | Selected − anchor)`` does NOT
        exceed anchor's conditional relevance × ``(1 + swap_gain_threshold)``

    No mutation occurs until ``commit_swap`` is called with the returned
    decision (Critic1/B-2 pre-confirmation guarantee).
    """
    cluster = state.cluster_anchors.get(anchor, set())
    if len(cluster) < max(int(state.min_cluster_size),
                           int(state.cluster_size_threshold)):
        return SwapDecision(accept=False)
    members = [anchor] + sorted(cluster)
    if X_raw is None:
        X_raw = state.X_raw_ref
    if X_raw is None:
        return SwapDecision(accept=False)
    cols = state.cols
    if cols is None or len(cols) <= max(members):
        return SwapDecision(accept=False)
    try:
        from ._cluster_aggregate import (
            _standardize_align, _derive_weights, _continuous_cols,
        )
    except Exception as exc:
        logger.warning(f"DCD swap: failed to import cluster_aggregate helpers: {exc!r}")
        return SwapDecision(accept=False)
    try:
        member_names = [cols[m] for m in members]
        # Resolve raw columns; X_raw may be a DataFrame or ndarray.
        if hasattr(X_raw, "columns"):
            present = [c for c in member_names if c in X_raw.columns]
            if len(present) < 2:
                return SwapDecision(accept=False)
            M = X_raw[present].to_numpy(dtype=np.float64, copy=True)
        else:
            arr = np.asarray(X_raw)
            if arr.ndim != 2 or arr.shape[1] < max(members) + 1:
                return SwapDecision(accept=False)
            M = arr[:, members].astype(np.float64, copy=True)
        # Drop columns containing NaN / Inf to keep PC1 stable.
        finite_mask = np.isfinite(M).all(axis=0)
        if finite_mask.sum() < 2:
            return SwapDecision(accept=False)
        M = M[:, finite_mask]
        Z, mean, std, signs = _standardize_align(M, ref_col=0)
        weights = _derive_weights(Z, state.swap_method)
        if weights is None:
            return SwapDecision(accept=False)
        rep_continuous = (Z @ weights).astype(np.float64)
        # Bin the rep via quantile/uniform to integer codes.
        rep_binned = _binarize_aggregate(
            rep_continuous, method=state.quantization_method,
            n_bins=state.quantization_nbins, dtype=state.quantization_dtype,
        )
    except Exception as exc:
        logger.warning(f"DCD swap: PC1 fit failed: {exc!r}")
        return SwapDecision(accept=False)
    # Build a candidate matrix with rep appended.
    new_col_idx = int(state.factors_data.shape[1])
    data_with_rep = np.column_stack([state.factors_data, rep_binned])
    nbins_with_rep = np.concatenate([
        np.asarray(state.factors_nbins, dtype=np.int64),
        [int(rep_binned.max()) + 1 if rep_binned.size else int(state.quantization_nbins)],
    ])
    # Relevance comparison: conditional MI against Selected − {anchor}.
    target = (state.target_indices if state.target_indices is not None and
              state.target_indices.size > 0 else target_y)
    if target is None:
        return SwapDecision(accept=False)
    # Build conditioning set Selected − {anchor}.
    S_minus_anchor = [int(s) for s in selected_vars if int(s) != int(anchor)]
    try:
        from .info_theory import mi, conditional_mi
        if S_minus_anchor:
            rep_relevance = float(conditional_mi(
                factors_data=data_with_rep,
                x=np.array([new_col_idx], dtype=np.int64),
                y=np.asarray(target, dtype=np.int64),
                z=np.array(S_minus_anchor, dtype=np.int64),
                var_is_nominal=None,
                factors_nbins=nbins_with_rep,
                entropy_cache=entropy_cache,
                can_use_x_cache=False, can_use_y_cache=True,
            ))
            anchor_rel = float(conditional_mi(
                factors_data=state.factors_data,
                x=np.array([int(anchor)], dtype=np.int64),
                y=np.asarray(target, dtype=np.int64),
                z=np.array(S_minus_anchor, dtype=np.int64),
                var_is_nominal=None,
                factors_nbins=state.factors_nbins,
                entropy_cache=entropy_cache,
                can_use_x_cache=False, can_use_y_cache=True,
            ))
        else:
            # First-selected case: use unconditional MI.
            rep_relevance = float(mi(
                data_with_rep, np.array([new_col_idx], dtype=np.int64),
                np.asarray(target, dtype=np.int64), nbins_with_rep,
            ))
            anchor_rel = float(mi(
                state.factors_data, np.array([int(anchor)], dtype=np.int64),
                np.asarray(target, dtype=np.int64), state.factors_nbins,
            ))
    except Exception as exc:
        logger.warning(f"DCD swap: relevance estimation failed: {exc!r}")
        return SwapDecision(accept=False)
    deterministic_gate = rep_relevance > anchor_rel * (1.0 + float(state.swap_gain_threshold))
    if not deterministic_gate:
        return SwapDecision(
            accept=False,
            rep_relevance=rep_relevance,
            anchor_relevance_in_ctx=anchor_rel,
        )
    # 2026-05-30 Wave 9.1 fix (loop iter 3): permutation null on rep.
    # The deterministic point-MI gate is upward-biased on small / noisy data
    # because rep (a continuous PC1 projection re-binned with quantization_nbins
    # bins, often > anchor's bin count) gets more degrees of freedom than the
    # raw anchor. Without a null check, swap accepts spurious aggregates on
    # pure noise. Plan v2 B-2 mandated this null but it was never implemented.
    #
    # Null hypothesis: rep_binned has no real conditional dependence on y
    # given Selected\{anchor}. Reject when observed rep_relevance lies in
    # the upper tail of the shuffled-rep distribution.
    perm_p_value = 0.0
    B = int(full_npermutations or 0)
    if B > 0:
        try:
            rng = np.random.default_rng(int(getattr(state, "_perm_seed", 0)) + int(anchor))
            # Persist rolling seed so successive swaps don't reuse the same null draws.
            state._perm_seed = int(getattr(state, "_perm_seed", 0)) + B + 1
            target_arr = np.asarray(target, dtype=np.int64)
            n_exceed = 0
            data_with_rep_perm = data_with_rep.copy()
            for _ in range(B):
                rep_shuffled = rep_binned.copy()
                rng.shuffle(rep_shuffled)
                data_with_rep_perm[:, new_col_idx] = rep_shuffled
                if S_minus_anchor:
                    null_rel = float(conditional_mi(
                        factors_data=data_with_rep_perm,
                        x=np.array([new_col_idx], dtype=np.int64),
                        y=target_arr,
                        z=np.array(S_minus_anchor, dtype=np.int64),
                        var_is_nominal=None,
                        factors_nbins=nbins_with_rep,
                        entropy_cache=None,
                        can_use_x_cache=False, can_use_y_cache=False,
                    ))
                else:
                    null_rel = float(mi(
                        data_with_rep_perm, np.array([new_col_idx], dtype=np.int64),
                        target_arr, nbins_with_rep,
                    ))
                if null_rel >= rep_relevance:
                    n_exceed += 1
            perm_p_value = (n_exceed + 1) / (B + 1)
        except Exception as exc:
            logger.warning(f"DCD swap: permutation null failed (B={B}): {exc!r}")
            perm_p_value = 1.0  # conservative: fail closed
    accept = deterministic_gate and (B <= 0 or perm_p_value < float(state.swap_alpha))
    if not accept:
        return SwapDecision(
            accept=False,
            rep_relevance=rep_relevance,
            anchor_relevance_in_ctx=anchor_rel,
            perm_p_value=perm_p_value,
        )
    aggregate_name = (
        f"_dcd_pc1_{'_'.join(str(cols[m])[:6] for m in members[:3])}"
        f"_n{len(members)}_a{anchor}"
    )
    return SwapDecision(
        accept=True,
        new_col_idx=new_col_idx,
        aggregate_name=aggregate_name,
        binned_rep=rep_binned,
        new_nbins=int(nbins_with_rep[-1]),
        recipe_obj={"method": state.swap_method, "members": members,
                     "mean": mean.tolist(), "std": std.tolist(),
                     "signs": signs.tolist(),
                     "weights": weights.tolist()},
        rep_relevance=rep_relevance,
        anchor_relevance_in_ctx=anchor_rel,
        perm_p_value=perm_p_value,
    )


def _binarize_aggregate(values: np.ndarray, *, method: str, n_bins: int,
                         dtype) -> np.ndarray:
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
            binned = np.clip(((values - mn) / (mx - mn) * n_bins).astype(np.int64),
                             0, n_bins - 1)
    else:
        mn, mx = float(finite.min()), float(finite.max())
        if mx <= mn:
            return np.zeros(values.shape, dtype=dtype)
        binned = np.clip(((values - mn) / (mx - mn) * n_bins).astype(np.int64),
                         0, n_bins - 1)
    return binned.astype(dtype)


# =============================================================================
# Atomic commit (state mutation happens here)
# =============================================================================


def commit_swap(
    state: DCDState,
    anchor: int,
    decision: SwapDecision,
    *,
    selected_vars: list,
    data_ref: dict,
    engineered_recipes: Optional[dict] = None,
    predictors_log: Optional[list] = None,
) -> int:
    """Atomic mutation of state, the host MRMR data structures, and the
    cluster bookkeeping. ``data_ref`` is a dict containing references the
    caller wants updated: keys ``data``, ``cols``, ``nbins`` map to the
    np.ndarray / list / np.ndarray objects to be replaced in-place via
    re-assignment to the SAME dict (caller reads them back).
    """
    if not decision.accept:
        return -1
    new_idx = int(decision.new_col_idx)
    # 1. Extend matrix.
    new_data = np.column_stack([state.factors_data, decision.binned_rep])
    new_cols = list(state.cols) + [str(decision.aggregate_name)]
    new_nbins = np.concatenate([
        np.asarray(state.factors_nbins, dtype=np.int64),
        [int(decision.new_nbins)],
    ])
    state.factors_data = new_data
    state.cols = new_cols
    state.factors_nbins = new_nbins
    # Expand pool_pruned_mask to match (new column is implicitly NOT pruned).
    state.pool_pruned_mask = np.concatenate([
        state.pool_pruned_mask, np.zeros(1, dtype=bool),
    ])
    # 2. Update caller's data references.
    if data_ref is not None:
        data_ref["data"] = new_data
        data_ref["cols"] = new_cols
        data_ref["nbins"] = new_nbins
    # 3. Replace anchor in selected_vars with new aggregate col idx.
    try:
        pos = selected_vars.index(int(anchor))
        selected_vars[pos] = new_idx
    except ValueError:
        # Anchor not in selected_vars (interaction-tuple case); append.
        selected_vars.append(new_idx)
    # 4. Move cluster_anchors[anchor] under new key; mark anchor pruned.
    cluster_members = state.cluster_anchors.pop(int(anchor), set())
    state.cluster_anchors[new_idx] = cluster_members
    for m in cluster_members:
        state.member_to_anchor[int(m)] = new_idx
    if int(anchor) < state.pool_pruned_mask.shape[0]:
        state.pool_pruned_mask[int(anchor)] = True
    # 5. Persist recipe + log.
    if engineered_recipes is not None and decision.aggregate_name:
        engineered_recipes[decision.aggregate_name] = decision.recipe_obj
    if predictors_log is not None:
        predictors_log.append({
            "dcd_swap": True,
            "anchor": int(anchor),
            "new_col_idx": new_idx,
            "aggregate_name": decision.aggregate_name,
            "n_members": len(cluster_members),
        })
    state.swap_log.append({
        "anchor": int(anchor),
        "new_col_idx": new_idx,
        "aggregate_name": decision.aggregate_name,
        "n_members": len(cluster_members),
        "rep_relevance": float(decision.rep_relevance),
        "anchor_relevance_in_ctx": float(decision.anchor_relevance_in_ctx),
    })
    return new_idx


# =============================================================================
# Public summary helper
# =============================================================================


def dcd_summary(state: Optional[DCDState]) -> Optional[dict]:
    """Return a JSON-serialisable summary of the DCD run for ``MRMR.dcd_``
    artifact. Returns None when ``state is None`` (DCD disabled)."""
    if state is None:
        return None
    n_pruned = int(state.pool_pruned_mask.sum()) if state.pool_pruned_mask is not None else 0
    return {
        "n_anchors": len(state.cluster_anchors),
        "n_pruned": n_pruned,
        "n_swaps": len(state.swap_log),
        "n_su_calls": int(state.n_su_calls),
        "n_cache_hits": int(state.n_cache_hits),
        "n_cache_misses": int(state.n_cache_misses),
        "cache_hit_rate": (state.n_cache_hits / max(state.n_su_calls, 1)),
        "cache_size_final": len(state.pairwise_su_cache),
        "swap_log": state.swap_log,
        "cluster_anchors": {int(k): sorted(int(v) for v in vs)
                             for k, vs in state.cluster_anchors.items()},
    }


__all__ = [
    "DCDState", "SwapDecision",
    "make_dcd_state",
    "pair_su",
    "should_be_pruned",
    "discover_cluster_members",
    "evaluate_swap_candidate",
    "commit_swap",
    "dcd_summary",
    "use_dcd", "set_dcd_active",
]
