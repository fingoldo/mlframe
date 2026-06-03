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
    perform a commit_swap if accepted.

    Layer 45 (2026-05-31) adds the ``branch`` discriminator and the
    ``member_col_idx`` / ``member_relevance`` payload so commit_swap can
    take the cheap member-swap path (no new column) when a cluster
    member out-CMIs both the anchor and the candidate aggregate. Three
    branches:
      - ``"none"`` — anchor stays; no swap fires
      - ``"member"`` — anchor replaced by a cluster member (no aggregate built)
      - ``"aggregate"`` — anchor replaced by the candidate aggregate (existing behaviour)
    """
    accept: bool
    new_col_idx: int = -1
    aggregate_name: str = ""
    binned_rep: Optional[np.ndarray] = None
    new_nbins: int = 0
    recipe_obj: Any = None
    rep_relevance: float = 0.0
    anchor_relevance_in_ctx: float = 0.0
    perm_p_value: float = 1.0
    branch: str = "none"
    member_col_idx: int = -1
    member_relevance: float = 0.0


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
    # iter589: cached fn_arr (int64 view of factors_nbins) + 2-element
    # pair-index scratch buffer. Pre-fix, every pair_su(a, b) call
    # allocated ``np.array([a, b], dtype=np.int64)`` for the joint
    # merge_vars argument AND re-ran ``np.asarray(fn, dtype=np.int64)``.
    # Both are no-ops when ``fn`` is already int64 (typical) but pay the
    # numpy dispatch cost (~1-3 us each) every call. Reusing a single
    # 2-element ``int64`` buffer and a cached ``fn_arr`` saves a few us
    # per call -- 1.02x on the all-pairs bench; small but bit-equivalent
    # and aligns with the W6 audit's drift-discipline that says cheap
    # cleanups landing alongside a measurable hotspot stay.
    _fn_arr_cached: Optional[np.ndarray] = None
    _pair_idx_buf: Optional[np.ndarray] = None
    pool_pruned_mask: Optional[np.ndarray] = None                    # bool[p_initial]; True == pruned
    swap_log: list = field(default_factory=list)
    n_su_calls: int = 0
    n_cache_hits: int = 0
    n_cache_misses: int = 0
    # -- gates and runtime flags --
    # -- tunables (forwarded from MRMR ctor) --
    tau_cluster: float = 0.7
    distance: str = "su"
    # 2026-05-31 Layer 42: default held at 4 (members beyond anchor) pending
    # the post-swap aggregate -> _engineered_recipes_ wiring. Lowering to 2
    # actually triggers the PC1 swap on the 3-feature redundancy cluster
    # (anchor + 2 dups, the production-canonical pattern), but commit_swap
    # is currently called with engineered_recipes=None at
    # _screen_predictors.py L718 so the new aggregate column is silently
    # dropped from the final ``support_`` / ``get_feature_names_out``.
    # Until the recipe-propagation hookup lands, threshold=2 net-shrinks
    # ``support_`` on real data. Opt in via the MRMR kwarg.
    cluster_size_threshold: int = 4
    swap_gain_threshold: float = 0.05
    swap_method: str = "pca_pc1"
    pairwise_cache_max: int = 50_000
    min_cluster_size: int = 2
    max_cluster_size: int = 12
    swap_alpha: float = 0.05                                          # permutation-null p-value threshold for swap accept
    # 2026-06-03 (audit dcd-core-1 / dcd-swap-null-1/2): the swap permutation
    # null needs its OWN draw count, decoupled from the screening-confidence
    # ``full_npermutations`` (MRMR default 3). At B=3 the smallest attainable
    # p-value is (0+1)/(3+1)=0.25, so ``perm_p_value < swap_alpha`` (0.05) was
    # arithmetically impossible and EVERY swap (aggregate + member) was silently
    # rejected -- the whole supervised swap subsystem was dead on the default
    # path. 199 gives a min-p of 1/200=0.005, comfortably resolving alpha=0.05;
    # ``evaluate_swap_candidate`` also auto-raises it to ceil(1/swap_alpha) as a
    # backstop so the null can never be structurally un-rejectable.
    swap_npermutations: int = 199
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
    # Layer 47 (2026-05-31): tau-auto calibration diagnostics. None when the
    # caller passed a numeric ``tau_cluster``; populated by ``make_dcd_state``
    # when ``tau_cluster='auto'`` to expose the bimodality detection result
    # (mode, valley_su, sampled scores, mean/std) so users can audit the
    # selection. Surfaced on ``MRMR.dcd_["tau_calibration"]`` via
    # ``dcd_summary``.
    tau_calibration: Optional[dict] = None

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


# =============================================================================
# Layer 47 (2026-05-31): auto-tau calibration via small SU sweep on the data
# =============================================================================
# Implementation lives in the sibling module ``_dcd_tau_auto`` (LOC budget).
# Re-exported here so the parent's ``__all__`` and downstream import paths
# (``from ._dynamic_cluster_discovery import _calibrate_tau_auto``) continue
# to work unchanged.
from ._dcd_tau_auto import (
    _calibrate_tau_auto,
    _detect_valley_between_modes,
    _DCD_DEFAULT_TAU,
    _DCD_AUTO_TAU_DEFAULT_N_PAIRS,
    _DCD_AUTO_TAU_FALLBACK,
    _DCD_AUTO_TAU_MIN,
    _DCD_AUTO_TAU_MAX,
)

# Layer 51 (2026-05-31): batched pairwise-SU dispatcher. Sibling module
# keeps the parent under the LOC budget; re-exported here so the
# downstream import path
# ``from ._dynamic_cluster_discovery import pair_su_batch`` continues
# to work alongside ``pair_su``.
from ._dcd_pair_su_batch import pair_su_batch


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
        "swap_npermutations",
        "suppress_legacy_postoc",
    ):
        if key in dcd_config:
            setattr(state, key, dcd_config[key])
    # Layer 47 (2026-05-31): ``tau_cluster='auto'`` calibration. When the
    # caller passes the sentinel string ``'auto'``, run a small SU sweep
    # over random feature pairs on the actual factors_data, detect a valley
    # between the two modes of the SU distribution, and set tau to that
    # valley. Falls back to ``_DCD_AUTO_TAU_FALLBACK`` (0.7) when the
    # distribution is unimodal -- no clear clusters to pick a valley from.
    # The diagnostics are recorded on ``state.tau_calibration`` and surfaced
    # via ``dcd_summary``.
    n_pairs = int(dcd_config.get("tau_calibration_n_pairs",
                                  _DCD_AUTO_TAU_DEFAULT_N_PAIRS))
    seed = int(dcd_config.get("tau_calibration_seed", 0))
    tau_val = state.tau_cluster
    if isinstance(tau_val, str) and tau_val.lower() == "auto":
        calibrated, cal_diag = _calibrate_tau_auto(
            factors_data=factors_data,
            factors_nbins=factors_nbins,
            distance=str(state.distance),
            n_pairs=n_pairs,
            seed=seed,
            fallback=_DCD_AUTO_TAU_FALLBACK,
        )
        state.tau_cluster = float(calibrated)
        state.tau_calibration = {
            "requested": "auto",
            "n_pairs_sampled": cal_diag["n_pairs_sampled"],
            "n_pairs_finite": cal_diag["n_pairs_finite"],
            "mode": cal_diag["mode"],
            "tau": cal_diag["tau"],
            "valley_su": cal_diag["valley_su"],
            "su_mean": cal_diag["su_mean"],
            "su_std": cal_diag["su_std"],
        }
        # NOTE: persisting the calibrated tau back to kernel_tuning_cache is
        # intentionally NOT done here. KernelTuningCache.update() takes a
        # ``regions`` list shape, not a flat ``(key, value)`` pair, so a
        # round-trip persist requires the upstream bench-style mutate that
        # owns the cache file. The lookup-only path in the
        # ``_kernel_tuning_cache_lookup_tau`` helper above is the existing
        # half of that loop; users wanting cross-run persistence should run
        # the dedicated calibration bench and write a single regions entry.
        return state
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
        pair_buf[0] = a
        pair_buf[1] = b
        _, freqs_ab, _ = merge_vars(
            fd, pair_buf, None, fn_arr, dtype=dtype,
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
        from .info_theory import symmetric_uncertainty
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


# =============================================================================
# Layer 46 (2026-05-31): raw VI accessor (in nats) for diagnostic use
# =============================================================================


def pair_vi(state: DCDState, a: int, b: int,
            factors_data=None, factors_nbins=None,
            dtype=np.int32) -> float:
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
    from .info_theory import mi, entropy, merge_vars
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
# Layer 43 (PART B): auto-method selection via K-fold OOF MI bake-off
# =============================================================================


# Layer 44 (2026-05-31): expand the DCD auto bake-off pool from 3 to 7
# candidate aggregators so the K-fold MI selector has materially more
# combiner shapes to pick from. The original 3 cover homogeneous linear
# averaging; the 4 additions extend the menu with:
#   - ``pca_pc2``: secondary principal component (correlated multi-latent
#     clusters where PC1 leaves shared structure on the table)
#   - ``median_z``: row-robust to outlier members
#   - ``signed_max_abs``: surfaces the loudest single member's signal
#   - ``signed_l2_sum``: signed quadratic combiner
_AUTO_METHOD_CANDIDATES = (
    "mean_z", "mean_inv_var", "pca_pc1",
    "pca_pc2", "median_z", "signed_max_abs", "signed_l2_sum",
)


def _select_swap_method_auto(
    *,
    state: DCDState,
    Z: np.ndarray,
    target_y,
    member_names: tuple,
    n_folds: int = 5,
) -> tuple:
    """K-fold OOF MI bake-off over the three linear-combiner methods.

    For each method in ``_AUTO_METHOD_CANDIDATES``:
      1. Split rows into ``n_folds`` folds (deterministic by hash of member_names).
      2. For each fold: derive weights on the train rows of Z, project the
         held-out rows to a scalar aggregate, bin via the quantization recipe.
      3. Compute MI(aggregate_oof; y_held_out) on the held-out rows.
      4. Mean the per-fold MIs.
    Returns ``(winner_method, scores_dict)`` where ``scores_dict`` maps every
    method to its mean OOF MI. Caches under ``state._auto_method_cache`` keyed
    by ``member_names`` for cheap re-evaluation.
    """
    from ._cluster_aggregate import _standardize_align, _derive_weights
    # Cache lookup -- same cluster, same bake-off result.
    cache = getattr(state, "_auto_method_cache", None)
    if cache is None:
        cache = {}
        state._auto_method_cache = cache  # type: ignore[attr-defined]
    cache_key = tuple(member_names)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    # Resolve target -- prefer state's target index column.
    y_arr = None
    if state.target_indices is not None and state.target_indices.size > 0 and state.factors_data is not None:
        try:
            y_arr = np.asarray(state.factors_data[:, int(state.target_indices[0])], dtype=np.int64)
        except Exception:
            y_arr = None
    if y_arr is None and target_y is not None:
        y_arr = np.asarray(target_y, dtype=np.int64).ravel()
    if y_arr is None or y_arr.size != Z.shape[0]:
        # Cannot K-fold without a per-row target -> fall back to pca_pc1.
        result = ("pca_pc1", {})
        cache[cache_key] = result
        return result

    n_samples = Z.shape[0]
    n_folds_eff = max(2, min(int(n_folds), n_samples))
    # Deterministic shuffle seeded by member_names so repeat runs are stable.
    seed_material = abs(hash(cache_key)) & 0xFFFFFFFF
    rng = np.random.default_rng(int(seed_material))
    perm = rng.permutation(n_samples)
    fold_sizes = np.full(n_folds_eff, n_samples // n_folds_eff, dtype=np.int64)
    fold_sizes[: n_samples % n_folds_eff] += 1
    fold_bounds = np.concatenate([[0], np.cumsum(fold_sizes)])

    # Layer 50 (2026-05-31): loop folds-outer / methods-inner with a per-fold
    # SVD cache. Pre-fix the loop was methods-outer; 4 of the 7 candidates
    # (``mean_inv_var``, ``pca_pc1``, ``pca_pc2``, ``factor_score``) each
    # re-SVD'd the SAME Z_train independently -- 4x redundant SVD work per
    # fold per cluster. Profile on p=200/n=5000/10 latents attributed
    # 0.444s tottime / 150 calls to ``np.linalg.svd`` alone; with the cache
    # the 4 SVDs collapse to 1 (~4x reduction on the SVD line item, ~2x
    # reduction on the auto-bake-off cumtime). Bit-equivalent: every method
    # consumes the SAME vt[0] / vt[1] / Zc / communalities arrays it would
    # have computed independently; the cache just hands back the precomputed
    # result instead of redoing it.
    scores = {m: [] for m in _AUTO_METHOD_CANDIDATES}
    # Per-fold Z_train cache slot (reset on every new fold). The dict is
    # reused as a sentinel: cleared at the top of each fold so methods
    # within the fold see the same SVD; methods across folds get a fresh
    # cache (Z_train differs by row partition).
    svd_cache: dict = {}
    for f in range(n_folds_eff):
        test_idx = perm[fold_bounds[f]: fold_bounds[f + 1]]
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_idx] = False
        if train_mask.sum() < 3 or test_idx.size < 2:
            continue
        Z_train = Z[train_mask]
        Z_test = Z[test_idx]
        svd_cache.clear()  # fresh per-fold SVD slot.
        for method in _AUTO_METHOD_CANDIDATES:
            try:
                w = _derive_weights(Z_train, method, svd_cache=svd_cache)
                if w is None:
                    # Layer 44: non-linear / row-reduction combiners (median /
                    # median_z / signed_max_abs / signed_l2_sum) have no weight
                    # vector. Apply the same row-reducer the recipe will use at
                    # replay so the K-fold MI estimate matches the production
                    # path.
                    from ._cluster_aggregate import (
                        _apply_method_nonlinear, _NONLINEAR_METHODS,
                    )
                    if method not in _NONLINEAR_METHODS:
                        continue
                    rep_test = _apply_method_nonlinear(Z_test, method)
                else:
                    rep_test = Z_test @ np.asarray(w, dtype=np.float64)
                rep_test = np.nan_to_num(rep_test, nan=0.0, posinf=0.0, neginf=0.0)
                # Bin rep_test with the recipe's quantization (uses test-fold
                # edges -- cheap, fold-local).
                rep_binned = _binarize_aggregate(
                    rep_test, method=state.quantization_method,
                    n_bins=state.quantization_nbins, dtype=state.quantization_dtype,
                )
                y_test = y_arr[test_idx]
                # MI(rep_binned; y_test). Use the mlframe info_theory.mi with
                # a 2-col data block (rep_binned, y_test).
                from .info_theory import mi as _mi_func
                _data = np.column_stack([
                    rep_binned.astype(np.int64), y_test.astype(np.int64),
                ])
                _nb_rep = int(rep_binned.max()) + 1 if rep_binned.size else int(state.quantization_nbins)
                _nb_y = int(y_test.max()) + 1 if y_test.size else 2
                _nbins_arr = np.array([_nb_rep, _nb_y], dtype=np.int64)
                _mi_val = float(_mi_func(
                    _data, np.array([0], dtype=np.int64),
                    np.array([1], dtype=np.int64), _nbins_arr,
                ))
                scores[method].append(_mi_val)
            except Exception:
                continue

    # Reduce per-method per-fold lists to mean OOF MI; tie-break by candidate
    # order (mean_z first). Bit-equivalent to the pre-Layer-50 flow: same
    # MI values, same averaging, same tie-break -- the only change was the
    # loop nesting order to enable per-fold SVD caching.
    scores = {m: (float(np.mean(v)) if v else 0.0) for m, v in scores.items()}
    winner = max(_AUTO_METHOD_CANDIDATES, key=lambda m: (scores.get(m, 0.0), -_AUTO_METHOD_CANDIDATES.index(m)))
    result = (winner, scores)
    cache[cache_key] = result
    return result


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
    """Decide whether (and how) to swap ``anchor`` for a better cluster
    representative.

    Returns a SwapDecision with ``accept=False`` if either:
      - the cluster is below ``min_cluster_size`` members
      - aggregate fit fails (degenerate / NaN variance)
      - neither the candidate aggregate nor the best cluster member's
        conditional relevance ``I(. ; y | Selected − anchor)`` exceeds
        anchor's conditional relevance × ``(1 + swap_gain_threshold)``

    Layer 45 (2026-05-31): three exclusive branches are evaluated:

      A. No swap — anchor's CMI already dominates the cluster; ``branch="none"``.
      B. Member swap — a cluster member's CMI dominates the anchor's AND
         the candidate aggregate's. The anchor index in ``selected_vars`` is
         simply replaced by the member index (no new aggregate column);
         ``branch="member"``.
      C. Aggregate swap — the aggregate's CMI dominates both the anchor's
         and every member's. Existing behaviour; ``branch="aggregate"``.

    The branch with the highest CMI wins (tie-break order: aggregate >
    member > none, because aggregate already passed the explicit
    permutation null when present). Both the aggregate AND the member
    branch run the permutation null when one is requested (B > 0): the
    Wave 9.1 iter-3 follow-up closed the member-branch side door that
    previously bypassed it. The null's draw count is sourced from
    ``state.swap_npermutations`` (NOT the screening-confidence
    ``full_npermutations``, which only acts as the on/off switch); see the
    ``swap_npermutations`` field comment for why decoupling is mandatory.

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
        # 2026-05-31 Layer 43 (PART B): ``auto`` swap method.
        # Run a K-fold (n_folds=5) OOF MI bake-off over the three linear-
        # combiner methods and pick the per-cluster winner. The chosen method
        # is recorded in the recipe + swap_log so replay uses it bit-identically
        # (no y at transform time). K-fold scores are cached on
        # state._auto_method_cache keyed by tuple(member_names) so successive
        # re-evaluations of the same cluster reuse the bake-off.
        if str(state.swap_method) == "auto":
            chosen_method, kfold_scores = _select_swap_method_auto(
                state=state, Z=Z, target_y=target_y,
                member_names=tuple(member_names),
            )
        else:
            chosen_method = str(state.swap_method)
            kfold_scores = None
        # Layer 44: route all non-linear / row-reduction methods through the
        # shared ``_apply_method_nonlinear`` (median / median_z / signed_max_abs
        # / signed_l2_sum). Linear methods stay on the ``Z @ weights`` fast path.
        from ._cluster_aggregate import (
            _apply_method_nonlinear, _NONLINEAR_METHODS,
        )
        if chosen_method in _NONLINEAR_METHODS:
            weights = None
            rep_continuous = _apply_method_nonlinear(Z, chosen_method).astype(np.float64)
        else:
            weights = _derive_weights(Z, chosen_method)
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
    # 2026-05-31 Layer 45: ANCHOR REFINEMENT. Pre-fix the decision was a
    # binary gate (aggregate vs anchor). When the FIRST-picked feature
    # was a high-MI noisy spike rather than the cluster's center, the
    # aggregate ended up sign-aligned to a sub-optimal reference and the
    # swap either fired weakly or never fired even though a sibling
    # member carried strictly more information about y.
    #
    # The refinement: score each cluster member with the SAME conditional
    # MI(member; y | Selected − anchor) used for the anchor. Pick the
    # best member; if it dominates the anchor by the swap gain, it's a
    # viable swap target on its own. The final branch is whichever of
    # ``aggregate`` / ``best_member`` has the higher CMI -- both have to
    # individually beat the anchor by ``swap_gain_threshold``.
    from .info_theory import mi as _mi_func, conditional_mi as _cmi_func
    member_relevances: dict = {}
    best_member_idx = -1
    best_member_rel = float("-inf")
    target_arr = np.asarray(target, dtype=np.int64)
    for m_idx in sorted(cluster):
        try:
            if S_minus_anchor:
                m_rel = float(_cmi_func(
                    factors_data=state.factors_data,
                    x=np.array([int(m_idx)], dtype=np.int64),
                    y=target_arr,
                    z=np.array(S_minus_anchor, dtype=np.int64),
                    var_is_nominal=None,
                    factors_nbins=state.factors_nbins,
                    entropy_cache=entropy_cache,
                    can_use_x_cache=False, can_use_y_cache=True,
                ))
            else:
                m_rel = float(_mi_func(
                    state.factors_data,
                    np.array([int(m_idx)], dtype=np.int64),
                    target_arr, state.factors_nbins,
                ))
        except Exception:
            m_rel = 0.0
        member_relevances[int(m_idx)] = m_rel
        if m_rel > best_member_rel:
            best_member_rel = m_rel
            best_member_idx = int(m_idx)
    gain_factor = 1.0 + float(state.swap_gain_threshold)
    aggregate_gate = rep_relevance > anchor_rel * gain_factor
    member_gate = (
        best_member_idx >= 0
        and best_member_rel > anchor_rel * gain_factor
    )
    if not aggregate_gate and not member_gate:
        # Branch A: no swap candidate beats the anchor.
        return SwapDecision(
            accept=False,
            rep_relevance=rep_relevance,
            anchor_relevance_in_ctx=anchor_rel,
            branch="none",
            member_col_idx=best_member_idx,
            member_relevance=(best_member_rel if best_member_idx >= 0 else 0.0),
        )
    # Both gates active -> pick the higher CMI as the candidate branch.
    # When only one is active, that one wins by definition.
    prefer_aggregate = aggregate_gate and (
        not member_gate or rep_relevance >= best_member_rel
    )
    # 2026-06-03 (audit dcd-core-1 / dcd-swap-null-1/2): resolve the effective
    # permutation-null draw count. ``full_npermutations`` (the screening
    # confidence, default 3) acts ONLY as the on/off switch -- 0 means the
    # caller opted out of every null. When a null is requested we source the
    # actual count from ``state.swap_npermutations`` and auto-raise it to
    # ceil(1/swap_alpha) so 1/(B+1) < swap_alpha holds; otherwise the gate is
    # arithmetically un-passable (B=3 -> min-p 0.25 >> 0.05) and every swap is
    # silently rejected. Both the aggregate and member nulls use this B_eff.
    if int(full_npermutations or 0) <= 0:
        B_eff = 0
    else:
        B_eff = int(getattr(state, "swap_npermutations", 199) or 0)
        if B_eff <= 0:
            B_eff = int(full_npermutations)
        _swap_alpha = float(state.swap_alpha)
        if _swap_alpha > 0.0:
            _min_B = int(np.ceil(1.0 / _swap_alpha))  # 1/(B_eff+1) < swap_alpha
            if B_eff < _min_B:
                B_eff = _min_B
    # Wave 9.1 iter-3 follow-up: when the caller requested a permutation
    # null (``full_npermutations > 0``), apply the SAME null to the member
    # candidate too. The point-CMI gate is upward-biased on small/noisy
    # data; if the swap is firing on pure noise the null catches it for
    # the aggregate path -- the member branch must not be a side door
    # that bypasses the same check.
    def _run_member_null(member_idx: int, member_rel: float, B_: int) -> float:
        if B_ <= 0:
            return 0.0
        try:
            rng_m = np.random.default_rng(int(getattr(state, "_perm_seed", 0)) + int(anchor) * 7919 + int(member_idx))
            state._perm_seed = int(getattr(state, "_perm_seed", 0)) + B_ + 1
            n_rows = state.factors_data.shape[0]
            data_perm = state.factors_data.copy()
            member_col_orig = data_perm[:, member_idx].copy()
            target_arr_m = np.asarray(target, dtype=np.int64)
            n_exceed_m = 0
            for _ in range(B_):
                shuffled = member_col_orig.copy()
                rng_m.shuffle(shuffled)
                data_perm[:, member_idx] = shuffled
                if S_minus_anchor:
                    null_rel_m = float(conditional_mi(
                        factors_data=data_perm,
                        x=np.array([member_idx], dtype=np.int64),
                        y=target_arr_m,
                        z=np.array(S_minus_anchor, dtype=np.int64),
                        var_is_nominal=None,
                        factors_nbins=state.factors_nbins,
                        entropy_cache=None,
                        can_use_x_cache=False, can_use_y_cache=False,
                    ))
                else:
                    null_rel_m = float(mi(
                        data_perm, np.array([member_idx], dtype=np.int64),
                        target_arr_m, state.factors_nbins,
                    ))
                if null_rel_m >= member_rel:
                    n_exceed_m += 1
            return (n_exceed_m + 1) / (B_ + 1)
        except Exception as exc:
            logger.warning(f"DCD swap: member permutation null failed (B={B_}): {exc!r}")
            return 1.0  # conservative: fail closed
    if not prefer_aggregate:
        # Branch B: member swap. Apply permutation null when requested
        # (B>0) -- otherwise this branch silently bypasses the check the
        # caller asked for on the swap as a whole.
        member_p = _run_member_null(int(best_member_idx), float(best_member_rel), B_eff)
        if B_eff > 0 and member_p >= float(state.swap_alpha):
            return SwapDecision(
                accept=False,
                rep_relevance=rep_relevance,
                anchor_relevance_in_ctx=anchor_rel,
                perm_p_value=member_p,
                branch="none",
                member_col_idx=int(best_member_idx),
                member_relevance=float(best_member_rel),
            )
        return SwapDecision(
            accept=True,
            new_col_idx=int(best_member_idx),
            aggregate_name="",
            binned_rep=None,
            new_nbins=0,
            recipe_obj=None,
            rep_relevance=rep_relevance,
            anchor_relevance_in_ctx=anchor_rel,
            perm_p_value=member_p,
            branch="member",
            member_col_idx=int(best_member_idx),
            member_relevance=float(best_member_rel),
        )
    # Branch C continues -- aggregate must still pass the permutation null.
    deterministic_gate = aggregate_gate
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
    B = B_eff
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
        # Layer 45: aggregate failed its permutation null. If the
        # best-member gate also held (member_gate True), fall through to
        # the member-swap branch -- but apply the SAME permutation null
        # the caller requested via ``full_npermutations``. Wave 9.1 iter-3
        # follow-up: the prior bypass made the member branch a side door
        # past the null on pure-noise candidates.
        if member_gate and best_member_idx >= 0:
            member_p2 = _run_member_null(int(best_member_idx), float(best_member_rel), B_eff)
            if B_eff > 0 and member_p2 >= float(state.swap_alpha):
                return SwapDecision(
                    accept=False,
                    rep_relevance=rep_relevance,
                    anchor_relevance_in_ctx=anchor_rel,
                    perm_p_value=max(perm_p_value, member_p2),
                    branch="none",
                    member_col_idx=int(best_member_idx),
                    member_relevance=float(best_member_rel),
                )
            return SwapDecision(
                accept=True,
                new_col_idx=int(best_member_idx),
                aggregate_name="",
                binned_rep=None,
                new_nbins=0,
                recipe_obj=None,
                rep_relevance=rep_relevance,
                anchor_relevance_in_ctx=anchor_rel,
                perm_p_value=member_p2,
                branch="member",
                member_col_idx=int(best_member_idx),
                member_relevance=float(best_member_rel),
            )
        return SwapDecision(
            accept=False,
            rep_relevance=rep_relevance,
            anchor_relevance_in_ctx=anchor_rel,
            perm_p_value=perm_p_value,
            branch="none",
            member_col_idx=best_member_idx,
            member_relevance=(best_member_rel if best_member_idx >= 0 else 0.0),
        )
    aggregate_name = (
        f"_dcd_pc1_{'_'.join(str(cols[m])[:6] for m in members[:3])}"
        f"_n{len(members)}_a{anchor}"
    )
    # 2026-05-31 Layer 43 (PART B): record the chosen method (the actual
    # combiner used to build ``rep_continuous``) in the recipe, NOT the user-
    # facing ``state.swap_method`` string. When ``auto`` was active, the
    # chosen method is the K-fold OOF winner; when a specific method was
    # pinned, chosen_method == state.swap_method. Replay reads recipe.method
    # so the transform-time aggregate is bit-identical with fit.
    recipe_obj = {
        "method": chosen_method, "members": members,
        "mean": mean.tolist(), "std": std.tolist(),
        "signs": signs.tolist(),
    }
    if weights is not None:
        recipe_obj["weights"] = weights.tolist()
    if kfold_scores is not None:
        recipe_obj["kfold_scores"] = {k: float(v) for k, v in kfold_scores.items()}
        recipe_obj["auto_winner"] = chosen_method
    return SwapDecision(
        accept=True,
        new_col_idx=new_col_idx,
        aggregate_name=aggregate_name,
        binned_rep=rep_binned,
        new_nbins=int(nbins_with_rep[-1]),
        recipe_obj=recipe_obj,
        rep_relevance=rep_relevance,
        anchor_relevance_in_ctx=anchor_rel,
        perm_p_value=perm_p_value,
        branch="aggregate",
        member_col_idx=best_member_idx,
        member_relevance=(best_member_rel if best_member_idx >= 0 else 0.0),
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
    # 2026-05-31 Layer 45: member-swap branch. The decision points at an
    # existing factors_data column (a cluster member), not a new aggregate.
    # No matrix extension, no recipe, no pool_pruned_mask resize. We
    # simply unprune the chosen member (discover_cluster_members had
    # pruned every member when it grew the cluster), replace the anchor
    # index in selected_vars, and reseat the cluster bookkeeping under
    # the member as the new anchor. The rest of the pipeline -- bin counts,
    # transform replay, support_ resolution -- already trusts the column.
    is_member_swap = (
        getattr(decision, "branch", "aggregate") == "member"
        and not decision.aggregate_name
        and decision.binned_rep is None
    )
    if is_member_swap:
        member_idx = new_idx
        # Replace anchor in selected_vars with the chosen member.
        try:
            pos = selected_vars.index(int(anchor))
            selected_vars[pos] = member_idx
        except ValueError:
            selected_vars.append(member_idx)
        # Move cluster ownership to the new anchor; drop the chosen
        # member from its own membership set (it IS the anchor now).
        cluster_members = state.cluster_anchors.pop(int(anchor), set())
        new_member_set = {int(m) for m in cluster_members if int(m) != member_idx}
        # Add old anchor as a member of the new cluster (it's still
        # SU-redundant with the new anchor and should stay pruned).
        new_member_set.add(int(anchor))
        state.cluster_anchors[member_idx] = new_member_set
        for m in new_member_set:
            state.member_to_anchor[int(m)] = member_idx
        # Mark old anchor as pruned (it's now a member); unprune the new
        # anchor (it must be eligible for downstream confirm/select).
        if int(anchor) < state.pool_pruned_mask.shape[0]:
            state.pool_pruned_mask[int(anchor)] = True
        if 0 <= member_idx < state.pool_pruned_mask.shape[0]:
            state.pool_pruned_mask[member_idx] = False
        # Remove the new anchor from member_to_anchor since it is itself
        # the anchor now (anchors aren't tracked there).
        state.member_to_anchor.pop(member_idx, None)
        if predictors_log is not None:
            predictors_log.append({
                "dcd_swap": True,
                "dcd_swap_branch": "member",
                "anchor": int(anchor),
                "new_col_idx": member_idx,
                "aggregate_name": "",
                "n_members": len(new_member_set),
            })
        state.swap_log.append({
            "anchor": int(anchor),
            "new_col_idx": member_idx,
            "aggregate_name": "",
            "n_members": len(new_member_set),
            "rep_relevance": float(decision.rep_relevance),
            "anchor_relevance_in_ctx": float(decision.anchor_relevance_in_ctx),
            "branch": "member",
            "member_relevance": float(decision.member_relevance),
        })
        return member_idx
    # Branch C below: aggregate swap (existing behaviour).
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
    # 2026-06-03 (audit dcd-core-2): invalidate the int64 view of factors_nbins
    # cached by pair_su / pair_su_batch / pair_vi. It is built lazily at the
    # PRE-swap length and then reused while IGNORING the passed factors_nbins
    # (lines 451-454 / 511-514 / 559-562 / 654-657); without this reset, any
    # pair_su on the new aggregate column index would index the stale, too-short
    # array and raise IndexError (re-raised as a fit-aborting RuntimeError at
    # _screen_predictors.py). Invalidating the derived cache where the source
    # mutates is the runtime-cache contract. The next call rebuilds it at the
    # new length.
    state._fn_arr_cached = None
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
    # 2026-05-31 Layer 43 (PART A): when an ``engineered_recipes`` dict is
    # supplied (host MRMR's name->EngineeredRecipe map), upgrade the swap
    # ``recipe_obj`` (a plain dict carrying method/members/mean/std/signs/
    # weights) to a frozen ``EngineeredRecipe`` of kind ``cluster_aggregate``
    # so the ``_mrmr_fit_impl`` remap routes it into ``self._engineered_recipes_``
    # and ``get_feature_names_out`` / ``transform`` replay the PC1 aggregate
    # column. Pre-fix this stored a raw dict and the remap dropped the
    # aggregate from ``support_``/output silently.
    if engineered_recipes is not None and decision.aggregate_name:
        _recipe = decision.recipe_obj
        if isinstance(_recipe, dict):
            try:
                from .engineered_recipes import build_cluster_aggregate_recipe
                _src_names = tuple(
                    str(state.cols[m]) if 0 <= m < len(state.cols) else f"col_{m}"
                    for m in _recipe.get("members", [])
                )
                _quant = {
                    "nbins": int(state.quantization_nbins),
                    "method": str(state.quantization_method),
                    "dtype": np.dtype(state.quantization_dtype).str,
                }
                # Persist fit-time bin edges so ``_apply_cluster_aggregate``
                # uses identical edges at transform-time (no re-quantile drift).
                _bin = np.asarray(decision.binned_rep)
                if _bin.size > 0:
                    n_bins_eff = int(_bin.max()) + 1
                    _q_arr = np.linspace(0.0, 100.0, n_bins_eff + 1)
                    # Reconstruct continuous rep so we can derive edges
                    # consistent with the fit-time binning.
                    try:
                        # Use the same standardized + sign-aligned matrix the
                        # swap was evaluated against. X_raw_ref + src_names are
                        # the canonical sources.
                        if state.X_raw_ref is not None and hasattr(state.X_raw_ref, "columns"):
                            _M = state.X_raw_ref[list(_src_names)].to_numpy(
                                dtype=np.float64, copy=True,
                            )
                            _mean = np.asarray(_recipe["mean"], dtype=np.float64)
                            _std_raw = np.asarray(_recipe["std"], dtype=np.float64)
                            _std = np.where(_std_raw > 0.0, _std_raw, 1.0)
                            _signs = np.asarray(_recipe["signs"], dtype=np.float64)
                            _Z = ((_M - _mean) / _std) * _signs
                            # Layer 44: non-linear / row-reduction methods
                            # (median / median_z / signed_max_abs / signed_l2_sum)
                            # rebuild via the shared reducer; linear methods
                            # stay on ``Z @ weights``.
                            from ._cluster_aggregate import (
                                _apply_method_nonlinear, _NONLINEAR_METHODS,
                            )
                            _m = _recipe.get("method")
                            if _m in _NONLINEAR_METHODS:
                                _cont = _apply_method_nonlinear(_Z, _m)
                            else:
                                _cont = _Z @ np.asarray(_recipe["weights"], dtype=np.float64)
                            _cont = np.nan_to_num(
                                _cont, copy=False, nan=0.0, posinf=0.0, neginf=0.0,
                            )
                            _edges = np.nanpercentile(_cont, _q_arr)
                            _quant["edges"] = _edges.tolist()
                    except Exception:
                        pass
                engineered_recipe = build_cluster_aggregate_recipe(
                    name=str(decision.aggregate_name),
                    src_names=_src_names,
                    method=str(_recipe.get("method", "pca_pc1")),
                    member_mean=np.asarray(_recipe["mean"], dtype=np.float64),
                    member_std=np.asarray(_recipe["std"], dtype=np.float64),
                    signs=np.asarray(_recipe["signs"], dtype=np.float64),
                    weights=(
                        np.asarray(_recipe["weights"], dtype=np.float64)
                        if "weights" in _recipe else None
                    ),
                    quantization=_quant,
                )
                engineered_recipes[decision.aggregate_name] = engineered_recipe
            except Exception as _build_exc:
                logger.warning(
                    "DCD commit_swap: failed to build EngineeredRecipe "
                    "(falling back to dict): %r", _build_exc,
                )
                engineered_recipes[decision.aggregate_name] = _recipe
        else:
            engineered_recipes[decision.aggregate_name] = _recipe
    if predictors_log is not None:
        predictors_log.append({
            "dcd_swap": True,
            "anchor": int(anchor),
            "new_col_idx": new_idx,
            "aggregate_name": decision.aggregate_name,
            "n_members": len(cluster_members),
        })
    # 2026-05-31 Layer 43 (PART B): record the chosen swap method (and any
    # K-fold OOF bake-off scores when ``auto`` ran) in the swap_log entry so
    # downstream callers can audit per-cluster method selection.
    _swap_log_entry = {
        "anchor": int(anchor),
        "new_col_idx": new_idx,
        "aggregate_name": decision.aggregate_name,
        "n_members": len(cluster_members),
        "rep_relevance": float(decision.rep_relevance),
        "anchor_relevance_in_ctx": float(decision.anchor_relevance_in_ctx),
        # Layer 45: branch discriminator on every swap_log entry. Aggregate
        # path here; member-swap path emits "branch":"member" above.
        "branch": "aggregate",
        "member_relevance": float(decision.member_relevance),
    }
    if isinstance(decision.recipe_obj, dict):
        _method = decision.recipe_obj.get("method")
        if _method is not None:
            _swap_log_entry["method"] = str(_method)
        _kfold = decision.recipe_obj.get("kfold_scores")
        if _kfold is not None:
            _swap_log_entry["kfold_scores"] = {k: float(v) for k, v in _kfold.items()}
        _winner = decision.recipe_obj.get("auto_winner")
        if _winner is not None:
            _swap_log_entry["auto_winner"] = str(_winner)
    state.swap_log.append(_swap_log_entry)
    return new_idx


# =============================================================================
# Public summary helper
# =============================================================================


def dcd_summary(state: Optional[DCDState]) -> Optional[dict]:
    """Return a JSON-serialisable summary of the DCD run for ``MRMR.dcd_``
    artifact. Returns None when ``state is None`` (DCD disabled).

    Layer 41: in addition to the integer-indexed ``cluster_anchors`` map
    (anchor_col_idx -> sorted_member_col_idx_list), the summary now exposes
    a parallel ``cluster_anchors_names`` map (anchor_col_name -> sorted
    member_col_name list) AND per-cluster diagnostics (``cluster_diagnostics``:
    ``size``, ``min_pair_su``, ``mean_pair_su``, ``max_pair_su`` over the
    anchor+member set). The integer map stays so legacy consumers keep
    working; the name map makes the summary self-describing without forcing
    callers to remember the column ordering at fit time.
    """
    if state is None:
        return None
    n_pruned = int(state.pool_pruned_mask.sum()) if state.pool_pruned_mask is not None else 0
    cols = list(getattr(state, "cols", []) or [])
    n_cols = len(cols)
    # Integer-indexed map (legacy contract, preserved bit-identical).
    int_anchors = {int(k): sorted(int(v) for v in vs)
                    for k, vs in state.cluster_anchors.items()}
    # Layer 41: name-indexed map. Resolve each integer col index against
    # ``state.cols`` (which is kept up-to-date through commit_swap, so
    # post-swap aggregate names like ``_dcd_pc1_*`` resolve naturally).
    # Falls back to ``f"col_{idx}"`` when the index is out of bounds for
    # the captured cols list (defensive: should not happen on normal fits
    # but guards against malformed states from partial swaps).
    def _name(idx: int) -> str:
        i = int(idx)
        if 0 <= i < n_cols:
            return str(cols[i])
        return f"col_{i}"
    name_anchors: dict = {}
    for anchor_idx, member_idx_list in int_anchors.items():
        name_anchors[_name(anchor_idx)] = [_name(m) for m in member_idx_list]
    # Layer 41: per-cluster diagnostics. The minimum within-cluster SU is the
    # most useful single number for judging whether ``tau_cluster`` is set
    # too high (min_pair_su near tau means borderline) or too low
    # (min_pair_su well below tau means false-positive cluster). Cheap:
    # all pair-SU values were already computed during cluster discovery
    # and live in ``state.pairwise_su_cache`` — we just look them up.
    cluster_diagnostics: dict = {}
    su_cache = state.pairwise_su_cache
    for anchor_idx, member_idx_list in int_anchors.items():
        members_all = [int(anchor_idx)] + [int(m) for m in member_idx_list]
        size = len(members_all)
        # Pairwise SU among ALL members of the cluster (anchor + members).
        # Pull from cache; missing entries are skipped (cluster discovery
        # only computes SU(member, anchor), so anchor-pair entries are
        # always present, but member-member pairs may be absent -- that's
        # fine, the min/mean/max are over what's available, which always
        # includes at least the anchor-member SU values).
        pair_sus: list = []
        for i in range(size):
            for j in range(i + 1, size):
                a, b = members_all[i], members_all[j]
                key = (a, b) if a < b else (b, a)
                v = su_cache.get(key)
                if v is not None:
                    pair_sus.append(float(v))
        if pair_sus:
            cluster_diagnostics[_name(anchor_idx)] = {
                "size": int(size),
                "min_pair_su": float(min(pair_sus)),
                "mean_pair_su": float(sum(pair_sus) / len(pair_sus)),
                "max_pair_su": float(max(pair_sus)),
                "n_pairs_evaluated": int(len(pair_sus)),
            }
        else:
            cluster_diagnostics[_name(anchor_idx)] = {
                "size": int(size),
                "min_pair_su": None,
                "mean_pair_su": None,
                "max_pair_su": None,
                "n_pairs_evaluated": 0,
            }
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
        "cluster_anchors": int_anchors,
        # Layer 41 additions (additive; all legacy keys preserved above).
        "cluster_anchors_names": name_anchors,
        "cluster_diagnostics": cluster_diagnostics,
        "tau_cluster": float(state.tau_cluster),
        # Layer 47 (2026-05-31): auto-tau calibration diagnostics. None when
        # the user passed a numeric tau (calibration didn't run).
        "tau_calibration": getattr(state, "tau_calibration", None),
    }


__all__ = [
    "DCDState", "SwapDecision",
    "make_dcd_state",
    "pair_su",
    "pair_su_batch",
    "pair_vi",
    "should_be_pruned",
    "discover_cluster_members",
    "evaluate_swap_candidate",
    "commit_swap",
    "dcd_summary",
    "use_dcd", "set_dcd_active",
    # Layer 47 (2026-05-31): tau-auto calibration helpers (exported for
    # white-box testing + downstream tooling).
    "_calibrate_tau_auto", "_detect_valley_between_modes",
]
