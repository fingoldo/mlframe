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
    cluster_anchors: dict = field(default_factory=dict)  # anchor_idx -> set[member_col]
    member_to_anchor: dict = field(default_factory=dict)  # member_col -> anchor_idx
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
    column_entropy_cache: dict = field(default_factory=dict)  # int -> float
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
    pool_pruned_mask: Optional[np.ndarray] = None  # bool[p_initial]; True == pruned
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
    swap_alpha: float = 0.05  # permutation-null p-value threshold for swap accept
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
    # Monotone-warp linear-usability tie-break. A strictly-monotone warp g=exp(4f) of an informative f is rank-identical to f, so binned MI(g;y)==MI(f;y) and the redundancy gate keeps EXACTLY
    # ONE of {f, g}; the survivor is otherwise decided by column order alone. f and g are equivalent for trees but f is strictly more linearly-usable. When ON, at the cluster-pruning point, if a
    # candidate about to be pruned as SU-redundant with its anchor is a strictly-monotone twin (raw rank-corr >= warp_twin_rank_corr) AND ties the anchor in SU (within warp_tiebreak_su_band) AND is
    # strictly more linearly-usable (|corr(col, rank(col))| on the RAW column), the candidate DISPLACES the anchor (the anchor is pruned instead). One leg is kept either way, so this can never empty
    # support_ nor swap in an unvalidated column; non-twin / non-tie pairs are byte-identical to the order-decided default.
    warp_tiebreak_prefer_linear: bool = True
    warp_twin_rank_corr: float = 0.99                                # raw rank-corr floor to treat (c, anchor) as a strictly-monotone twin (NOT coarse-binned codes -- those create false twins at small nbins)
    warp_tiebreak_su_band: float = 0.02                              # SU tie band: |SU(c, anchor) - SU(anchor, c)| is ~0, so the band gates how close anchor's own redundancy must be (kept for symmetry / future asymmetric metrics)
    warp_linear_margin: float = 0.05                                 # minimum linear-usability advantage (|corr(c, rank c)| - |corr(anchor, rank anchor)|) for the candidate to displace the anchor
    # -- references to host MRMR matrix (mutated on swap) --
    X_raw_ref: Any = None  # pd.DataFrame or np.ndarray
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


def _kernel_tuning_cache_lookup_tau(factors_data, factors_nbins, fallback: float = 0.7) -> float:
    """Wave 9.1: route ``dcd_tau_cluster`` through pyutilz kernel_tuning_cache.

    Looks up a calibrated tau by ``(n_samples, n_features, mean_pairwise_su_proxy)``
    fingerprint. Falls back to the constructor-supplied value when the cache
    is cold / unavailable. Per memory rule: hardcoded thresholds should route
    through ``pyutilz.performance.kernel_tuning.cache`` so per-host calibration
    persists across runs.
    """
    try:
        from .._kernel_tuning import get_kernel_tuning_cache
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
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        logger.debug("suppressed in __init__.py:238: %s", e)
        pass
    return float(fallback)


# =============================================================================
# Layer 47 (2026-05-31): auto-tau calibration via small SU sweep on the data
# =============================================================================
# Implementation lives in the sibling module ``_dcd_tau_auto`` (LOC budget).
# Re-exported here so the parent's ``__all__`` and downstream import paths
# (``from ._dynamic_cluster_discovery import _calibrate_tau_auto``) continue
# to work unchanged.
from .._dcd_tau_auto import (
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
from .._dcd_pair_su_batch import pair_su_batch


def _carry_forward_dcd_bookkeeping(state: "DCDState", existing_state: "DCDState", p_new: int) -> None:
    """Copy the cluster bookkeeping from a prior screen pass into a freshly
    built ``state`` whose matrix has ``p_new`` columns (>= the prior width).

    Structures keyed by stable column indices are carried: the anchor→members
    graph, the member→anchor map, the swap_log, the SU counters AND the
    pairwise-SU / per-column-entropy caches. The pruned-mask is right-padded so
    the newly appended FE / aggregate columns start un-pruned while every prior
    prune is preserved. Carrying the SU/entropy caches is SAFE because raw
    column indices are stable across passes and any aggregate column the prior
    pass created is adopted back into the rescreen matrix at the SAME index with
    the same data — so cached SU(a,b) / H(X_a) values stay bit-correct. It is
    also REQUIRED: ``dcd_summary`` builds ``cluster_diagnostics`` by looking up
    each cluster's pairwise SU in ``pairwise_su_cache``; a carried cluster whose
    SU pairs were computed in the prior pass would report ``n_pairs_evaluated=0``
    / ``min_pair_su=None`` if the cache were dropped, breaking the
    diagnostics-consistent-with-tau invariant.
    """
    try:
        prev_mask = existing_state.pool_pruned_mask
        if prev_mask is not None:
            prev_mask = np.asarray(prev_mask, dtype=bool)
            keep = min(int(prev_mask.shape[0]), int(p_new))
            state.pool_pruned_mask[:keep] = prev_mask[:keep]
        state.cluster_anchors = {int(k): set(int(v) for v in vs) for k, vs in (existing_state.cluster_anchors or {}).items()}
        state.member_to_anchor = {int(k): int(v) for k, v in (existing_state.member_to_anchor or {}).items()}
        state.swap_log = list(existing_state.swap_log or [])
        state.n_su_calls = int(getattr(existing_state, "n_su_calls", 0) or 0)
        state.n_cache_hits = int(getattr(existing_state, "n_cache_hits", 0) or 0)
        state.n_cache_misses = int(getattr(existing_state, "n_cache_misses", 0) or 0)
        prev_su = getattr(existing_state, "pairwise_su_cache", None)
        if prev_su:
            for k, v in prev_su.items():
                state.pairwise_su_cache[k] = v
            state.cache_evict_lru()
        prev_ent = getattr(existing_state, "column_entropy_cache", None)
        if prev_ent:
            state.column_entropy_cache.update(prev_ent)
        # Preserve the rolling permutation seed so re-screen swap nulls do not
        # reuse the prior pass's exact draws.
        _ps = getattr(existing_state, "_perm_seed", None)
        if _ps is not None:
            state._perm_seed = int(_ps)
    except Exception as exc:
        logger.warning("DCD: failed to carry forward prior-pass bookkeeping: %r", exc)


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
    existing_state: Optional["DCDState"] = None,
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

    ``existing_state``: when MRMR.fit re-screens after feature engineering
    (the confirm-rescreen loop), the matrix only GROWS — raw column indices
    are stable and FE/aggregate columns are appended. Building a fresh
    DCDState each pass discards the cluster discovery from the prior pass:
    the dup cluster that screen-1 found (anchor + members pruned) is lost,
    so the published ``dcd_["cluster_anchors"]`` / ``n_pruned`` / ``swap_log``
    reflect only the final pass — which on a 3-dup fixture clusters nothing
    (the FE column selected on the rescreen is not SU-redundant with the raw
    dups). Threading the prior state forward CARRIES the anchor graph,
    member→anchor map, pool-pruned mask (extended to the new width), swap_log
    and SU counters so the summary accumulates across passes. Raw indices are
    preserved verbatim; only the pruned-mask is right-padded with ``False``
    for the newly appended columns.
    """
    p_initial = int(factors_data.shape[1]) if factors_data is not None else 0
    state = DCDState(
        pool_pruned_mask=np.zeros(p_initial, dtype=bool),
        X_raw_ref=X_raw,
        factors_data=factors_data,
        factors_nbins=np.asarray(factors_nbins),
        cols=list(cols) if cols is not None else [],
        nbins=np.asarray(nbins) if nbins is not None else np.array([], dtype=np.int64),
        target_indices=np.asarray(target_indices) if target_indices is not None else np.array([], dtype=np.int64),
        quantization_method=str(quantization_method),
        quantization_nbins=int(quantization_nbins),
        quantization_dtype=quantization_dtype,
    )
    if existing_state is not None:
        _carry_forward_dcd_bookkeeping(state, existing_state, p_initial)
    # Forward optional tunables.
    for key in (
        "tau_cluster", "distance", "cluster_size_threshold",
        "swap_gain_threshold", "swap_method", "pairwise_cache_max",
        "min_cluster_size", "max_cluster_size", "swap_alpha",
        "swap_npermutations",
        "warp_tiebreak_prefer_linear", "warp_twin_rank_corr",
        "warp_tiebreak_su_band", "warp_linear_margin",
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
    n_pairs = int(dcd_config.get("tau_calibration_n_pairs", _DCD_AUTO_TAU_DEFAULT_N_PAIRS))
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


# Pairwise metrics + pruning predicate carved to ``_dcd_metrics.py`` (LOC budget); re-exported
# here so ``from ._dynamic_cluster_discovery import pair_su`` (and siblings) keep working.
from ._dcd_metrics import pair_su, pair_vi, should_be_pruned, _binarize_aggregate

# Anchor->aggregate swap machinery carved to ``_dcd_swap.py``; re-exported for the same reason.
from ._dcd_swap import (
    _AUTO_METHOD_CANDIDATES, _select_swap_method_auto, evaluate_swap_candidate, commit_swap,
)

# =============================================================================
# Cluster discovery (no mutation of candidates list)
# =============================================================================


def _raw_column(state: DCDState, idx: int) -> Optional[np.ndarray]:
    """Pull raw column ``idx`` (by name from ``state.cols``) from ``X_raw_ref`` as a 1-D float64
    array, or None when unavailable (no raw ref, name missing, ndarray out of range)."""
    X_raw = state.X_raw_ref
    if X_raw is None or state.cols is None or idx < 0 or idx >= len(state.cols):
        return None
    try:
        if hasattr(X_raw, "columns"):
            name = state.cols[idx]
            if name not in X_raw.columns:
                return None
            col = X_raw[name]
            if hasattr(col, "to_numpy"):
                col = col.to_numpy()
            arr = np.asarray(col, dtype=np.float64)
        else:
            arr = np.asarray(X_raw)
            if arr.ndim != 2 or idx >= arr.shape[1]:
                return None
            arr = arr[:, idx].astype(np.float64, copy=False)
    except Exception:
        return None
    return arr if arr.ndim == 1 else None


def _linear_usability(arr: np.ndarray) -> Optional[float]:
    """|corr(arr, rank(arr))| on the finite RAW values -- ~1.0 for a linear-usable column, ~0.14 for
    exp(4 f). Returns None when the column is degenerate (constant / <8 finite rows)."""
    fin = np.isfinite(arr)
    if fin.sum() < 8:
        return None
    a = arr[fin]
    if a.std() <= 1e-12:
        return None
    import pandas as _pd
    ranks = _pd.Series(a).rank(method="average").to_numpy()
    if ranks.std() <= 1e-12:
        return None
    c = np.corrcoef(a, ranks)[0, 1]
    return abs(float(c)) if np.isfinite(c) else None


def _raw_rank_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """|Spearman(a, b)| on the rows finite in BOTH -- the strictly-monotone-twin detector. Uses RAW
    values (NOT coarse-binned codes, which manufacture false twins at small nbins)."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 8:
        return None
    aa, bb = a[mask], b[mask]
    if aa.std() <= 1e-12 or bb.std() <= 1e-12:
        return None
    import pandas as _pd
    ra = _pd.Series(aa).rank(method="average").to_numpy()
    rb = _pd.Series(bb).rank(method="average").to_numpy()
    if ra.std() <= 1e-12 or rb.std() <= 1e-12:
        return None
    c = np.corrcoef(ra, rb)[0, 1]
    return abs(float(c)) if np.isfinite(c) else None


def discover_cluster_members(
    state: DCDState,
    just_selected,
    candidate_pool,
    entropy_cache: Optional[dict] = None,
    factors_data=None,
    factors_nbins=None,
    selected_vars: Optional[list] = None,
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
    # Exclude the temporarily-injected target column(s) from cluster membership: a leak/decoy column (``decoy ~ y``) gives the target SU ~ 1.0, so the target would be clustered and folded into a PC1-aggregate recipe whose ``src_names`` reference ``targ_<id>`` -- which is dropped after fit, so transform() KeyErrors on replay (layer6 / layer49).
    _tgt_set = set(int(t) for t in state.target_indices) if state.target_indices is not None else set()
    if anchor in _tgt_set:
        return set()
    anchors = state.cluster_anchors.setdefault(anchor, set())
    newly_added: set = set()
    # Batch-warm the pairwise-SU cache for every statically-eligible
    # candidate against ``anchor`` in ONE prange-over-pairs joint-entropy
    # pass, then let the loop below read those values as cache hits. The
    # loop body (filters, max_cluster_size break, warp-tiebreak anchor
    # displacement, all mutations) is UNCHANGED: pair_su returns the exact
    # cached float on a hit, so selection is byte-identical; a mid-loop
    # anchor swap simply misses and recomputes fresh. The K candidates all
    # share the anchor column, so the batch amortises one thread-spawn
    # across all K (5-8.5x vs the serial per-pair joint; bench
    # bench_pair_su_batch_over_pairs, maxdiff 0.0). Only the su-distance
    # path is batched (the kernel computes SU joints); other distances
    # fall through to the untouched per-pair loop.
    if getattr(state, "distance", "su") == "su":
        _tgt = _tgt_set
        _pruned = state.pool_pruned_mask
        _m2a = state.member_to_anchor
        _warm_pairs = []
        for _c in candidate_pool:
            try:
                _ci = int(_c)
            except (TypeError, ValueError):
                continue
            if _ci == anchor or _ci in _tgt:
                continue
            if _ci < 0 or _ci >= n_cols:
                continue
            if _pruned[_ci] or _ci in _m2a:
                continue
            _warm_pairs.append((_ci, anchor))
        if len(_warm_pairs) > 1:
            try:
                pair_su_batch(
                    state, _warm_pairs,
                    factors_data=factors_data, factors_nbins=factors_nbins,
                    entropy_cache=entropy_cache,
                )
            except Exception:  # nosec B110 - warmup is best-effort; the loop recomputes on miss
                pass
    for c in candidate_pool:
        try:
            c_int = int(c)
        except (TypeError, ValueError):
            continue
        if c_int == anchor or c_int in _tgt_set:
            continue
        if c_int < 0 or c_int >= n_cols:
            continue
        if state.pool_pruned_mask[c_int]:
            continue
        if c_int in state.member_to_anchor:
            continue
        if len(anchors) >= int(state.max_cluster_size):
            break
        su = pair_su(state, c_int, anchor, entropy_cache=entropy_cache, factors_data=factors_data, factors_nbins=factors_nbins)
        if su > float(state.tau_cluster):
            # Monotone-warp linear-usability tie-break. The candidate ``c`` is about to be pruned as SU-redundant with the already-selected ``anchor``; the SU/MI gate is monotone-invariant, so for a
            # strictly-monotone twin (e.g. anchor=g=exp(4f), c=raw f) the survivor is decided purely by which leg the greedy loop selected first (column order). When ON, if ``c`` is a strictly-monotone
            # twin of the anchor AND strictly more linearly-usable, DISPLACE the anchor: prune the anchor and keep ``c`` selected instead. Exactly one leg is kept either way, so support_ can never empty
            # and no unvalidated column is introduced. Guarded so a degenerate / non-twin / non-tie pair falls through to the order-decided default (byte-identical selection on those).
            if getattr(state, "warp_tiebreak_prefer_linear", False) and selected_vars is not None and int(anchor) in [int(s) for s in selected_vars]:
                _a_raw = _raw_column(state, anchor)
                _c_raw = _raw_column(state, c_int)
                if _a_raw is not None and _c_raw is not None and _a_raw.shape == _c_raw.shape:
                    _rc = _raw_rank_corr(_a_raw, _c_raw)
                    if _rc is not None and _rc >= float(state.warp_twin_rank_corr):
                        _lin_a = _linear_usability(_a_raw)
                        _lin_c = _linear_usability(_c_raw)
                        if _lin_a is not None and _lin_c is not None and _lin_c - _lin_a > float(state.warp_linear_margin):
                            # ``c`` is the linear-usable leg of a monotone twin -> swap roles with the anchor.
                            try:
                                _pos = [int(s) for s in selected_vars].index(int(anchor))
                                selected_vars[_pos] = c_int
                            except ValueError:
                                selected_vars.append(c_int)
                            if int(anchor) < n_cols:
                                state.pool_pruned_mask[int(anchor)] = True
                            state.pool_pruned_mask[c_int] = False
                            # Reseat the cluster under the new anchor ``c``: the old anchor becomes a pruned member.
                            _existing = state.cluster_anchors.pop(int(anchor), set())
                            _new_members = {int(m) for m in _existing if int(m) != c_int}
                            _new_members.add(int(anchor))
                            state.cluster_anchors[c_int] = _new_members
                            for _m in _new_members:
                                state.member_to_anchor[int(_m)] = c_int
                            state.member_to_anchor.pop(c_int, None)
                            anchor = c_int
                            anchors = state.cluster_anchors[c_int]
                            continue
            anchors.add(c_int)
            state.member_to_anchor[c_int] = anchor
            state.pool_pruned_mask[c_int] = True
            newly_added.add(c_int)
    return newly_added

def reattach_raw_representative_after_aggregate_swap(
    state: DCDState,
    aggregate_idx: int,
    selected_vars: list,
) -> int:
    """After an AGGREGATE swap, ensure the collapsed cluster keeps ONE raw
    column in ``selected_vars`` (and hence in the raw ``support_``).

    The denoised aggregate is an engineered column that surfaces only via
    ``get_feature_names_out`` / ``transform`` — it can never live in the raw
    integer ``support_`` (which indexes ``feature_names_in_``). When the
    swapped anchor was the cluster's ONLY selected raw column, replacing it
    with the aggregate index erases the cluster's latent from ``support_``
    entirely: a consumer that inspects ``support_`` (e.g. the embedding
    cross-terms contract) sees the latent as completely dropped even though
    its denoised aggregate survives in the transform output. This re-attaches
    the cluster's best raw member as the single raw representative so the
    latent is visible in BOTH views, at the cost of exactly one raw column per
    orphaned cluster (no net growth over the redundant raw block the aggregate
    replaced — the other members stay pruned).

    Returns the re-attached raw column index, or -1 when no re-attach was
    needed (a raw cluster sibling is already selected) or possible (empty
    cluster).
    """
    if state is None:
        return -1
    members = state.cluster_anchors.get(int(aggregate_idx), set())
    if not members:
        return -1
    sel = {int(s) for s in selected_vars}
    # A raw sibling already represents the cluster -> nothing to do.
    if any(int(m) in sel for m in members):
        return -1
    # Pick the member with the highest marginal relevance to the target as the
    # cluster's raw stand-in; fall back to the smallest index for determinism.
    target = state.target_indices if state.target_indices is not None and state.target_indices.size > 0 else None
    best_idx = -1
    best_rel = float("-inf")
    if target is not None and state.factors_data is not None:
        try:
            from ..info_theory import mi as _mi_func
            tgt = np.asarray(target, dtype=np.int64)
            fn = np.asarray(state.factors_nbins, dtype=np.int64)
            for m in sorted(members):
                mi_idx = int(m)
                if mi_idx < 0 or mi_idx >= state.factors_data.shape[1]:
                    continue
                rel = float(_mi_func(
                    state.factors_data,
                    np.array([mi_idx], dtype=np.int64), tgt, fn,
                ))
                if rel > best_rel:
                    best_rel = rel
                    best_idx = mi_idx
        except Exception as exc:
            logger.warning("DCD: raw-representative MI ranking failed: %r", exc)
            best_idx = -1
    if best_idx < 0:
        best_idx = min(int(m) for m in members)
    # Un-prune so the raw stand-in is a legitimate selected column, and record
    # it in selected_vars right after the aggregate.
    if 0 <= best_idx < state.pool_pruned_mask.shape[0]:
        state.pool_pruned_mask[best_idx] = False
    if best_idx not in sel:
        try:
            pos = selected_vars.index(int(aggregate_idx))
            selected_vars.insert(pos + 1, best_idx)
        except ValueError:
            selected_vars.append(best_idx)
    # Drop the re-attached member from the aggregate's membership set so the
    # cluster bookkeeping does not double-count it as both a pruned member and
    # the visible representative.
    members.discard(best_idx)
    state.member_to_anchor.pop(best_idx, None)
    return best_idx


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
    # Integer-indexed map. Only REAL clusters (>= 1 member) are reported: an
    # anchor with an empty member set is a column that was selected but had no
    # SU-redundant sibling — it is not a cluster, and reporting it pollutes
    # ``cluster_members_`` / the L48 hierarchy with singleton noise. This also
    # keeps the count stable across the confirm-rescreen: each pass may select
    # different (e.g. engineered) columns that each open an empty anchor entry,
    # and accumulating those empties across passes (now that prior-pass cluster
    # bookkeeping is carried forward) would otherwise inflate ``n_anchors``.
    # ``cluster_diagnostics`` already restricts itself to >= 2-member clusters,
    # so dropping empties here makes the three views mutually consistent.
    int_anchors = {int(k): sorted(int(v) for v in vs) for k, vs in state.cluster_anchors.items() if vs}

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
        # Count only REAL clusters (>= 1 member); empty singleton anchors are
        # excluded above and must not inflate the reported anchor count.
        "n_anchors": len(int_anchors),
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
    "_binarize_aggregate",
    "discover_cluster_members",
    "_select_swap_method_auto",
    "_AUTO_METHOD_CANDIDATES",
    "evaluate_swap_candidate",
    "commit_swap",
    "reattach_raw_representative_after_aggregate_swap",
    "dcd_summary",
    "use_dcd", "set_dcd_active",
    # Layer 47 (2026-05-31): tau-auto calibration helpers (exported for
    # white-box testing + downstream tooling).
    "_calibrate_tau_auto", "_detect_valley_between_modes",
    "_DCD_DEFAULT_TAU", "_DCD_AUTO_TAU_DEFAULT_N_PAIRS", "_DCD_AUTO_TAU_FALLBACK",
    "_DCD_AUTO_TAU_MIN", "_DCD_AUTO_TAU_MAX",
]
