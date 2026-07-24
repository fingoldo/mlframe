"""``ScreenContext`` dataclass, carved out of ``_confirm_predictor.py`` (X_EFFICIENCY_ARCHITECTURE-1
fix, mrmr_audit_2026-07-22) to clear the repo's enforced hard 1000-LOC CI gate (that file had crept
back to 1007 lines). Behaviour preserved bit-for-bit; the parent re-exports the class so every existing
``from ._confirm_predictor import ScreenContext`` (or equivalent) import keeps working unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class ScreenContext:
    """Bundle of shared/static screening state threaded into the confirmation primitives.

    Static fields (data, algorithm parameters, thresholds) are set once by the orchestrator;
    the four caches are mutated in place across candidates; the per-order fields
    (``candidates`` .. ``failed_candidates``) are reassigned by the caller between predictors
    and interaction orders.
    """

    # --- data / target ---
    factors_data: np.ndarray
    factors_nbins: Sequence[int]
    factors_names: Sequence[str]
    y: Sequence[int]
    data_copy: np.ndarray
    classes_y: np.ndarray
    classes_y_safe: object
    freqs_y: np.ndarray
    freqs_y_safe: object
    # --- algorithm params ---
    mrmr_relevance_algo: str
    mrmr_redundancy_algo: str
    reduce_gain_on_subelement_chosen: bool
    max_veteranes_interactions_order: int
    only_unknown_interactions: bool
    use_gpu: bool
    use_simple_mode: bool
    extra_x_shuffling: bool
    engineered_lineage: object
    # --- performance ---
    n_workers: int
    workers_pool: object
    parallel_kwargs: dict
    # --- confidence / stopping ---
    baseline_npermutations: int
    full_npermutations: int
    min_nonzero_confidence: float
    max_failed: int
    min_relevance_gain: float
    max_consec_unconfirmed: int
    max_runtime_mins: float
    max_confirmation_cand_nbins: int
    random_seed: int
    # --- misc ---
    verbose: int
    ndigits: int
    start_time: float
    num_possible_candidates: int
    # --- shared MI caches (mutated in place) ---
    cached_MIs: dict
    cached_confident_MIs: dict
    cached_cond_MIs: object
    # 2026-06-19: JMIM joint-MI cache (numba typed dict) + 1-elem int64 hit-counter
    # array, mirroring ``cached_cond_MIs``. Lazily built in ``score_candidates`` on
    # first use so they persist across greedy rounds (cross-round reuse of the
    # ``{X} u Z`` multiset key). ``None`` default keeps direct ScreenContext callers
    # backward-compatible. Both are converted to plain values at any pickling
    # boundary (the typed dict never escapes onto an instance).
    cached_jmim_MIs: object = None
    jmim_hit_counter: object = None
    entropy_cache: object = None
    # 2026-07-13 (Wave 13 finding 1): (candidates, cand_names, name_rank) tuple cached across the
    # ~100-150 confirm_one_predictor calls per interactions_order -- the pool (``ctx.candidates``,
    # thousands of entries) is reassigned only once per order, so the identity check below is exact.
    _name_rank_cache: object = None
    # 2026-07-13 (Wave 13 finding 2): (candidates, {parent_name: [idx, ...]}) index cached the same
    # way, for ``_confirmable_engineered_child``'s per-winner parent-name rescan (prefer_engineered_rel_eps path).
    _engineered_parent_index_cache: object = None
    # 2026-07-16 (wellbore-100k profiling): (z_key, z_classes, z_nclasses) cache for
    # ``_conditioning_rows_per_cell``'s Z=selected_vars encoding -- see that function's docstring.
    _z_merge_cache: object = None
    # 2026-07-19 (MRMR wellbore-4.1M diagnosability): cumulative ``score_candidates`` call count
    # and wall time across the WHOLE ``screen_predictors`` call (all interactions_orders), so a
    # future log can directly show whether wall time is dominated by pool_size x |Z| scaling
    # instead of requiring another multi-agent investigation to reconstruct it after the fact.
    sc_calls: int = 0
    sc_wall: float = 0.0
    # --- per-interactions-order / per-node mutable state ---
    # These five are ALWAYS populated by the orchestrator before use (never read as None at
    # runtime); ``None`` is only a dataclass placeholder default (a required-after-defaulted-
    # field constraint), so they stay typed as their concrete container rather than Optional
    # -- which would force null-checks onto every one of their many non-nullable call sites.
    candidates: list = None  # type: ignore[assignment]
    # 2026-05-30 Wave 9 — DCD state forwarded into ``should_skip_candidate``
    # for pool_pruned_mask check. ``None`` preserves legacy bit-stable.
    dcd_state: object = None
    interactions_order: int = 1
    selected_vars: list = None  # type: ignore[assignment]
    selected_interactions_vars: list = None  # type: ignore[assignment]
    partial_gains: dict = None  # type: ignore[assignment]
    added_candidates: set = field(default=None)  # type: ignore[arg-type]
    failed_candidates: set = field(default=None)  # type: ignore[arg-type]
    # 2026-06-02 — directed-FE tie-break. ``raw_feature_names`` is the set of
    # ORIGINAL (pre-FE) column names; any ``factors_names[idx]`` not in it is an
    # engineered transform of its raw parent(s). On a near-tie in selection gain
    # (within ``prefer_engineered_rel_eps`` relative tolerance) the greedy pick
    # deterministically prefers the engineered candidate over a raw one: an
    # engineered column is a function of its parent, so on an MI-tie it dominates
    # representationally (a shallow downstream can use x1**2-1 but not raw x1),
    # and the deterministic rule removes the njit-vs-njit_par pick nondeterminism
    # that the prior index-order tie-break introduced. ``None`` raw-name set
    # falls back to the syntactic heuristic (a name containing ``(`` or ``__`` is
    # engineered) so direct callers still get deterministic behaviour. Setting
    # the rel-eps to ``0.0`` restores the legacy pure-index tie-break.
    raw_feature_names: object = None
    prefer_engineered_rel_eps: float = 0.0
    # 2026-06-02 RC2 — sample-size-aware Fleuret confirmation. When the
    # conditioning joint ``(X u selected_vars)`` has fewer than this many rows
    # per occupied cell, the conditional-MI permutation test is finite-sample
    # unreliable (shuffled-y null CMI ~= real CMI -> over-rejection / premature
    # stop) so ``confirm_candidate`` falls back to a MARGINAL-MI permutation
    # test (the X-marginal joint is far better sampled). ``0.0`` restores the
    # strict legacy conditional test for every candidate. Default ``5.0`` set
    # by the MRMR ctor (new behaviour ON since ``use_simple_mode=False`` is the
    # default). Dedup is unaffected (handled by the gain's redundancy term).
    fe_confirm_undersample_rows_per_cell: float = 0.0
