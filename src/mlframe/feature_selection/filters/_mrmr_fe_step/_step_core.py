"""MRMR._run_fe_step -- the FE-step orchestrator for mlframe.feature_selection.filters.mrmr.

The single irreducible ~1.36k-line _run_fe_step function lives here, verbatim. The package
__init__ re-exports it (and the small _helpers symbols), and mrmr.py binds it back onto
the MRMR class at its module bottom, so self._run_fe_step(...) call sites are unchanged.

The parent module helpers (_lazy_chunks etc.) and every other intra-filters dependency are
imported lazily in-body (from ..mrmr import ... etc.) since .mrmr re-imports this package at
its bottom for method binding -- a top-level import would create a hard cycle. The self-contained FE
sub-blocks (synergy bootstrap, order-2 maxT floor, FE summary log, cluster-aggregate emission) live in
the sibling _mrmr_fe_step_helpers module; the two small operand-pool helpers live in ._helpers.

bench-note (2026-06-24, MRMR-FE wall /loop): _run_fe_step's OWN body is AT FLOOR -- it is pure
orchestration. cProfile F2 100k full fit: this function's tottime = 0.001s (cumtime ~13s over 2 calls is
ALL in carved-sibling callees). line_profiler over the body: glue (every getattr resolution, the
synergy-budget dict-comp at ~300, the original_cols dict-comp at ~464, sort_dict_by_value, the prewarp/
gate-med/rejection-ledger merge loops) sums to 0.166ms = 0.149% of the function's 11.16s; the other
99.85% is the 9 call sites into _step_pool / _step_pairmi / _step_pairs_rank / check_prospective_fe_pairs
/ _step_score. The remaining wall lives in (a) GPU-dispatched routes (cupy .get 2.0s,
gpu_materialise_discretize_codes_host 2.77s cum) and (b) njit compute at CPU-optimum
(_plugin_mi_classif_batch_njit 1.8s, _combine_factorize_njit 0.85s, _pair_combo_mi_njit_table_parallel
0.82s, conditional_mi 0.64s) -- none in this body. No safe wall win exists at the orchestrator level; do
not re-profile the body. Next dominant wall hotspot = the _score_one_pair / check_prospective_fe_pairs
permutation-null + cupy-host-transfer path (already GPU-routed; iter9/11b/15).
"""
from __future__ import annotations

import logging

import numpy as np
from joblib import delayed

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

from .._mrmr_fe_step_helpers import (
    apply_interaction_information_routing,
    log_fe_summary,
    run_cluster_aggregate_emission,
)
from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection
from ._helpers import _synergy_bootstrap_can_supply_pool


def _should_serialize_fe_pair_check(n_prospective_pairs: int, gpu_fe_active: bool, serial_min_pairs_per_worker: int) -> bool:
    """Route ``check_prospective_fe_pairs`` to the serial-main-thread branch instead of the joblib
    threading fan-out. True when the GPU-discretize path is active (N threads would otherwise all
    contend for the single device -- see the GPU-FE SERIALIZE comment at this module's main call site) OR
    when there are too few prospective pairs to fill the worker pool. Extracted as a pure, directly
    testable predicate (2026-07-09, MRMR audit finding #27 regression coverage); behavior is unchanged.
    """
    return bool(gpu_fe_active) or int(n_prospective_pairs) < int(serial_min_pairs_per_worker)


def _free_gpu_fe_mempool() -> bool:
    """Free the cupy default pool's UNUSED blocks at GPU FE-step teardown; return whether a free was issued.

    When the FE step ran GPU kernels (``MLFRAME_FE_GPU_STRICT`` / ``MLFRAME_CMI_GPU``), the cupy default pool
    RETAINS every block it cached this step for reuse. Across repeated fits (CV folds, repeated ``MRMR().fit``
    on a 4 GB card) the retained pool sits at its high-water mark near the device cap and the allocator thrashes
    cudaMalloc/sync -- measured: consecutive 100k f32 STRICT fits degrade 11.2s -> 31.8s -> 32.3s, while freeing
    the unused blocks at each step teardown holds the footprint flat (~2.9 GB) and the wall at ~11.2s.
    ``free_all_blocks`` only returns blocks with no live reference, so a resident operand table held across the
    fit is untouched; this is post-compute teardown, never mid-pipeline. Best-effort + env-gated so CPU users /
    other backends are not touched."""
    import os as _os
    if not (_os.environ.get("MLFRAME_FE_GPU_STRICT") or _os.environ.get("MLFRAME_CMI_GPU")):
        return False
    # FIX3 (2026-06-28): drop the resident y/z device cache in _cmi_cuda FIRST so its device arrays carry no
    # live reference -> free_all_blocks below can actually reclaim them (a fit-scoped cache, never persisted).
    try:
        from ..info_theory._cmi_cuda import clear_cmi_resident_cache
        clear_cmi_resident_cache()
    except Exception:  # nosec B110 - optional dependency import guard
        pass
    # FE operand resident cache: drop the fit-constant operand device copies (y / z / base columns) so their
    # device arrays carry no live reference -> free_all_blocks below can reclaim them (a fit-scoped cache).
    try:
        from .._fe_resident_operands import clear_fe_resident_operands
        clear_fe_resident_operands()
    except Exception:  # nosec B110 - optional dependency import guard
        pass
    try:
        import cupy as _cp
        _cp.get_default_memory_pool().free_all_blocks()
        return True
    except Exception:
        return False


def _run_fe_step(
    self,
    *,
    # Mutable state from MRMR.fit (returned updated as tuple).
    data, cols, nbins, X,
    target_names, target_indices,
    selected_vars, categorical_vars,
    classes_y, classes_y_safe, freqs_y,
    cached_MIs, cached_confident_MIs,
    unary_transformations, binary_transformations,
    engineered_features, checked_pairs,
    # Parallel dict (name -> EngineeredRecipe) populated as new columns are added to data / cols so
    # transform() can replay them on test data. Mutated in place; MRMR.fit reads it after the FE loop
    # and copies surviving recipes into self._engineered_recipes_. ``None`` skips recipe construction.
    engineered_recipes=None,
    times_spent,
    num_fs_steps,
    # Service.
    n_jobs, prefetch_factor, parallel_kwargs,
    _is_polars_input, verbose,
    # FE config (frozen per fit).
    fe_max_steps, fe_npermutations, fe_max_pair_features,
    fe_print_best_mis_only, fe_min_nonzero_confidence,
    fe_min_engineered_mi_prevalence,
    fe_good_to_best_feature_mi_threshold,
    fe_max_external_validation_factors,
    fe_min_pair_mi, fe_min_pair_mi_prevalence,
    fe_smart_polynom_iters, fe_smart_polynom_optimization_steps,
    fe_min_polynom_degree, fe_max_polynom_degree,
    fe_min_polynom_coeff, fe_max_polynom_coeff,
    # Preset-name snapshot so recipes can rebuild the correct registry at replay time. Default "minimal"
    # matches MRMR.__init__ defaults; callers that override via self.fe_unary_preset / self.fe_binary_preset
    # get the actual values threaded through by fit().
    fe_unary_preset: str = "medium",
    fe_binary_preset: str = "minimal",
):
    """One Feature Engineering iteration. Extracted from ``MRMR.fit`` for testability and FE experimentation.

    Returns ``None`` if the FE step should not run (empty-screen + ``fe_fallback_to_all=False``); otherwise
    ``(data, cols, nbins, X, selected_vars, n_recommended_features)``. ``n_recommended_features == 0`` signals
    the outer loop to stop. Private; external callers should use ``MRMR.fit()`` or ``MRMR.fit_transform()``.
    """
    # VRAM POOL CAP (2026-07-05): setup counterpart to the ``_free_gpu_fe_mempool`` teardown below. Cap MRMR's
    # OWN cupy default pool to ``MLFRAME_FE_GPU_POOL_FRACTION`` (default 0.6) of total VRAM ONCE per process at the
    # FE-step entry, so the pool cannot grow unbounded and eat a shared 4 GB card (starving the next launch / other
    # processes). Idempotent + best-effort (no-op without cupy); on exhaustion cupy raises OutOfMemoryError which
    # the GPU-FE try/excepts catch -> graceful CPU. Cheap: the once-flag short-circuits every call after the first.
    try:
        from .._fe_gpu_vram import ensure_fe_gpu_pool_limit as _ensure_fe_gpu_pool_limit
        _ensure_fe_gpu_pool_limit()
    except Exception as e:
        logger.debug("swallowed exception in _step_core.py: %s", e)
        pass
    # SEPARATE KTC-free GPU-RESIDENT FE step (MLFRAME_FE_GPU_STRICT + MLFRAME_FE_GPU_STRICT_RESIDENT, default
    # OFF). When the resident path is enabled it takes over the WHOLE FE step (operands uploaded once per
    # device, all compute on GPU kernels, no bulk D2H). Phase 0: the entry is a stub that raises
    # NotImplementedError, so this is INERT (falls through to the existing per-family path below) until later
    # phases implement it. ``locals()`` here (before any local is bound) is exactly the call params.
    from .._gpu_strict_fe import fe_gpu_strict_resident_enabled, run_fe_step_gpu_strict
    if fe_gpu_strict_resident_enabled():
        _fe_args = {k: v for k, v in locals().items() if k not in ("self", "fe_gpu_strict_resident_enabled", "run_fe_step_gpu_strict")}
        try:
            return run_fe_step_gpu_strict(self, **_fe_args)
        except NotImplementedError:
            pass  # Phase 0 scaffold / unported config -> existing per-family FE path below
    # GUARDED-ADAPTIVE PREVALENCE (2026-06-13, hardcoded-threshold conversion). When
    # ``fe_min_pair_mi_prevalence == "auto"``, keep the proven 1.05 ratio bar but apply it to the
    # MILLER-MADOW-DEBIASED pair MI (the analytic finite-sample joint-MI bias subtracted, the value
    # the maxT floor already uses) rather than the raw ``pair_mi``. Debiasing can ONLY LOWER the
    # observed joint MI, so the gate can only TIGHTEN -- it drops the best-of-pool finite-sample-noise
    # pairs a fixed 1.05 admits (the bench showed a higher effective bar helps the bilinear archetype:
    # 0.207 -> 0.092) while a genuine high-signal pair (joint MI >> bias, e.g. F2's c*d) is untouched.
    # Resolve to the float HERE so every downstream float use is byte-identical to the default path;
    # only the per-pair prevalence comparison below switches to the debiased observed value. An
    # explicit float (incl. the default 1.05) is honoured verbatim -> byte-identical to pre-conversion.
    # R1 STRATIFIED SUBSAMPLE flag, resolved ONCE per FE step (UNCONDITIONALLY -- both the polynom-pair
    # FE and the unary/binary ``check_prospective_fe_pairs`` calls below consume it, and the polynom
    # block is gated by ``if fe_smart_polynom_iters`` so this must live at function-body level). The
    # MRMR ``fe_subsample_stratify`` tri-state knob (None=auto / True / False) is resolved against the
    # best available target: the continuous y (``_fe_prewarp_y_continuous_``) when present (its
    # dtype/cardinality lets ``infer_classification`` pick clf-vs-reg and the regression heavy-tail
    # heuristic see the real distribution), else the discrete ``classes_y`` codes (classification).
    # Default-None auto-ON fires only on a small rare-class fraction / heavy-tailed target; otherwise
    # OFF -> byte-identical legacy uniform draw.
    from .._fe_subsample import _resolve_fe_subsample_stratify as _resolve_strat
    from .._fe_accuracy_gate import infer_classification as _infer_clf
    _strat_knob = getattr(self, "fe_subsample_stratify", None)
    _strat_yc = getattr(self, "_fe_prewarp_y_continuous_", None)
    if _strat_yc is not None and len(_strat_yc) == len(classes_y):
        _fe_subsample_stratify = _resolve_strat(_strat_knob, np.asarray(_strat_yc), is_clf=bool(_infer_clf(np.asarray(_strat_yc))))
    else:
        _fe_subsample_stratify = _resolve_strat(_strat_knob, np.asarray(classes_y), is_clf=True)

    _prevalence_debias_auto = isinstance(fe_min_pair_mi_prevalence, str) and fe_min_pair_mi_prevalence.strip().lower() == "auto"
    if _prevalence_debias_auto:
        fe_min_pair_mi_prevalence = 1.05
    # SYNERGY prevalence "auto" (conversion #3, 2026-06-13): the synergy-pair bar
    # (``max(fe_min_pair_mi_prevalence, fe_synergy_min_prevalence)``, default 1.5) gates the
    # bootstrap-added synergy operands. "auto" activates the SAME MM-debias mechanism for the
    # prevalence comparison and resolves the bar to its 1.5 default float -- so a synergy pair is
    # admitted only when its DEBIASED joint MI clears 1.5x the marginal sum, tightening against the
    # finite-sample noise that a fixed 1.5 on the RAW MI lets through. ``_synergy_prev_resolved`` is
    # the float used at the gate; an explicit float (incl. the 1.15/1.5 defaults) is honoured verbatim.
    _synergy_prev_raw = getattr(self, "fe_synergy_min_prevalence", 1.15)
    if isinstance(_synergy_prev_raw, str) and _synergy_prev_raw.strip().lower() == "auto":
        _prevalence_debias_auto = True  # share the prevalence-comparison debias (consistent mechanism)
        _synergy_prev_resolved = 1.5
    else:
        _synergy_prev_resolved = float(_synergy_prev_raw)
    # Lazy import: ``.mrmr`` re-imports this module at its bottom for method
    # binding -> any top-level ``from .mrmr import ...`` here creates a hard
    # import cycle that ``tests/test_meta/test_no_import_cycles.py`` flags.
    from ..mrmr import (
        check_prospective_fe_pairs,
        discretize_array,
        get_new_feature_name,
        parallel_run,
        sort_dict_by_value,
    )
    if verbose:
        logger.info("MRMR+ selected %d out of %d features before the Feature Engineering step.", len(selected_vars), self.n_features_in_)

    _screening_returned_empty = False
    if len(selected_vars) == 0:
        if self.fe_fallback_to_all:
            logger.info("Proceeding with all features though (fe_fallback_to_all=True).")
            # ``cols.index(col)`` while iterating ``cols`` itself is always just the loop position
            # (column names are unique by construction, matching every other name->index map in
            # the FE step) -- enumerate() gives the identical index without the O(F) rescan per item.
            selected_vars = np.array([i for i, col in enumerate(cols) if col not in target_names])
        elif getattr(self, "cluster_aggregate_enable", False) and num_fs_steps == 0:
            # cluster_aggregate operates on raw ``feature_names_in_`` columns
            # (correlation-based clusters) and does NOT need ``selected_vars``.
            # When screening returns 0 features (heavily-correlated reflection
            # clusters routinely trigger this, since every member's marginal
            # MI is near-zero), the legacy ``return None`` skipped the
            # cluster-aggregate step too -- the test contract (and the
            # documented "ON by default and fires on a clean reflection
            # cluster" promise) requires running it regardless. Continue with
            # an empty ``selected_vars`` and let the cluster-aggregate block
            # at the bottom of this function fire; the pair / hermite blocks
            # in between are no-ops on empty ``selected_vars``.
            logger.info(
                "Screening returned 0 features but cluster_aggregate_enable=True; "
                "running cluster-aggregate step anyway (operates on raw "
                "feature_names_in_, not on selected_vars).",
            )
            selected_vars = []
            # 2026-06-03 (wave-9 follow-up, default_filtering.py:165): screening
            # selected nothing but we continue (interaction-only signal / cluster
            # aggregate). Flag this so the smart-polynom optimiser does NOT treat
            # the raw-seeded pool as "speculative synergy" to withhold (which
            # would exclude EVERY pair and the polynom search would never fire).
            _screening_returned_empty = True
        elif _synergy_bootstrap_can_supply_pool(self, num_fs_steps, data):
            # INTERACTION-ONLY SIGNAL (2026-06-05): screening returned 0 features because every operand of a pure-interaction target (a*b + c*d + ...) has ~0 MARGINAL MI -- and the
            # empirical-null debiasing (Fix B) now correctly demotes those near-zero marginals to exactly 0, so even the weak pre-debiasing marginal that used to slip one operand
            # through the screen no longer does. The signal is entirely in the PAIRS, which the synergy bootstrap below surfaces by seeding the all-pairs joint-MI sweep with the raw
            # numeric columns. Continue into the FE step (rather than skipping it) so that sweep + the smart-polynom / pair search can recover the interaction features. Without this,
            # the default-on DCD suppresses the cluster_aggregate fallback branch above, so a pure-interaction frame would silently engineer NOTHING. ``_screening_returned_empty``
            # disables the speculative-synergy withhold downstream (every operand is bootstrap-added here, so the withhold would otherwise drop every pair).
            logger.info(
                "Screening returned 0 features but the synergy bootstrap can seed an interaction-only pool "
                "(fe_synergy_screen_max_features>0); running the FE pair / smart-polynom search anyway.",
            )
            selected_vars = []
            _screening_returned_empty = True
        else:
            logger.info("Skipping Feature Engineering (screening returned 0 features and fe_fallback_to_all=False).")
            return None

    # RAW-RETENTION capture (2026-06-03): record the SCREENING-confirmed genuine
    # raw features on the first FE step. The post-FE re-selection can drop a
    # screening-confirmed (permutation-validated) weak feature when an engineered
    # feature absorbs its signal as a redundant near-duplicate (measured: a genuine
    # X5 / pair operand absorbed into a noise-paired engineered feature, raw column
    # dropped). ``MRMR.fit`` re-adds these at support finalisation unless a
    # SINGLE-PARENT engineered child substitutes them (the prefer-engineered case).
    if num_fs_steps == 0:
        try:
            _raw_set = set(self.feature_names_in_)
            self._prefe_screened_raw_ = [cols[v] for v in selected_vars if cols[v] in _raw_set]
        except Exception:
            self._prefe_screened_raw_ = []

    n_recommended_features = 0
    if verbose >= 2:
        logger.info("Computing prospective FE pairs...")

    # FE operand-pool construction (carved 2026-06-22 to _step_pool.py): builds numeric_vars_to_consider
    # from selected_vars + runs the synergy / GBM / gradient seeders. Selection is byte-for-byte identical.
    from ._step_pool import build_fe_operand_pool
    numeric_vars_to_consider, _synergy_added_idx = build_fe_operand_pool(
        self,
        selected_vars=selected_vars,
        categorical_vars=categorical_vars,
        cols=cols, X=X, data=data, nbins=nbins,
        target_indices=target_indices,
        classes_y=classes_y, freqs_y=freqs_y,
        cached_MIs=cached_MIs,
        num_fs_steps=num_fs_steps,
        verbose=verbose,
    )
    # Operand-pool feed-forward cap + batch/per-pair joint-MI computation + order-2 maxT floor (carved
    # 2026-06-22 to _step_pairmi.py). Mutates cached_MIs in place and returns the re-bound pool + gate
    # state; selection is byte-for-byte identical.
    from ._step_pairmi import compute_pair_mis_and_floor
    (
        numeric_vars_to_consider, _eng_cap, _pair_maxt_floor, _pair_mm_bias, _prevalence_debias_auto,
    ) = compute_pair_mis_and_floor(
        self,
        data=data, cols=cols, nbins=nbins, X=X,
        classes_y=classes_y, classes_y_safe=classes_y_safe, freqs_y=freqs_y,
        target_indices=target_indices,
        cached_MIs=cached_MIs, cached_confident_MIs=cached_confident_MIs,
        numeric_vars_to_consider=numeric_vars_to_consider,
        _prevalence_debias_auto=_prevalence_debias_auto,
        n_jobs=n_jobs, prefetch_factor=prefetch_factor, parallel_kwargs=parallel_kwargs,
        fe_min_nonzero_confidence=fe_min_nonzero_confidence, fe_npermutations=fe_npermutations,
        fe_min_pair_mi=fe_min_pair_mi, fe_min_pair_mi_prevalence=fe_min_pair_mi_prevalence,
        verbose=verbose,
    )
    # ---------------------------------------------------------------------------------------------------------------
    # For every pair of factors (A,B), select ones having MI((A,B),Y)>MI(A,Y)+MI(B,Y). Such ones must possess more special connection!
    # ---------------------------------------------------------------------------------------------------------------

    # Per-pair joint-MI uplift + order-2 maxT scoring of the candidate pairs (carved 2026-06-22 to
    # _step_pairs_rank.py). Returns the prospective_pairs ranking dict + the prevalence-failed synergy
    # rescue ledger; selection is byte-for-byte identical.
    from ._step_pairs_rank import score_prospective_pairs
    prospective_pairs, _prevalence_failed_synergy = score_prospective_pairs(
        self,
        cached_MIs=cached_MIs,
        numeric_vars_to_consider=numeric_vars_to_consider,
        checked_pairs=checked_pairs,
        _pair_mm_bias=_pair_mm_bias,
        _pair_maxt_floor=_pair_maxt_floor,
        _synergy_added_idx=_synergy_added_idx,
        fe_min_pair_mi_prevalence=fe_min_pair_mi_prevalence,
        _synergy_prev_resolved=_synergy_prev_resolved,
        _prevalence_debias_auto=_prevalence_debias_auto,
        data=data,
        classes_y=classes_y,
        X=X,
        cols=cols,
        num_fs_steps=num_fs_steps,
        verbose=verbose,
        sort_dict_by_value=sort_dict_by_value,
    )

    # SIGNED INTERACTION-INFORMATION ROUTING (2026-06-09, backlog idea #8). Among the
    # pairs that PASSED the ratio gate + order-2 maxT floor above, separate genuine
    # synergy (II > null floor) from the ADDITIVE cross-mix (II <= floor: a feeds one
    # independent term of y, b a DIFFERENT one -- the weak-F2 ``add(invqubed(a),
    # invsqrt(c))`` surrogate that mixes an (a,b)-term operand with a (c,d)-term operand)
    # and from REDUNDANCY (II < 0). ``II(a;b;y) = I((a,b);y) - I(a;y) - I(b;y)`` is the
    # signed difference of terms the gate ALREADY computed (cached marginals + pair_mi),
    # Miller-Madow corrected per term so the joint's nbins_a*nbins_b-bin finite-sample
    # bias does not masquerade as synergy. Positive II is floored by a permutation null on
    # the per-shuffle MAX II (the additive-completion + finite-sample chance ceiling).
    # DEMOTE additive speculative (synergy-added) cross-mix pairs out of the FE search so
    # no cross-mix surrogate is built; this is a ROUTING change, the maxT floor + ratio
    # gate stay as the detection guards. Runs BEFORE the synergy budget cap so demoted
    # pairs do not consume budget. SELF-GATING => byte-stable on narrow pools / when
    # disabled. See ``_interaction_information.py``.
    #
    # bench-rejected (2026-06-09) as a DEFAULT (now ``fe_ii_routing_enable=False``): on the
    # user's WEAK F2 the cross-mix pair (b,c) has HIGHER interaction information than the
    # genuine (a,b) a**2/b pair on every cross-mix seed (II +0.0132/+0.0135/+0.0139 vs
    # +0.0114/+0.0120/+0.0132 at n=20000, seeds 0/6/8), and the y-shuffle null floor sits at
    # ~0.0007 -- so no II threshold demotes the cross-mix without also dropping the genuine
    # pair. F2 10-seed result NEUTRAL (cross_mix 3/10 -> 3/10). The router is correct on
    # STRONG synergy (synthetic synergy II +0.55 vs additive +0.03 below floor) and stays an
    # opt-in; see the default-off rationale in ``mrmr.py`` (fe_ii_routing_enable). The call
    # below is a structural no-op while disabled (returns the un-routed pairs unchanged).
    prospective_pairs = apply_interaction_information_routing(
        self,
        prospective_pairs=prospective_pairs,
        cached_MIs=cached_MIs,
        nbins=nbins,
        freqs_y=freqs_y,
        classes_y=classes_y,
        data=data,
        synergy_added_idx=_synergy_added_idx,
        verbose=verbose,
    )

    # SYNERGY-PAIR BUDGET (2026-06-02): cap the synergy pairs (>=1 operand is a
    # bootstrap-added unselected column) at ``fe_synergy_max_pairs`` top-joint-MI
    # pairs before the expensive per-pair search, so a noise-heavy frame cannot
    # flood ``check_prospective_fe_pairs``. Selected-selected pairs are kept in
    # full. ``key`` is ``(raw_vars_pair, pair_mi)``; rank synergy pairs by pair_mi.
    if _synergy_added_idx:
        _synergy_budget = int(getattr(self, "fe_synergy_max_pairs", 16) or 0)
        _synergy_keys = [k for k in prospective_pairs if (k[0][0] in _synergy_added_idx or k[0][1] in _synergy_added_idx)]
        if _synergy_budget >= 0 and len(_synergy_keys) > _synergy_budget:
            _keep_synergy = set(sorted(_synergy_keys, key=lambda k: k[1], reverse=True)[:_synergy_budget])
            _dropped = 0
            for k in _synergy_keys:
                if k not in _keep_synergy:
                    del prospective_pairs[k]
                    _dropped += 1
            if verbose and _dropped:
                logger.info(
                    "MRMR FE synergy bootstrap: kept top %d synergy pairs by joint MI, "
                    "dropped %d below budget (fe_synergy_max_pairs) to bound FE search cost.",
                    min(_synergy_budget, len(_synergy_keys)), _dropped,
                )

    # Now need to sort prospective_pairs by the uplift, to check most promising pairs within the time budget.
    # Also need to sort them by their members usage frequency+members ids sum. this way, their splitting will benefit more from caching.
    prospective_pairs = sort_dict_by_value(prospective_pairs, reverse=True)

    # SUCCESSIVE-HALVING / RUNG-SCHEDULE FE-search budget (2026-06-10, backlog #16).
    # ON by default. Before the EXPENSIVE per-pair operator search below
    # (``check_prospective_fe_pairs`` -- all unary x binary transforms / CMA-ES / full
    # discretize / prewarp, ~4-50s per pair), run a CHEAP rung-0 SCREEN: rank the
    # gate-surviving prospective pairs by their JOINT MI ``pair_mi`` (key[1] -- a
    # monotone-ish proxy of the operator-search outcome ALREADY computed by the pair-MI
    # gate, so the screen is FREE) and keep only the top fraction (UNION a relative-MI
    # floor so a moderate-MI genuine winner is never cut). The expensive search then runs
    # on the survivors only. GATES UNCHANGED -- this changes WHERE the compute goes, not
    # admission (a pair the rung-0 screen drops simply never gets the operator search,
    # exactly as the synergy budget already caps synergy pairs; this generalises that to
    # the whole pool). Measured 1.7-2.2x at keep_frac=0.5 with NO genuine signal pair
    # dropped (n=5000/p=40 canonical fixture + noise, 5 seeds). Self-gates to a no-op
    # below ``fe_rung_min_pairs`` pairs / all-zero pair_mi (byte-identical flat sweep).
    if bool(getattr(self, "fe_rung_schedule_enable", True)) and len(prospective_pairs) >= int(getattr(self, "fe_rung_min_pairs", 6)):
        from .._fe_rung_schedule import apply_rung_schedule
        _rung_n_rows = int(data.shape[0]) if hasattr(data, "shape") else 0
        prospective_pairs, _rung_info = apply_rung_schedule(
            prospective_pairs,
            n_rows=_rung_n_rows,
            keep_frac=getattr(self, "fe_rung_keep_frac", None),
            rel_floor=float(getattr(self, "fe_rung_rel_floor", 0.40)),
            min_pairs=int(getattr(self, "fe_rung_min_pairs", 6)),
            verbose=verbose,
        )

    # cols-space indices of polynom-pair engineered columns appended by the
    # ``run_polynom_pair_fe`` block below; promoted into ``selected_vars``
    # alongside the unary/binary indices so a polynom feature that cleared the
    # FE gates actually reaches ``support_`` (see promotion at the bottom).
    _polynom_engineered_indices: list[int] = []
    if fe_smart_polynom_iters:
        # Orthogonal-polynomial pair FE: Chebyshev default basis (empirically robust); tight coef range [-2, 2],
        # fixed degree per study, L2 regularisation, identity-baseline filter. Override basis via
        # ``self.fe_polynomial_basis``. See feature_selection.filters.hermite_fe and bench_polynomial_bases.
        #
        # 2026-05-18: extracted from inline ~200 LOC block into
        # ``polynom_pair_fe.run_polynom_pair_fe`` (joblib-threaded pair
        # eval + serial inject). ``self._hermite_features_`` is fed
        # through as a target list so the helper stays method-free.
        from ..polynom_pair_fe import run_polynom_pair_fe
        if not hasattr(self, "_hermite_features_"):
            self._hermite_features_ = []
        # SYNERGY pairs feed the STANDARD unary/binary+prewarp search below, but NOT
        # this orthogonal-poly optimiser. The synergy bootstrap adds SPECULATIVE
        # pairs (>=1 unselected, often-noise operand); the cma_batch / optuna
        # optimiser here is powerful enough to fit a high-MI atan2/poly cell to
        # PURE NOISE on such a pair (the saturating-penalty relaxation makes that
        # reachable), fabricating a spurious feature. Every measured synergy WIN
        # (sign_prod, gauss_prod, ratio_abs, ...) is recovered by the standard
        # mul/div search, NOT a _polynom_ cell, so withholding synergy pairs from
        # the optimiser loses no recovery while keeping the pure-noise control clean.
        _prospective_for_polynom = prospective_pairs
        # Withhold SPECULATIVE synergy pairs from the powerful poly optimiser
        # (it can fit a high-MI cell to pure noise on a noise-operand pair) --
        # but ONLY when there was a genuine selected pool to augment. When
        # screening returned 0 features (interaction-only signal), EVERY operand
        # is "synergy-added", so this exclusion would withhold every pair and the
        # polynom search would never fire (default_filtering.py:165). In that
        # case keep the pairs: they ARE the signal, and the synergy max-pairs cap
        # + the downstream pair-MI / engineered-MI / uplift gates already bound
        # the pure-noise risk.
        if _synergy_added_idx and not _screening_returned_empty:
            _filtered_for_polynom = {k: v for k, v in prospective_pairs.items() if not (k[0][0] in _synergy_added_idx or k[0][1] in _synergy_added_idx)}
            # 2026-06-03 (wave-9 follow-up, default_filtering.py:165): apply the
            # speculative-synergy exclusion ONLY if it leaves a non-empty pool.
            # When the selected pool is too small to form any NON-synergy pair
            # (screening kept 0-1 features on an interaction-only target, so
            # every surviving pair has a synergy-added operand), excluding them
            # would withhold EVERY pair and silently disable the polynom search
            # -- yet those pairs ARE the signal. Keep them in that case; the
            # synergy max-pairs cap + the downstream pair-MI / engineered-MI /
            # uplift gates already bound the pure-noise risk.
            if _filtered_for_polynom:
                _prospective_for_polynom = _filtered_for_polynom
        # None / 0 / negative all map to "no subsample" (use full data).
        _subsample_raw = getattr(self, "fe_smart_polynom_subsample_n", 0)
        _subsample_n = int(_subsample_raw) if _subsample_raw and _subsample_raw > 0 else 0
        # ONE shared FE subsample for this fit (2026-06-25): resolve + cache the single row-index draw
        # ONCE here, so the polynom path, the pair-search and the sufficient-summary floor all REUSE it
        # (same rows everywhere) instead of each drawing its own. ``classes_y`` is the discrete fit
        # target; the helper returns None at small n (<= unified screen size) -> legacy per-call draw.
        try:
            from .._fe_sufficient_summary import _get_shared_fe_subsample_idx
            _shared_fe_idx = _get_shared_fe_subsample_idx(self, np.asarray(classes_y), int(getattr(X, "shape", [len(X)])[0]))
        except Exception as _sub_exc:
            # Full-n fallback is safe but ~33x slower at n~1M -> log so it is never a silent mystery.
            logger.warning("mrmr: shared FE subsample resolution failed in FE step; running at FULL n: %r", _sub_exc, exc_info=True)
            _shared_fe_idx = None
        # Capture cols width before the polynom block so we can promote the
        # polynom-injected engineered column indices into ``selected_vars``
        # below (same "ROOT CAUSE 5" promotion the unary/binary block does for
        # its own appended cols). Without this, a polynom-pair feature that
        # cleared every polynom-FE gate (pair-MI prevalence + engineered-MI
        # prevalence + uplift) was appended to ``data``/``cols`` and tracked in
        # ``_hermite_features_`` but never reached ``support_`` under the default
        # single-step path, because only the unary/binary indices were promoted.
        _n_cols_before_polynom = len(cols)
        data, nbins, cols, X = run_polynom_pair_fe(
            X=X, is_polars_input=_is_polars_input,
            prospective_pairs=_prospective_for_polynom,
            classes_y=classes_y,
            cols=cols, nbins=nbins, data=data,
            engineered_features=engineered_features,
            engineered_recipes=engineered_recipes,
            hermite_features_list=self._hermite_features_,
            feature_names_in=self.feature_names_in_,
            fe_smart_polynom_iters=fe_smart_polynom_iters,
            fe_smart_polynom_optimization_steps=fe_smart_polynom_optimization_steps,
            fe_min_polynom_degree=fe_min_polynom_degree,
            fe_max_polynom_degree=fe_max_polynom_degree,
            fe_min_polynom_coeff=fe_min_polynom_coeff,
            fe_max_polynom_coeff=fe_max_polynom_coeff,
            fe_min_engineered_mi_prevalence=fe_min_engineered_mi_prevalence,
            fe_hermite_l2_penalty=getattr(self, "fe_hermite_l2_penalty", 0.05),
            fe_polynomial_basis=getattr(self, "fe_polynomial_basis", "chebyshev"),
            fe_mi_estimator=getattr(self, "fe_mi_estimator", "plugin"),
            # 2026-05-22: cma_batch is the new default (20.58x faster than
            # optuna, 1.09x faster than per-solution cma, within_1%=1.00
            # on a 12-pair benchmark). See profiling/bench_polynom_optimizers.py.
            fe_optimizer=getattr(self, "fe_optimizer", "cma_batch"),
            fe_warm_start=getattr(self, "fe_warm_start", True),
            fe_multi_fidelity=getattr(self, "fe_multi_fidelity", True),
            quantization_nbins=self.quantization_nbins,
            quantization_method=self.quantization_method,
            quantization_dtype=self.quantization_dtype,
            n_jobs=int(n_jobs) if n_jobs and n_jobs > 0 else 1,
            verbose=int(verbose),
            subsample_n=_subsample_n,
            fe_subsample_stratify=_fe_subsample_stratify,
            shared_subsample_idx=_shared_fe_idx,
            # Cheap-first dispatch: skip the expensive CMA/Optuna search for pairs
            # whose trivial baseline already saturates the joint-MI ceiling. getattr
            # default keeps the knob optional (no ctor flag required); 1.0 disables.
            poly_cheap_skip_ratio=float(getattr(self, "fe_poly_cheap_skip_ratio", 0.97)),
            # Linear-usability guard on the skip: only skip when the trivial feature
            # is ALSO linearly useful (|corr| to y >= this), so an MI-saturated-but-
            # non-linear trivial (atan2, etc.) still falls through to the optimiser.
            poly_cheap_skip_min_corr=float(getattr(self, "fe_poly_cheap_skip_min_corr", 0.90)),
        )
        # Columns appended by the polynom block (its own gates already accepted
        # them). Promote into selected_vars below so they reach support_.
        _polynom_engineered_indices = list(range(_n_cols_before_polynom, len(cols)))

    # The standard check_prospective_fe_pairs path used to live in
    # ``else:`` of the Hermite block, which meant enabling
    # ``fe_smart_polynom_iters > 0`` silently DISABLED all standard
    # unary/binary FE (cbrt, sqrt, log, hypot, atan2, ...). De-dented the
    # block so the standard pipeline always runs after the Hermite block;
    # users get the unary/binary FE they asked for via
    # ``fe_unary_preset='medium'`` regardless of whether Hermite ran.
    # feature_names_in_ is an ndarray (sklearn convention) -- list() once so .index() works (ndarray has none)
    # and the comprehension below doesn't rebuild it per element.
    _fni_list = list(self.feature_names_in_)
    # name -> index map built once (O(F)) instead of an ``in`` test + ``.index()`` rescan of
    # ``_fni_list`` per ``col`` (O(F) each) -- turns the O(K*F) lookup below into O(K+F).
    _fni_idx = {nm: i for i, nm in enumerate(_fni_list)}
    original_cols = {i: _fni_idx[col] for i, col in enumerate(cols) if col in _fni_idx}
    if verbose >= 1:
        logger.info("Checking %d most prospective_pairs for feature engineering...", len(prospective_pairs))

    # PER-OPERAND PRE-WARP (2026-06-02): read the opt-in flag + knobs off the
    # MRMR instance (getattr keeps _run_fe_step's signature stable, mirroring
    # ``fe_check_pairs_subsample_n``). When enabled, the discretised target
    # (``classes_y`` -- the SAME codes the MI sweep scores against) is handed
    # to ``check_prospective_fe_pairs`` so it can fit a learned 1-D pre-warp
    # per operand; ``_prewarp_specs`` collects the fitted coeffs (by cols-space
    # var index) for leak-safe recipe construction. Default OFF.
    _prewarp_enable = bool(getattr(self, "fe_pair_prewarp_enable", False))
    # CONTINUOUS ALS RECONSTRUCTION TARGET (2026-06-11): the raw continuous y
    # stashed by ``_fit_impl`` so the rank-1 ALS warp reconstructs against the
    # faithful continuous target instead of the coarse target-rebin-guard codes.
    # Aligned to the FE-step row count (full-n; ``check_prospective_fe_pairs``
    # handles any internal subsample). None -> ALS falls back to ``classes_y``.
    _prewarp_y_cont = None
    if _prewarp_enable:
        _pwc = getattr(self, "_fe_prewarp_y_continuous_", None)
        if _pwc is not None and len(_pwc) == len(classes_y):
            _prewarp_y_cont = _pwc
    # LINEAR-USABILITY GUARD TARGET (2026-06-17): the leader tie-break + noise-wrap |corr|
    # guard must score against CONTINUOUS y regardless of prewarp -- the binned ``classes_y``
    # fallback INVERTS linear usability on heavy-tailed targets (picks ``a/sqrt(b)`` over
    # ``a**2/b``). Threaded unconditionally; None for classification/non-numeric y (correct
    # ``classes_y`` fallback inside ``check_prospective_fe_pairs``).
    _usab_y_cont = None
    _uyc = getattr(self, "_fe_prewarp_y_continuous_", None)
    if _uyc is not None and len(_uyc) == len(classes_y):
        _usab_y_cont = _uyc
    _prewarp_basis = str(getattr(self, "fe_pair_prewarp_basis", "chebyshev"))
    _prewarp_max_degree = int(getattr(self, "fe_pair_prewarp_max_degree", 4))
    _prewarp_uplift = float(getattr(self, "fe_pair_prewarp_uplift_threshold", 1.20))
    _prewarp_min_val_corr = float(getattr(self, "fe_pair_prewarp_min_val_corr", 0.08))
    # PREWARP-SPEC PERSISTENCE (2026-06-08 fix). ``_mrmr_fe_step`` is re-entered
    # once per FE-bearing MRMR iteration, and each call's ``check_prospective_fe_pairs``
    # fits prewarp specs ONLY for the operands of THIS iteration's prospective pairs
    # (a var whose pair isn't prospective this round is not re-fit). But the recipe
    # build below runs per iteration over ``this_pair_features`` and can construct a
    # recipe whose operand used the ``prewarp`` pseudo-unary in an EARLIER iteration --
    # if ``_prewarp_specs`` were a fresh per-call dict, that operand's coeffs would be
    # absent and the recipe would be built with ``prewarp_a/b=None``, producing a
    # recipe that names ``prewarp`` but lacks ``prewarp_*_coef`` in ``extra`` -> a
    # KeyError at transform()-time replay (observed: ``sub(prewarp(informative_3),
    # prewarp(noise_5))`` with informative_3's spec missing). Back the local dict with
    # a SELF-LEVEL accumulator (mirrors ``self._engineered_continuous_``) so every
    # spec fit in any prior iteration stays available for recipe construction. Seeded
    # from the accumulator and written back after each call; specs are keyed by
    # cols-space var index, which is stable across iterations (no cat reorder mid-fit).
    _prewarp_specs: dict | None = getattr(self, "_prewarp_specs_accum_", None)
    if _prewarp_specs is None:
        _prewarp_specs = {}
        self._prewarp_specs_accum_ = _prewarp_specs

    # PER-OPERAND MEDIAN GATE (2026-06-04): opt-in flag off the MRMR instance
    # (getattr keeps the signature stable, mirroring the prewarp wiring). When
    # enabled, ``check_prospective_fe_pairs`` fits one TRAIN median per operand
    # and exposes a ``gate_med`` pseudo-unary; ``_gate_med_specs`` collects the
    # fitted medians (by cols-space var index) for leak-safe recipe
    # construction. Default OFF -> byte-identical legacy path. Same cross-iteration
    # persistence as the prewarp specs above (a gate_med operand selected in an
    # earlier iteration must keep its median available for recipe replay).
    _gate_med_enable = bool(getattr(self, "fe_gate_med_enable", False))
    # MULTI-CANDIDATE DIVERSE EMISSION (2026-06-12): per pair, emit up to this many
    # DISTINCT engineered forms (MI is rank-blind to linear usability, so a single
    # MI-winner can be tree-friendly-but-linearly-useless while a lower-MI form is the
    # linearly-usable one; both should survive for the downstream model to choose).
    _multi_emit_max = int(getattr(self, "fe_multi_emit_max_per_pair", 1))
    _multi_emit_floor = float(getattr(self, "fe_multi_emit_mi_floor", 0.5))
    _multi_emit_div_corr = float(getattr(self, "fe_multi_emit_diversity_corr", 0.90))
    _gate_med_specs: dict | None = getattr(self, "_gate_med_specs_accum_", None)
    if _gate_med_specs is None:
        _gate_med_specs = {}
        self._gate_med_specs_accum_ = _gate_med_specs

    # SERIAL-vs-JOBLIB DISPATCH (2026-06-08 narrow+tall fix). The ``else`` branch
    # spreads the prospective PAIRS across ``n_jobs`` joblib ``backend="threading"``
    # workers, each running the SERIAL (no-prange) FE kernels (a numba prange would
    # nest inside the threading layer and deadlock -- see ``_fe_use_parallel_kernels``).
    #
    # BACKEND BENCH (2026-06-19, n=8000 p=150 FE pair-search, 3-rep medians, peak RSS incl children):
    #   joblib-threading 8.98s/1599MB  <  serial 10.75s/1321MB  <  cf-ThreadPool 10.22s/1898MB
    #   joblib-loky 39.1s/2948MB (3.9x slower, +1.7GB frame copies); multiprocessing/ProcessPool OOM-cascade.
    # => the current joblib backend="threading" is the MEASURED-FASTEST option (~16% over serial, the njit MI
    # kernels release the GIL enough to win despite the Python orchestration); process backends are a
    # non-starter (frame deep-copy memory + Windows spawn cascade). Do NOT serial-ize or switch to processes.
    # (A single-run measurement misleadingly showed serial ahead; the 3-rep median is authoritative.)
    # That is the right lever ONLY when there are enough pairs to actually fill the
    # worker pool. In the NARROW+TALL regime (few features -> 1-2 prospective pairs,
    # but n large) the joblib path span only 1-2 jobs over 16 threads (14 idle) AND
    # each job is pinned to a single-core serial kernel -> ~1-2 of 16 cores busy, the
    # measured 56s-per-pair / fully-idle-HW pathology. The serial-main-thread path
    # instead runs the WHOLE candidate sweep on the main thread where a numba prange
    # is safe, so the materialise / searchsorted / MI kernels dispatch to their
    # BYTE-IDENTICAL ``parallel=True`` column-prange twins and spread each pair's K
    # candidates across ALL cores. So route to it whenever there are FEWER pairs than
    # workers (joblib-over-pairs cannot saturate the pool): selection is unchanged
    # (same ``check_prospective_fe_pairs`` over the same pairs; only kernel
    # parallelism + chunk-merge differ, both byte-identical). The pair-count crossover
    # subsumes the old fixed ``len(X) < 50000`` row gate -- a tall narrow frame now
    # takes the all-cores path it always should have.
    _fe_serial_min_pairs_per_worker = max(2, int(n_jobs) if n_jobs and n_jobs > 0 else 1)
    # GPU-FE SERIALIZE (2026-06-20): when the per-pair candidate materialise/binning runs on the
    # GPU, the single device is the bottleneck resource -- spreading the pair-chunks across joblib
    # ``backend="threading"`` workers does NOT parallelize the GPU work, it just makes every worker
    # CONTEND for the one device while the joblib parent burns the wait in its ``_retrieve`` sleep-
    # poll. Measured (canonical n=100k FE fit): default multi-worker = 105s of which ~84s is
    # ``time.sleep`` in joblib ``_retrieve``; the serial-main-thread path (no joblib, prange kernels,
    # GPU work issued from one thread) = 63s. So route to the serial path whenever the GPU discretize
    # path is active, exactly as we already do when there are fewer pairs than workers. The threading
    # win the ``else`` branch was tuned for is the CPU-njit-MI regime (small n, GPU gate OFF), which
    # this predicate leaves untouched. Selection is unchanged (same check_prospective_fe_pairs over
    # the same pairs; only the kernel parallelism + chunk-merge differ, both byte-identical).
    # Ensure the ONE shared FE subsample draw is bound on the pair-search path too (the polynom
    # block above that first resolves it may be skipped when the smart-polynom search is off). The
    # helper caches on the instance keyed by n, so this is idempotent -- same cached draw, no re-draw.
    try:
        from .._fe_sufficient_summary import _get_shared_fe_subsample_idx
        _shared_fe_idx = _get_shared_fe_subsample_idx(self, np.asarray(classes_y), int(getattr(X, "shape", [len(X)])[0]))
    except Exception as _sub_exc:
        # Full-n fallback is safe but ~33x slower at n~1M -> log so it is never a silent mystery.
        logger.warning("mrmr: shared FE subsample resolution failed in FE step; running at FULL n: %r", _sub_exc, exc_info=True)
        _shared_fe_idx = None
    try:
        from .._feature_engineering_pairs._pairs_core import _fe_gpu_discretize_enabled as _fe_gpu_disc_gate
        # Representative candidate-count for the gate's n*K crossover (a pair generates dozens-to-
        # hundreds of unary/binary candidates); the auto gate is CUDA-presence-gated internally.
        _gpu_fe_active = bool(_fe_gpu_disc_gate(int(getattr(X, "shape", [0])[0] or 0), 256))
    except Exception:
        _gpu_fe_active = False
    if _should_serialize_fe_pair_check(len(prospective_pairs), _gpu_fe_active, _fe_serial_min_pairs_per_worker):
        prospective_additions = check_prospective_fe_pairs(
            prospective_pairs,
            X,
            unary_transformations,
            binary_transformations,
            classes_y,
            classes_y_safe,
            freqs_y,
            num_fs_steps,
            cols,
            original_cols,
            fe_max_steps,
            fe_npermutations,
            fe_max_pair_features,
            fe_print_best_mis_only,
            fe_min_nonzero_confidence,
            fe_min_engineered_mi_prevalence,
            fe_good_to_best_feature_mi_threshold,
            fe_max_external_validation_factors,
            numeric_vars_to_consider,
            self.quantization_nbins,
            self.quantization_method,
            self.quantization_dtype,
            times_spent,
            verbose,
            subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
            subsample_seed=int(getattr(self, "random_seed", 0) or 0),
            fe_subsample_stratify=_fe_subsample_stratify,
            shared_subsample_idx=_shared_fe_idx,
            prewarp_enable=_prewarp_enable,
            prewarp_y=classes_y if _prewarp_enable else None,
            prewarp_y_continuous=_prewarp_y_cont if _prewarp_enable else None,
            usability_y_continuous=_usab_y_cont,
            prewarp_basis=_prewarp_basis,
            prewarp_max_degree=_prewarp_max_degree,
            prewarp_uplift_threshold=_prewarp_uplift,
            prewarp_min_val_corr=_prewarp_min_val_corr,
            prewarp_specs_out=_prewarp_specs,
            fe_gate_med_enable=_gate_med_enable,
            fe_multi_emit_max_per_pair=_multi_emit_max,
            fe_multi_emit_mi_floor=_multi_emit_floor,
            fe_multi_emit_diversity_corr=_multi_emit_div_corr,
            fe_pair_usability_admission_enable=bool(getattr(self, "fe_pair_usability_admission_enable", True)),
            fe_pair_usability_admission_min_corr=float(getattr(self, "fe_pair_usability_admission_min_corr", 0.6)),
            fe_pair_usability_admission_pairness_margin=float(getattr(self, "fe_pair_usability_admission_pairness_margin", 1.05)),
            gate_med_specs_out=_gate_med_specs,
            # OPT-A (2026-06-07): THIS is the serial-main-thread branch -- the whole FE
            # search runs here with NO joblib threads (the ``else`` below is the
            # ``len(X) >= 50000`` joblib ``backend="threading"`` path). On this branch a
            # numba prange does not nest inside Python threads, so the FE materialise /
            # searchsorted kernels may dispatch to their byte-identical ``parallel=True``
            # column-prange twins (gated further by the per-host crossover). The joblib
            # path below leaves ``serial_main_thread`` at its default False -> serial kernels.
            serial_main_thread=True,
            # ENGINEERED-OPERAND FEED-FORWARD (2026-06-08): resolve engineered operands by
            # name so (eng_i, eng_j) composites materialise. Off when the cap is 0 (raw-only
            # pool); the pool already excludes them in that case. ``engineered_operand_values``
            # supplies the CONTINUOUS engineered values so the composite is built on them
            # rather than the lossy bin codes.
            allow_engineered_operands=(_eng_cap != 0),
            engineered_operand_values=getattr(self, "_engineered_continuous_", None),
            # MM-DEBIAS (2026-06-09, backlog #1 + #4): debias the joint-prevalence
            # ratio gate (see check_prospective_fe_pairs). Co-updated with the maxT
            # floor below (IRON RULE). Default-on; ``fe_mm_debias_prevalence=False``
            # byte-reproduces pre-2026-06-09 fits.
            fe_mm_debias_prevalence=bool(getattr(self, "fe_mm_debias_prevalence", False)),
        )
    else:

        prospective_additions = {}
        desired_nitems = max(1, len(prospective_pairs) // (n_jobs * prefetch_factor))

        jobs_list = []

        nitems = 0
        cur_dict = {}
        for key, value in prospective_pairs.items():
            nitems += 1
            cur_dict[key] = value
            if nitems >= desired_nitems:
                jobs_list.append(cur_dict)
                nitems = 0
                cur_dict = {}
        if cur_dict:
            jobs_list.append(cur_dict)

        if verbose:
            logger.info(
                "Using %d items per thread for checking %d prospective_pairs with gain>%.2f.",
                desired_nitems, len(prospective_pairs), fe_min_pair_mi_prevalence,
            )

        dicts = parallel_run(
            [
                delayed(check_prospective_fe_pairs)(
                    chunk,
                    X,
                    unary_transformations,
                    binary_transformations,
                    classes_y,
                    classes_y_safe,
                    freqs_y,
                    num_fs_steps,
                    cols,
                    original_cols,
                    fe_max_steps,
                    fe_npermutations,
                    fe_max_pair_features,
                    fe_print_best_mis_only,
                    fe_min_nonzero_confidence,
                    fe_min_engineered_mi_prevalence,
                    fe_good_to_best_feature_mi_threshold,
                    fe_max_external_validation_factors,
                    numeric_vars_to_consider,
                    self.quantization_nbins,
                    self.quantization_method,
                    self.quantization_dtype,
                    times_spent,
                    verbose,
                    subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                    subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                    fe_subsample_stratify=_fe_subsample_stratify,
                    shared_subsample_idx=_shared_fe_idx,
                    prewarp_enable=_prewarp_enable,
                    prewarp_y=classes_y if _prewarp_enable else None,
                    prewarp_y_continuous=_prewarp_y_cont if _prewarp_enable else None,
                    usability_y_continuous=_usab_y_cont,
                    prewarp_basis=_prewarp_basis,
                    prewarp_max_degree=_prewarp_max_degree,
                    prewarp_uplift_threshold=_prewarp_uplift,
                    prewarp_min_val_corr=_prewarp_min_val_corr,
                    prewarp_specs_out=None,  # loky: recovered from result dict below
                    fe_gate_med_enable=_gate_med_enable,
                    fe_multi_emit_max_per_pair=_multi_emit_max,
                    fe_multi_emit_mi_floor=_multi_emit_floor,
                    fe_multi_emit_diversity_corr=_multi_emit_div_corr,
                    fe_pair_usability_admission_enable=bool(getattr(self, "fe_pair_usability_admission_enable", True)),
                    fe_pair_usability_admission_min_corr=float(getattr(self, "fe_pair_usability_admission_min_corr", 0.6)),
                    fe_pair_usability_admission_pairness_margin=float(getattr(self, "fe_pair_usability_admission_pairness_margin", 1.05)),
                    gate_med_specs_out=None,  # loky: recovered from result dict below
                    # ENGINEERED-OPERAND FEED-FORWARD (2026-06-08): see the serial branch above.
                    allow_engineered_operands=(_eng_cap != 0),
                    engineered_operand_values=getattr(self, "_engineered_continuous_", None),
                    # MM-DEBIAS (2026-06-09, backlog #1 + #4): see the serial branch above.
                    fe_mm_debias_prevalence=bool(getattr(self, "fe_mm_debias_prevalence", False)),
                    # LARGE-N PEAK-MEMORY FIX (2026-06-08): joblib ``backend="threading"``
                    # runs up to ``n_jobs`` of these CONCURRENTLY in the shared address space,
                    # each allocating its own candidate/chunk/disc/MI buffers; divide the
                    # per-call RAM budget by the worker count so N threads don't collectively
                    # blow past 0.4*available and OOM a worker.
                    concurrent_workers=int(n_jobs) if n_jobs and n_jobs > 0 else 1,
                    # OPT-A n_jobs=1 EXTENSION (2026-06-20). This ``else`` branch is the joblib
                    # path, but joblib ``Parallel(n_jobs=1)`` runs every ``delayed`` chunk
                    # SEQUENTIALLY in the CALLING thread (joblib's SequentialBackend short-circuit
                    # -- no worker thread is spawned, regardless of the requested backend), so
                    # there is NO Python-threading nest a numba prange could deadlock. Mark the
                    # call serial-main-thread when n_jobs<=1 so the FE materialise / searchsorted /
                    # discretise kernels may dispatch to their BYTE-IDENTICAL ``parallel=True``
                    # column-prange twins (gated further by the per-host kernel_tuning crossover) --
                    # the canonical n=100k/n_jobs=1 fit (>=2 pairs -> this branch, but 1 joblib
                    # worker) otherwise ran the serial nogil kernels with every other core idle.
                    # Selection is unchanged (the twins are verified maxdiff 0). For n_jobs>1 the
                    # backend genuinely spawns threads -> stays False (serial nogil, no nesting).
                    serial_main_thread=(n_jobs is None or int(n_jobs) <= 1),
                )
                for chunk in jobs_list
            ],
            # max_nbytes=0,
            n_jobs=n_jobs,
            **parallel_kwargs,
        )
        # PREWARP/GATE-MED SPEC-MERGE FIX (2026-06-08). Each chunk's result dict
        # carries its OWN fitted-spec payload under the SAME reserved key
        # (``_PREWARP_SPECS_RESULT_KEY`` / ``_GATE_MED_SPECS_RESULT_KEY``). A bare
        # ``prospective_additions.update(next_dict)`` lets each chunk's reserved-key
        # entry CLOBBER the previous chunk's -- only the LAST chunk's specs survive,
        # so an operand whose ``prewarp`` warp was fit in an EARLIER chunk loses its
        # coeffs. The recipe builder below then constructs a recipe that names
        # ``prewarp`` but has no ``prewarp_*_coef`` in ``extra`` -> a KeyError at
        # transform()-time replay (observed with n_jobs=1, which still chunks the
        # pairs into multiple sequential ``parallel_run`` jobs: e.g.
        # ``sub(prewarp(informative_3),prewarp(noise_5))`` with informative_3's spec
        # dropped). Fix: MERGE each chunk's reserved spec payload into the
        # accumulators DURING the merge loop, before ``update`` overwrites the key.
        from .._feature_engineering_pairs import _PREWARP_SPECS_RESULT_KEY, _GATE_MED_SPECS_RESULT_KEY, _FE_REJECTION_RESULT_KEY
        for next_dict in dicts:
            _pw_chunk = next_dict.pop(_PREWARP_SPECS_RESULT_KEY, None)
            if _pw_chunk:
                _prewarp_specs.update(_pw_chunk)
            _gm_chunk = next_dict.pop(_GATE_MED_SPECS_RESULT_KEY, None)
            if _gm_chunk:
                _gate_med_specs.update(_gm_chunk)
            # REJECTION LEDGER (additive): drain each chunk's per-pair-gate drops.
            _rej_chunk = next_dict.pop(_FE_REJECTION_RESULT_KEY, None)
            if _rej_chunk:
                for _rr in _rej_chunk:
                    _record_fe_rejection(
                        self, gate=_rr.get("gate", "engineered_mi_prevalence"),
                        candidate=_rr.get("candidate"), operands=_rr.get("operands"),
                        operator=_rr.get("operator"),
                        observed=_rr.get("observed", float("nan")),
                        threshold=_rr.get("threshold", float("nan")),
                        reason=_rr.get("reason", ""), step=int(num_fs_steps),
                    )
            prospective_additions.update(next_dict)

    # Extract any reserved pre-warp / gate-med spec entry the SERIAL path may have
    # left in ``prospective_additions`` (the serial branch returns a single dict that
    # also carries the reserved key) and merge it, so the pair loop below never
    # treats it as a ``raw_vars_pair``. The joblib branch already drained the key
    # per-chunk above; this pop is then a harmless no-op.
    from .._feature_engineering_pairs import _PREWARP_SPECS_RESULT_KEY, _GATE_MED_SPECS_RESULT_KEY, _FE_REJECTION_RESULT_KEY
    _pw_from_res = prospective_additions.pop(_PREWARP_SPECS_RESULT_KEY, None)
    if _pw_from_res:
        _prewarp_specs.update(_pw_from_res)
    # Same recovery for the per-operand TRAIN medians (loky-parallel path).
    _gm_from_res = prospective_additions.pop(_GATE_MED_SPECS_RESULT_KEY, None)
    if _gm_from_res:
        _gate_med_specs.update(_gm_from_res)
    # REJECTION LEDGER (additive): drain the SERIAL path's per-pair-gate drops (the
    # joblib branch already drained per-chunk above, so this pop is then a no-op).
    _rej_from_res = prospective_additions.pop(_FE_REJECTION_RESULT_KEY, None)
    if _rej_from_res:
        for _rr in _rej_from_res:
            _record_fe_rejection(
                self, gate=_rr.get("gate", "engineered_mi_prevalence"),
                candidate=_rr.get("candidate"), operands=_rr.get("operands"),
                operator=_rr.get("operator"),
                observed=_rr.get("observed", float("nan")),
                threshold=_rr.get("threshold", float("nan")),
                reason=_rr.get("reason", ""), step=int(num_fs_steps),
            )

    # Per-candidate scoring / quantile-discretization materialise stage (carved 2026-06-22 to
    # _step_score.py to bring _step_core.py under the 1k-LOC ceiling). Threads the loop locals
    # explicitly and returns the re-bound values; engineered_features / checked_pairs /
    # engineered_recipes are mutated in place. Selection is byte-for-byte identical.
    from ._step_score import materialise_and_finalise_fe_candidates
    (
        prospective_additions, data, cols, nbins, X, selected_vars, n_recommended_features,
    ) = materialise_and_finalise_fe_candidates(
        self,
        prospective_additions=prospective_additions,
        prospective_pairs=prospective_pairs,
        _prevalence_failed_synergy=_prevalence_failed_synergy,
        _pair_maxt_floor=_pair_maxt_floor,
        _polynom_engineered_indices=_polynom_engineered_indices,
        data=data, cols=cols, nbins=nbins, X=X,
        classes_y=classes_y,
        selected_vars=selected_vars,
        engineered_features=engineered_features,
        engineered_recipes=engineered_recipes,
        checked_pairs=checked_pairs,
        n_recommended_features=n_recommended_features,
        num_fs_steps=num_fs_steps,
        fe_max_steps=fe_max_steps,
        fe_unary_preset=fe_unary_preset,
        fe_binary_preset=fe_binary_preset,
        _is_polars_input=_is_polars_input,
        verbose=verbose,
        discretize_array=discretize_array,
        get_new_feature_name=get_new_feature_name,
        # ND-1: the poly_<coef> hermite-coefficient subset so a poly-FE recipe can persist its coef for replay.
        _poly_coefs={_k: _v for _k, _v in unary_transformations.items() if isinstance(_k, str) and _k.startswith("poly_")},
    )

    log_fe_summary(
        prospective_pairs=prospective_pairs,
        prospective_additions=prospective_additions,
        n_recommended_features=n_recommended_features,
        fe_min_pair_mi_prevalence=fe_min_pair_mi_prevalence,
        fe_min_engineered_mi_prevalence=fe_min_engineered_mi_prevalence,
        fe_min_nonzero_confidence=fe_min_nonzero_confidence,
        fe_good_to_best_feature_mi_threshold=fe_good_to_best_feature_mi_threshold,
        fe_acceptance=str(getattr(self, "fe_acceptance", "conditional_mi")),
        verbose=verbose,
    )

    data, cols, nbins, X, selected_vars, n_recommended_features = run_cluster_aggregate_emission(
        self,
        data=data,
        cols=cols,
        nbins=nbins,
        X=X,
        target_indices=target_indices,
        categorical_vars=categorical_vars,
        cached_MIs=cached_MIs,
        engineered_recipes=engineered_recipes,
        selected_vars=selected_vars,
        n_recommended_features=n_recommended_features,
        num_fs_steps=num_fs_steps,
        _is_polars_input=_is_polars_input,
        verbose=verbose,
    )

    # GPU memory-pool teardown (2026-06-27): when the FE step ran GPU kernels (STRICT / CMI_GPU), the cupy
    # default pool RETAINS every block it cached this step for reuse. Across repeated fits (CV folds, repeated
    # MRMR().fit on a 4 GB card) the pool grows toward the device cap and the allocator starts thrashing
    # cudaMalloc/sync -- measured: consecutive 100k f32 STRICT fits degrade 11.2s -> 31.8s -> 32.3s, while
    # freeing the UNUSED blocks at each step teardown holds it flat at ~11.2s (pool total stays well under the
    # cap). free_all_blocks only returns blocks with no live reference, so a resident operand table held across
    # the fit is untouched; this is post-compute teardown, never mid-pipeline. Guarded + best-effort.
    _free_gpu_fe_mempool()

    return data, cols, nbins, X, selected_vars, n_recommended_features
