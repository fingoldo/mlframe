"""Helper blocks extracted from the over-long functions of ``_step_core`` (function-length gate); behaviour is unchanged."""
from __future__ import annotations

from typing import Any


import numpy as np
from joblib import delayed

from .._mrmr_fe_step_helpers import (
    apply_interaction_information_routing,
    log_fe_summary,
    run_cluster_aggregate_emission,
)
from .._fe_rejection_ledger import _drain_fe_rejection_records
from ._helpers import _synergy_bootstrap_can_supply_pool
from ._step_pair_order import order_prospective_pairs


from ._step_core_helpers import (
    logger,
    _run_fe_step_impl_full_key_raw_vars,
    _run_fe_step_impl_below_fe_rung_min,
    _run_fe_step_impl_pure_noise_risk,
    _run_fe_step_impl_handles_any_internal_subsample,
    _run_fe_step_impl_key_value_prospective_pairs,
    _run_fe_step_impl_accumulators_during_merge_loop,
    _run_fe_step_impl_joblib_branch_already_drained,
)


def _run_fe_step_impl_part7(
    verbose: Any, selected_vars: Any, self: Any, st: Any, cols: Any, target_names: Any, num_fs_steps: Any, data: Any, categorical_vars: Any, X: Any,
    nbins: Any, target_indices: Any, classes_y: Any, freqs_y: Any, cached_MIs: Any, classes_y_safe: Any, cached_confident_MIs: Any, n_jobs: Any,
    prefetch_factor: Any, parallel_kwargs: Any, fe_min_nonzero_confidence: Any, fe_npermutations: Any, fe_min_pair_mi: Any, fe_min_pair_mi_prevalence: Any,
    checked_pairs: Any,
) -> tuple[Any, ...]:
    """Extracted block of ``_run_fe_step_impl`` (part 7); returns the names the caller still uses."""
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
    _ret_run_fe_step_impl_part3 = _run_fe_step_impl_part3(
        verbose, selected_vars, self, st, cols, target_names, num_fs_steps, data, categorical_vars, X, nbins, target_indices, classes_y, freqs_y, cached_MIs,
        classes_y_safe, cached_confident_MIs, n_jobs, prefetch_factor, parallel_kwargs, fe_min_nonzero_confidence, fe_npermutations, fe_min_pair_mi,
        fe_min_pair_mi_prevalence, checked_pairs, sort_dict_by_value,
    )
    if _ret_run_fe_step_impl_part3[0]:
        return True, (_ret_run_fe_step_impl_part3[1])
    selected_vars = _ret_run_fe_step_impl_part3[1]

    # SIGNED INTERACTION-INFORMATION ROUTING (backlog idea #8). Among the
    # pairs that PASSED the ratio gate + order-2 maxT floor above, separate genuine
    # synergy (II > null floor) from the ADDITIVE cross-mix (II <= floor: a feeds one
    # independent term of y, b a DIFFERENT one - the weak-F2 ``add(invqubed(a),
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
    # bench-rejected as a DEFAULT (now ``fe_ii_routing_enable=False``): on the
    # user's WEAK F2 the cross-mix pair (b,c) has HIGHER interaction information than the
    # genuine (a,b) a**2/b pair on every cross-mix seed (II +0.0132/+0.0135/+0.0139 vs
    # +0.0114/+0.0120/+0.0132 at n=20000, seeds 0/6/8), and the y-shuffle null floor sits at
    # ~0.0007 - so no II threshold demotes the cross-mix without also dropping the genuine
    # pair. F2 10-seed result NEUTRAL (cross_mix 3/10 -> 3/10). The router is correct on
    # STRONG synergy (synthetic synergy II +0.55 vs additive +0.03 below floor) and stays an
    # opt-in; see the default-off rationale in ``mrmr.py`` (fe_ii_routing_enable). The call
    # below is a structural no-op while disabled (returns the un-routed pairs unchanged).
    st.prospective_pairs = apply_interaction_information_routing(
        self,
        prospective_pairs=st.prospective_pairs,
        cached_MIs=cached_MIs,
        nbins=nbins,
        freqs_y=freqs_y,
        classes_y=classes_y,
        data=data,
        synergy_added_idx=st._synergy_added_idx,
        verbose=verbose,
    )
    return False, check_prospective_fe_pairs, discretize_array, get_new_feature_name, parallel_run, selected_vars


def _run_fe_step_impl_part6(
    materialise_and_finalise_fe_candidates: Any, self: Any, st: Any, data: Any, cols: Any, nbins: Any, X: Any, classes_y: Any, selected_vars: Any,
    engineered_features: Any, engineered_recipes: Any, checked_pairs: Any, num_fs_steps: Any, fe_max_steps: Any, fe_unary_preset: str, fe_binary_preset: str,
    _is_polars_input: Any, verbose: Any, discretize_array: Any, get_new_feature_name: Any, unary_transformations: Any, fe_min_pair_mi_prevalence: Any,
    fe_min_engineered_mi_prevalence: Any, fe_min_nonzero_confidence: Any, fe_good_to_best_feature_mi_threshold: Any, target_indices: Any,
    categorical_vars: Any, cached_MIs: Any,
) -> tuple[Any, ...]:
    """Extracted block of ``_run_fe_step_impl`` (part 6); returns the names the caller still uses."""
    (
        st.prospective_additions, data, cols, nbins, X, selected_vars, st.n_recommended_features,
    ) = materialise_and_finalise_fe_candidates(
        self,
        prospective_additions=st.prospective_additions,
        prospective_pairs=st.prospective_pairs,
        _prevalence_failed_synergy=st._prevalence_failed_synergy,
        _pair_maxt_floor=st._pair_maxt_floor,
        _polynom_engineered_indices=st._polynom_engineered_indices,
        data=data, cols=cols, nbins=nbins, X=X,
        classes_y=classes_y,
        selected_vars=selected_vars,
        engineered_features=engineered_features,
        engineered_recipes=engineered_recipes,
        checked_pairs=checked_pairs,
        n_recommended_features=st.n_recommended_features,
        num_fs_steps=num_fs_steps,
        fe_max_steps=fe_max_steps,
        fe_unary_preset=fe_unary_preset,
        fe_binary_preset=fe_binary_preset,
        _is_polars_input=_is_polars_input,
        verbose=verbose,
        discretize_array=discretize_array,
        get_new_feature_name=get_new_feature_name,
        # The poly_<coef> hermite-coefficient subset, so a poly-FE recipe can persist its coef for replay.
        _poly_coefs={_k: _v for _k, _v in unary_transformations.items() if isinstance(_k, str) and _k.startswith("poly_")},
    )

    log_fe_summary(
        prospective_pairs=st.prospective_pairs,
        prospective_additions=st.prospective_additions,
        n_recommended_features=st.n_recommended_features,
        fe_min_pair_mi_prevalence=fe_min_pair_mi_prevalence,
        fe_min_engineered_mi_prevalence=fe_min_engineered_mi_prevalence,
        fe_min_nonzero_confidence=fe_min_nonzero_confidence,
        fe_good_to_best_feature_mi_threshold=fe_good_to_best_feature_mi_threshold,
        fe_acceptance=str(getattr(self, "fe_acceptance", "conditional_mi")),
        verbose=verbose,
    )

    data, cols, nbins, X, selected_vars, st.n_recommended_features = run_cluster_aggregate_emission(
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
        n_recommended_features=st.n_recommended_features,
        num_fs_steps=num_fs_steps,
        _is_polars_input=_is_polars_input,
        verbose=verbose,
    )

    # GPU memory-pool teardown: when the FE step ran GPU kernels (STRICT / CMI_GPU), the cupy
    # default pool RETAINS every block it cached this step for reuse. Across repeated fits (CV folds, repeated
    # MRMR().fit on a 4 GB card) the pool grows toward the device cap and the allocator starts thrashing
    # cudaMalloc/sync - measured: consecutive 100k f32 STRICT fits degrade 11.2s -> 31.8s -> 32.3s, while
    # freeing the UNUSED blocks at each step teardown holds it flat at ~11.2s (pool total stays well under the
    # cap). free_all_blocks only returns blocks with no live reference, so a resident operand table held across
    # the fit is untouched; this is post-compute teardown, never mid-pipeline. Guarded + best-effort.
    # (moved into a try/finally in the ``_run_fe_step`` wrapper above this function, so it
    # also runs on the exception path - this was previously the only call site and skipped on any raise.)

    return True, (data, cols, nbins, X, selected_vars, st.n_recommended_features)


def _run_fe_step_impl_part5(
    self: Any, st: Any, cols: Any, verbose: Any, classes_y: Any, n_jobs: Any, X: Any, check_prospective_fe_pairs: Any, unary_transformations: Any,
    binary_transformations: Any, classes_y_safe: Any, freqs_y: Any, num_fs_steps: Any, fe_max_steps: Any, fe_npermutations: Any, fe_max_pair_features: Any,
    fe_print_best_mis_only: Any, fe_min_nonzero_confidence: Any, fe_min_engineered_mi_prevalence: Any, fe_good_to_best_feature_mi_threshold: Any,
    fe_max_external_validation_factors: Any, times_spent: Any, prefetch_factor: Any, fe_min_pair_mi_prevalence: Any, parallel_run: Any, parallel_kwargs: Any,
) -> Any:
    """Extracted block of ``_run_fe_step_impl`` (part 5); returns the names the caller still uses."""
    from ._step_core import _should_serialize_fe_pair_check
    _run_fe_step_impl_part1(self, st, cols, verbose, classes_y, n_jobs, X)
    try:
        from .._feature_engineering_pairs._pairs_core import _fe_gpu_discretize_enabled as _fe_gpu_disc_gate
        # Representative candidate-count for the gate's n*K crossover (a pair generates dozens-to-
        # hundreds of unary/binary candidates); the auto gate is CUDA-presence-gated internally.
        st._gpu_fe_active = bool(_fe_gpu_disc_gate(int(getattr(X, "shape", [0])[0] or 0), 256))
    except Exception as exc:
        # Correct but slower path; say so, as the full-n fallback just above does.
        logger.warning("mrmr: GPU-FE-active probe failed (%s: %s); using the serial pair-check dispatch", type(exc).__name__, exc)
        st._gpu_fe_active = False
    if _should_serialize_fe_pair_check(len(st.prospective_pairs), st._gpu_fe_active, st._fe_serial_min_pairs_per_worker):
        st.prospective_additions = check_prospective_fe_pairs(
            st.prospective_pairs,
            X,
            unary_transformations,
            binary_transformations,
            classes_y,
            classes_y_safe,
            freqs_y,
            num_fs_steps,
            cols,
            st.original_cols,
            fe_max_steps,
            fe_npermutations,
            fe_max_pair_features,
            fe_print_best_mis_only,
            fe_min_nonzero_confidence,
            fe_min_engineered_mi_prevalence,
            fe_good_to_best_feature_mi_threshold,
            fe_max_external_validation_factors,
            st.numeric_vars_to_consider,
            self.quantization_nbins,
            self.quantization_method,
            self.quantization_dtype,
            times_spent,
            verbose,
            subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
            subsample_seed=int(getattr(self, "random_seed", 0) or 0),
            fe_subsample_stratify=st._fe_subsample_stratify,
            shared_subsample_idx=st._shared_fe_idx,
            prewarp_enable=st._prewarp_enable,
            prewarp_y=classes_y if st._prewarp_enable else None,
            prewarp_y_continuous=st._prewarp_y_cont if st._prewarp_enable else None,
            usability_y_continuous=st._usab_y_cont,
            prewarp_basis=st._prewarp_basis,
            prewarp_max_degree=st._prewarp_max_degree,
            prewarp_uplift_threshold=st._prewarp_uplift,
            prewarp_min_val_corr=st._prewarp_min_val_corr,
            prewarp_specs_out=st._prewarp_specs,
            fe_gate_med_enable=st._gate_med_enable,
            fe_multi_emit_max_per_pair=st._multi_emit_max,
            fe_multi_emit_mi_floor=st._multi_emit_floor,
            fe_multi_emit_diversity_corr=st._multi_emit_div_corr,
            fe_pair_usability_admission_enable=bool(getattr(self, "fe_pair_usability_admission_enable", True)),
            fe_pair_usability_admission_min_corr=float(getattr(self, "fe_pair_usability_admission_min_corr", 0.6)),
            fe_pair_usability_admission_pairness_margin=float(getattr(self, "fe_pair_usability_admission_pairness_margin", 1.05)),
            gate_med_specs_out=st._gate_med_specs,
            # THIS is the serial-main-thread branch - the whole FE
            # search runs here with NO joblib threads (the ``else`` below is the
            # ``len(X) >= 50000`` joblib ``backend="threading"`` path). On this branch a
            # numba prange does not nest inside Python threads, so the FE materialise /
            # searchsorted kernels may dispatch to their byte-identical ``parallel=True``
            # column-prange twins (gated further by the per-host crossover). The joblib
            # path below leaves ``serial_main_thread`` at its default False -> serial kernels.
            serial_main_thread=True,
            # ENGINEERED-OPERAND FEED-FORWARD: resolve engineered operands by
            # name so (eng_i, eng_j) composites materialise. Off when the cap is 0 (raw-only
            # pool); the pool already excludes them in that case. ``engineered_operand_values``
            # supplies the CONTINUOUS engineered values so the composite is built on them
            # rather than the lossy bin codes.
            allow_engineered_operands=(st._eng_cap != 0),
            engineered_operand_values=getattr(self, "_engineered_continuous_", None),
            # MILLER-MADOW DEBIAS: debias the joint-prevalence
            # ratio gate (see check_prospective_fe_pairs). Co-updated with the maxT
            # floor below (IRON RULE). Default-on; ``fe_mm_debias_prevalence=False``
            # byte-reproduces pre- fits.
            fe_mm_debias_prevalence=bool(getattr(self, "fe_mm_debias_prevalence", False)),
        )
    else:

        _PREWARP_SPECS_RESULT_KEY, _GATE_MED_SPECS_RESULT_KEY, _FE_REJECTION_RESULT_KEY = _run_fe_step_impl_part4(
            st, n_jobs, prefetch_factor, verbose, fe_min_pair_mi_prevalence, parallel_run, check_prospective_fe_pairs, X, unary_transformations,
            binary_transformations, classes_y, classes_y_safe, freqs_y, num_fs_steps, cols, fe_max_steps, fe_npermutations, fe_max_pair_features,
            fe_print_best_mis_only, fe_min_nonzero_confidence, fe_min_engineered_mi_prevalence, fe_good_to_best_feature_mi_threshold,
            fe_max_external_validation_factors, self, times_spent, parallel_kwargs,
        )

    # Extract any reserved pre-warp / gate-med spec entry the SERIAL path may have
    # left in ``prospective_additions`` (the serial branch returns a single dict that
    # also carries the reserved key) and merge it, so the pair loop below never
    # treats it as a ``raw_vars_pair``. The joblib branch already drained the key
    # per-chunk above; this pop is then a harmless no-op.
    from .._feature_engineering_pairs import _PREWARP_SPECS_RESULT_KEY, _GATE_MED_SPECS_RESULT_KEY, _FE_REJECTION_RESULT_KEY
    st._pw_from_res = st.prospective_additions.pop(_PREWARP_SPECS_RESULT_KEY, None)
    if st._pw_from_res:
        st._prewarp_specs.update(st._pw_from_res)
    # Same recovery for the per-operand TRAIN medians (loky-parallel path).
    st._gm_from_res = st.prospective_additions.pop(_GATE_MED_SPECS_RESULT_KEY, None)
    if st._gm_from_res:
        st._gate_med_specs.update(st._gm_from_res)
    # REJECTION LEDGER (additive): drain the SERIAL path's per-pair-gate drops (the
    # joblib branch already drained per-chunk above, so this pop is then a no-op).
    st._rej_from_res = st.prospective_additions.pop(_FE_REJECTION_RESULT_KEY, None)
    _run_fe_step_impl_joblib_branch_already_drained(self, st._rej_from_res, num_fs_steps)

    # Per-candidate scoring / quantile-discretization materialise stage (carved to
    # _step_score.py to bring _step_core.py under the 1k-LOC ceiling). Threads the loop locals
    # explicitly and returns the re-bound values; engineered_features / checked_pairs /
    # engineered_recipes are mutated in place. Selection is byte-for-byte identical.
    from ._step_score import materialise_and_finalise_fe_candidates
    return materialise_and_finalise_fe_candidates


def _run_fe_step_impl_part4(
    st: Any, n_jobs: Any, prefetch_factor: Any, verbose: Any, fe_min_pair_mi_prevalence: Any, parallel_run: Any, check_prospective_fe_pairs: Any, X: Any,
    unary_transformations: Any, binary_transformations: Any, classes_y: Any, classes_y_safe: Any, freqs_y: Any, num_fs_steps: Any, cols: Any,
    fe_max_steps: Any, fe_npermutations: Any, fe_max_pair_features: Any, fe_print_best_mis_only: Any, fe_min_nonzero_confidence: Any,
    fe_min_engineered_mi_prevalence: Any, fe_good_to_best_feature_mi_threshold: Any, fe_max_external_validation_factors: Any, self: Any, times_spent: Any,
    parallel_kwargs: Any,
) -> tuple[Any, ...]:
    """Extracted block of ``_run_fe_step_impl`` (part 4); returns the names the caller still uses."""
    st.prospective_additions = {}
    desired_nitems = max(1, len(st.prospective_pairs) // (n_jobs * prefetch_factor))

    jobs_list: list[Any] = []

    nitems = 0
    cur_dict: dict[Any, Any] = {}
    cur_dict = _run_fe_step_impl_key_value_prospective_pairs(st.prospective_pairs, nitems, cur_dict, desired_nitems, jobs_list)
    if cur_dict:
        jobs_list.append(cur_dict)

    if verbose:
        logger.debug(
            "Using %d items per thread for checking %d prospective_pairs with gain>%.2f.",
            desired_nitems, len(st.prospective_pairs), fe_min_pair_mi_prevalence,
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
                st.original_cols,
                fe_max_steps,
                fe_npermutations,
                fe_max_pair_features,
                fe_print_best_mis_only,
                fe_min_nonzero_confidence,
                fe_min_engineered_mi_prevalence,
                fe_good_to_best_feature_mi_threshold,
                fe_max_external_validation_factors,
                st.numeric_vars_to_consider,
                self.quantization_nbins,
                self.quantization_method,
                self.quantization_dtype,
                times_spent,
                verbose,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                fe_subsample_stratify=st._fe_subsample_stratify,
                shared_subsample_idx=st._shared_fe_idx,
                prewarp_enable=st._prewarp_enable,
                prewarp_y=classes_y if st._prewarp_enable else None,
                prewarp_y_continuous=st._prewarp_y_cont if st._prewarp_enable else None,
                usability_y_continuous=st._usab_y_cont,
                prewarp_basis=st._prewarp_basis,
                prewarp_max_degree=st._prewarp_max_degree,
                prewarp_uplift_threshold=st._prewarp_uplift,
                prewarp_min_val_corr=st._prewarp_min_val_corr,
                prewarp_specs_out=None,  # loky: recovered from result dict below
                fe_gate_med_enable=st._gate_med_enable,
                fe_multi_emit_max_per_pair=st._multi_emit_max,
                fe_multi_emit_mi_floor=st._multi_emit_floor,
                fe_multi_emit_diversity_corr=st._multi_emit_div_corr,
                fe_pair_usability_admission_enable=bool(getattr(self, "fe_pair_usability_admission_enable", True)),
                fe_pair_usability_admission_min_corr=float(getattr(self, "fe_pair_usability_admission_min_corr", 0.6)),
                fe_pair_usability_admission_pairness_margin=float(getattr(self, "fe_pair_usability_admission_pairness_margin", 1.05)),
                gate_med_specs_out=None,  # loky: recovered from result dict below
                # ENGINEERED-OPERAND FEED-FORWARD: see the serial branch above.
                allow_engineered_operands=(st._eng_cap != 0),
                engineered_operand_values=getattr(self, "_engineered_continuous_", None),
                # MILLER-MADOW DEBIAS: see the serial branch above.
                fe_mm_debias_prevalence=bool(getattr(self, "fe_mm_debias_prevalence", False)),
                # LARGE-N PEAK-MEMORY FIX: joblib ``backend="threading"``
                # runs up to ``n_jobs`` of these CONCURRENTLY in the shared address space,
                # each allocating its own candidate/chunk/disc/MI buffers; divide the
                # per-call RAM budget by the worker count so N threads don't collectively
                # blow past 0.4*available and OOM a worker.
                concurrent_workers=int(n_jobs) if n_jobs and n_jobs > 0 else 1,
                # The n_jobs=1 extension. This ``else`` branch is the joblib
                # path, but joblib ``Parallel(n_jobs=1)`` runs every ``delayed`` chunk
                # SEQUENTIALLY in the CALLING thread (joblib's SequentialBackend short-circuit
                # - no worker thread is spawned, regardless of the requested backend), so
                # there is NO Python-threading nest a numba prange could deadlock. Mark the
                # call serial-main-thread when n_jobs<=1 so the FE materialise / searchsorted /
                # discretise kernels may dispatch to their BYTE-IDENTICAL ``parallel=True``
                # column-prange twins (gated further by the per-host kernel_tuning crossover) -
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
    # PREWARP/GATE-MED SPEC-MERGE FIX. Each chunk's result dict
    # carries its OWN fitted-spec payload under the SAME reserved key
    # (``_PREWARP_SPECS_RESULT_KEY`` / ``_GATE_MED_SPECS_RESULT_KEY``). A bare
    # ``prospective_additions.update(next_dict)`` lets each chunk's reserved-key
    # entry CLOBBER the previous chunk's - only the LAST chunk's specs survive,
    # so an operand whose ``prewarp`` warp was fit in an EARLIER chunk loses its
    # coeffs. The recipe builder below then constructs a recipe that names
    # ``prewarp`` but has no ``prewarp_*_coef`` in ``extra`` -> a KeyError at
    # transform()-time replay (observed with n_jobs=1, which still chunks the
    # pairs into multiple sequential ``parallel_run`` jobs: e.g.
    # ``sub(prewarp(informative_3),prewarp(noise_5))`` with informative_3's spec
    # dropped). Fix: MERGE each chunk's reserved spec payload into the
    # accumulators DURING the merge loop, before ``update`` overwrites the key.
    from .._feature_engineering_pairs import _PREWARP_SPECS_RESULT_KEY, _GATE_MED_SPECS_RESULT_KEY, _FE_REJECTION_RESULT_KEY
    _run_fe_step_impl_accumulators_during_merge_loop(self, dicts, st._prewarp_specs, st._gate_med_specs, num_fs_steps, st.prospective_additions)
    return _PREWARP_SPECS_RESULT_KEY, _GATE_MED_SPECS_RESULT_KEY, _FE_REJECTION_RESULT_KEY


def _run_fe_step_impl_part3(
    verbose: Any, selected_vars: Any, self: Any, st: Any, cols: Any, target_names: Any, num_fs_steps: Any, data: Any, categorical_vars: Any, X: Any,
    nbins: Any, target_indices: Any, classes_y: Any, freqs_y: Any, cached_MIs: Any, classes_y_safe: Any, cached_confident_MIs: Any, n_jobs: Any,
    prefetch_factor: Any, parallel_kwargs: Any, fe_min_nonzero_confidence: Any, fe_npermutations: Any, fe_min_pair_mi: Any, fe_min_pair_mi_prevalence: Any,
    checked_pairs: Any, sort_dict_by_value: Any,
) -> tuple[Any, ...]:
    """Extracted block of ``_run_fe_step_impl`` (part 3); returns the names the caller still uses."""
    from ._step_core import _capture_prefe_screened_raw
    if verbose:
        logger.info("MRMR+ selected %d out of %d features before the Feature Engineering step.", len(selected_vars), self.n_features_in_)

    st._screening_returned_empty = False
    if len(selected_vars) == 0:
        if self.fe_fallback_to_all:
            logger.info("Proceeding with all features though (fe_fallback_to_all=True).")
            # ``cols.index(col)`` while iterating ``cols`` itself is always just the loop position
            # (column names are unique by construction, matching every other name->index map in
            # the FE step) - enumerate() gives the identical index without the O(F) rescan per item.
            selected_vars = np.array([i for i, col in enumerate(cols) if col not in target_names])
        elif getattr(self, "cluster_aggregate_enable", False) and num_fs_steps == 0:
            # cluster_aggregate operates on raw ``feature_names_in_`` columns
            # (correlation-based clusters) and does NOT need ``selected_vars``.
            # When screening returns 0 features (heavily-correlated reflection
            # clusters routinely trigger this, since every member's marginal
            # MI is near-zero), the legacy ``return None`` skipped the
            # cluster-aggregate step too - the test contract (and the
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
            selected_vars = np.array([], dtype=np.int64)  # Match the ndarray type the fe_fallback_to_all branch (and the normal caller-provided path) use.
            # (see default_filtering.py:165): screening
            # selected nothing but we continue (interaction-only signal / cluster
            # aggregate). Flag this so the smart-polynom optimiser does NOT treat
            # the raw-seeded pool as "speculative synergy" to withhold (which
            # would exclude EVERY pair and the polynom search would never fire).
            st._screening_returned_empty = True
        elif _synergy_bootstrap_can_supply_pool(self, num_fs_steps, data):
            # INTERACTION-ONLY SIGNAL: screening returned 0 features because every operand of a pure-interaction target (a*b + c*d + ...) has ~0 MARGINAL MI - and the
            # empirical-null debiasing (Fix B) now correctly demotes those near-zero marginals to exactly 0, so even the weak pre-debiasing marginal that used to slip one operand
            # through the screen no longer does. The signal is entirely in the PAIRS, which the synergy bootstrap below surfaces by seeding the all-pairs joint-MI sweep with the raw
            # numeric columns. Continue into the FE step (rather than skipping it) so that sweep + the smart-polynom / pair search can recover the interaction features. Without this,
            # the default-on DCD suppresses the cluster_aggregate fallback branch above, so a pure-interaction frame would silently engineer NOTHING. ``_screening_returned_empty``
            # disables the speculative-synergy withhold downstream (every operand is bootstrap-added here, so the withhold would otherwise drop every pair).
            logger.info(
                "Screening returned 0 features but the synergy bootstrap can seed an interaction-only pool "
                "(fe_synergy_screen_max_features>0); running the FE pair / smart-polynom search anyway.",
            )
            selected_vars = np.array([], dtype=np.int64)  # Match the ndarray type the fe_fallback_to_all branch (and the normal caller-provided path) use.
            st._screening_returned_empty = True
        else:
            logger.info("Skipping Feature Engineering (screening returned 0 features and fe_fallback_to_all=False).")
            return True, (None)

    # RAW-RETENTION capture: record the SCREENING-confirmed genuine
    # raw features on the first FE step. The post-FE re-selection can drop a
    # screening-confirmed (permutation-validated) weak feature when an engineered
    # feature absorbs its signal as a redundant near-duplicate (measured: a genuine
    # X5 / pair operand absorbed into a noise-paired engineered feature, raw column
    # dropped). ``MRMR.fit`` re-adds these at support finalisation unless a
    # SINGLE-PARENT engineered child substitutes them (the prefer-engineered case).
    if num_fs_steps == 0:
        _capture_prefe_screened_raw(self, cols, selected_vars)

    st.n_recommended_features = 0
    if verbose >= 2:
        logger.info("Computing prospective FE pairs...")

    # FE operand-pool construction (carved to _step_pool.py): builds numeric_vars_to_consider
    # from selected_vars + runs the synergy / GBM / gradient seeders. Selection is byte-for-byte identical.
    from ._step_pool import build_fe_operand_pool
    st.numeric_vars_to_consider, st._synergy_added_idx = build_fe_operand_pool(
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
    # to _step_pairmi.py). Mutates cached_MIs in place and returns the re-bound pool + gate
    # state; selection is byte-for-byte identical.
    from ._step_pairmi import compute_pair_mis_and_floor
    (
        st.numeric_vars_to_consider, st._eng_cap, st._pair_maxt_floor, st._pair_mm_bias, st._prevalence_debias_auto,
    ) = compute_pair_mis_and_floor(
        self,
        data=data, cols=cols, nbins=nbins, X=X,
        classes_y=classes_y, classes_y_safe=classes_y_safe, freqs_y=freqs_y,
        target_indices=target_indices,
        cached_MIs=cached_MIs, cached_confident_MIs=cached_confident_MIs,
        numeric_vars_to_consider=st.numeric_vars_to_consider,
        _prevalence_debias_auto=st._prevalence_debias_auto,
        n_jobs=n_jobs, prefetch_factor=prefetch_factor, parallel_kwargs=parallel_kwargs,
        fe_min_nonzero_confidence=fe_min_nonzero_confidence, fe_npermutations=fe_npermutations,
        fe_min_pair_mi=fe_min_pair_mi, fe_min_pair_mi_prevalence=fe_min_pair_mi_prevalence,
        verbose=verbose,
    )
    # ---------------------------------------------------------------------------------------------------------------
    # For every pair of factors (A,B), select ones having MI((A,B),Y)>MI(A,Y)+MI(B,Y). Such ones must possess more special connection!
    # ---------------------------------------------------------------------------------------------------------------

    # Per-pair joint-MI uplift + order-2 maxT scoring of the candidate pairs (carved to
    # _step_pairs_rank.py). Returns the prospective_pairs ranking dict + the prevalence-failed synergy
    # rescue ledger; selection is byte-for-byte identical.
    from ._step_pairs_rank import score_prospective_pairs
    st.prospective_pairs, st._prevalence_failed_synergy = score_prospective_pairs(
        self,
        cached_MIs=cached_MIs,
        numeric_vars_to_consider=st.numeric_vars_to_consider,
        checked_pairs=checked_pairs,
        _pair_mm_bias=st._pair_mm_bias,
        _pair_maxt_floor=st._pair_maxt_floor,
        _synergy_added_idx=st._synergy_added_idx,
        fe_min_pair_mi_prevalence=fe_min_pair_mi_prevalence,
        _synergy_prev_resolved=st._synergy_prev_resolved,
        _prevalence_debias_auto=st._prevalence_debias_auto,
        data=data,
        classes_y=classes_y,
        X=X,
        cols=cols,
        num_fs_steps=num_fs_steps,
        verbose=verbose,
        sort_dict_by_value=sort_dict_by_value,
    )
    return False, selected_vars


def _run_fe_step_impl_part2(
    self: Any, st: Any, verbose: Any, data: Any, fe_smart_polynom_iters: Any, classes_y: Any, X: Any, cols: Any, _is_polars_input: Any, nbins: Any,
    engineered_features: Any, engineered_recipes: Any, fe_smart_polynom_optimization_steps: Any, fe_min_polynom_degree: Any, fe_max_polynom_degree: Any,
    fe_min_polynom_coeff: Any, fe_max_polynom_coeff: Any, fe_min_engineered_mi_prevalence: Any, n_jobs: Any, num_fs_steps: Any,
) -> tuple[Any, ...]:
    """Extracted block of ``_run_fe_step_impl`` (part 2); returns the names the caller still uses."""
    # SYNERGY-PAIR BUDGET: cap the synergy pairs (>=1 operand is a
    # bootstrap-added unselected column) at ``fe_synergy_max_pairs`` top-joint-MI
    # pairs before the expensive per-pair search, so a noise-heavy frame cannot
    # flood ``check_prospective_fe_pairs``. Selected-selected pairs are kept in
    # full. ``key`` is ``(raw_vars_pair, pair_mi)``; rank synergy pairs by pair_mi.
    _run_fe_step_impl_full_key_raw_vars(self, st._synergy_added_idx, st.prospective_pairs, verbose)

    # Now need to sort prospective_pairs by the uplift, to check most promising pairs within the time budget.
    # Also need to sort them by their members usage frequency+members ids sum. this way, their splitting will benefit more from caching.
    st.prospective_pairs = order_prospective_pairs(st.prospective_pairs)  # reuse-counter ties break by pair MI: strongest of a tie group first

    # SUCCESSIVE-HALVING / RUNG-SCHEDULE FE-search budget.
    # ON by default. Before the EXPENSIVE per-pair operator search below
    # (``check_prospective_fe_pairs`` - all unary x binary transforms / CMA-ES / full
    # discretize / prewarp, ~4-50s per pair), run a CHEAP rung-0 SCREEN: rank the
    # gate-surviving prospective pairs by their JOINT MI ``pair_mi`` (key[1] - a
    # monotone-ish proxy of the operator-search outcome ALREADY computed by the pair-MI
    # gate, so the screen is FREE) and keep only the top fraction (UNION a relative-MI
    # floor so a moderate-MI genuine winner is never cut). The expensive search then runs
    # on the survivors only. GATES UNCHANGED - this changes WHERE the compute goes, not
    # admission (a pair the rung-0 screen drops simply never gets the operator search,
    # exactly as the synergy budget already caps synergy pairs; this generalises that to
    # the whole pool). Measured 1.7-2.2x at keep_frac=0.5 with NO genuine signal pair
    # dropped (n=5000/p=40 canonical fixture + noise, 5 seeds). Self-gates to a no-op
    # below ``fe_rung_min_pairs`` pairs / all-zero pair_mi (byte-identical flat sweep).
    st.prospective_pairs = _run_fe_step_impl_below_fe_rung_min(self, st.prospective_pairs, data, verbose)

    # cols-space indices of polynom-pair engineered columns appended by the
    # ``run_polynom_pair_fe`` block below; promoted into ``selected_vars``
    # alongside the unary/binary indices so a polynom feature that cleared the
    # FE gates actually reaches ``support_`` (see promotion at the bottom).
    st._polynom_engineered_indices = []
    if fe_smart_polynom_iters:
        # Orthogonal-polynomial pair FE: Chebyshev default basis (empirically robust); tight coef range [-2, 2],
        # fixed degree per study, L2 regularisation, identity-baseline filter. Override basis via
        # ``self.fe_polynomial_basis``. See feature_selection.filters.hermite_fe and bench_polynomial_bases.
        #
        # Extracted from inline ~200 LOC block into
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
        _prospective_for_polynom = st.prospective_pairs
        # Withhold SPECULATIVE synergy pairs from the powerful poly optimiser
        # (it can fit a high-MI cell to pure noise on a noise-operand pair) -
        # but ONLY when there was a genuine selected pool to augment. When
        # screening returned 0 features (interaction-only signal), EVERY operand
        # is "synergy-added", so this exclusion would withhold every pair and the
        # polynom search would never fire (default_filtering.py:165). In that
        # case keep the pairs: they ARE the signal, and the synergy max-pairs cap
        # + the downstream pair-MI / engineered-MI / uplift gates already bound
        # the pure-noise risk.
        _prospective_for_polynom = _run_fe_step_impl_pure_noise_risk(st._synergy_added_idx, st._screening_returned_empty, st.prospective_pairs, _prospective_for_polynom)
        # None / 0 / negative all map to "no subsample" (use full data).
        _subsample_raw = getattr(self, "fe_smart_polynom_subsample_n", 0)
        _subsample_n = int(_subsample_raw) if _subsample_raw and _subsample_raw > 0 else 0
        # ONE shared FE subsample for this fit: resolve + cache the single row-index draw
        # ONCE here, so the polynom path, the pair-search and the sufficient-summary floor all REUSE it
        # (same rows everywhere) instead of each drawing its own. ``classes_y`` is the discrete fit
        # target; the helper returns None at small n (<= unified screen size) -> legacy per-call draw.
        try:
            from .._fe_sufficient_summary import _get_shared_fe_subsample_idx
            st._shared_fe_idx = _get_shared_fe_subsample_idx(self, np.asarray(classes_y), int(getattr(X, "shape", [len(X)])[0]))
        except Exception as _sub_exc:
            # Full-n fallback is safe but ~33x slower at n~1M -> log so it is never a silent mystery.
            logger.warning("mrmr: shared FE subsample resolution failed in FE step; running at FULL n: %r", _sub_exc, exc_info=True)
            st._shared_fe_idx = None
        # Capture cols width before the polynom block so we can promote the
        # polynom-injected engineered column indices into ``selected_vars``
        # below (the same promotion the unary/binary block does for
        # its own appended cols). Without this, a polynom-pair feature that
        # cleared every polynom-FE gate (pair-MI prevalence + engineered-MI
        # prevalence + uplift) was appended to ``data``/``cols`` and tracked in
        # ``_hermite_features_`` but never reached ``support_`` under the default
        # single-step path, because only the unary/binary indices were promoted.
        _n_cols_before_polynom = len(cols)
        _polynom_rejections: list = []
        data, nbins, cols, X = run_polynom_pair_fe(
            rejection_records=_polynom_rejections,
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
            # cma_batch is the new default (20.58x faster than
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
            fe_subsample_stratify=st._fe_subsample_stratify,
            shared_subsample_idx=st._shared_fe_idx,
            # Cheap-first dispatch: skip the expensive CMA/Optuna search for pairs
            # whose trivial baseline already saturates the joint-MI ceiling. getattr
            # default keeps the knob optional (no ctor flag required); 1.0 disables.
            poly_cheap_skip_ratio=float(getattr(self, "fe_poly_cheap_skip_ratio", 0.97)),
            # Linear-usability guard on the skip: only skip when the trivial feature
            # is ALSO linearly useful (|corr| to y >= this), so an MI-saturated-but-
            # non-linear trivial (atan2, etc.) still falls through to the optimiser.
            poly_cheap_skip_min_corr=float(getattr(self, "fe_poly_cheap_skip_min_corr", 0.90)),
        )
        _drain_fe_rejection_records(self, _polynom_rejections, num_fs_steps)
        # Columns appended by the polynom block (its own gates already accepted
        # them). Promote into selected_vars below so they reach support_.
        st._polynom_engineered_indices = list(range(_n_cols_before_polynom, len(cols)))
    return data, X, cols, nbins


def _run_fe_step_impl_part1(self: Any, st: Any, cols: Any, verbose: Any, classes_y: Any, n_jobs: Any, X: Any) -> None:
    """Extracted block of ``_run_fe_step_impl`` (part 1); returns the names the caller still uses."""
    # The standard check_prospective_fe_pairs path used to live in
    # ``else:`` of the Hermite block, which meant enabling
    # ``fe_smart_polynom_iters > 0`` silently DISABLED all standard
    # unary/binary FE (cbrt, sqrt, log, hypot, atan2, ...). De-dented the
    # block so the standard pipeline always runs after the Hermite block;
    # users get the unary/binary FE they asked for via
    # ``fe_unary_preset='medium'`` regardless of whether Hermite ran.
    # feature_names_in_ is an ndarray (sklearn convention) - list() once so .index() works (ndarray has none)
    # and the comprehension below doesn't rebuild it per element.
    st._fni_list = list(self.feature_names_in_)
    # name -> index map built once (O(F)) instead of an ``in`` test + ``.index()`` rescan of
    # ``_fni_list`` per ``col`` (O(F) each) - turns the O(K*F) lookup below into O(K+F).
    st._fni_idx = {nm: i for i, nm in enumerate(st._fni_list)}
    st.original_cols = {i: st._fni_idx[col] for i, col in enumerate(cols) if col in st._fni_idx}
    if verbose >= 1:
        logger.debug("Checking %d most prospective_pairs for feature engineering...", len(st.prospective_pairs))

    # PER-OPERAND PRE-WARP: read the opt-in flag + knobs off the
    # MRMR instance (getattr keeps _run_fe_step's signature stable, mirroring
    # ``fe_check_pairs_subsample_n``). When enabled, the discretised target
    # (``classes_y`` - the SAME codes the MI sweep scores against) is handed
    # to ``check_prospective_fe_pairs`` so it can fit a learned 1-D pre-warp
    # per operand; ``_prewarp_specs`` collects the fitted coeffs (by cols-space
    # var index) for leak-safe recipe construction. Default OFF.
    st._prewarp_enable = bool(getattr(self, "fe_pair_prewarp_enable", False))
    # CONTINUOUS ALS RECONSTRUCTION TARGET: the raw continuous y
    # stashed by ``_fit_impl`` so the rank-1 ALS warp reconstructs against the
    # faithful continuous target instead of the coarse target-rebin-guard codes.
    # Aligned to the FE-step row count (full-n; ``check_prospective_fe_pairs``
    # handles any internal subsample). None -> ALS falls back to ``classes_y``.
    st._prewarp_y_cont = None
    st._prewarp_y_cont = _run_fe_step_impl_handles_any_internal_subsample(self, st._prewarp_enable, classes_y, st._prewarp_y_cont)
    # LINEAR-USABILITY GUARD TARGET: the leader tie-break + noise-wrap |corr|
    # guard must score against CONTINUOUS y regardless of prewarp - the binned ``classes_y``
    # fallback INVERTS linear usability on heavy-tailed targets (picks ``a/sqrt(b)`` over
    # ``a**2/b``). Threaded unconditionally; None for classification/non-numeric y (correct
    # ``classes_y`` fallback inside ``check_prospective_fe_pairs``).
    st._usab_y_cont = None
    st._uyc = getattr(self, "_fe_prewarp_y_continuous_", None)
    if st._uyc is not None and len(st._uyc) == len(classes_y):
        st._usab_y_cont = st._uyc
    st._prewarp_basis = str(getattr(self, "fe_pair_prewarp_basis", "chebyshev"))
    st._prewarp_max_degree = int(getattr(self, "fe_pair_prewarp_max_degree", 4))
    st._prewarp_uplift = float(getattr(self, "fe_pair_prewarp_uplift_threshold", 1.20))
    st._prewarp_min_val_corr = float(getattr(self, "fe_pair_prewarp_min_val_corr", 0.08))
    # PREWARP-SPEC PERSISTENCE. ``_mrmr_fe_step`` is re-entered
    # once per FE-bearing MRMR iteration, and each call's ``check_prospective_fe_pairs``
    # fits prewarp specs ONLY for the operands of THIS iteration's prospective pairs
    # (a var whose pair isn't prospective this round is not re-fit). But the recipe
    # build below runs per iteration over ``this_pair_features`` and can construct a
    # recipe whose operand used the ``prewarp`` pseudo-unary in an EARLIER iteration -
    # if ``_prewarp_specs`` were a fresh per-call dict, that operand's coeffs would be
    # absent and the recipe would be built with ``prewarp_a/b=None``, producing a
    # recipe that names ``prewarp`` but lacks ``prewarp_*_coef`` in ``extra`` -> a
    # KeyError at transform()-time replay (observed: ``sub(prewarp(informative_3),
    # prewarp(noise_5))`` with informative_3's spec missing). Back the local dict with
    # a SELF-LEVEL accumulator (mirrors ``self._engineered_continuous_``) so every
    # spec fit in any prior iteration stays available for recipe construction. Seeded
    # from the accumulator and written back after each call; specs are keyed by
    # cols-space var index, which is stable across iterations (no cat reorder mid-fit).
    st._prewarp_specs = getattr(self, "_prewarp_specs_accum_", None)
    if st._prewarp_specs is None:
        st._prewarp_specs = {}
        self._prewarp_specs_accum_ = st._prewarp_specs

    # PER-OPERAND MEDIAN GATE: opt-in flag off the MRMR instance
    # (getattr keeps the signature stable, mirroring the prewarp wiring). When
    # enabled, ``check_prospective_fe_pairs`` fits one TRAIN median per operand
    # and exposes a ``gate_med`` pseudo-unary; ``_gate_med_specs`` collects the
    # fitted medians (by cols-space var index) for leak-safe recipe
    # construction. Default OFF -> byte-identical legacy path. Same cross-iteration
    # persistence as the prewarp specs above (a gate_med operand selected in an
    # earlier iteration must keep its median available for recipe replay).
    st._gate_med_enable = bool(getattr(self, "fe_gate_med_enable", False))
    # MULTI-CANDIDATE DIVERSE EMISSION: per pair, emit up to this many
    # DISTINCT engineered forms (MI is rank-blind to linear usability, so a single
    # MI-winner can be tree-friendly-but-linearly-useless while a lower-MI form is the
    # linearly-usable one; both should survive for the downstream model to choose).
    st._multi_emit_max = int(getattr(self, "fe_multi_emit_max_per_pair", 1))
    st._multi_emit_floor = float(getattr(self, "fe_multi_emit_mi_floor", 0.5))
    st._multi_emit_div_corr = float(getattr(self, "fe_multi_emit_diversity_corr", 0.90))
    st._gate_med_specs = getattr(self, "_gate_med_specs_accum_", None)
    if st._gate_med_specs is None:
        st._gate_med_specs = {}
        self._gate_med_specs_accum_ = st._gate_med_specs

    # SERIAL-vs-JOBLIB DISPATCH. The ``else`` branch
    # spreads the prospective PAIRS across ``n_jobs`` joblib ``backend="threading"``
    # workers, each running the SERIAL (no-prange) FE kernels (a numba prange would
    # nest inside the threading layer and deadlock - see ``_fe_use_parallel_kernels``).
    #
    # BACKEND BENCH (n=8000 p=150 FE pair-search, 3-rep medians, peak RSS incl children):
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
    # subsumes the old fixed ``len(X) < 50000`` row gate - a tall narrow frame now
    # takes the all-cores path it always should have.
    st._fe_serial_min_pairs_per_worker = max(2, int(n_jobs) if n_jobs and n_jobs > 0 else 1)
    # GPU-FE SERIALIZE: when the per-pair candidate materialise/binning runs on the
    # GPU, the single device is the bottleneck resource - spreading the pair-chunks across joblib
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
    # helper caches on the instance keyed by n, so this is idempotent - same cached draw, no re-draw.
    try:
        from .._fe_sufficient_summary import _get_shared_fe_subsample_idx
        st._shared_fe_idx = _get_shared_fe_subsample_idx(self, np.asarray(classes_y), int(getattr(X, "shape", [len(X)])[0]))
    except Exception as _sub_exc:
        # Full-n fallback is safe but ~33x slower at n~1M -> log so it is never a silent mystery.
        logger.warning("mrmr: shared FE subsample resolution failed in FE step; running at FULL n: %r", _sub_exc, exc_info=True)
        st._shared_fe_idx = None
