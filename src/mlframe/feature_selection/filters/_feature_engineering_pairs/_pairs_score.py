"""Per-pair candidate-scoring + acceptance/external-validation body of
``check_prospective_fe_pairs`` (carved 2026-06-22, Tier E).

This module holds the verbatim per-pair loop body lifted out of
``_pairs_core.check_prospective_fe_pairs`` so the parent orchestration function
drops back under the 1k-LOC monolith ceiling. It is a STRAIGHT carve - the
block was moved unchanged except that:

  * the loop locals it reads are now EXPLICIT keyword parameters (no closure
    capture from the parent frame);
  * the six chunk-materialise state vars that must persist ACROSS pairs are
    threaded through one mutable ``chunk_state`` dict (the parent owns it and
    passes the SAME dict every pair, so the lazy per-chunk load / reset
    semantics are byte-for-byte identical to the in-loop version);
  * the per-pair rejection records append into the caller-owned
    ``rejection_records`` list (was ``_rejection_records``);
  * the single ``res[raw_vars_pair] = (...)`` write became a returned
    ``_pair_res_entry`` (the parent stores it under ``raw_vars_pair``).

The four per-call memo helpers (``_extval_raw_col`` / ``_safe_abs_corr`` /
``_operand_marginal_mi`` / ``_operand_discretized``) and the seven framework
callables lazily imported in the parent are passed in as parameters so the
math / RNG sequence / chunk iteration order are unchanged. Selection is
byte-for-byte identical to the pre-carve in-loop body.
"""
from __future__ import annotations

from typing import Any


import numpy as np

from ._pairs_dispatch import _dispatch_batch_mi_with_noise_gate
from ._pairs_materialise import (
    _fe_use_parallel_kernels,
    _materialise_extval_njit,
    _narrow_code_dtype,
)
from ._pairs_emit import _emit_pair_features

# DEGENERATE-PAIR |corr| threshold. A prospective pair is DEGENERATE when its winning
# composite is numerically ~= a SINGLE one of its own (warped) operands - the other operand's transform
# collapsed to ~constant so the binary op is just a re-wrap of one source and adds NO genuine joint
# information (e.g. ``mul(prewarp(b),prewarp(a__L2))`` where ``prewarp(b)`` is ~constant -> the column is
# essentially ``prewarp(a__L2)`` ~= a**2). Such a pair keeps |corr|~=1.0 with that one operand, clears the
# joint-prevalence gate, and DISPLACES the clean single-source univariate basis (a__L2). A GENUINE 2-var
# pair (a/b, a*b) has |corr| well below this with EITHER single operand, so the bar leaves it untouched.
# 0.999 is intentionally near-1.0: it fires only on a true single-operand re-wrap, never on a real pair.
from ._pairs_score_helpers import (  # noqa: F401  -- carved helpers
    logger,
    _should_demote_prewarp,
    _score_one_pair_fe_gpu_binning_enabled,
    _score_one_pair_gpu_fused_done,
    _score_one_pair_fut_none,
    _score_one_pair_prefetch_next_chunk_into,
    _score_one_pair_transformations_pair_combs,
    _score_one_pair_non_analytic_branch_su,
    _score_one_pair_fe_mi_arr_none,
    _score_one_pair_fe_mi_best_mi,
    _score_one_pair_iteration_serialization_point_under,
    _score_one_pair_monotone_equivalent_warp_scores,
    _score_one_pair_mi_threshold_ratio_fix,
    _score_one_pair_uplift_fires_prewarp_config,
    _score_one_pair_warp_spec_round_trip,
    _score_one_pair_flips,
    _score_one_pair_computed_once_per_distinct,
    _score_one_pair_side,
    _score_one_pair_all_values_were_already,
)
from types import SimpleNamespace as _SimpleNamespace


from ._pairs_score_steps import (  # noqa: F401  -- carved helpers
    _DEGENERATE_PAIR_SINGLE_OPERAND_CORR,
    _score_one_pair_step1_reads_buffer_have,
    _score_one_pair_step2_def_resolve_col,
    _score_one_pair_step3_resolve_col_treat,
    _score_one_pair_step3_step1_pair_phase_produced,
    _score_one_pair_step3_step2_transformations_pair_st,
    _score_one_pair_step4_iteration_serialization_point,
    _score_one_pair_step5_any_error_leaves,
    _score_one_pair_step6_wide_fraction_margin,
)


def _score_one_pair(
    *,
    raw_vars_pair,
    pair_mi,
    chunk_state,
    rejection_records,
    rejection_ledger_out,
    # --- frames / candidate tables ---
    X,
    transformed_vars,
    vars_transformations,
    binary_transformations,
    unary_transformations,
    pair_combs,
    # --- buffer / chunk dispatch ---
    final_transformed_vals_shared,
    _need_recompute_map,
    _chunk_global_batch,
    _chunk_buffer,
    _pair_to_chunk,
    _fe_chunks,
    _pair_valid_combs,
    _fe_defer_float,
    # --- per-call hoists (env-gate / op-code table / MI tie-band): resolved ONCE by the caller (invariant for the whole ``check_prospective_fe_pairs`` call),
    # NOT recomputed per pair ---
    _gpu_mat_on,
    _op_code_arr,
    _op_code_arr_all,
    _mi_band,
    _fe_env_gate,
    # --- target / estimator ---
    classes_y,
    classes_y_safe,
    freqs_y,
    fe_npermutations,
    fe_min_nonzero_confidence,
    quantization_nbins,
    quantization_method,
    quantization_dtype,
    # --- gates / thresholds ---
    num_fs_steps,
    fe_min_engineered_mi_prevalence,
    fe_good_to_best_feature_mi_threshold,
    fe_max_external_validation_factors,
    numeric_vars_to_consider,
    fe_max_steps,
    fe_print_best_mis_only,
    fe_mm_debias_prevalence,
    _prewarp_active,
    prewarp_uplift_threshold,
    _PREWARP_UNARY,
    _corr_y_cont,
    _corr_y_cont_finite,
    _NOISE_WRAP_CORR_COLLAPSE_FRAC,
    _NOISE_WRAP_MIN_OPERAND_CORR,
    fe_multi_emit_max_per_pair,
    fe_multi_emit_mi_floor,
    fe_multi_emit_diversity_corr,
    fe_pair_usability_admission_enable=True,
    fe_pair_usability_admission_min_corr=0.6,
    fe_pair_usability_admission_pairness_margin=1.05,
    # --- cols / subsample / fitted state ---
    cols,
    original_cols,
    _use_subsample,
    _X_full,
    _full_n_rows,
    _prewarp_spec_by_var,
    _gate_med_median_by_var,
    engineered_operand_values,
    # --- rng / timing / misc ---
    _rng_extval,
    _n_workers,
    times_spent,
    verbose,
    serial_main_thread,
    # --- per-call memo helpers (closures from the parent frame) ---
    _extval_raw_col,
    _safe_abs_corr,
    _raw_operand_abs_corr,
    _transformed_operand_abs_corr,
    _operand_marginal_mi,
    _operand_discretized,
    # --- framework callables (lazily imported in the parent) ---
    batch_mi_with_noise_gate,
    use_su_normalization,
    discretize_array,
    discretize_2d_quantile_batch,
    mi_direct,
    get_new_feature_name,
    _rebuild_full_survivor_col,
    _can_hoist_shared_buffer,
    _fe_gpu_discretize_enabled,
):
    """Score ONE prospective pair: run the per-candidate MI sweep + the
    joint-prevalence / prewarp / marginal-uplift acceptance gates + the
    noise-wrap veto + (when admitted) the external-validation tie-break and
    survivor materialisation. Returns ``(_pair_res_entry, best_config, best_mi)``
    where ``_pair_res_entry`` is the ``res[raw_vars_pair]`` tuple or ``None`` when
    the pair was rejected (no feature emitted). Mutates ``chunk_state`` (lazy
    chunk load/reset, shared across pairs), ``rejection_records`` /
    ``rejection_ledger_out`` (drops), and ``times_spent`` (per-bin_func wall)."""
    st = _SimpleNamespace()  # long-lived locals of this function (see the stage helpers below)
    final_transformed_vals: Any = None
    st._pair_res_entry = None
    st.messages = []

    st.combs = pair_combs[raw_vars_pair]

    st.best_config, st.best_mi = None, -1.0
    st.this_pair_features = set()
    st.var_pairs_perf = {}
    # Pre-warp uplift tracking: the best engineered MI achievable with ONLY the elementary library unaries (no ``prewarp`` operand) vs the best USING a prewarp
    # operand. A 1-D engineered summary of a 2-D pair cannot retain ``fe_min_engineered_mi_prevalence`` of the 2-D JOINT MI, so on a non-monotone inner
    # distortion (where the elementary library is representationally blind) the prewarp winner is rejected by the joint prevalence gate despite being a large,
    # real uplift over the best the library can do. The alternative acceptance path below admits a prewarp winner when it beats the best non-prewarp engineered
    # MI by a margin - directed (only fires where the prewarp adds representational power) and noise-safe (on linear/monotone/noise data the prewarp does not
    # beat the elementary library, so the margin is never cleared).
    st.best_nonprewarp_mi = -1.0
    st.best_nonprewarp_config = None
    st.best_prewarp_config, st.best_prewarp_mi = None, -1.0

    # CRITICAL #2 dispatch: hoist path uses the shared buffer (writes into ``[:, i]``); recompute-fallback path uses a tiny 1D scratch + a config-by-i map for
    # on-demand survivor recomputation later. CROSS-PAIR: when this pair was batched across the chunk, its survivor columns live in the wide ``_chunk_buffer``
    # (the config's ``i`` is the chunk-buffer column), so point ``final_transformed_vals`` at it. The chunk is materialised LAZILY: pairs are processed in
    # chunk-plan order, so when we reach the FIRST pair of a not-yet-loaded chunk we fill the buffer + MI cache
    # for that whole chunk in ONE batched pass. By the time the next chunk's first
    # pair arrives, all of this chunk's pairs (incl. their survivor packing, which
    # reads the buffer) have already been processed -> safe to overwrite.
    st._chunk_entry = None
    _score_one_pair_step1_reads_buffer_have(_chunk_global_batch, _chunk_buffer, _pair_to_chunk, raw_vars_pair, chunk_state, _fe_chunks, _pair_valid_combs, vars_transformations, transformed_vars, binary_transformations, quantization_nbins, quantization_dtype, classes_y, classes_y_safe, freqs_y, fe_npermutations, fe_min_nonzero_confidence, batch_mi_with_noise_gate, use_su_normalization, _PREWARP_UNARY, discretize_2d_quantile_batch, serial_main_thread, _fe_defer_float, _op_code_arr_all, _gpu_mat_on, _fe_env_gate, st)
    # When the chunk DEFERRED its float D2H the buffer is unfilled - point the reads at None so they
    # take the GPU re-materialise branch in ``_resolve_col`` (bit-identical to the buffer value).
    _this_chunk_deferred = (st._chunk_entry is not None) and chunk_state["float_deferred"]
    if st._chunk_entry is not None:
        final_transformed_vals = None if _this_chunk_deferred else chunk_state.get("active_buffer", _chunk_buffer)
    else:
        final_transformed_vals = final_transformed_vals_shared
    st._col_buf_1d = np.empty(len(X), dtype=np.float32) if _need_recompute_map else None
    _config_by_i: dict[int, tuple] | None = {} if _need_recompute_map else None

    _resolve_col = _score_one_pair_step2_def_resolve_col(final_transformed_vals, chunk_state, transformed_vars)

    st.i = 0
    # Per-pair thread-local timing accumulator; merged into the shared
    # ``times_spent`` under the lock once per pair (see end of pair loop).
    st._local_times = {}

    # BATCHED-DISCRETIZE dispatch: per-candidate ``discretize_array``
    # (np.linspace + np.nanpercentile->partition + searchsorted) is the FE-pair-search
    # hotspot - millions of tiny per-column numpy calls -> serial-dispatch-bound,
    # idle CPU. On the HOIST path (shared buffer present) with the quantile method we
    # split this pair's sweep into 3 phases: (1) materialise ALL candidate columns into
    # the buffer + nan_to_num + record (config, idx, uses_pw); (2) batch-discretise the
    # filled buffer slice in ONE ``np.nanpercentile(axis=0)`` (amortises dispatch over K
    # columns - bit-identical to per-column, see ``discretize_2d_quantile_batch``);
    # (3) replay the EXACT per-candidate mi_direct + best/prewarp/config tracking.
    # MI stays per-candidate (mi_direct's permutation confidence is NOT batched). The
    # recompute-fallback (no buffer) and the uniform method keep the original
    # per-candidate path verbatim - only the hoist+quantile case is batched.
    # A deferred chunk has final_transformed_vals=None (buffer not filled) but still drives the
    # cross-pair replay path (its MI is precomputed, its columns re-materialise on demand via
    # _resolve_col) - so treat a present _chunk_entry as batch-disc-eligible even when deferred.
    _deferred_chunk_entry = st._chunk_entry is not None and _this_chunk_deferred
    st._use_batch_disc = False
    if quantization_method == "quantile":
        st._use_batch_disc = _deferred_chunk_entry if final_transformed_vals is None else True

    st._fe_defer_float = _fe_defer_float
    st._pair_defer_state = None
    _score_one_pair_step3_resolve_col_treat(st, final_transformed_vals, _op_code_arr, binary_transformations, vars_transformations, _PREWARP_UNARY, quantization_nbins, quantization_dtype, transformed_vars, _fe_gpu_discretize_enabled, classes_y, classes_y_safe, freqs_y, fe_npermutations, fe_min_nonzero_confidence, use_su_normalization, discretize_2d_quantile_batch, serial_main_thread, batch_mi_with_noise_gate, _fe_env_gate, _need_recompute_map, _config_by_i, fe_print_best_mis_only, verbose, pair_mi, discretize_array, quantization_method, mi_direct)
    if st._pair_defer_state is not None:
        # The GPU fused materialise left the float candidate matrix on the device: nothing is copied to the host, and the survivor / usability / emit reads below
        # take the same deferred path a deferred chunk does, re-materialising only the columns they touch.
        final_transformed_vals = None
        _this_chunk_deferred = True
        _resolve_col = _score_one_pair_step2_def_resolve_col(None, st._pair_defer_state, transformed_vars)

    # Merge this pair's per-bin_func timings into the shared accumulator in
    # ONE locked pass (the increment was previously locked per inner
    # iteration - a serialization point under the parallel pair dispatch).
    _score_one_pair_step4_iteration_serialization_point(st, times_spent, verbose, raw_vars_pair, _corr_y_cont, _PREWARP_UNARY, fe_good_to_best_feature_mi_threshold, final_transformed_vals, _this_chunk_deferred, _resolve_col, _config_by_i, transformed_vars, vars_transformations, binary_transformations, _safe_abs_corr, pair_mi, fe_mm_debias_prevalence, classes_y, freqs_y, quantization_nbins, discretize_array, quantization_method, quantization_dtype, _operand_discretized, fe_min_engineered_mi_prevalence, num_fs_steps, _operand_marginal_mi, _prewarp_active, prewarp_uplift_threshold)
    _score_one_pair_step5_any_error_leaves(fe_pair_usability_admission_enable, _corr_y_cont, pair_mi, st, final_transformed_vals, _this_chunk_deferred, _extval_raw_col, raw_vars_pair, fe_pair_usability_admission_min_corr, fe_pair_usability_admission_pairness_margin, _resolve_col, _safe_abs_corr, _raw_operand_abs_corr, _mi_band, verbose)

    # NOISE-WRAP CORR-COLLAPSE VETO. Whatever path admitted the winner, VETO it when the
    # winning composite WRAPS a strong, clean operand with a (near-)noise operand: its |corr| with the
    # target collapses to a small fraction of the best single operand's |corr| while that operand is
    # genuinely strong on its own. This is the ``sub(log(e),invqubed(a__T2))`` failure - an extreme
    # heavy-tailed transform inflates the binned ``best_mi/pair_mi`` so it clears the joint-prevalence
    # gate, yet the column carries ~0 linear/monotone signal (|corr|~0.02) versus the clean operand's
    # |corr|~0.99, so it would DISPLACE the clean univariate basis from the support and kill recovery.
    # Genuine synergy (a*b, log(c)*sin(d)) keeps the engineered column tracking y (no collapse), so the
    # wide 2x fraction margin never condemns it. Pure-noise pairs never reach here (upstream screens).
    _score_one_pair_step6_wide_fraction_margin(st, _corr_y_cont, final_transformed_vals, _this_chunk_deferred, _resolve_col, _safe_abs_corr, vars_transformations, _transformed_operand_abs_corr, _raw_operand_abs_corr, raw_vars_pair, _NOISE_WRAP_MIN_OPERAND_CORR, _NOISE_WRAP_CORR_COLLAPSE_FRAC, verbose, transformed_vars)

    # REJECTION LEDGER (additive): record a pair the per-pair acceptance gate is about
    # to DROP - the joint-prevalence floor declined AND both the prewarp and the
    # marginal-uplift (abs-MAD / joint-recovery) fallbacks declined. Attribute to whichever
    # floor it primarily missed: the engineered-MI prevalence floor (the 0.97 floor the
    # session hand-diagnoses) is the primary gate; if the ratio DID clear that bar (so the
    # prewarp/uplift path declined for another reason) tag the marginal-uplift floor.
    # All values were already computed above (no recompute).
    _score_one_pair_all_values_were_already(st._passes_joint_gate, st._prewarp_accept, st._marginal_uplift_accept, st._usability_accept, fe_min_engineered_mi_prevalence, num_fs_steps, st.best_config, raw_vars_pair, cols, st._gate_ratio, rejection_records, rejection_ledger_out)

    if st._passes_joint_gate or st._prewarp_accept or st._marginal_uplift_accept or st._usability_accept:  # Best transformation is good enough
        st._pair_res_entry = _emit_pair_features(
            raw_vars_pair=raw_vars_pair,
            pair_mi=pair_mi,
            best_mi=st.best_mi,
            best_config=st.best_config,
            var_pairs_perf=st.var_pairs_perf,
            this_pair_features=st.this_pair_features,
            _passes_joint_gate=st._passes_joint_gate,
            _prewarp_accept=st._prewarp_accept,
            _marginal_uplift_accept=st._marginal_uplift_accept,
            _usability_primary=st._usability_primary,
            final_transformed_vals=final_transformed_vals,
            _this_chunk_deferred=_this_chunk_deferred,
            _config_by_i=_config_by_i,
            _resolve_col=_resolve_col,
            _corr_y_cont=_corr_y_cont,
            _safe_abs_corr=_safe_abs_corr,
            transformed_vars=transformed_vars,
            vars_transformations=vars_transformations,
            binary_transformations=binary_transformations,
            unary_transformations=unary_transformations,
            numeric_vars_to_consider=numeric_vars_to_consider,
            fe_good_to_best_feature_mi_threshold=fe_good_to_best_feature_mi_threshold,
            fe_max_external_validation_factors=fe_max_external_validation_factors,
            fe_max_steps=fe_max_steps,
            fe_multi_emit_max_per_pair=fe_multi_emit_max_per_pair,
            fe_multi_emit_mi_floor=fe_multi_emit_mi_floor,
            fe_multi_emit_diversity_corr=fe_multi_emit_diversity_corr,
            quantization_nbins=quantization_nbins,
            quantization_method=quantization_method,
            quantization_dtype=quantization_dtype,
            classes_y=classes_y,
            classes_y_safe=classes_y_safe,
            freqs_y=freqs_y,
            fe_npermutations=fe_npermutations,
            fe_min_nonzero_confidence=fe_min_nonzero_confidence,
            _mi_band=_mi_band,
            _op_code_arr=_op_code_arr_all,
            _fe_env_gate=_fe_env_gate,
            cols=cols,
            original_cols=original_cols,
            _use_subsample=_use_subsample,
            _X_full=_X_full,
            _full_n_rows=_full_n_rows,
            _prewarp_spec_by_var=_prewarp_spec_by_var,
            _gate_med_median_by_var=_gate_med_median_by_var,
            engineered_operand_values=engineered_operand_values,
            _extval_raw_col=_extval_raw_col,
            _rng_extval=_rng_extval,
            _n_workers=_n_workers,
            serial_main_thread=serial_main_thread,
            verbose=verbose,
            messages=st.messages,
            discretize_array=discretize_array,
            discretize_2d_quantile_batch=discretize_2d_quantile_batch,
            mi_direct=mi_direct,
            get_new_feature_name=get_new_feature_name,
            _rebuild_full_survivor_col=_rebuild_full_survivor_col,
            _can_hoist_shared_buffer=_can_hoist_shared_buffer,
            _narrow_code_dtype=_narrow_code_dtype,
            _materialise_extval_njit=_materialise_extval_njit,
            _fe_use_parallel_kernels=_fe_use_parallel_kernels,
            _dispatch_batch_mi_with_noise_gate=_dispatch_batch_mi_with_noise_gate,
            batch_mi_with_noise_gate=batch_mi_with_noise_gate,
            use_su_normalization=use_su_normalization,
            X=X,
        )
    return st._pair_res_entry, st.best_config, st.best_mi
