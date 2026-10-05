"""Helpers carved out of ``_pairs_score`` to keep that module under its size budget."""
from __future__ import annotations

from typing import Any

import logging
from timeit import default_timer as timer
from functools import partial

import numpy as np

from ._pairs_operand_floor import _beats_the_larger_operand
from ._pairs_abs_corr_gpu import abs_corr_or_none, candidate_abs_corr
from ._pairs_chunks import _compute_one_fe_chunk
from ._pairs_dispatch import _dispatch_batch_mi_with_noise_gate
from ._pairs_materialise import (
    _narrow_code_dtype,
)
from .._fe_usability_signal import (  # shared leaf detectors (numpy-only, no cycle)
    pair_is_tail_concentrated_rankaware,
    tail_concentration_form_override,
)
from mlframe.utils.log_throttle import log_throttle

# DEGENERATE-PAIR |corr| threshold. A prospective pair is DEGENERATE when its winning
# composite is numerically ~= a SINGLE one of its own (warped) operands - the other operand's transform
# collapsed to ~constant so the binary op is just a re-wrap of one source and adds NO genuine joint
# information (e.g. ``mul(prewarp(b),prewarp(a__L2))`` where ``prewarp(b)`` is ~constant -> the column is
# essentially ``prewarp(a__L2)`` ~= a**2). Such a pair keeps |corr|~=1.0 with that one operand, clears the
# joint-prevalence gate, and DISPLACES the clean single-source univariate basis (a__L2). A GENUINE 2-var
# pair (a/b, a*b) has |corr| well below this with EITHER single operand, so the bar leaves it untouched.
# 0.999 is intentionally near-1.0: it fires only on a true single-operand re-wrap, never on a real pair.
_DEGENERATE_PAIR_SINGLE_OPERAND_CORR: float = 0.999


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


def _score_one_pair_step1_reads_buffer_have(_chunk_global_batch, _chunk_buffer, _pair_to_chunk, raw_vars_pair, chunk_state, _fe_chunks, _pair_valid_combs, vars_transformations, transformed_vars, binary_transformations, quantization_nbins, quantization_dtype, classes_y, classes_y_safe, freqs_y, fe_npermutations, fe_min_nonzero_confidence, batch_mi_with_noise_gate, use_su_normalization, _PREWARP_UNARY, discretize_2d_quantile_batch, serial_main_thread, _fe_defer_float, _op_code_arr_all, _gpu_mat_on, _fe_env_gate, st):
    """Step 1 of _score_one_pair: lines starting at ``if _chunk_global_batch and (_chunk_buffer is not None):``."""
    if _chunk_global_batch and (_chunk_buffer is not None):
        _my_chunk = _pair_to_chunk.get(raw_vars_pair)
        if _my_chunk is not None:
            if _my_chunk != chunk_state["loaded_idx"]:
                # CHUNK PIPELINE (2026-07-02, max-GPU phase): under the strict-resident gate the driver put a
                # single-worker executor + a SECOND chunk buffer in ``chunk_state`` - chunk c+1's whole
                # produce (materialise+bin+MI, GIL-releasing GPU/njit work) runs on the worker thread into the
                # alternate buffer WHILE the main thread replays chunk c's pairs, so the GPU no longer idles
                # through the host consume phase. Depth-1 double buffer: producing c+1 reuses slot (c-1)%2,
                # whose pairs were fully consumed before c+1 was submitted. Selection identical: the SAME
                # ``_compute_one_fe_chunk`` runs with the SAME inputs, chunks resolve in plan order, and the
                # consumer never starts a chunk before its future resolves. Any worker fault -> synchronous
                # inline compute for that chunk (never a regression). Non-pipelined path byte-identical.
                _bufs = chunk_state.get("pipeline_buffers")
                _ex = chunk_state.get("pipeline_ex")
                _target_buf = _bufs[_my_chunk % 2] if _bufs is not None else _chunk_buffer

                def _produce(_ci, _buf):
                    """Compute one FE chunk's engineered values into ``_buf``, for the pipelined worker to prefetch ahead of the consumer."""
                    return _compute_one_fe_chunk(
                        chunk_pairs=_fe_chunks[_ci],
                        pair_valid_combs=_pair_valid_combs,
                        chunk_buffer=_buf,
                        vars_transformations=vars_transformations,
                        transformed_vars=transformed_vars,
                        binary_transformations=binary_transformations,
                        quantization_nbins=quantization_nbins,
                        quantization_dtype=quantization_dtype,
                        classes_y=classes_y,
                        classes_y_safe=classes_y_safe,
                        freqs_y=freqs_y,
                        fe_npermutations=fe_npermutations,
                        fe_min_nonzero_confidence=fe_min_nonzero_confidence,
                        batch_mi_kernel=batch_mi_with_noise_gate,
                        use_su=use_su_normalization(),
                        prewarp_unary=_PREWARP_UNARY,
                        logger=logger,
                        discretize_2d_quantile_batch=discretize_2d_quantile_batch,
                        serial_main_thread=serial_main_thread,  # OPT-A
                        defer_float=_fe_defer_float,
                        op_code_arr=_op_code_arr_all,
                        gpu_mat_enabled=_gpu_mat_on,
                        env_gate=_fe_env_gate,
                    )

                _mi_cache = None
                _fut = (chunk_state.get("pipeline_futures") or {}).pop(_my_chunk, None)
                _mi_cache = _score_one_pair_fut_none(_fut, _my_chunk, _mi_cache)
                if _mi_cache is None:
                    _mi_cache = _produce(_my_chunk, _target_buf)
                chunk_state["mi_cache"] = _mi_cache
                chunk_state["active_buffer"] = _target_buf
                chunk_state["loaded_idx"] = _my_chunk
                # Reset the per-chunk re-materialise caches + capture the deferral signal/metadata.
                chunk_state["float_deferred"] = bool(chunk_state["mi_cache"].get("__float_deferred__", False))
                chunk_state["defer_meta"] = chunk_state["mi_cache"].get("__defer_meta__")
                chunk_state["tv_gpu"] = None
                chunk_state["resolved_cols"] = {}
                # PREFETCH the next chunk into the alternate buffer while this chunk's pairs replay.
                _score_one_pair_prefetch_next_chunk_into(_ex, _bufs, _my_chunk, _fe_chunks, _produce, chunk_state)
            st._chunk_entry = chunk_state["mi_cache"].get(raw_vars_pair)


def _score_one_pair_step2_def_resolve_col(final_transformed_vals, chunk_state, transformed_vars):
    """Step 2 of _score_one_pair: lines starting at ``def _resolve_col(_buf_col):``."""
    def _resolve_col(_buf_col):
        """Continuous candidate column ``_buf_col`` for the intermediate (subsample) scoring reads.
        Reads the filled chunk buffer when present; on the DEFERRED-float GPU path the buffer is
        unfilled, so RE-MATERIALISE the column on the GPU via ``_fe_materialise_block_gpu`` - the SAME
        kernel that filled the bulk buffer, so the bytes are BIT-IDENTICAL (no cupy-vs-numpy ULP shift).
        The operand table is uploaded ONCE per chunk and cached; resolved columns are cached per
        buf_col (a column may be read several times). Returns a host float32 (n,) array, nan_to_num'd
        exactly as the bulk materialise (the kernel scrubs inline + we nan_to_num the D2H copy)."""
        if final_transformed_vals is not None:
            return final_transformed_vals[:, _buf_col]
        _cached = chunk_state["resolved_cols"].get(_buf_col)
        if _cached is not None:
            return _cached
        import cupy as cp
        from mlframe.feature_selection.filters._gpu_resident_fe import _fe_materialise_block_gpu, _resident_operand_table  # type: ignore[attr-defined]  # dynamically re-exported via globals() from _gpu_resident_materialise
        if chunk_state["tv_gpu"] is None:
            # per-step weakref-cached operand table (shared with gpu_materialise) -> one H2D/step
            chunk_state["tv_gpu"] = _resident_operand_table(cp, transformed_vars)
        _a, _b, _ops = chunk_state["defer_meta"]
        _cand = _fe_materialise_block_gpu(
            chunk_state["tv_gpu"], _a[_buf_col:_buf_col + 1], _b[_buf_col:_buf_col + 1], _ops[_buf_col:_buf_col + 1],
        )
        _col = np.ascontiguousarray(cp.asnumpy(_cand)[:, 0])
        np.nan_to_num(_col, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        chunk_state["resolved_cols"][_buf_col] = _col
        del _cand
        return _col

    def _device_col(_buf_col):
        """Candidate column ``_buf_col`` as a float32 DEVICE array (same kernel and bytes as ``_resolve_col``, no copy back), or ``None`` when the
        column is host-resident or the device path faults."""
        if final_transformed_vals is not None:
            return None
        try:
            import cupy as cp
            from mlframe.feature_selection.filters._gpu_resident_fe import _fe_materialise_block_gpu, _resident_operand_table  # type: ignore[attr-defined]  # dynamically re-exported via globals() from _gpu_resident_materialise

            if chunk_state["tv_gpu"] is None:
                chunk_state["tv_gpu"] = _resident_operand_table(cp, transformed_vars)
            _a, _b, _ops = chunk_state["defer_meta"]
            return _fe_materialise_block_gpu(chunk_state["tv_gpu"], _a[_buf_col:_buf_col + 1], _b[_buf_col:_buf_col + 1], _ops[_buf_col:_buf_col + 1])[:, 0]
        except Exception as e:
            logger.debug("device candidate column failed, falling back to the host column: %s", e)
            return None

    def _abs_corr(_buf_col, _y, _y_finite):
        """|corr(y)| of candidate column ``_buf_col`` computed ON the device (only the scalar crosses to the host), or ``None`` when the column is
        already on the host or the device path faults (the caller then reads ``_resolve_col``)."""
        if final_transformed_vals is not None or chunk_state["resolved_cols"].get(_buf_col) is not None:
            return None
        from ._pairs_abs_corr_gpu import abs_corr_or_none

        _col_dev = _device_col(_buf_col)
        return None if _col_dev is None else abs_corr_or_none(_col_dev, _y, _y_finite)

    _resolve_col.abs_corr = _abs_corr  # type: ignore[attr-defined]  # optional capabilities probed with getattr by the emit step
    _resolve_col.device = _device_col  # type: ignore[attr-defined]
    return _resolve_col


def _score_one_pair_step3_resolve_col_treat(st, final_transformed_vals, _op_code_arr, binary_transformations, vars_transformations, _PREWARP_UNARY, quantization_nbins, quantization_dtype, transformed_vars, _fe_gpu_discretize_enabled, classes_y, classes_y_safe, freqs_y, fe_npermutations, fe_min_nonzero_confidence, use_su_normalization, discretize_2d_quantile_batch, serial_main_thread, batch_mi_with_noise_gate, _fe_env_gate, _need_recompute_map, _config_by_i, fe_print_best_mis_only, verbose, pair_mi, discretize_array, quantization_method, mi_direct):
    """Step 3 of _score_one_pair: lines starting at ``if st._use_batch_disc:``."""
    if st._use_batch_disc:
        # CROSS-PAIR fast path: this pair was batched together with the rest of its
        # chunk in ``_compute_one_fe_chunk``. Its candidate columns already live
        # in ``_chunk_buffer`` (the buf_col index is the config ``i``), its MI is
        # already computed, and its per-bin_func materialise timings are recorded.
        # We only replay the EXACT per-candidate tracking below. Bit-identical: the
        # chunk's ONE discretize_2d + ONE batch_mi score each column independently,
        # and the candidate order per pair is the SAME (combs x bin_funcs) order the
        # per-pair Phase 1 produced.
        _batch_candidates, _fe_mi_arr = _score_one_pair_step3_step1_pair_phase_produced(st, final_transformed_vals, _op_code_arr, binary_transformations, vars_transformations, _PREWARP_UNARY, quantization_nbins, quantization_dtype, transformed_vars, _fe_gpu_discretize_enabled, classes_y, classes_y_safe, freqs_y, fe_npermutations, fe_min_nonzero_confidence, use_su_normalization, discretize_2d_quantile_batch, serial_main_thread, batch_mi_with_noise_gate, _fe_env_gate)

        # Replay best/prewarp/config tracking in the SAME order candidates were
        # produced -> identical tie-break behaviour. ``_fe_mi_arr`` is indexed by the
        # buffer column (per-pair: 0..K-1; cross-pair: the chunk-buffer column).
        if _batch_candidates and _fe_mi_arr is not None:
            for transformations_pair, bin_func_name, _ci, _uses_pw in _batch_candidates:
                # Cast to Python float so ``var_pairs_perf`` / downstream tracking see
                # the same scalar type ``mi_direct`` returned (numba njit returns a
                # python float at the call boundary). Value is bit-identical.
                fe_mi = float(_fe_mi_arr[_ci])

                config = (transformations_pair, bin_func_name, _ci)
                st.var_pairs_perf[config] = fe_mi
                if _need_recompute_map:
                    assert _config_by_i is not None
                    _config_by_i[_ci] = (transformations_pair[0], transformations_pair[1], bin_func_name)

                if fe_mi > st.best_mi:
                    st.best_mi = fe_mi
                    st.best_config = config
                if _uses_pw:
                    if fe_mi > st.best_prewarp_mi:
                        st.best_prewarp_mi = fe_mi
                        st.best_prewarp_config = config
                else:
                    if fe_mi > st.best_nonprewarp_mi:
                        st.best_nonprewarp_mi = fe_mi
                        st.best_nonprewarp_config = config
                _score_one_pair_fe_mi_best_mi(fe_mi, st.best_mi, fe_print_best_mis_only, verbose, bin_func_name, transformations_pair, pair_mi)
    else:
        _score_one_pair_step3_step2_transformations_pair_st(st, vars_transformations, transformed_vars, _PREWARP_UNARY, binary_transformations, final_transformed_vals, discretize_array, quantization_nbins, quantization_method, quantization_dtype, mi_direct, classes_y, classes_y_safe, freqs_y, fe_min_nonzero_confidence, fe_npermutations, _need_recompute_map, _config_by_i, fe_print_best_mis_only, verbose, pair_mi)


def _score_one_pair_step3_step1_pair_phase_produced(st, final_transformed_vals, _op_code_arr, binary_transformations, vars_transformations, _PREWARP_UNARY, quantization_nbins, quantization_dtype, transformed_vars, _fe_gpu_discretize_enabled, classes_y, classes_y_safe, freqs_y, fe_npermutations, fe_min_nonzero_confidence, use_su_normalization, discretize_2d_quantile_batch, serial_main_thread, batch_mi_with_noise_gate, _fe_env_gate):
    """Step 1 of _score_one_pair_step3_resolve_col_treat: lines starting at ``if st._chunk_entry is not None:``."""
    if st._chunk_entry is not None:
        _batch_candidates, _fe_mi_by_col, _pair_local_times = st._chunk_entry
        for _bf_name, _dt in _pair_local_times.items():
            st._local_times[_bf_name] = st._local_times.get(_bf_name, 0.0) + _dt
        # ``_fe_mi_arr`` is indexed by the chunk-buffer column (buf_col), so the
        # replay's ``_fe_mi_arr[_ci]`` lookup is correct without re-indexing.
        _fe_mi_arr = _fe_mi_by_col
    else:
        assert final_transformed_vals is not None  # _use_batch_disc with no _chunk_entry implies the buffer is filled (see _use_batch_disc definition above)
        # bench-attempt-rejected (2026-06-23, MRMR FE wall /loop iter10): the F2 100k cProfile
        # attributes ~40s tottime to THIS function and ~6.6s to ``_safe_div``, suggesting "Python
        # per-candidate orchestration overhead". A line_profiler pass (line-by-line, F2 100k warm)
        # disproves that: line 310 (``final_transformed_vals[:, i] = bin_func(param_a, param_b)``)
        # is 72.2% of the body's time, the Phase-1b ``nan_to_num`` 12.1%, GPU binning 7.4%, batch-MI
        # dispatch 4.6% - ALL numeric kernels (njit ufuncs / compiled div / GPU), already routed by
        # prior iterations. The Python bookkeeping is negligible: ``_local_times`` dict 0.4%,
        # ``_batch_candidates.append`` 0.2%, the ``binary_transformations.items()`` loop 0.2%, the
        # per-pair-comb ``np.errstate`` 1.0%; the replay loop / ``var_pairs_perf`` dict / recipe-name
        # building / clean-form-demotion loop are each <0.2% (sum of all Python orchestration < 2.5%).
        # ``feature_engineering._safe_div`` is NOT pure-Python overhead: it IS njit-compiled
        # in-run (4 signatures compile during the fit, incl. the float32 'A' strided-column layout this
        # path feeds it) and its 6.6s is genuine compiled float32->float64 ratio compute over 9720
        # 100k-row columns; cProfile attributes the dispatcher's call to the py_func code-object
        # location, which LOOKS like a Python frame but is not. The float64 upcast inside ``_safe_div``
        # (then downcast on the float32 store here) is a real but UNSAFE lever: a native float32 divide
        # rounds differently from float64-divide-then-cast at ULP and this is selection-critical FE
        # scoring, so it is NOT changed. Verdict: ``_score_one_pair`` is already lean on Python overhead
        # - the wall IS candidate-column materialisation COMPUTE. iter11 RESOLVES that wall:
        # the line-310 CPU ``bin_func`` materialise is now routed to the GPU FUSED materialise+bin
        # (the GPU-fused branch directly below), measured F2 100k 83.8s -> 30.0s (2.8x), selection
        # bit-identical - so the "no win" tail of the iter10 verdict is SUPERSEDED for this path.
        # Phase 1: materialise + nan_to_num + record. ``i`` advances exactly as in the
        # per-candidate path so ``config``'s buffer index and ``_config_by_i`` are identical.
        _batch_candidates = []  # (transformations_pair, bin_func_name, i, uses_pw)
        _disc_2d = None  # set by the GPU FUSED materialise+bin path below; None -> CPU Phase 2 bins it

        # GPU FUSED PER-PAIR MATERIALISE+BIN. The CPU
        # ``bin_func`` strided materialise (the line ``final_transformed_vals[:, i] = bin_func(param_a,
        # param_b)`` below) is the per-pair FE-scan WALL: on the F2 100k fit it is 71.9% of
        # ``_score_one_pair`` (line_profiler 45.8s of 63.7s). The chunk path already routes its
        # materialise to the GPU FUSED ``gpu_materialise_discretize_codes_host`` (4.40x vs CPU
        # njit-mat+CPU-bin / 2.03x vs CPU njit-mat+GPU-bin, GTX 1050 Ti n=100k), but it only fires
        # when a chunk holds >1 pair (``_chunk_buffer`` allocated); at the canonical 100k fit the
        # RAM-budgeted chunk width is barely one pair wide (chunkmax~3561 vs pairwidth 1944), so
        # ``_chunk_buffer`` stays None and EVERY pair falls to THIS per-pair CPU path - 100% of
        # candidate materialise (measured: F2 seed-7 = 58320/58320 cols on the CPU line below).
        # So we route the per-pair materialise through the SAME fused GPU kernel here. It builds the
        # (n,K) float candidate matrix on-device from (a_cols, b_cols, op_codes), fills the host buffer
        # (``out_cand`` - the downstream survivor/usability/ext-val stages still read the continuous
        # columns) AND bins it RESIDENT, returning ``_disc_2d`` so the separate Phase-2 binning is
        # skipped. BIT-IDENTICAL selection: ``gpu_materialise_discretize_codes_host`` mirrors
        # ``_materialise_chunk_njit`` (maxdiff 0), which is itself bit-identical to the numpy
        # ``bin_func`` (the chunk-batch invariant), and the inline kernel nan-scrub == the per-column
        # ``np.nan_to_num`` below. PER-OP GATE: routed only when ``_njit_binary_op_codes`` covers EVERY
        # op in the registry (None -> any hypot/scipy.special op -> stay on the bit-safe CPU loop). Gated
        # by the dedicated GPU-binning crossover (if binning wins, the fused mat+bin certainly wins) +
        # the ``MLFRAME_FE_GPU_MATERIALISE`` escape hatch (same knob the chunk path uses). Any GPU
        # failure falls through to the CPU loop below (never a regression).
        # ``_gpu_mat_on``/``_op_code_arr`` are resolved ONCE by the caller (call-invariant), not
        # recomputed for every pair - see the per-call hoist comment in ``check_prospective_fe_pairs``.
        _gpu_fused_done = False
        try:
            if _op_code_arr is not None:
                # Build candidate specs + per-candidate (a_col, b_col, op_code) in the SAME
                # (combs x bin_func) order the CPU path produces -> identical buffer index ``i``.
                _name_list = list(binary_transformations.keys())
                _a_cols: list = []
                _b_cols: list = []
                _ops: list = []
                _gpu_cands: list[Any] = []
                st.i = _score_one_pair_transformations_pair_combs(st.combs, vars_transformations, _PREWARP_UNARY, _name_list, _a_cols, _b_cols, _ops, _op_code_arr, _gpu_cands, st.i)
                _K = st.i
                from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core import _fe_gpu_binning_enabled
                _batch_candidates, _disc_2d, _gpu_fused_done, _defer_meta = _score_one_pair_fe_gpu_binning_enabled(_K, _fe_gpu_binning_enabled, final_transformed_vals, quantization_nbins, quantization_dtype, transformed_vars, _a_cols, _b_cols, _ops, _gpu_cands, _name_list, st._local_times, _batch_candidates, _disc_2d, _gpu_fused_done, st._fe_defer_float)
                if _defer_meta is not None:
                    # same keys the chunk path keeps in ``chunk_state``: the operand table is uploaded once and resolved columns are cached
                    st._pair_defer_state = {"defer_meta": _defer_meta, "tv_gpu": None, "resolved_cols": {}}
        except Exception:
            logger.debug("FE per-pair GPU fused materialise+bin failed; CPU materialise", exc_info=True)
            _gpu_fused_done = False
            _disc_2d = None
            _batch_candidates = []
            st.i = 0

        st.i = _score_one_pair_gpu_fused_done(_gpu_fused_done, st.combs, vars_transformations, transformed_vars, _PREWARP_UNARY, binary_transformations, final_transformed_vals, st._local_times, _batch_candidates, st.i)

        # Phase 2: ONE batch discretisation over the materialised columns [:, :n].
        # Bit-identical to per-column ``discretize_array(method='quantile')`` - the
        # buffer dtype (float32) is NOT cast; per-column edges/codes match exactly.
        _fe_mi_arr = None
        if _batch_candidates:
            # ``i`` advanced once per materialised candidate from 0 (reset per raw-pair),
            # so the filled buffer slice is exactly [:, :i], densely packed 0..i-1.
            _code_dtype = _narrow_code_dtype(quantization_nbins, quantization_dtype)  # OPT-B narrow codes
            _fe_mi_arr = None
            # GPU-resident FE candidate MI (size+HW gated, default OFF via MLFRAME_FE_GPU_DISCRETIZE).
            # At large n*K the per-pair binning + observed-MI counting is the dominant FE-scan cost;
            # the GPU path runs BOTH on-device and returns fe_mi BIT-IDENTICAL to the production
            # analytic dispatch (GPU binning == CPU discretize, maxdiff 0; GPU observed-MI == CPU,
            # maxdiff 0; same analytic chi2 gate), so the FE selection is identical. Returns None for
            # the non-analytic branch (SU / sparse / small-n) -> falls through to the CPU dispatch.
            # a deferred pair has no host float matrix to hand the GPU pair-MI path; its resident codes go to the noise gate below
            _fe_mi_arr = None if st._pair_defer_state is not None else _score_one_pair_non_analytic_branch_su(_fe_gpu_discretize_enabled, final_transformed_vals, st.i, quantization_nbins, classes_y, classes_y_safe, freqs_y, fe_npermutations, fe_min_nonzero_confidence, use_su_normalization, _fe_mi_arr)
            _disc_2d = _score_one_pair_fe_mi_arr_none(_fe_mi_arr, _disc_2d, final_transformed_vals, st.i, quantization_nbins, _code_dtype, discretize_2d_quantile_batch, serial_main_thread)

            # Phase 3: BATCHED MI + permutation noise-gate across ALL K candidate
            # columns in ONE kernel call. Bit-identical to the per-candidate
            # ``mi_direct`` loop on the default FE path (parallelism='outer',
            # n_workers=1 -> parallel_mi_prange, base_seed=0): every candidate is
            # tested against the SAME npermutations shuffles of y (the shuffle is
            # seeded by (base_seed, perm_index) ONLY, never by classes_x), so a single
            # batched kernel can shuffle y once per permutation and score all columns
            # against it - amortising both the MI compute and the shuffle across K.
            # ``_dispatch_batch_mi_with_noise_gate`` routes CPU-njit vs a GPU batched
            # path by n*K via the kernel_tuning_cache (no hardcoded threshold).
            # Skipped when the GPU pair-MI path above already produced ``_fe_mi_arr``.
            if _fe_mi_arr is None:
                assert _disc_2d is not None  # populated by the GPU-fused/GPU-binning path or the CPU discretize_2d_quantile_batch fallback above
                _fe_mi_arr = _dispatch_batch_mi_with_noise_gate(
                    disc_2d=_disc_2d,
                    quantization_nbins=quantization_nbins,
                    classes_y=classes_y,
                    classes_y_safe=classes_y_safe,
                    freqs_y=freqs_y,
                    npermutations=fe_npermutations,
                    min_nonzero_confidence=fe_min_nonzero_confidence,
                    use_su=use_su_normalization(),
                    batch_mi_kernel=batch_mi_with_noise_gate,
                    env_gate=_fe_env_gate,
                )
    return _batch_candidates, _fe_mi_arr


def _score_one_pair_step3_step2_transformations_pair_st(st, vars_transformations, transformed_vars, _PREWARP_UNARY, binary_transformations, final_transformed_vals, discretize_array, quantization_nbins, quantization_method, quantization_dtype, mi_direct, classes_y, classes_y_safe, freqs_y, fe_min_nonzero_confidence, fe_npermutations, _need_recompute_map, _config_by_i, fe_print_best_mis_only, verbose, pair_mi):
    """Step 2 of _score_one_pair_step3_resolve_col_treat: lines starting at ``for transformations_pair in st.combs:``."""
    for transformations_pair in st.combs:
        if (transformations_pair[0] not in vars_transformations) or (transformations_pair[1] not in vars_transformations):
            continue
        param_a = transformed_vars[:, vars_transformations[transformations_pair[0]]]
        param_b = transformed_vars[:, vars_transformations[transformations_pair[1]]]

        # A config "uses prewarp" iff either operand's unary name is the
        # pseudo-unary. Invariant across the bin_func loop -> compute once.
        _uses_pw = transformations_pair[0][1] == _PREWARP_UNARY or transformations_pair[1][1] == _PREWARP_UNARY

        # ``bin_func`` produces NaN/+-inf on extreme Optuna-picked params
        # (overflow in mul/exp, divide-by-zero in log); the downstream
        # nan_to_num + MI gate already sanitise, so the bare numpy
        # RuntimeWarnings carry zero diagnostic value. Suppress them for the
        # whole binary-transform sweep: entering np.errstate per inner
        # iteration cost ~6.8us/iter (measured ~490ms over 72k iters),
        # dwarfing the bin_func work itself; one context per pair-comb
        # removes that. numba kernels (discretize/mi_direct) ignore errstate
        # and nan_to_num emits nothing, so the wider scope is value-identical.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for bin_func_name, bin_func in binary_transformations.items():

                start = timer()
                try:
                    if final_transformed_vals is not None:
                        final_transformed_vals[:, st.i] = bin_func(param_a, param_b)
                        _col_view = final_transformed_vals[:, st.i]
                    else:
                        # Recompute fallback: write into the shared 1D scratch.
                        # bin_func returns a fresh ndarray; copy into the scratch
                        # so downstream nan_to_num + discretize see contiguous
                        # data. Avoids accumulating one alloc per inner iter.
                        assert st._col_buf_1d is not None  # recompute fallback only reached when _need_recompute_map allocated it
                        st._col_buf_1d[:] = bin_func(param_a, param_b)
                        _col_view = st._col_buf_1d
                except Exception:
                    # Failed transform: the buffer slot may still hold a prior column's data. Null it so it is
                    # never scored, and skip the scoring ``else`` (no candidate recorded for this bin_func).
                    log_throttle(logger, "pairs_score_transform_recompute_failed", logging.ERROR, "Error when performing %s", bin_func, exc_info=True)
                    if final_transformed_vals is not None:
                        final_transformed_vals[:, st.i] = np.nan
                else:
                    np.nan_to_num(_col_view, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                    # Wave 27 P1: ``times_spent`` is shared across mrmr.py's
                    # parallel threading dispatch. Accumulate this pair's
                    # per-bin_func timings in a thread-LOCAL dict and merge them
                    # under ``_TIMES_SPENT_LOCK`` once per pair (below); the old
                    # per-inner-iteration lock was a serialization point on the
                    # hot path. Totals are identical.
                    st._local_times[bin_func_name] = st._local_times.get(bin_func_name, 0.0) + (timer() - start)

                    discretized_transformed_values = discretize_array(
                        arr=_col_view, n_bins=quantization_nbins, method=quantization_method, dtype=quantization_dtype
                    )
                    fe_mi, _fe_conf = mi_direct(
                        discretized_transformed_values.reshape(-1, 1),
                        x=np.array([0], dtype=np.int64),
                        y=None,
                        factors_nbins=np.array([quantization_nbins], dtype=np.int64),
                        classes_y=classes_y,
                        classes_y_safe=classes_y_safe,
                        freqs_y=freqs_y,
                        min_nonzero_confidence=fe_min_nonzero_confidence,
                        npermutations=fe_npermutations,
                    )

                    config = (transformations_pair, bin_func_name, st.i)
                    st.var_pairs_perf[config] = fe_mi
                    if _need_recompute_map:
                        # Map i -> (a_key, b_key, bin_func_name) for downstream
                        # rebuild; bin_func is looked up via the original dict.
                        assert _config_by_i is not None
                        _config_by_i[st.i] = (transformations_pair[0], transformations_pair[1], bin_func_name)

                    if fe_mi > st.best_mi:
                        st.best_mi = fe_mi
                        st.best_config = config
                    # Track best-with-prewarp vs best-without so the alternative
                    # uplift gate below can decide whether the prewarp earned its
                    # place (``_uses_pw`` hoisted above the bin_func loop).
                    if _uses_pw:
                        if fe_mi > st.best_prewarp_mi:
                            st.best_prewarp_mi = fe_mi
                            st.best_prewarp_config = config
                    else:
                        if fe_mi > st.best_nonprewarp_mi:
                            st.best_nonprewarp_mi = fe_mi
                            st.best_nonprewarp_config = config
                    if fe_mi > st.best_mi * 0.85:
                        if not fe_print_best_mis_only or (fe_mi == st.best_mi):
                            if verbose > 2:
                                logger.debug("MI of transformed pair %s(%s)=%.4f, MI of the plain pair %.4f", bin_func_name, transformations_pair, fe_mi, pair_mi)
                    st.i += 1


def _score_one_pair_step4_iteration_serialization_point(st, times_spent, verbose, raw_vars_pair, _corr_y_cont, _PREWARP_UNARY, fe_good_to_best_feature_mi_threshold, final_transformed_vals, _this_chunk_deferred, _resolve_col, _config_by_i, transformed_vars, vars_transformations, binary_transformations, _safe_abs_corr, pair_mi, fe_mm_debias_prevalence, classes_y, freqs_y, quantization_nbins, discretize_array, quantization_method, quantization_dtype, _operand_discretized, fe_min_engineered_mi_prevalence, num_fs_steps, _operand_marginal_mi, _prewarp_active, prewarp_uplift_threshold):
    """Step 4 of _score_one_pair: lines starting at ``_score_one_pair_iteration_serialization_point_under(st._local_times, t``."""
    _score_one_pair_iteration_serialization_point_under(st._local_times, times_spent)

    if verbose > 2:
        logger.debug("For pair %s, best config is %s with best mi= %s", raw_vars_pair, st.best_config, st.best_mi)

    # CLEAN-FORM DEMOTION over the per-pair MI winner. The ``prewarp``
    # pseudo-unary fits a learned 1-D orthogonal-poly warp per operand; on a target whose
    # inner function is already LIBRARY-expressible up to a MONOTONE distortion (e.g.
    # ``log(c)*sin(d)`` - ``mul(log(c),sin(d))`` is the clean form, while the warp learns a
    # monotone re-expression ``mul(prewarp(c),sin(d))``) the prewarp form has IDENTICAL
    # ordering -> bit-equal binned MI, but MI is RANK-only so it cannot prefer the clean leg.
    # The warp then wins ``best_config`` by an MI tie/epsilon and propagates a DISTORTED form
    # (``log(div(sqr(a),neg(b)))`` for the a/b half, double-``prewarp(c)`` for the c/d half)
    # that the step-k>1 composite chains, displacing the clean additive compound AND dragging
    # in a redundant raw operand. Demote: when the global winner USES a prewarp operand but a
    # clean elementary-library (non-prewarp) form scores essentially the SAME target MI, keep
    # the prewarp winner ONLY if it has a real LINEAR-USABILITY uplift (|corr(continuous y)|)
    # over the best clean form. This preserves prewarp's INTENDED case - a genuinely
    # non-monotone inner (``a**3-2a``) where the warp's reconstruction is MORE linearly usable
    # than any library form, so the uplift is real and the prewarp form is kept - while making
    # the monotone-equivalent case (no |corr| uplift) fall back to the clean library compound.
    if (
        st.best_config is not None
        and st.best_nonprewarp_config is not None
        and st.best_config is not st.best_nonprewarp_config
        and st.best_nonprewarp_mi > 0.0
        and _corr_y_cont is not None
    ):
        _bc_uses_pw = (
            isinstance(st.best_config[0], (tuple, list))
            and len(st.best_config[0]) == 2
            and (st.best_config[0][0][1] == _PREWARP_UNARY or st.best_config[0][1][1] == _PREWARP_UNARY)
        )
        # Only act when the winner is a prewarp form AND the clean form is MI-equivalent
        # (within the same 0.85 leaders band already used as the equivalence notion). A
        # strictly-higher-MI prewarp winner is left untouched - the prewarp/marginal gates
        # below decide it on its own merits; demotion is for the MI-tie monotone case only.
        if _bc_uses_pw and st.best_nonprewarp_mi >= st.best_mi * fe_good_to_best_feature_mi_threshold:

            def _config_corr(_cfg):
                """|corr(continuous y)| of a config's materialised continuous column; -1.0 when
                it cannot be rebuilt (so an unrecoverable form never wins the comparison)."""
                try:
                    _ci = _cfg[2]
                    if final_transformed_vals is not None:
                        _v = final_transformed_vals[:, _ci]
                    elif _this_chunk_deferred:
                        # DEFERRED-float GPU path: re-materialise this column on the GPU (bit-identical
                        # to the bulk buffer -> the clean-form demotion uses the EXACT same |corr| it
                        # would have under the host buffer; a numpy recompute here flips it at ULP).
                        _dev_corr = candidate_abs_corr(_resolve_col, _safe_abs_corr, _ci)
                        if _dev_corr is not None:
                            return _dev_corr
                        _v = _resolve_col(_ci)
                    elif _config_by_i is not None and _ci in _config_by_i:
                        _ak, _bk, _bn = _config_by_i[_ci]
                        _pa = transformed_vars[:, vars_transformations[_ak]]
                        _pb = transformed_vars[:, vars_transformations[_bk]]
                        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                            _v = binary_transformations[_bn](_pa, _pb)
                        _v = np.nan_to_num(np.asarray(_v, dtype=np.float32), copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    else:
                        return -1.0
                    return _safe_abs_corr(_v)
                except Exception as e:
                    # Unmeasured, not unrecoverable: -1.0 here used to switch the clean-form demotion OFF when the CLEAN side failed.
                    log_throttle(
                        logger, "pairs_score_config_corr_failed", logging.WARNING,
                        "_config_corr: could not measure |corr| for config %r (%s: %s); the prewarp demotion treats it as unmeasured",
                        _cfg, type(e).__name__, e,
                    )
                    return None

            _pw_corr = _config_corr(st.best_config)
            _clean_corr = _config_corr(st.best_nonprewarp_config)
            # Demote to the clean form unless the prewarp form is MEANINGFULLY more linearly
            # usable. ``1.05`` = the prewarp must beat the clean |corr| by >= 5% to justify the
            # distorted re-expression; a genuinely non-monotone inner clears this comfortably
            # (its warp reconstruction is the ONLY linearly-aligned form), while a
            # monotone-equivalent warp scores <= the clean form and is demoted.
            st.best_config, st.best_mi = _score_one_pair_monotone_equivalent_warp_scores(_pw_corr, _clean_corr, st.best_nonprewarp_config, st.best_nonprewarp_mi, st.var_pairs_perf, _PREWARP_UNARY, verbose, st.messages, st.best_config, st.best_mi)

    # experiment-rejected (2026-06-03): a held-out-CV firewall here (score
    # per-combo MI on a TRAIN stride slice for honest selection, then keep the
    # winner only if its held-out VAL-slice MI retains >= ratio of train MI) was
    # implemented and benched END-TO-END on Layer-49 - NO gain. In an isolated
    # probe it separated cleanly (genuine synergy val/train 0.90-1.04 vs noise-FE
    # 0.12-0.36), BUT in the real pipeline the tighter prevalence-gate defaults
    # (fe_synergy_min_prevalence 1.5 / fe_min_engineered_mi_prevalence 0.97)
    # already remove the pure noise*noise products, and the RESIDUAL "noise" FE
    # are signal*noise combos (e.g. max(log(L4_s2),noise_3) - L4_s2 is a real
    # sensor) that genuinely generalise (val/train > 0.5) and SHOULD be kept; the
    # firewall's train-based selection (half the rows) then merely added selection
    # noise (+1 support). Prevalence gating subsumes the win -> not shipped.
    # Standard acceptance: the best engineered MI clears the configured
    # fraction of the 2-D pair-joint MI.
    #
    # MILLER-MADOW DEBIAS (2026-06-09, + #4). The RAW ratio
    # ``best_mi / pair_mi`` compares a 1-D engineered MI (over ~``quantization_nbins``
    # bins) against a 2-D joint MI (over ~``nbins^2`` bins). Both are plug-in MIs whose
    # positive bias is ``(k_x-1)(k_y-1)/2n``; the JOINT denominator's term is ~``nbins``x
    # larger, so the raw ratio is structurally depressed below 1.0 even when the 1-D
    # feature captures all the joint information (worst at small/moderate n) - this is
    # exactly the documented reason the marginal-uplift fallback gate had to be added.
    # When ``fe_mm_debias_prevalence`` we subtract the MM MI-bias term from BOTH sides,
    # using the OCCUPIED bin counts (#4: nominal ``nbins`` over-corrects heavy-tailed
    # columns that collapse), with a denominator-positivity guard that defers to the raw
    # ratio when the joint bias term swamps the finite-sample joint MI. ``->`` raw ratio
    # as ``n -> inf`` (bias terms vanish) => large-n selection byte-untouched. The order-2
    # maxT floor (the outer guard) is MM-debiased CONSISTENTLY upstream (the IRON RULE),
    # so admitting more pairs here does NOT weaken the best-of-pool noise floor.
    #
    # bench-attempt-rejected (2026-06-09, FS "permutation-null-calibrated
    # prevalence bar"). The idea: REPLACE the hardcoded ``fe_min_engineered_mi_prevalence``
    # (0.90) with a SELF-CALIBRATING per-pool null ratio - in the SAME K y-shuffles the
    # order-2 maxT floor runs, ALSO mirror the max-over-transforms search (discretise the
    # elementary binary bank mul/add/sub/div/max/min over the CONTINUOUS operands ONCE -
    # permutation-invariant - then per shuffle take max 1-D engineered MI / joint pair MI),
    # and gate ``best_mi/pair_mi`` against the q95 of that null-ratio distribution (the chance
    # ceiling), admitting only ABOVE it. Unlike #1 (a DETERMINISTIC bias subtraction that
    # uniformly relaxes the bar) the null ratio is calibrated to what NOISE actually produces.
    # MEASURED (standalone probe, N_BINS=8, K=25, q=0.95):
    #   * PURE NOISE (n=2000, p=12): null q95 ratio ~0.16; real noise-pair ratios <=0.17 ->
    #     ~5% admitted = pure (1-q) chance rate. The HARD noise-FP gate PASSES on clean noise.
    #   * He2(a)*b genuine synergy (n=500/2000/8000): real ratio 0.28/0.275/0.268 >> null
    #     0.15/0.16/0.17 -> ADMIT, while the hardcoded 0.90 bar REJECTS at every n. In a mixed
    #     He2-signal+8-noise frame the null bar admits the genuine (a,b) pair and 0-1/28 noise
    #     pairs (chance rate). So in ISOLATION #5 is a genuine improvement over #1.
    # BUT bench-REJECTED on the case that matters - the user's WEAK F2
    # (``0.2*a**2/b + log(c*2)*sin(d/3)``, the SAME target that rejected #1/#8/#19): the null
    # ceiling is ~0.167 (calibrated to clean-noise pairs, ratio ~0.13-0.17), but EVERY weak-F2
    # pair sits FAR above it (5 seeds, n=20000): genuine_ab ~0.81, genuine_cd ~0.73, AND all four
    # cross-mix pairs 0.56-0.72 (cross(b,d) ~0.717 >= genuine_cd). So the null bar ADMITS every
    # cross-mix on every seed - the IRON-RULE failure mode, identical to #1 (cross-mix 3/10 ->
    # 9/10). ROOT CAUSE is the documented fundamental detectability limit (see
    # ``test_mrmr_weak_f2_seed_stability.py`` "THREE DIRECT LEVERS EXHAUSTED"): the cross-mix
    # smuggles the dominant MONOTONE predictor ``c`` across the pair boundary, so its 1-D
    # engineered summary recovers a large fraction of its (real, cross) joint - a HIGH ratio
    # indistinguishable from genuine synergy by ANY MI threshold. The null bar measures the
    # noise floor, but the weak-F2 problem is NOT noise admission; it is a real-monotone-predictor
    # cross-mix whose ratio is nowhere near the noise floor. AND the existing marginal-uplift /
    # prewarp FALLBACK already recovers the genuine pairs end-to-end at n=500/2000/8000, so #5
    # adds ZERO incremental recovery while WEAKENING cross-mix rejection. #5 is structurally a
    # 4th MI-threshold lever and fails by construction like #1/#8/#19; do NOT re-attempt an
    # MI-threshold/ratio fix here. Numbers + verdict in D:/Temp/null_prev_results.md.
    st._gate_ratio = (st.best_mi / pair_mi) if pair_mi > 0.0 else 0.0
    st._gate_ratio = _score_one_pair_mi_threshold_ratio_fix(fe_mm_debias_prevalence, pair_mi, st.best_config, classes_y, freqs_y, quantization_nbins, discretize_array, _resolve_col, quantization_method, quantization_dtype, _operand_discretized, raw_vars_pair, st.best_mi, st._gate_ratio)
    st._passes_joint_gate = st._gate_ratio > fe_min_engineered_mi_prevalence * (1.0 if num_fs_steps < 1 else 1.025)
    st._passes_joint_gate = st._passes_joint_gate and _beats_the_larger_operand(st.best_mi, _operand_marginal_mi, raw_vars_pair, st.messages if verbose else None, pair_mi)

    # Alternative pre-warp acceptance: the joint-prevalence gate
    # structurally rejects a 1-D summary of a 2-D pair on a non-monotone inner
    # distortion. Admit the prewarp winner when it beats the best NON-prewarp
    # engineered MI by ``prewarp_uplift_threshold`` AND clears the pair-MI
    # noise floor (its MI must exceed the larger individual operand MI - the
    # same notion the smart_polynom baseline uplift uses), so it cannot fire
    # on noise (where prewarp does not beat the library) or pure-linear data
    # (where the elementary library already saturates and the prewarp adds no
    # uplift). When it fires, the prewarp config becomes the winner.
    st._prewarp_accept = False
    st._prewarp_accept, st.best_config, st.best_mi = _score_one_pair_uplift_fires_prewarp_config(_prewarp_active, st._passes_joint_gate, st.best_prewarp_config, st.best_nonprewarp_mi, st.best_prewarp_mi, prewarp_uplift_threshold, verbose, st.messages, pair_mi, fe_min_engineered_mi_prevalence, st._prewarp_accept, st.best_config, st.best_mi)

    # MARGINAL-UPLIFT alternative acceptance: admit a pair the joint-prevalence
    # gate rejects when its best ELEMENTARY-LIBRARY (non-prewarp) engineered column
    # beats the LARGER individual operand marginal MI by ``_FE_MARGINAL_UPLIFT_MIN_RATIO``
    # AND still recovers at least ``_FE_MARGINAL_UPLIFT_MIN_JOINT_RATIO`` of the inflated
    # 2-D joint. Rationale + thresholds: see the module-level constants. Genuine synergy
    # pairs (a**2/b, log(c)*sin(d)) clear both; cross-pair artefacts that merely recapture
    # one operand's marginal fail the uplift bar, and structureless noise pairs never reach
    # here (the upstream pair screen + order-2 maxT floor remove them). Only fires when the
    # primary joint gate AND the prewarp path both declined, so it is purely additive recall
    # for genuine pairs the strict joint bar drops. We score + promote the best NON-PREWARP
    # winner: the prewarp pseudo-unary has its own dedicated acceptance path above
    # (``_prewarp_accept``), and promoting a prewarp form here would require the per-operand
    # warp spec to round-trip into the recipe - which is only guaranteed on the prewarp path.
    st._marginal_uplift_accept = False
    st._marginal_uplift_accept, st.best_config, st.best_mi = _score_one_pair_warp_spec_round_trip(st._passes_joint_gate, st._prewarp_accept, st.best_nonprewarp_config, st.best_nonprewarp_mi, pair_mi, _operand_marginal_mi, raw_vars_pair, verbose, st.messages, fe_min_engineered_mi_prevalence, st._marginal_uplift_accept, st.best_config, st.best_mi)

    # TAIL-CONCENTRATION USABILITY ADMISSION + WINNER PROMOTION. Under heavy operand outliers a
    # genuine ratio (a**2/b) is TAIL-CONCENTRATED: in the clean bulk its rank association with y collapses
    # (Spearman ~0), so its rank-MI sits in a tight noise band where a SPURIOUS form out-ranks the true ratio
    # by a REAL margin the tie band cannot bridge (F2 with_outliers: min(reciproc(a),sign(b)) rank-MI 0.073
    # beats the true div(sqr(a),b) 0.063), AND the whole pair's best rank-MI fails the engineered-MI joint /
    # marginal-uplift gates (0.073/0.094 = 0.78 < the 0.90+ bar). Both failures are the SAME rank-MI blind
    # spot: the distinguishing signal is LINEAR (|corr(continuous y)| 0.986 true vs 0.371 spurious, corr
    # outlier-inflated - exactly right for a tail signal). Detect tail concentration on the RAW operands vs
    # the continuous y (``pair_is_tail_concentrated`` - best bivariate-form |corr| clears the bar AND beats
    # the best single-operand form by the pairness margin, so cross-mix / lone-dominant-operand / noise pairs
    # are rejected). When detected, PROMOTE the |corr|-best engineered FORM as the winner: this (a) admits the
    # pair when every rank-MI gate declined (``_usability_accept``) and (b) both widens the emit leader set to
    # include the true form - its rank-MI is BELOW ``fe_good_to_best_feature_mi_threshold`` x the spurious
    # leader, so it would otherwise be excluded - and flips the winner-selection primary key to |corr|
    # (``_usability_primary``) so the true ratio wins. The rank-vs-|corr| DISAGREEMENT gate
    # (``tail_concentration_form_override``) makes it a strict no-op on the 4 passing F2 profiles + canonical
    # fixtures (there the ratio is BOTH the rank-MI and |corr| leader, so nothing is promoted). Best-effort:
    # any error leaves the rank-MI decision untouched.
    st._usability_accept = False
    st._usability_primary = False


def _score_one_pair_step5_any_error_leaves(fe_pair_usability_admission_enable, _corr_y_cont, pair_mi, st, final_transformed_vals, _this_chunk_deferred, _extval_raw_col, raw_vars_pair, fe_pair_usability_admission_min_corr, fe_pair_usability_admission_pairness_margin, _resolve_col, _safe_abs_corr, _raw_operand_abs_corr, _mi_band, verbose):
    """Step 5 of _score_one_pair: lines starting at ``if (``."""
    if (
        bool(fe_pair_usability_admission_enable)
        and _corr_y_cont is not None
        and pair_mi > 0.0
        and st.best_config is not None
        and st.var_pairs_perf
        and (final_transformed_vals is not None or _this_chunk_deferred)
    ):
        try:
            _tc_x0 = _extval_raw_col(raw_vars_pair[0])
            _tc_x1 = _extval_raw_col(raw_vars_pair[1])
            # PART A (cheap raw-operand pre-filter): the pair must be genuinely tail-concentrated - its best
            # RAW bivariate form clears ``min_corr`` AND beats the best single-operand form by the pairness
            # margin. FALSE for noise pairs (best-form |corr| ~0.02-0.2) and lone-dominant-operand pairs, so
            # the per-form materialise in PART B is skipped for the common case.
            if (
                _tc_x0 is not None and _tc_x1 is not None
                and np.asarray(_tc_x0).shape[0] == _corr_y_cont.shape[0]
                and np.asarray(_tc_x1).shape[0] == _corr_y_cont.shape[0]
                and pair_is_tail_concentrated_rankaware(
                    _corr_y_cont, _tc_x0, _tc_x1,
                    min_corr=fe_pair_usability_admission_min_corr,
                    pairness_margin=fe_pair_usability_admission_pairness_margin,
                )
            ):
                # PART B (rank-vs-|corr| DISAGREEMENT): materialise each candidate form's |corr(continuous y)|
                # and ask ``tail_concentration_form_override`` for the form to promote. It returns a form ONLY
                # when the rank-MI leader is NOT the |corr| leader AND their rank-MI gap EXCEEDS the
                # Miller-Madow tie band (a REAL rank lead the usability tie-break cannot already resolve). On
                # the 4 passing F2 profiles + canonical fixtures the ratio is BOTH the rank-MI and |corr|
                # leader (they AGREE), so this returns None -> NO promotion, NO admission change -> byte-
                # identical. Only the true tail-concentration case (spurious rank leader, true |corr| leader)
                # flips it.
                _use_map: dict[Any, Any] = {}
                _score_one_pair_flips(st.var_pairs_perf, _resolve_col, _safe_abs_corr, _use_map)
                # Memoised by raw var key: ``_tc_x0``/``_tc_x1`` (fetched via ``_extval_raw_col`` above)
                # are the SAME raw operands the noise-wrap veto below scores, and either can recur
                # across every admitted pair sharing that var - see ``_raw_operand_abs_corr``.
                _best_single_corr = max(
                    _raw_operand_abs_corr(raw_vars_pair[0]), _raw_operand_abs_corr(raw_vars_pair[1]),
                )
                # ``_mi_band`` is the caller-hoisted, call-invariant tie band (see the per-call hoist
                # comment in ``check_prospective_fe_pairs``) - not recomputed per pair.
                _override_cfg = tail_concentration_form_override(
                    st.var_pairs_perf, _use_map,
                    min_corr=fe_pair_usability_admission_min_corr,
                    pairness_margin=fe_pair_usability_admission_pairness_margin,
                    mi_band=_mi_band,
                    best_single_corr=_best_single_corr,
                )
                if _override_cfg is not None:
                    st.best_config = _override_cfg
                    st.best_mi = float(st.var_pairs_perf.get(_override_cfg, st.best_mi))
                    st._usability_primary = True
                    if not (st._passes_joint_gate or st._prewarp_accept or st._marginal_uplift_accept):
                        st._usability_accept = True
                    if verbose:
                        st.messages.append(
                            f"tail-concentration usability admission: pair {raw_vars_pair} is tail-concentrated "
                            f"(rank-MI collapsed in the bulk, rank leader disagrees with the |corr(y)| leader "
                            f"beyond the tie band); promoted the |corr|-best engineered form "
                            f"(|corr|={_use_map.get(_override_cfg, 0.0):.3f}) as the winner over the spurious "
                            f"rank-MI leader."
                        )
        except Exception as e:
            logger.debug("usability-acceptance computation failed, defaulting to reject: %s", e)
            st._usability_accept = False
            st._usability_primary = False


def _zerofill_corr_dev(win_dev, _unused_win_host, operand_host) -> float:
    """``_abs_corr_zerofill_njit``-shaped callable that scores the DEVICE winner column against a host operand (the host winner slot is unused)."""
    from ._pairs_abs_corr_gpu import abs_corr_zerofill_gpu

    return abs_corr_zerofill_gpu(win_dev, operand_host)


def _score_one_pair_step6_wide_fraction_margin(st, _corr_y_cont, final_transformed_vals, _this_chunk_deferred, _resolve_col, _safe_abs_corr, vars_transformations, _transformed_operand_abs_corr, _raw_operand_abs_corr, raw_vars_pair, _NOISE_WRAP_MIN_OPERAND_CORR, _NOISE_WRAP_CORR_COLLAPSE_FRAC, verbose, transformed_vars):
    """Step 6 of _score_one_pair: lines starting at ``if (st._passes_joint_gate or st._prewarp_accept or st._marginal_uplift``."""
    if (st._passes_joint_gate or st._prewarp_accept or st._marginal_uplift_accept or st._usability_accept) and _corr_y_cont is not None and st.best_config is not None:
        try:
            # DEFERRED-float GPU path: score the winner against the target and its operands ON the device (``_win_dev``); the host column is only read
            # when the device path is unavailable.
            _win_dev = getattr(_resolve_col, "device", lambda *_: None)(st.best_config[2]) if (final_transformed_vals is None and _this_chunk_deferred) else None
            _win_vals = None
            _win_corr = None
            if _win_dev is not None:
                _tgt = getattr(_safe_abs_corr, "target", None)
                if _tgt is not None and _tgt[0] is not None:
                    _win_corr = abs_corr_or_none(_win_dev, _tgt[0], _tgt[1])
            if _win_corr is None and (final_transformed_vals is not None or _this_chunk_deferred):
                _win_vals = _resolve_col(st.best_config[2])
                _win_corr = _safe_abs_corr(_win_vals)
            # Compare against the strongest CLEAN per-operand column the winner actually used: each operand
            # under its CHOSEN unary (``sqr(a)`` for the ``a`` side, not raw ``a`` - raw ``a`` is ~0 corr
            # for an even target like ``exp(-a**2)``), falling back to the raw operand value. This is the
            # genuine single-source signal the wrap is diluting.
            # Memoised by operand key (``_transformed_operand_abs_corr``/``_raw_operand_abs_corr``): the
            # SAME operand recurs across every admitted pair it participates in, so its |corr| is
            # computed once per distinct operand for the whole call rather than once per pair.
            _op_corr = 0.0
            _tp = st.best_config[0]
            _op_corr = _score_one_pair_computed_once_per_distinct(_tp, vars_transformations, _op_corr, _transformed_operand_abs_corr, _raw_operand_abs_corr, raw_vars_pair)
            if _win_corr is not None and _op_corr >= _NOISE_WRAP_MIN_OPERAND_CORR and _win_corr < _op_corr * _NOISE_WRAP_CORR_COLLAPSE_FRAC:
                st._passes_joint_gate = st._prewarp_accept = st._marginal_uplift_accept = False
                st._usability_accept = st._usability_primary = False
                if verbose:
                    st.messages.append(
                        f"noise-wrap corr-collapse veto: winning composite |corr| with target "
                        f"{_win_corr:.3f} collapsed below {_NOISE_WRAP_CORR_COLLAPSE_FRAC:.2f}x the "
                        f"best operand |corr| {_op_corr:.3f}; the pair wraps a clean strong operand with "
                        f"a near-noise operand (binned-MI inflated by an extreme transform) -- rejecting "
                        f"so it cannot displace the clean operand."
                    )

            # DEGENERATE-PAIR (single-operand re-wrap) VETO. The noise-wrap veto above catches a
            # composite whose |corr| with the TARGET collapsed; this catches the dual failure where the
            # composite numerically EQUALS a single one of its own (warped) operands - the other operand's
            # transform collapsed to ~constant so the binary op carries no genuine joint information. Such a
            # "pair" keeps |corr|~=1.0 with that one operand and full target tracking, so the noise-wrap veto
            # never fires, yet it DISPLACES the clean single-source univariate basis (``a__L2``) it re-wraps
            # (``mul(prewarp(b),prewarp(a__L2))`` ~= ``prewarp(a__L2)`` ~= a**2). Compare the winning column
            # to EACH operand's chosen-unary continuous values: if it is ~= ONE operand (|corr| >= 0.999) the
            # pair adds nothing a single warped operand does not - veto so the clean single-source form wins.
            # A genuine 2-var pair (a/b, a*b) sits FAR below 0.999 with EITHER operand and is untouched.
            if (st._passes_joint_gate or st._prewarp_accept or st._marginal_uplift_accept or st._usability_accept) and (_win_vals is not None or _win_dev is not None):
                from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core import _abs_corr_zerofill_njit
                _tp2 = st.best_config[0]
                _max_single_op_corr = 0.0
                if _win_dev is not None:
                    _corr_fn = partial(_zerofill_corr_dev, _win_dev)
                    _max_single_op_corr = _score_one_pair_side(_tp2, vars_transformations, transformed_vars, _corr_fn, None, _max_single_op_corr)
                else:
                    _win_vals_f64 = np.asarray(_win_vals, dtype=np.float64)
                    _max_single_op_corr = _score_one_pair_side(_tp2, vars_transformations, transformed_vars, _abs_corr_zerofill_njit, _win_vals_f64, _max_single_op_corr)
                if _max_single_op_corr >= _DEGENERATE_PAIR_SINGLE_OPERAND_CORR:
                    st._passes_joint_gate = st._prewarp_accept = st._marginal_uplift_accept = False
                    st._usability_accept = st._usability_primary = False
                    if verbose:
                        st.messages.append(
                            f"degenerate-pair veto: winning composite is numerically ~= a SINGLE warped "
                            f"operand (|corr|={_max_single_op_corr:.4f} >= {_DEGENERATE_PAIR_SINGLE_OPERAND_CORR}); "
                            f"the other operand's transform collapsed to ~constant so the pair adds no genuine "
                            f"joint information -- rejecting so it cannot displace the clean single-source form."
                        )
        except Exception:
            # Load-bearing selection logic (the noise-wrap corr-collapse veto): log rather than swallow
            # silently, so a failure that lets a noise-wrapped pair through is visible at debug level.
            logger.debug("noise-wrap corr-collapse veto failed; pair not vetoed", exc_info=True)
