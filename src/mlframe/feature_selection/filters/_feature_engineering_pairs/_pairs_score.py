"""Per-pair candidate-scoring + acceptance/external-validation body of
``check_prospective_fe_pairs`` (carved 2026-06-22, Tier E).

This module holds the verbatim per-pair loop body lifted out of
``_pairs_core.check_prospective_fe_pairs`` so the parent orchestration function
drops back under the 1k-LOC monolith ceiling. It is a STRAIGHT carve -- the
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

import logging
from timeit import default_timer as timer

import numpy as np

from ._pairs_chunks import _compute_one_fe_chunk
from ._pairs_common import _TIMES_SPENT_LOCK
from ._pairs_dispatch import _dispatch_batch_mi_with_noise_gate
from ._pairs_gates import (
    _FE_MARGINAL_UPLIFT_MIN_JOINT_RATIO,
    _FE_MARGINAL_UPLIFT_MIN_RATIO,
    _FE_MARGINAL_UPLIFT_STRICT_JOINT_RATIO,
    _FE_MARGINAL_UPLIFT_SYNERGY_UPLIFT,
)
from ._pairs_materialise import (
    _fe_use_parallel_kernels,
    _materialise_extval_njit,
    _narrow_code_dtype,
    _njit_binary_op_codes,
)
from ._pairs_emit import _emit_pair_features

logger = logging.getLogger(__name__)

# DEGENERATE-PAIR |corr| threshold (2026-06-27). A prospective pair is DEGENERATE when its winning
# composite is numerically ~= a SINGLE one of its own (warped) operands -- the other operand's transform
# collapsed to ~constant so the binary op is just a re-wrap of one source and adds NO genuine joint
# information (e.g. ``mul(prewarp(b),prewarp(a__L2))`` where ``prewarp(b)`` is ~constant -> the column is
# essentially ``prewarp(a__L2)`` ~= a**2). Such a pair keeps |corr|~=1.0 with that one operand, clears the
# joint-prevalence gate, and DISPLACES the clean single-source univariate basis (a__L2). A GENUINE 2-var
# pair (a/b, a*b) has |corr| well below this with EITHER single operand, so the bar leaves it untouched.
# 0.999 is intentionally near-1.0: it fires only on a true single-operand re-wrap, never on a real pair.
_DEGENERATE_PAIR_SINGLE_OPERAND_CORR: float = 0.999


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
    _pair_res_entry = None
    messages = []

    combs = pair_combs[raw_vars_pair]

    best_config, best_mi = None, -1
    this_pair_features = set()
    var_pairs_perf = {}
    # Pre-warp uplift tracking (2026-06-02): the best engineered MI achievable
    # with ONLY the elementary library unaries (no ``prewarp`` operand) vs the
    # best USING a prewarp operand. A 1-D engineered summary of a 2-D pair
    # cannot retain ``fe_min_engineered_mi_prevalence`` of the 2-D JOINT MI,
    # so on a non-monotone inner distortion (where the elementary library is
    # representationally blind) the prewarp winner is rejected by the joint
    # prevalence gate despite being a large, real uplift over the best the
    # library can do. The alternative acceptance path below admits a prewarp
    # winner when it beats the best non-prewarp engineered MI by a margin --
    # directed (only fires where the prewarp adds representational power) and
    # noise-safe (on linear/monotone/noise data the prewarp does not beat the
    # elementary library, so the margin is never cleared).
    best_nonprewarp_mi = -1.0
    best_nonprewarp_config = None
    best_prewarp_config, best_prewarp_mi = None, -1.0

    # CRITICAL #2 dispatch: hoist path uses the shared buffer (writes into
    # ``[:, i]``); recompute-fallback path uses a tiny 1D scratch + a
    # config-by-i map for on-demand survivor recomputation later.
    # CROSS-PAIR: when this pair was batched across the chunk, its survivor
    # columns live in the wide ``_chunk_buffer`` (the config's ``i`` is the
    # chunk-buffer column), so point ``final_transformed_vals`` at it. The chunk
    # is materialised LAZILY: pairs are processed in chunk-plan order, so when we
    # reach the FIRST pair of a not-yet-loaded chunk we fill the buffer + MI cache
    # for that whole chunk in ONE batched pass. By the time the next chunk's first
    # pair arrives, all of this chunk's pairs (incl. their survivor packing, which
    # reads the buffer) have already been processed -> safe to overwrite.
    _chunk_entry = None
    if _chunk_global_batch and (_chunk_buffer is not None):
        _my_chunk = _pair_to_chunk.get(raw_vars_pair)
        if _my_chunk is not None:
            if _my_chunk != chunk_state["loaded_idx"]:
                # CHUNK PIPELINE (2026-07-02, max-GPU phase): under the strict-resident gate the driver put a
                # single-worker executor + a SECOND chunk buffer in ``chunk_state`` -- chunk c+1's whole
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
                    )

                _mi_cache = None
                _fut = (chunk_state.get("pipeline_futures") or {}).pop(_my_chunk, None)
                if _fut is not None:
                    try:
                        _mi_cache = _fut.result()
                    except Exception:
                        logger.debug("pipelined chunk %d producer failed; inline recompute", _my_chunk, exc_info=True)
                        _mi_cache = None
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
                if _ex is not None and _bufs is not None and (_my_chunk + 1) < len(_fe_chunks):
                    try:
                        chunk_state.setdefault("pipeline_futures", {})[_my_chunk + 1] = _ex.submit(
                            _produce, _my_chunk + 1, _bufs[(_my_chunk + 1) % 2])
                    except Exception:
                        logger.debug("chunk prefetch submit failed; falling back to lazy compute", exc_info=True)
            _chunk_entry = chunk_state["mi_cache"].get(raw_vars_pair)
    # When the chunk DEFERRED its float D2H the buffer is unfilled -- point the reads at None so they
    # take the GPU re-materialise branch in ``_resolve_col`` (bit-identical to the buffer value).
    _this_chunk_deferred = (_chunk_entry is not None) and chunk_state["float_deferred"]
    if _chunk_entry is not None:
        final_transformed_vals = None if _this_chunk_deferred else chunk_state.get("active_buffer", _chunk_buffer)
    else:
        final_transformed_vals = final_transformed_vals_shared
    _col_buf_1d: np.ndarray | None = (
        np.empty(len(X), dtype=np.float32) if _need_recompute_map else None
    )
    _config_by_i: dict[int, tuple] = {} if _need_recompute_map else None

    def _resolve_col(_buf_col):
        """Continuous candidate column ``_buf_col`` for the intermediate (subsample) scoring reads.
        Reads the filled chunk buffer when present; on the DEFERRED-float GPU path the buffer is
        unfilled, so RE-MATERIALISE the column on the GPU via ``_fe_materialise_block_gpu`` -- the SAME
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
        from .._gpu_resident_fe import _fe_materialise_block_gpu, _resident_operand_table
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

    i = 0
    # Per-pair thread-local timing accumulator; merged into the shared
    # ``times_spent`` under the lock once per pair (see end of pair loop).
    _local_times: dict = {}

    # BATCHED-DISCRETIZE dispatch (2026-06-04): per-candidate ``discretize_array``
    # (np.linspace + np.nanpercentile->partition + searchsorted) is the FE-pair-search
    # hotspot -- millions of tiny per-column numpy calls -> serial-dispatch-bound,
    # idle CPU. On the HOIST path (shared buffer present) with the quantile method we
    # split this pair's sweep into 3 phases: (1) materialise ALL candidate columns into
    # the buffer + nan_to_num + record (config, idx, uses_pw); (2) batch-discretise the
    # filled buffer slice in ONE ``np.nanpercentile(axis=0)`` (amortises dispatch over K
    # columns -- bit-identical to per-column, see ``discretize_2d_quantile_batch``);
    # (3) replay the EXACT per-candidate mi_direct + best/prewarp/config tracking.
    # MI stays per-candidate (mi_direct's permutation confidence is NOT batched). The
    # recompute-fallback (no buffer) and the uniform method keep the original
    # per-candidate path verbatim -- only the hoist+quantile case is batched.
    # A deferred chunk has final_transformed_vals=None (buffer not filled) but still drives the
    # cross-pair replay path (its MI is precomputed, its columns re-materialise on demand via
    # _resolve_col) -- so treat a present _chunk_entry as batch-disc-eligible even when deferred.
    _use_batch_disc = (
        (final_transformed_vals is not None) or (_chunk_entry is not None and _this_chunk_deferred)
    ) and (quantization_method == "quantile")

    if _use_batch_disc:
        # CROSS-PAIR fast path: this pair was batched together with the rest of its
        # chunk in ``_compute_one_fe_chunk``. Its candidate columns already live
        # in ``_chunk_buffer`` (the buf_col index is the config ``i``), its MI is
        # already computed, and its per-bin_func materialise timings are recorded.
        # We only replay the EXACT per-candidate tracking below. Bit-identical: the
        # chunk's ONE discretize_2d + ONE batch_mi score each column independently,
        # and the candidate order per pair is the SAME (combs x bin_funcs) order the
        # per-pair Phase 1 produced.
        if _chunk_entry is not None:
            _batch_candidates, _fe_mi_by_col, _pair_local_times = _chunk_entry
            for _bf_name, _dt in _pair_local_times.items():
                _local_times[_bf_name] = _local_times.get(_bf_name, 0.0) + _dt
            # ``_fe_mi_arr`` is indexed by the chunk-buffer column (buf_col), so the
            # replay's ``_fe_mi_arr[_ci]`` lookup is correct without re-indexing.
            _fe_mi_arr = _fe_mi_by_col
        else:
            # bench-attempt-rejected (2026-06-23, MRMR FE wall /loop iter10): the F2 100k cProfile
            # attributes ~40s tottime to THIS function and ~6.6s to ``_safe_div``, suggesting "Python
            # per-candidate orchestration overhead". A line_profiler pass (line-by-line, F2 100k warm)
            # disproves that: line 310 (``final_transformed_vals[:, i] = bin_func(param_a, param_b)``)
            # is 72.2% of the body's time, the Phase-1b ``nan_to_num`` 12.1%, GPU binning 7.4%, batch-MI
            # dispatch 4.6% -- ALL numeric kernels (njit ufuncs / compiled div / GPU), already routed by
            # prior iterations. The Python bookkeeping is negligible: ``_local_times`` dict 0.4%,
            # ``_batch_candidates.append`` 0.2%, the ``binary_transformations.items()`` loop 0.2%, the
            # per-pair-comb ``np.errstate`` 1.0%; the replay loop / ``var_pairs_perf`` dict / recipe-name
            # building / clean-form-demotion loop are each <0.2% (sum of all Python orchestration < 2.5%).
            # ``_safe_div`` (feature_engineering.py:738) is NOT pure-Python overhead: it IS njit-compiled
            # in-run (4 signatures compile during the fit, incl. the float32 'A' strided-column layout this
            # path feeds it) and its 6.6s is genuine compiled float32->float64 ratio compute over 9720
            # 100k-row columns; cProfile attributes the dispatcher's call to the py_func code-object
            # location, which LOOKS like a Python frame but is not. The float64 upcast inside ``_safe_div``
            # (then downcast on the float32 store here) is a real but UNSAFE lever: a native float32 divide
            # rounds differently from float64-divide-then-cast at ULP and this is selection-critical FE
            # scoring, so it is NOT changed. Verdict: ``_score_one_pair`` is already lean on Python overhead
            # -- the wall IS candidate-column materialisation COMPUTE. iter11 (2026-06-23) RESOLVES that wall:
            # the line-310 CPU ``bin_func`` materialise is now routed to the GPU FUSED materialise+bin
            # (the GPU-fused branch directly below), measured F2 100k 83.8s -> 30.0s (2.8x), selection
            # bit-identical -- so the "no win" tail of the iter10 verdict is SUPERSEDED for this path.
            # Phase 1: materialise + nan_to_num + record. ``i`` advances exactly as in the
            # per-candidate path so ``config``'s buffer index and ``_config_by_i`` are identical.
            _batch_candidates = []  # (transformations_pair, bin_func_name, i, uses_pw)
            _disc_2d = None  # set by the GPU FUSED materialise+bin path below; None -> CPU Phase 2 bins it

            # GPU FUSED PER-PAIR MATERIALISE+BIN (2026-06-23, MRMR FE wall /loop iter11). The CPU
            # ``bin_func`` strided materialise (the line ``final_transformed_vals[:, i] = bin_func(param_a,
            # param_b)`` below) is the per-pair FE-scan WALL: on the F2 100k fit it is 71.9% of
            # ``_score_one_pair`` (line_profiler 45.8s of 63.7s). The chunk path already routes its
            # materialise to the GPU FUSED ``gpu_materialise_discretize_codes_host`` (4.40x vs CPU
            # njit-mat+CPU-bin / 2.03x vs CPU njit-mat+GPU-bin, GTX 1050 Ti n=100k), but it only fires
            # when a chunk holds >1 pair (``_chunk_buffer`` allocated); at the canonical 100k fit the
            # RAM-budgeted chunk width is barely one pair wide (chunkmax~3561 vs pairwidth 1944), so
            # ``_chunk_buffer`` stays None and EVERY pair falls to THIS per-pair CPU path -- 100% of
            # candidate materialise (measured: F2 seed-7 = 58320/58320 cols on the CPU line below).
            # So we route the per-pair materialise through the SAME fused GPU kernel here. It builds the
            # (n,K) float candidate matrix on-device from (a_cols, b_cols, op_codes), fills the host buffer
            # (``out_cand`` -- the downstream survivor/usability/ext-val stages still read the continuous
            # columns) AND bins it RESIDENT, returning ``_disc_2d`` so the separate Phase-2 binning is
            # skipped. BIT-IDENTICAL selection: ``gpu_materialise_discretize_codes_host`` mirrors
            # ``_materialise_chunk_njit`` (maxdiff 0), which is itself bit-identical to the numpy
            # ``bin_func`` (the chunk-batch invariant), and the inline kernel nan-scrub == the per-column
            # ``np.nan_to_num`` below. PER-OP GATE: routed only when ``_njit_binary_op_codes`` covers EVERY
            # op in the registry (None -> any hypot/scipy.special op -> stay on the bit-safe CPU loop). Gated
            # by the dedicated GPU-binning crossover (if binning wins, the fused mat+bin certainly wins) +
            # the ``MLFRAME_FE_GPU_MATERIALISE`` escape hatch (same knob the chunk path uses). Any GPU
            # failure falls through to the CPU loop below (never a regression).
            _gpu_fused_done = False
            try:
                import os as _os
                _gpu_mat_on = _os.environ.get("MLFRAME_FE_GPU_MATERIALISE", "1").strip().lower() not in (
                    "0", "false", "no", "off",
                )
                # Probe K (candidate count for this pair) without materialising, to size the binning gate.
                if _gpu_mat_on:
                    from ._pairs_materialise import _njit_binary_op_codes as _njit_op_codes_fn
                    _op_code_arr = _njit_op_codes_fn(binary_transformations)
                else:
                    _op_code_arr = None
                if _op_code_arr is not None:
                    # Build candidate specs + per-candidate (a_col, b_col, op_code) in the SAME
                    # (combs x bin_func) order the CPU path produces -> identical buffer index ``i``.
                    _name_list = list(binary_transformations.keys())
                    _a_cols: list = []
                    _b_cols: list = []
                    _ops: list = []
                    _gpu_cands = []
                    for transformations_pair in combs:
                        if (transformations_pair[0] not in vars_transformations) or (
                            transformations_pair[1] not in vars_transformations
                        ):
                            continue
                        _ai = vars_transformations[transformations_pair[0]]
                        _bi = vars_transformations[transformations_pair[1]]
                        _uses_pw = (
                            transformations_pair[0][1] == _PREWARP_UNARY
                            or transformations_pair[1][1] == _PREWARP_UNARY
                        )
                        for _opn, bin_func_name in enumerate(_name_list):
                            _a_cols.append(_ai)
                            _b_cols.append(_bi)
                            _ops.append(int(_op_code_arr[_opn]))
                            _gpu_cands.append((transformations_pair, bin_func_name, i, _uses_pw))
                            i += 1
                    _K = i
                    from ._pairs_core import _fe_gpu_binning_enabled
                    if _K > 0 and _fe_gpu_binning_enabled(final_transformed_vals.shape[0], _K):
                        _code_dtype = _narrow_code_dtype(quantization_nbins, quantization_dtype)
                        _start = timer()
                        from .._gpu_resident_fe import gpu_materialise_discretize_codes_host
                        _disc_2d = gpu_materialise_discretize_codes_host(
                            transformed_vars,
                            np.asarray(_a_cols, dtype=np.int64),
                            np.asarray(_b_cols, dtype=np.int64),
                            np.asarray(_ops, dtype=np.int8),
                            int(quantization_nbins),
                            dtype=_code_dtype,
                            out_cand=final_transformed_vals[:, :_K],
                        )
                        _batch_candidates = _gpu_cands
                        # Attribute the fused materialise time across the bin_funcs (one event per pair).
                        _dt_each = (timer() - _start) / max(1, len(_name_list))
                        for bin_func_name in _name_list:
                            _local_times[bin_func_name] = _local_times.get(bin_func_name, 0.0) + _dt_each
                        _gpu_fused_done = True
            except Exception:
                logger.debug("FE per-pair GPU fused materialise+bin failed; CPU materialise", exc_info=True)
                _gpu_fused_done = False
                _disc_2d = None
                _batch_candidates = []
                i = 0

            if not _gpu_fused_done:
                # The GPU-prep candidate loop above advances ``i`` to size the fused launch; if the GPU-binning gate declines (no device / below crossover) we reach here with NO exception,
                # so the ``except``-path ``i = 0`` reset never ran -- restart the column cursor before the CPU materialise, else it overruns ``final_transformed_vals`` (``_batch_candidates``/``_disc_2d`` keep their pre-try init).
                i = 0
                for transformations_pair in combs:
                    if (transformations_pair[0] not in vars_transformations) or (transformations_pair[1] not in vars_transformations):
                        continue
                    param_a = transformed_vars[:, vars_transformations[transformations_pair[0]]]
                    param_b = transformed_vars[:, vars_transformations[transformations_pair[1]]]
                    _uses_pw = (
                        transformations_pair[0][1] == _PREWARP_UNARY
                        or transformations_pair[1][1] == _PREWARP_UNARY
                    )
                    # Same wide errstate scope as the original per-pair-comb path.
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        for bin_func_name, bin_func in binary_transformations.items():
                            start = timer()
                            try:
                                final_transformed_vals[:, i] = bin_func(param_a, param_b)
                            except Exception:
                                # The transform raised AFTER (or instead of) writing column ``i``; the buffer slot may
                                # still hold a prior column's data. Overwrite with NaN so the failed column is never
                                # scored against stale/garbage values, and skip recording it as a candidate.
                                logger.exception("Error when performing %s", bin_func)
                                final_transformed_vals[:, i] = np.nan
                            else:
                                # DEFER the NaN/inf scrub to ONE vectorised pass over the packed
                                # buffer slice [:, :i] below (was a per-column ``nan_to_num`` here:
                                # K tiny 50k-element isposinf/isneginf calls per pair -> profiled at
                                # 16.5s / 5834 calls on the 5-feat x 50000-row repro, pure serial
                                # numpy dispatch with the cores idle). ``nan_to_num`` is elementwise
                                # so scrubbing the whole [:, :i] block at once is byte-identical to
                                # scrubbing each column as it is written, and runs one C loop over a
                                # contiguous (n x K) buffer instead of K strided ones.
                                _local_times[bin_func_name] = _local_times.get(bin_func_name, 0.0) + (timer() - start)
                                _batch_candidates.append((transformations_pair, bin_func_name, i, _uses_pw))
                                i += 1

                # Phase 1b: ONE vectorised NaN/inf scrub over every materialised column
                # [:, :i] (replaces the per-column ``nan_to_num`` removed above). Elementwise
                # -> byte-identical to the per-column scrub; one contiguous-block C pass.
                # SKIPPED on the GPU fused path (the kernel scrubs NaN/inf inline -- bit-identical).
                if i > 0:
                    np.nan_to_num(final_transformed_vals[:, :i], copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            # Phase 2: ONE batch discretisation over the materialised columns [:, :n].
            # Bit-identical to per-column ``discretize_array(method='quantile')`` -- the
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
                if _fe_gpu_discretize_enabled(final_transformed_vals.shape[0], i):
                    try:
                        from .._gpu_resident_fe import gpu_pairs_fe_mi
                        _fe_mi_arr = gpu_pairs_fe_mi(
                            final_transformed_vals[:, :i], int(quantization_nbins),
                            classes_y, classes_y_safe, freqs_y,
                            fe_npermutations, fe_min_nonzero_confidence, use_su_normalization(),
                        )
                    except Exception:
                        logger.debug("FE GPU pair-MI failed; falling back to CPU", exc_info=True)
                        _fe_mi_arr = None
                if _fe_mi_arr is None:
                    # ``_disc_2d`` may ALREADY be the codes from the GPU FUSED materialise+bin path
                    # (Phase 1 above) -- in that case skip re-binning (the fused kernel produced the
                    # SAME codes ``gpu_discretize_codes_host`` would). Only bin here when it is None.
                    # GPU BINNING (2026-06-23): the per-pair Phase-2 binning gets the SAME dedicated
                    # binning crossover the chunk path now uses. ``gpu_discretize_codes_host`` is
                    # bit-identical to the CPU njit binning (verified maxdiff 0) and 17-24x faster at
                    # n=100k; it was previously reachable here ONLY through the full ``gpu_pairs_fe_mi``
                    # path (declined on the non-analytic branch -> CPU njit). Any GPU failure falls back
                    # to the CPU discretise below (never a regression; selection bit-identical).
                    if _disc_2d is None:
                        try:
                            from ._pairs_core import _fe_gpu_binning_enabled
                            if _fe_gpu_binning_enabled(final_transformed_vals.shape[0], i):
                                from .._gpu_resident_fe import gpu_discretize_codes_host
                                # defer_host_fill: the codes flow straight into _dispatch_batch_mi_with_noise_gate,
                                # whose resident-CUDA gate consumes the DEVICE codes in place (take_resident_codes)
                                # and only triggers the lazy host fill (ensure_host_codes_filled) on a host-reading
                                # branch. Skips the (n, K) codes D2H -- the fit's single largest -- whenever the
                                # resident gate is the consumer. Bit-identical (host buffer, if read, == device.get()).
                                _disc_2d = gpu_discretize_codes_host(
                                    final_transformed_vals[:, :i], int(quantization_nbins), dtype=_code_dtype,
                                    defer_host_fill=True,
                                )
                        except Exception:
                            logger.debug("FE per-pair GPU binning failed; CPU discretise", exc_info=True)
                            _disc_2d = None
                    if _disc_2d is None:
                        _disc_2d = discretize_2d_quantile_batch(
                            final_transformed_vals[:, :i], n_bins=quantization_nbins,
                            dtype=_code_dtype,
                            # OPT-A extension (2026-06-07): same main-thread parallel searchsorted
                            # gate as the chunk + marginal-uplift discretise -- byte-identical
                            # column-prange twin when serial_main_thread (no joblib nest).
                            parallel=_fe_use_parallel_kernels(i, serial_main_thread),
                            # The ``np.nan_to_num(..., copy=False)`` directly above scrubbed this exact
                            # buffer slice, so the per-call ``np.isnan().any()`` scan inside the discretiser
                            # is guaranteed-False wasted work; skip it (bit-identical on a NaN-free buffer).
                            assume_finite=True,
                        )

                # Phase 3: BATCHED MI + permutation noise-gate across ALL K candidate
                # columns in ONE kernel call. Bit-identical to the per-candidate
                # ``mi_direct`` loop on the default FE path (parallelism='outer',
                # n_workers=1 -> parallel_mi_prange, base_seed=0): every candidate is
                # tested against the SAME npermutations shuffles of y (the shuffle is
                # seeded by (base_seed, perm_index) ONLY, never by classes_x), so a single
                # batched kernel can shuffle y once per permutation and score all columns
                # against it -- amortising both the MI compute and the shuffle across K.
                # ``_dispatch_batch_mi_with_noise_gate`` routes CPU-njit vs a GPU batched
                # path by n*K via the kernel_tuning_cache (no hardcoded threshold).
                # Skipped when the GPU pair-MI path above already produced ``_fe_mi_arr``.
                if _fe_mi_arr is None:
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
                    )

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
                var_pairs_perf[config] = fe_mi
                if _need_recompute_map:
                    _config_by_i[_ci] = (transformations_pair[0], transformations_pair[1], bin_func_name)

                if fe_mi > best_mi:
                    best_mi = fe_mi
                    best_config = config
                if _uses_pw:
                    if fe_mi > best_prewarp_mi:
                        best_prewarp_mi = fe_mi
                        best_prewarp_config = config
                else:
                    if fe_mi > best_nonprewarp_mi:
                        best_nonprewarp_mi = fe_mi
                        best_nonprewarp_config = config
                if fe_mi > best_mi * 0.85:
                    if not fe_print_best_mis_only or (fe_mi == best_mi):
                        if verbose > 2:
                            print(f"MI of transformed pair {bin_func_name}({transformations_pair})={fe_mi:.4f}, MI of the plain pair {pair_mi:.4f}")
    else:
        for transformations_pair in combs:
            if (transformations_pair[0] not in vars_transformations) or (transformations_pair[1] not in vars_transformations):
                continue
            param_a = transformed_vars[:, vars_transformations[transformations_pair[0]]]
            param_b = transformed_vars[:, vars_transformations[transformations_pair[1]]]

            # A config "uses prewarp" iff either operand's unary name is the
            # pseudo-unary. Invariant across the bin_func loop -> compute once.
            _uses_pw = (
                transformations_pair[0][1] == _PREWARP_UNARY
                or transformations_pair[1][1] == _PREWARP_UNARY
            )

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
                            final_transformed_vals[:, i] = bin_func(param_a, param_b)
                            _col_view = final_transformed_vals[:, i]
                        else:
                            # Recompute fallback: write into the shared 1D scratch.
                            # bin_func returns a fresh ndarray; copy into the scratch
                            # so downstream nan_to_num + discretize see contiguous
                            # data. Avoids accumulating one alloc per inner iter.
                            _col_buf_1d[:] = bin_func(param_a, param_b)
                            _col_view = _col_buf_1d
                    except Exception:
                        # Failed transform: the buffer slot may still hold a prior column's data. Null it so it is
                        # never scored, and skip the scoring ``else`` (no candidate recorded for this bin_func).
                        logger.exception("Error when performing %s", bin_func)
                        if final_transformed_vals is not None:
                            final_transformed_vals[:, i] = np.nan
                    else:
                        np.nan_to_num(_col_view, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                        # Wave 27 P1: ``times_spent`` is shared across mrmr.py's
                        # parallel threading dispatch. Accumulate this pair's
                        # per-bin_func timings in a thread-LOCAL dict and merge them
                        # under ``_TIMES_SPENT_LOCK`` once per pair (below); the old
                        # per-inner-iteration lock was a serialization point on the
                        # hot path. Totals are identical.
                        _local_times[bin_func_name] = _local_times.get(bin_func_name, 0.0) + (timer() - start)

                        discretized_transformed_values = discretize_array(
                            arr=_col_view, n_bins=quantization_nbins, method=quantization_method, dtype=quantization_dtype
                        )
                        fe_mi, fe_conf = mi_direct(
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

                        config = (transformations_pair, bin_func_name, i)
                        var_pairs_perf[config] = fe_mi
                        if _need_recompute_map:
                            # Map i -> (a_key, b_key, bin_func_name) for downstream
                            # rebuild; bin_func is looked up via the original dict.
                            _config_by_i[i] = (transformations_pair[0], transformations_pair[1], bin_func_name)

                        if fe_mi > best_mi:
                            best_mi = fe_mi
                            best_config = config
                        # Track best-with-prewarp vs best-without so the alternative
                        # uplift gate below can decide whether the prewarp earned its
                        # place (``_uses_pw`` hoisted above the bin_func loop).
                        if _uses_pw:
                            if fe_mi > best_prewarp_mi:
                                best_prewarp_mi = fe_mi
                                best_prewarp_config = config
                        else:
                            if fe_mi > best_nonprewarp_mi:
                                best_nonprewarp_mi = fe_mi
                                best_nonprewarp_config = config
                        if fe_mi > best_mi * 0.85:
                            if not fe_print_best_mis_only or (fe_mi == best_mi):
                                if verbose > 2:
                                    print(f"MI of transformed pair {bin_func_name}({transformations_pair})={fe_mi:.4f}, MI of the plain pair {pair_mi:.4f}")
                        i += 1

    # Merge this pair's per-bin_func timings into the shared accumulator in
    # ONE locked pass (the increment was previously locked per inner
    # iteration -- a serialization point under the parallel pair dispatch).
    if _local_times:
        with _TIMES_SPENT_LOCK:
            for _bf, _dt in _local_times.items():
                times_spent[_bf] += _dt

    if verbose > 2:
        print(f"For pair {raw_vars_pair}, best config is {best_config} with best mi= {best_mi}")

    # CLEAN-FORM DEMOTION over the per-pair MI winner (2026-06-20). The ``prewarp``
    # pseudo-unary fits a learned 1-D orthogonal-poly warp per operand; on a target whose
    # inner function is already LIBRARY-expressible up to a MONOTONE distortion (e.g.
    # ``log(c)*sin(d)`` -- ``mul(log(c),sin(d))`` is the clean form, while the warp learns a
    # monotone re-expression ``mul(prewarp(c),sin(d))``) the prewarp form has IDENTICAL
    # ordering -> bit-equal binned MI, but MI is RANK-only so it cannot prefer the clean leg.
    # The warp then wins ``best_config`` by an MI tie/epsilon and propagates a DISTORTED form
    # (``log(div(sqr(a),neg(b)))`` for the a/b half, double-``prewarp(c)`` for the c/d half)
    # that the step-k>1 composite chains, displacing the clean additive compound AND dragging
    # in a redundant raw operand. Demote: when the global winner USES a prewarp operand but a
    # clean elementary-library (non-prewarp) form scores essentially the SAME target MI, keep
    # the prewarp winner ONLY if it has a real LINEAR-USABILITY uplift (|corr(continuous y)|)
    # over the best clean form. This preserves prewarp's INTENDED case -- a genuinely
    # non-monotone inner (``a**3-2a``) where the warp's reconstruction is MORE linearly usable
    # than any library form, so the uplift is real and the prewarp form is kept -- while making
    # the monotone-equivalent case (no |corr| uplift) fall back to the clean library compound.
    if (
        best_config is not None
        and best_nonprewarp_config is not None
        and best_config is not best_nonprewarp_config
        and best_nonprewarp_mi > 0.0
        and _corr_y_cont is not None
    ):
        _bc_uses_pw = (
            isinstance(best_config[0], (tuple, list))
            and len(best_config[0]) == 2
            and (best_config[0][0][1] == _PREWARP_UNARY or best_config[0][1][1] == _PREWARP_UNARY)
        )
        # Only act when the winner is a prewarp form AND the clean form is MI-equivalent
        # (within the same 0.85 leaders band already used as the equivalence notion). A
        # strictly-higher-MI prewarp winner is left untouched -- the prewarp/marginal gates
        # below decide it on its own merits; demotion is for the MI-tie monotone case only.
        if _bc_uses_pw and best_nonprewarp_mi >= best_mi * fe_good_to_best_feature_mi_threshold:

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
                except Exception:
                    return -1.0

            _pw_corr = _config_corr(best_config)
            _clean_corr = _config_corr(best_nonprewarp_config)
            # Demote to the clean form unless the prewarp form is MEANINGFULLY more linearly
            # usable. ``1.05`` = the prewarp must beat the clean |corr| by >= 5% to justify the
            # distorted re-expression; a genuinely non-monotone inner clears this comfortably
            # (its warp reconstruction is the ONLY linearly-aligned form), while a
            # monotone-equivalent warp scores <= the clean form and is demoted.
            if _clean_corr >= 0.0 and _pw_corr < _clean_corr * 1.05:
                best_config, best_mi = best_nonprewarp_config, best_nonprewarp_mi
                # The single-best emission path (below) does NOT read ``best_config``
                # directly -- it rebuilds the leaders band from ``var_pairs_perf`` and
                # re-picks via ``_select_single_best`` whose PRIMARY key is exact target
                # MI, so a prewarp form that beats the clean form by an MI EPSILON would be
                # re-selected and the usability tie-break (gated on EQUAL MI) would never
                # engage. So CAP every prewarp-using config's recorded MI at just-below the
                # best clean-form MI: it stays in the leaders band (so its column can still
                # be emitted if the model wants a tree-friendly twin via multi-emit) but can
                # no longer OUT-RANK the clean library form on the primary key, and the
                # already-wired ``_leader_usability`` tie-break then picks the clean leg. A
                # prewarp form with genuine |corr| uplift (the non-monotone intended case)
                # never reaches here (the ``_pw_corr < _clean_corr*1.05`` guard above fails),
                # so its MI rank is untouched and it keeps winning.
                _pw_cap = best_nonprewarp_mi * 0.999
                for _cfg in list(var_pairs_perf.keys()):
                    _ctp = _cfg[0]
                    if (
                        isinstance(_ctp, (tuple, list)) and len(_ctp) == 2
                        and (_ctp[0][1] == _PREWARP_UNARY or _ctp[1][1] == _PREWARP_UNARY)
                        and var_pairs_perf[_cfg] > _pw_cap
                    ):
                        var_pairs_perf[_cfg] = _pw_cap
                if verbose:
                    messages.append(
                        f"clean-form demotion: prewarp winner |corr(y)|={_pw_corr:.3f} did not beat "
                        f"the best clean library form |corr(y)|={_clean_corr:.3f} by >= 5% (MI-equivalent "
                        f"monotone re-expression); demoting to the clean form and capping prewarp-form "
                        f"MI at the clean form's so it cannot out-rank it on the primary key."
                    )

    # experiment-rejected (2026-06-03): a held-out-CV firewall here (score
    # per-combo MI on a TRAIN stride slice for honest selection, then keep the
    # winner only if its held-out VAL-slice MI retains >= ratio of train MI) was
    # implemented and benched END-TO-END on Layer-49 -- NO gain. In an isolated
    # probe it separated cleanly (genuine synergy val/train 0.90-1.04 vs noise-FE
    # 0.12-0.36), BUT in the real pipeline the tighter prevalence-gate defaults
    # (fe_synergy_min_prevalence 1.5 / fe_min_engineered_mi_prevalence 0.97)
    # already remove the pure noise*noise products, and the RESIDUAL "noise" FE
    # are signal*noise combos (e.g. max(log(L4_s2),noise_3) -- L4_s2 is a real
    # sensor) that genuinely generalise (val/train > 0.5) and SHOULD be kept; the
    # firewall's train-based selection (half the rows) then merely added selection
    # noise (+1 support). Prevalence gating subsumes the win -> not shipped.
    # Standard acceptance: the best engineered MI clears the configured
    # fraction of the 2-D pair-joint MI.
    #
    # MILLER-MADOW DEBIAS (2026-06-09, backlog #1 + #4). The RAW ratio
    # ``best_mi / pair_mi`` compares a 1-D engineered MI (over ~``quantization_nbins``
    # bins) against a 2-D joint MI (over ~``nbins^2`` bins). Both are plug-in MIs whose
    # positive bias is ``(k_x-1)(k_y-1)/2n``; the JOINT denominator's term is ~``nbins``x
    # larger, so the raw ratio is structurally depressed below 1.0 even when the 1-D
    # feature captures all the joint information (worst at small/moderate n) -- this is
    # exactly the documented reason the marginal-uplift fallback gate had to be added.
    # When ``fe_mm_debias_prevalence`` we subtract the MM MI-bias term from BOTH sides,
    # using the OCCUPIED bin counts (#4: nominal ``nbins`` over-corrects heavy-tailed
    # columns that collapse), with a denominator-positivity guard that defers to the raw
    # ratio when the joint bias term swamps the finite-sample joint MI. ``->`` raw ratio
    # as ``n -> inf`` (bias terms vanish) => large-n selection byte-untouched. The order-2
    # maxT floor (the outer guard) is MM-debiased CONSISTENTLY upstream (the IRON RULE),
    # so admitting more pairs here does NOT weaken the best-of-pool noise floor.
    #
    # bench-attempt-rejected (2026-06-09, FS backlog #5 "permutation-null-calibrated
    # prevalence bar"). The idea: REPLACE the hardcoded ``fe_min_engineered_mi_prevalence``
    # (0.90) with a SELF-CALIBRATING per-pool null ratio -- in the SAME K y-shuffles the
    # order-2 maxT floor runs, ALSO mirror the max-over-transforms search (discretise the
    # elementary binary bank mul/add/sub/div/max/min over the CONTINUOUS operands ONCE --
    # permutation-invariant -- then per shuffle take max 1-D engineered MI / joint pair MI),
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
    # BUT bench-REJECTED on the case that matters -- the user's WEAK F2
    # (``0.2*a**2/b + log(c*2)*sin(d/3)``, the SAME target that rejected #1/#8/#19): the null
    # ceiling is ~0.167 (calibrated to clean-noise pairs, ratio ~0.13-0.17), but EVERY weak-F2
    # pair sits FAR above it (5 seeds, n=20000): genuine_ab ~0.81, genuine_cd ~0.73, AND all four
    # cross-mix pairs 0.56-0.72 (cross(b,d) ~0.717 >= genuine_cd). So the null bar ADMITS every
    # cross-mix on every seed -- the IRON-RULE failure mode, identical to #1 (cross-mix 3/10 ->
    # 9/10). ROOT CAUSE is the documented fundamental detectability limit (see
    # ``test_mrmr_weak_f2_seed_stability.py`` "THREE DIRECT LEVERS EXHAUSTED"): the cross-mix
    # smuggles the dominant MONOTONE predictor ``c`` across the pair boundary, so its 1-D
    # engineered summary recovers a large fraction of its (real, cross) joint -- a HIGH ratio
    # indistinguishable from genuine synergy by ANY MI threshold. The null bar measures the
    # noise floor, but the weak-F2 problem is NOT noise admission; it is a real-monotone-predictor
    # cross-mix whose ratio is nowhere near the noise floor. AND the existing marginal-uplift /
    # prewarp FALLBACK already recovers the genuine pairs end-to-end at n=500/2000/8000, so #5
    # adds ZERO incremental recovery while WEAKENING cross-mix rejection. #5 is structurally a
    # 4th MI-threshold lever and fails by construction like #1/#8/#19; do NOT re-attempt an
    # MI-threshold/ratio fix here. Numbers + verdict in D:/Temp/null_prev_results.md.
    _gate_ratio = (best_mi / pair_mi) if pair_mi > 0.0 else 0.0
    if fe_mm_debias_prevalence and pair_mi > 0.0 and best_config is not None:
        from ._pairs_gates import _occupied_k, mm_debiased_prevalence_ratio
        _n_rows = int(len(classes_y))
        _k_y = int(np.asarray(freqs_y).shape[0])
        # Engineered winner occupied-K: discretise its CONTINUOUS column (the buffer
        # column ``best_config[2]``) with the SAME quantiser the MI was scored under.
        _k_eng = quantization_nbins
        try:
            _win_codes = discretize_array(
                arr=np.nan_to_num(_resolve_col(best_config[2])),
                n_bins=quantization_nbins, method=quantization_method, dtype=quantization_dtype,
            )
            _k_eng = _occupied_k(_win_codes)
        except Exception:
            _k_eng = quantization_nbins
        # 2-D joint occupied-K of the raw operands (bit-identical discretise to the
        # pair_mi compute); fall back to nominal ``nbins^2`` if either operand is
        # missing an identity transform.
        _ca = _operand_discretized(raw_vars_pair[0])
        _cb = _operand_discretized(raw_vars_pair[1])
        if _ca is not None and _cb is not None:
            _nb_b = int(np.asarray(_cb).max()) + 1 if np.asarray(_cb).size else quantization_nbins
            _joint_codes = np.asarray(_ca, dtype=np.int64) * _nb_b + np.asarray(_cb, dtype=np.int64)
            _k_joint = _occupied_k(_joint_codes)
        else:
            _k_joint = quantization_nbins * quantization_nbins
        _gate_ratio = mm_debiased_prevalence_ratio(
            best_mi, pair_mi, k_eng=_k_eng, k_joint=_k_joint, k_y=_k_y, n=_n_rows,
        )
    _passes_joint_gate = _gate_ratio > fe_min_engineered_mi_prevalence * (1.0 if num_fs_steps < 1 else 1.025)

    # Alternative pre-warp acceptance (2026-06-02): the joint-prevalence gate
    # structurally rejects a 1-D summary of a 2-D pair on a non-monotone inner
    # distortion. Admit the prewarp winner when it beats the best NON-prewarp
    # engineered MI by ``prewarp_uplift_threshold`` AND clears the pair-MI
    # noise floor (its MI must exceed the larger individual operand MI -- the
    # same notion the smart_polynom baseline uplift uses), so it cannot fire
    # on noise (where prewarp does not beat the library) or pure-linear data
    # (where the elementary library already saturates and the prewarp adds no
    # uplift). When it fires, the prewarp config becomes the winner.
    _prewarp_accept = False
    if (
        _prewarp_active
        and not _passes_joint_gate
        and best_prewarp_config is not None
        and best_nonprewarp_mi > 0.0
        and best_prewarp_mi >= best_nonprewarp_mi * float(prewarp_uplift_threshold)
    ):
        _prewarp_accept = True
        # Promote the prewarp winner to the pair's winner so the standard
        # leading-features / single-best materialisation path emits it.
        best_config, best_mi = best_prewarp_config, best_prewarp_mi
        if verbose:
            messages.append(
                f"pre-warp uplift gate: best prewarp MI={best_prewarp_mi:.4f} "
                f"beats best non-prewarp MI={best_nonprewarp_mi:.4f} by "
                f">= {float(prewarp_uplift_threshold):.2f}x (joint-prevalence "
                f"gate {best_mi / pair_mi:.3f} < {fe_min_engineered_mi_prevalence:.2f} "
                f"would have rejected it); admitting the prewarp feature."
            )

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
    # warp spec to round-trip into the recipe -- which is only guaranteed on the prewarp path.
    _marginal_uplift_accept = False
    if (
        not _passes_joint_gate
        and not _prewarp_accept
        and best_nonprewarp_config is not None
        and best_nonprewarp_mi > 0.0
        and pair_mi > 0.0
    ):
        _max_operand_marginal = max(
            _operand_marginal_mi(raw_vars_pair[0]),
            _operand_marginal_mi(raw_vars_pair[1]),
        )
        _joint_ratio = best_nonprewarp_mi / pair_mi
        _uplift_ratio = (best_nonprewarp_mi / _max_operand_marginal) if _max_operand_marginal > 0.0 else 0.0
        # HW-robust two-tier joint-recovery floor (see the constants above): a genuine
        # same-signal pair clears EITHER the strict joint floor on its own OR is a clear-synergy
        # pair (high uplift) that clears the relaxed base floor. A cross-signal artefact clears
        # neither, so a small cross-HW MI perturbation cannot flip it into the support.
        _joint_recovery_ok = (
            _joint_ratio >= _FE_MARGINAL_UPLIFT_STRICT_JOINT_RATIO
            or (
                _uplift_ratio >= _FE_MARGINAL_UPLIFT_SYNERGY_UPLIFT
                and _joint_ratio >= _FE_MARGINAL_UPLIFT_MIN_JOINT_RATIO
            )
        )
        if (
            _max_operand_marginal > 0.0
            and best_nonprewarp_mi >= _max_operand_marginal * _FE_MARGINAL_UPLIFT_MIN_RATIO
            and _joint_recovery_ok
        ):
            _marginal_uplift_accept = True
            # Promote the best non-prewarp form so the standard single-best
            # materialisation path emits a recipe-replayable winner.
            best_config, best_mi = best_nonprewarp_config, best_nonprewarp_mi
            if verbose:
                messages.append(
                    f"marginal-uplift gate: best non-prewarp engineered MI={best_nonprewarp_mi:.4f} "
                    f"beats the larger operand marginal MI={_max_operand_marginal:.4f} by "
                    f">= {_FE_MARGINAL_UPLIFT_MIN_RATIO:.2f}x and recovers {_joint_ratio:.3f} "
                    f"of the 2-D joint (joint-prevalence gate "
                    f"{fe_min_engineered_mi_prevalence:.2f} would have rejected it); "
                    f"admitting the genuine synergy pair."
                )

    # NOISE-WRAP CORR-COLLAPSE VETO (2026-06-15). Whatever path admitted the winner, VETO it when the
    # winning composite WRAPS a strong, clean operand with a (near-)noise operand: its |corr| with the
    # target collapses to a small fraction of the best single operand's |corr| while that operand is
    # genuinely strong on its own. This is the ``sub(log(e),invqubed(a__T2))`` failure -- an extreme
    # heavy-tailed transform inflates the binned ``best_mi/pair_mi`` so it clears the joint-prevalence
    # gate, yet the column carries ~0 linear/monotone signal (|corr|~0.02) versus the clean operand's
    # |corr|~0.99, so it would DISPLACE the clean univariate basis from the support and kill recovery.
    # Genuine synergy (a*b, log(c)*sin(d)) keeps the engineered column tracking y (no collapse), so the
    # wide 2x fraction margin never condemns it. Pure-noise pairs never reach here (upstream screens).
    if (
        (_passes_joint_gate or _prewarp_accept or _marginal_uplift_accept)
        and _corr_y_cont is not None
        and best_config is not None
    ):
        try:
            _win_vals = _resolve_col(best_config[2]) if (final_transformed_vals is not None or _this_chunk_deferred) else None
            _win_corr = _safe_abs_corr(_win_vals) if _win_vals is not None else None
            # Compare against the strongest CLEAN per-operand column the winner actually used: each operand
            # under its CHOSEN unary (``sqr(a)`` for the ``a`` side, not raw ``a`` -- raw ``a`` is ~0 corr
            # for an even target like ``exp(-a**2)``), falling back to the raw operand value. This is the
            # genuine single-source signal the wrap is diluting.
            _op_corr = 0.0
            _tp = best_config[0]
            for _side in (0, 1):
                _opk = _tp[_side] if isinstance(_tp, (tuple, list)) and len(_tp) > _side else None
                if _opk is not None and _opk in vars_transformations:
                    _op_corr = max(_op_corr, _safe_abs_corr(transformed_vars[:, vars_transformations[_opk]]))
                _op_corr = max(_op_corr, _safe_abs_corr(_extval_raw_col(raw_vars_pair[_side])))
            if (
                _win_corr is not None
                and _op_corr >= _NOISE_WRAP_MIN_OPERAND_CORR
                and _win_corr < _op_corr * _NOISE_WRAP_CORR_COLLAPSE_FRAC
            ):
                _passes_joint_gate = _prewarp_accept = _marginal_uplift_accept = False
                if verbose:
                    messages.append(
                        f"noise-wrap corr-collapse veto: winning composite |corr| with target "
                        f"{_win_corr:.3f} collapsed below {_NOISE_WRAP_CORR_COLLAPSE_FRAC:.2f}x the "
                        f"best operand |corr| {_op_corr:.3f}; the pair wraps a clean strong operand with "
                        f"a near-noise operand (binned-MI inflated by an extreme transform) -- rejecting "
                        f"so it cannot displace the clean operand."
                    )

            # DEGENERATE-PAIR (single-operand re-wrap) VETO (2026-06-27). The noise-wrap veto above catches a
            # composite whose |corr| with the TARGET collapsed; this catches the dual failure where the
            # composite numerically EQUALS a single one of its own (warped) operands -- the other operand's
            # transform collapsed to ~constant so the binary op carries no genuine joint information. Such a
            # "pair" keeps |corr|~=1.0 with that one operand and full target tracking, so the noise-wrap veto
            # never fires, yet it DISPLACES the clean single-source univariate basis (``a__L2``) it re-wraps
            # (``mul(prewarp(b),prewarp(a__L2))`` ~= ``prewarp(a__L2)`` ~= a**2). Compare the winning column
            # to EACH operand's chosen-unary continuous values: if it is ~= ONE operand (|corr| >= 0.999) the
            # pair adds nothing a single warped operand does not -- veto so the clean single-source form wins.
            # A genuine 2-var pair (a/b, a*b) sits FAR below 0.999 with EITHER operand and is untouched.
            if (
                (_passes_joint_gate or _prewarp_accept or _marginal_uplift_accept)
                and _win_vals is not None
            ):
                _tp2 = best_config[0]
                _max_single_op_corr = 0.0
                for _side in (0, 1):
                    _opk = _tp2[_side] if isinstance(_tp2, (tuple, list)) and len(_tp2) > _side else None
                    if _opk is not None and _opk in vars_transformations:
                        _ov = transformed_vars[:, vars_transformations[_opk]]
                        _r = np.corrcoef(
                            np.nan_to_num(np.asarray(_win_vals, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0),
                            np.nan_to_num(np.asarray(_ov, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0),
                        )[0, 1]
                        if np.isfinite(_r):
                            _max_single_op_corr = max(_max_single_op_corr, abs(float(_r)))
                if _max_single_op_corr >= _DEGENERATE_PAIR_SINGLE_OPERAND_CORR:
                    _passes_joint_gate = _prewarp_accept = _marginal_uplift_accept = False
                    if verbose:
                        messages.append(
                            f"degenerate-pair veto: winning composite is numerically ~= a SINGLE warped "
                            f"operand (|corr|={_max_single_op_corr:.4f} >= {_DEGENERATE_PAIR_SINGLE_OPERAND_CORR}); "
                            f"the other operand's transform collapsed to ~constant so the pair adds no genuine "
                            f"joint information -- rejecting so it cannot displace the clean single-source form."
                        )
        except Exception:
            # Load-bearing selection logic (the noise-wrap corr-collapse veto): log rather than swallow
            # silently, so a failure that lets a noise-wrapped pair through is visible at debug level.
            logger.debug("noise-wrap corr-collapse veto failed; pair not vetoed", exc_info=True)

    # REJECTION LEDGER (additive): record a pair the per-pair acceptance gate is about
    # to DROP -- the joint-prevalence floor declined AND both the prewarp and the
    # marginal-uplift (abs-MAD / joint-recovery) fallbacks declined. Attribute to whichever
    # floor it primarily missed: the engineered-MI prevalence floor (the 0.97 floor the
    # session hand-diagnoses) is the primary gate; if the ratio DID clear that bar (so the
    # prewarp/uplift path declined for another reason) tag the marginal-uplift floor.
    # All values were already computed above (no recompute).
    if not (_passes_joint_gate or _prewarp_accept or _marginal_uplift_accept):
        try:
            _rej_thr = float(fe_min_engineered_mi_prevalence) * (1.0 if num_fs_steps < 1 else 1.025)
            _rej_op = None
            if best_config is not None:
                try:
                    _rej_op = best_config[1]  # binary func name of the best engineered form
                except Exception:
                    _rej_op = None
            if not _passes_joint_gate:
                _rej_rec = {
                    "gate": "engineered_mi_prevalence",
                    "candidate": str(raw_vars_pair),
                    "operands": tuple(raw_vars_pair),
                    "operator": _rej_op,
                    "observed": float(_gate_ratio),
                    "threshold": _rej_thr,
                    "reason": "best_mi_over_pair_mi_below_floor",
                }
            else:
                _rej_rec = {
                    "gate": "marginal_uplift_floor",
                    "candidate": str(raw_vars_pair),
                    "operands": tuple(raw_vars_pair),
                    "operator": _rej_op,
                    "observed": float(_gate_ratio),
                    "threshold": _rej_thr,
                    "reason": "marginal_uplift_and_prewarp_declined",
                }
            rejection_records.append(_rej_rec)
            if rejection_ledger_out is not None:
                rejection_ledger_out.append(_rej_rec)
        except Exception:
            pass

    if _passes_joint_gate or _prewarp_accept or _marginal_uplift_accept:  # Best transformation is good enough
        _pair_res_entry = _emit_pair_features(
            raw_vars_pair=raw_vars_pair,
            pair_mi=pair_mi,
            best_mi=best_mi,
            best_config=best_config,
            var_pairs_perf=var_pairs_perf,
            this_pair_features=this_pair_features,
            _passes_joint_gate=_passes_joint_gate,
            _prewarp_accept=_prewarp_accept,
            _marginal_uplift_accept=_marginal_uplift_accept,
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
            messages=messages,
            discretize_array=discretize_array,
            discretize_2d_quantile_batch=discretize_2d_quantile_batch,
            mi_direct=mi_direct,
            get_new_feature_name=get_new_feature_name,
            _rebuild_full_survivor_col=_rebuild_full_survivor_col,
            _can_hoist_shared_buffer=_can_hoist_shared_buffer,
            _narrow_code_dtype=_narrow_code_dtype,
            _materialise_extval_njit=_materialise_extval_njit,
            _njit_binary_op_codes=_njit_binary_op_codes,
            _fe_use_parallel_kernels=_fe_use_parallel_kernels,
            _dispatch_batch_mi_with_noise_gate=_dispatch_batch_mi_with_noise_gate,
            batch_mi_with_noise_gate=batch_mi_with_noise_gate,
            use_su_normalization=use_su_normalization,
            X=X,
        )
    return _pair_res_entry, best_config, best_mi
