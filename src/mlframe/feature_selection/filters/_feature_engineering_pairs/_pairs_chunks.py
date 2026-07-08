"""Cross-pair (chunk) batching for the FE pair-search: chunk planning + the
one-batched-pass materialise -> discretize_2d -> batch_mi over a whole chunk."""
from __future__ import annotations

import logging

import numpy as np

from ._pairs_dispatch import _dispatch_batch_mi_with_noise_gate
from ._pairs_materialise import (
    _fe_use_parallel_kernels,
    _materialise_chunk_njit,
    _materialise_chunk_njit_parallel,
    _narrow_code_dtype,
    _njit_binary_op_codes,
)
from ..feature_engineering import _FE_BUFFER_RAM_BUDGET_RATIO

# NOTE: the authoritative ``_FE_BUFFER_RAM_BUDGET_RATIO`` (and the RAM-budget block comment
# documenting the hoist/recompute dispatch) lives in ``feature_engineering.py`` (value 0.3). The
# chunk module previously carried a stale duplicate (0.4) that nothing here read -- the chunk WIDTH
# is bounded by the ``chunk_max_cols`` param passed in by the caller, which the caller derives from
# ``_fe_effective_buffer_budget_bytes`` (the same authoritative ratio). The duplicate is removed; we
# re-export the authoritative constant so the package ``__init__`` surface (and any historical import)
# still resolves a single source of truth.

logger = logging.getLogger(__name__)

# CROSS-PAIR (CHUNK) BATCHING (2026-06-06). The per-pair 3-phase batch (materialise
# -> ONE discretize_2d -> ONE batch_mi) below is bit-identical to the per-candidate
# path but its batches are too SMALL to saturate the cores: the njit-prange kernels
# (discretize_2d_quantile_batch, batch_mi_with_noise_gate) release the GIL, yet one
# pair's K candidates (~tens-to-low-hundreds of columns) is too little work for the
# prange to spread across cores AND stays below the GPU dispatch threshold. So the
# GIL-bound Python orchestration BETWEEN kernel calls (combo materialisation,
# nan_to_num, per-candidate best/prewarp/config bookkeeping) serialises on ONE core
# and dominates -> ~15-21% CPU (1 of 8), GPU idle (mrmr.py keeps backend="threading"
# because loky memmaps crash Windows paging on 1M-row data).
#
# Fix: process a CHUNK of many raw-pairs together -- accumulate ALL the chunk's
# candidate columns into ONE wide buffer, run ONE big discretize_2d + ONE big
# batch_mi over the whole chunk (the prange now has K_chunk = sum of per-pair K
# columns of work -> enough to spread across cores via nogil, AND the batch crosses
# the GPU dispatch threshold so the cupy/cuda backend can engage), THEN replay the
# EXACT per-pair best/prewarp/config tracking by grouping the chunk's candidates back
# by pair. Bit-identical because BOTH kernels score each column INDEPENDENTLY
# (discretize: per-column percentile axis=0 / searchsorted; batch_mi: prange over
# columns with per-column joint histogram, and the permutation shuffle is seeded by
# (base_seed=0, perm_index) ONLY -- never by the column -- so concatenating columns
# from many pairs into one disc_2d yields the same per-column MI as scoring each pair
# separately). Only the hoist+quantile path is chunked; the recompute-fallback (no
# buffer) and uniform method keep the per-pair path verbatim.
#
# The chunk's buffer width is bounded by the SAME RAM budget the per-pair buffer uses
# (``n_rows * chunk_cols * 4`` bytes within ``_FE_BUFFER_RAM_BUDGET_RATIO`` of
# available RAM); we pack as many whole pairs as fit (a pair is never split across
# chunks so the per-pair replay is intact) but always at least one pair per chunk.
_FE_CHUNK_MAX_COLS_HARD_CAP: int = 65536


def _plan_fe_chunks(*, prospective_pairs, pair_combs, vars_transformations, n_binary, chunk_max_cols):
    """Partition ``prospective_pairs`` (in iteration order) into CHUNKS of raw-pairs
    whose summed candidate-column count fits ``chunk_max_cols``.

    A pair is NEVER split across chunks (so the per-pair replay stays intact); a single
    pair larger than the cap becomes its own chunk. Pairs with zero valid candidates
    (both-operand registration fails for every comb) are dropped from the plan -- the
    caller's per-pair path no-ops them identically.

    Returns ``(chunks, pair_valid_combs, buf_width)``:
      * ``chunks`` -- list of lists of ``raw_vars_pair`` (chunk membership, in order).
      * ``pair_valid_combs`` -- ``raw_vars_pair -> [valid transformations_pair, ...]``.
      * ``buf_width`` -- the column width the shared chunk buffer must have (the widest
        chunk's candidate count); 0 when nothing is chunkable.
    """
    chunk_max_cols = max(int(chunk_max_cols), 1)
    pair_valid_combs: dict = {}
    pair_cols: dict = {}
    for (raw_vars_pair, _pm), _u in prospective_pairs.items():
        combs = pair_combs.get(raw_vars_pair, [])
        valid = [tp for tp in combs if (tp[0] in vars_transformations) and (tp[1] in vars_transformations)]
        pair_valid_combs[raw_vars_pair] = valid
        pair_cols[raw_vars_pair] = len(valid) * n_binary

    chunks: list = []
    cur: list = []
    cur_cols = 0
    widest = 0
    for (raw_vars_pair, _pm), _u in prospective_pairs.items():
        need = pair_cols.get(raw_vars_pair, 0)
        if need == 0:
            continue
        if cur and (cur_cols + need > chunk_max_cols):
            chunks.append(cur)
            widest = max(widest, cur_cols)
            cur = []
            cur_cols = 0
        cur.append(raw_vars_pair)
        cur_cols += need
    if cur:
        chunks.append(cur)
        widest = max(widest, cur_cols)
    return chunks, pair_valid_combs, widest


def _compute_one_fe_chunk(
    *,
    chunk_pairs,
    pair_valid_combs,
    chunk_buffer,
    vars_transformations,
    transformed_vars,
    binary_transformations,
    quantization_nbins,
    quantization_dtype,
    classes_y,
    classes_y_safe,
    freqs_y,
    fe_npermutations,
    fe_min_nonzero_confidence,
    batch_mi_kernel,
    use_su,
    prewarp_unary,
    logger,
    discretize_2d_quantile_batch,
    serial_main_thread: bool = False,
    defer_float: bool = False,
):
    """Fill ``chunk_buffer[:, :K]`` with EVERY candidate column of EVERY pair in
    ``chunk_pairs``, then run ONE ``discretize_2d_quantile_batch`` + ONE
    ``_dispatch_batch_mi_with_noise_gate`` over the whole chunk. Returns a dict
    ``raw_vars_pair -> (candidates, fe_mi_by_col, local_times)``.

    This is the cross-pair batch: K = sum of the chunk's per-pair candidate counts, so
    the njit-prange / GPU dispatch sees the WHOLE chunk's work at once (saturates cores
    / can cross the GPU threshold). BIT-IDENTITY: both kernels score each column
    INDEPENDENTLY (per-column percentile/searchsorted; prange over columns with a
    per-column joint histogram and a permutation shuffle seeded by (base_seed=0,
    perm_index) ONLY -- never by the column), so a candidate's codes + MI are exactly
    what the per-pair batch produced. The materialise order (combs x bin_func) and
    nan_to_num/timing per pair mirror the per-pair Phase 1 exactly. The caller must
    extract any survivor columns from ``chunk_buffer`` BEFORE the next chunk overwrites
    it (the caller processes a chunk's pairs immediately after this returns).
    """
    from timeit import default_timer as timer

    col = 0
    chunk_records: dict = {}  # raw_vars_pair -> (candidates, local_times)
    _op_codes_by_name = _njit_binary_op_codes(binary_transformations)  # None if ANY op is not njit-coded
    _gpu_disc_2d = None  # GPU fused materialise+binning codes (njit-op path only); None -> CPU below
    # RESIDENCY DEFERRAL metadata (gated, ``defer_float``): when the GPU fused materialise SKIPPED the
    # (n,K) float D2H (out_cand=None), the chunk-buffer is unfilled and the caller must RE-MATERIALISE the
    # few columns it reads on the GPU. These chunk-wide arrays (indexed by buf_col, in the SAME order
    # ``col`` increments) carry the exact (a_col, b_col, op_code) ``_fe_materialise_block_gpu`` needs to
    # reproduce the EXACT bytes the bulk materialise would have written. None unless deferral fired.
    _deferred_float = False
    _defer_a_cols = None
    _defer_b_cols = None
    _defer_ops = None

    if _op_codes_by_name is not None:
        # NJIT PARALLEL PATH: record the candidate specs (NO per-candidate Python materialise), then fill the
        # ENTIRE chunk in ONE prange(nogil) kernel -> threads spread the work across cores. Candidate ORDER is the
        # SAME (pair x transformations_pair x bin_func) as the numpy path, so the config buffer index ``col`` is
        # identical, and the float32 op kernel is bit-identical to the numpy bin_funcs (see _materialise_chunk_njit).
        _name_list = list(binary_transformations.keys())
        _a_cols: list = []
        _b_cols: list = []
        _ops: list = []
        _cands_by_pair: dict = {}
        for raw_vars_pair in chunk_pairs:
            cands = []
            for transformations_pair in pair_valid_combs[raw_vars_pair]:
                _ai = vars_transformations[transformations_pair[0]]
                _bi = vars_transformations[transformations_pair[1]]
                uses_pw = transformations_pair[0][1] == prewarp_unary or transformations_pair[1][1] == prewarp_unary
                for _opn, bin_func_name in enumerate(_name_list):
                    _a_cols.append(_ai)
                    _b_cols.append(_bi)
                    _ops.append(int(_op_codes_by_name[_opn]))
                    cands.append((transformations_pair, bin_func_name, col, uses_pw))
                    col += 1
            _cands_by_pair[raw_vars_pair] = cands
        _t0 = timer()
        # GPU FUSED MATERIALISE+BINNING (2026-06-20). The candidate MATERIALISE
        # (``_materialise_chunk_njit``) is the #1 CPU FE hotspot at the canonical fit -- it is
        # MEMORY-BANDWIDTH bound on the strided operand gathers ``tv[r, ai]`` / ``tv[r, bi]``, not
        # compute (a CPU branch-hoist gave 0.95x and was reverted). The GPU has the bandwidth, and the
        # chunk-binning step right after this ALREADY runs on the GPU under the SAME size+HW gate
        # (``_fe_gpu_discretize_enabled``). So when that gate fires we generate the (n, K) candidate
        # matrix on the GPU and keep it RESIDENT to feed the resident discretize -- only the operand
        # columns go up and the small int codes come down; the float candidate matrix never crosses
        # the bus, removing both the CPU materialise AND the separate float H2D upload. BIT-IDENTICAL:
        # ``gpu_materialise_discretize_codes_host`` mirrors ``_materialise_chunk_njit``'s op semantics
        # EXACTLY (float32 ops, float64-promoted div/ratio_abs, nan_to_num) and bins via the same
        # resident cp.percentile path (verified maxdiff 0). Any GPU failure falls back to the CPU njit
        # materialise below (never a regression). ``_gpu_disc_2d`` carries the resident-path codes so
        # the discretize block below skips re-binning.
        #
        # NOTE the float candidate matrix is STILL brought to host (``out_cand=chunk_buffer``): the
        # downstream survivor / usability-corr / ext-val / multi-emit stages read the CONTINUOUS
        # candidate columns out of the chunk buffer, so a codes-ONLY resident path is not a drop-in
        # (it would leave the buffer uninitialised -> garbage survivor columns). So the GPU replaces the
        # bandwidth-bound CPU strided-gather materialise (and folds in the resident binning) but the
        # (n,K) float still D2Hs once for the buffer the rest of the pipeline expects.
        # BENCH (GTX 1050 Ti, n=100k, K=3600, median of 5, warm; this step in isolation):
        #   * vs CPU njit materialise + CPU binning : 30.3s -> 6.9s  = 4.40x
        #   * vs CPU njit materialise + GPU binning : 13.7s -> 6.7s  = 2.03x  (fair A/B, binning held on GPU)
        # End-to-end the FE materialise is one of several stages, so the fit-level delta is smaller and
        # contention-dependent; selection is BIT-IDENTICAL with the path on vs off (verified).
        _gpu_disc_2d = None
        _gpu_materialise_done = False
        # Escape hatch / A/B knob: MLFRAME_FE_GPU_MATERIALISE=0 disables ONLY the fused GPU materialise
        # (falls through to the CPU njit materialise + the existing GPU binning below). Default on -- the
        # path is gated by the SAME size+HW predicate as the GPU binning (_fe_gpu_discretize_enabled).
        import os as _os
        _gpu_mat_enabled = _os.environ.get("MLFRAME_FE_GPU_MATERIALISE", "1").strip().lower() not in (
            "0", "false", "no", "off",
        )
        if col > 0 and _gpu_mat_enabled:
            try:
                from ._pairs_core import _fe_gpu_discretize_enabled
                # RESIDENCY-OVERRIDE bench-attempt-rejected (2026-06-21): forcing the fused GPU path on
                # sub-crossover njit chunks (``... or fe_gpu_resident_codes_enabled()``) to keep their codes
                # on-device was a NO-OP at the canonical 100k fit -- A/B (seed 777): override on vs off =
                # IDENTICAL 60.27 MB H2D / 953 cp.asarray calls / 45.2-45.8s wall. Every njit chunk already
                # passes the speed crossover at the 30k screen subsample, so there are no sub-crossover njit
                # chunks to force; where it COULD fire (smaller fits) it would re-upload operand columns
                # (~60 MB pattern) instead of the compact int codes the CPU path uploads (~14 MB) -> a net
                # H2D LOSS. The real residency lever is making the per-chunk operand reads consume the
                # phase-1 resident operand table (the 953-call / 60 MB floor), not widening this gate.
                if _fe_gpu_discretize_enabled(transformed_vars.shape[0], col):
                    from .._gpu_resident_fe import gpu_materialise_discretize_codes_host  # type: ignore[attr-defined]  # dynamically re-exported via globals()
                    _code_dtype_gpu = _narrow_code_dtype(quantization_nbins, quantization_dtype)
                    _gpu_disc_2d = gpu_materialise_discretize_codes_host(
                        transformed_vars,
                        np.asarray(_a_cols, dtype=np.int64),
                        np.asarray(_b_cols, dtype=np.int64),
                        np.asarray(_ops, dtype=np.int8),
                        int(quantization_nbins),
                        dtype=_code_dtype_gpu,
                        # The downstream survivor / usability / ext-val stages read the CONTINUOUS
                        # candidate columns from the chunk buffer, so fill it with the GPU-materialised
                        # float matrix (this is the bandwidth-bound op the GPU replaces). The buffer is
                        # then exactly what the CPU njit path would have produced (bit-identical).
                        # RESIDENCY DEFERRAL (gated, ``defer_float``): skip this (n,K) float D2H entirely;
                        # the caller RE-MATERIALISES on the GPU (``_fe_materialise_block_gpu``) only the few
                        # buffer columns it actually reads, from the (a_col,b_col,op_code) metadata returned
                        # below -- BIT-IDENTICAL bytes (same kernel) so selection is unchanged. See the
                        # residency map in _gpu_resident_fe.py.
                        out_cand=None if defer_float else chunk_buffer[:, :col],
                    )
                    _gpu_materialise_done = True
                    if defer_float:
                        # The float buffer was NOT filled -> mark deferred + hand the caller the chunk-wide
                        # (a_col,b_col,op_code) arrays (same buf_col order as ``col``) for GPU re-materialise.
                        _deferred_float = True
                        _defer_a_cols = np.asarray(_a_cols, dtype=np.int64)
                        _defer_b_cols = np.asarray(_b_cols, dtype=np.int64)
                        _defer_ops = np.asarray(_ops, dtype=np.int8)
            except Exception:
                logger.debug("FE chunk GPU materialise+binning failed; CPU materialise", exc_info=True)
                _gpu_disc_2d = None
                _gpu_materialise_done = False
        if col > 0 and not _gpu_materialise_done:
            # OPT-A: on the serial-main-thread path (no joblib nest) use the byte-identical
            # column-prange twin above the per-host crossover; else the serial nogil kernel.
            _mat_args = (
                transformed_vars,
                np.asarray(_a_cols, dtype=np.int64),
                np.asarray(_b_cols, dtype=np.int64),
                np.asarray(_ops, dtype=np.int8),
                chunk_buffer[:, :col],
            )
            if _fe_use_parallel_kernels(col, serial_main_thread):
                _materialise_chunk_njit_parallel(*_mat_args)
            else:
                _materialise_chunk_njit(*_mat_args)
        # Fold the single batched materialise time evenly across the bin_func names (diagnostic only -- the
        # per-bin_func breakdown does not affect recovery; the MI/feature selection below is index-driven).
        _per = (timer() - _t0) / max(len(_name_list), 1)
        for raw_vars_pair in chunk_pairs:
            chunk_records[raw_vars_pair] = (_cands_by_pair[raw_vars_pair], {nm: _per for nm in _name_list})
    else:
        # NUMPY FALLBACK: a chunk op is not njit-coded (hypot / maximal-preset specials) -> per-candidate path.
        for raw_vars_pair in chunk_pairs:
            candidates = []
            local_times: dict = {}
            for transformations_pair in pair_valid_combs[raw_vars_pair]:
                param_a = transformed_vars[:, vars_transformations[transformations_pair[0]]]
                param_b = transformed_vars[:, vars_transformations[transformations_pair[1]]]
                uses_pw = transformations_pair[0][1] == prewarp_unary or transformations_pair[1][1] == prewarp_unary
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    for bin_func_name, bin_func in binary_transformations.items():
                        start = timer()
                        try:
                            chunk_buffer[:, col] = bin_func(param_a, param_b)
                        except Exception:
                            # Failed transform: the buffer slot may still hold a prior column's data. Null it so it is
                            # never scored, and skip recording it as a candidate (``col`` is not advanced here).
                            logger.exception("Error when performing %s", bin_func)
                            chunk_buffer[:, col] = np.nan
                            continue
                        # NaN/inf scrub DEFERRED to one vectorised pass over chunk_buffer[:, :col]
                        # below (was a per-column ``nan_to_num`` here -- the same per-column serial
                        # numpy hotspot the per-pair Phase-1 path had). Elementwise -> byte-identical.
                        local_times[bin_func_name] = local_times.get(bin_func_name, 0.0) + (timer() - start)
                        candidates.append((transformations_pair, bin_func_name, col, uses_pw))
                        col += 1
            chunk_records[raw_vars_pair] = (candidates, local_times)

    out: dict = {}
    if col == 0:
        for raw_vars_pair in chunk_pairs:
            candidates, local_times = chunk_records[raw_vars_pair]
            out[raw_vars_pair] = (candidates, None, local_times)
        return out

    # ONE vectorised NaN/inf scrub over every materialised chunk column [:, :col]
    # (replaces the per-column ``nan_to_num`` removed in the numpy-fallback loop above).
    # ONLY on the numpy-fallback path: the njit materialise kernel already scrubs inline
    # (see _materialise_chunk_njit), so re-scrubbing its output would be a wasted O(n*col)
    # pass. Elementwise -> byte-identical to the per-column scrub it replaces.
    if _op_codes_by_name is None:
        np.nan_to_num(chunk_buffer[:, :col], copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    _code_dtype = _narrow_code_dtype(quantization_nbins, quantization_dtype)  # OPT-B narrow codes
    disc_2d = _gpu_disc_2d  # GPU fused materialise+binning already produced the codes (njit-op path)
    if disc_2d is not None:
        # The fused GPU path generated the candidate matrix AND binned it RESIDENT (no CPU materialise,
        # no float H2D); ``disc_2d`` is the bit-identical int codes. Skip the CPU/GPU re-binning below.
        pass
    # GPU-binning decoupled from the analytic gate (2026-06-20): the cross-pair chunk binning
    # (_quantile_edges_2d_njit + _searchsorted_2d_right_njit_parallel) is the dominant CPU FE
    # hotspot at the default 30k screen-subsample (~4.4s of the warm canonical fit, on a single
    # (30000, ~3888) batch). The full GPU pair-MI path (``gpu_pairs_fe_mi``) DECLINES at n<50k
    # because its analytic chi2 noise-gate needs n >= analytic_null_min_n -- but the BINNING does
    # NOT depend on that gate. ``gpu_discretize_codes_host`` is bit-identical to the CPU
    # ``discretize_2d_quantile_batch`` (verified maxdiff 0; same cp.percentile linspace edges +
    # right-searchsorted) and ~1.7x faster at these screen sizes, so run ONLY the binning on the GPU
    # and feed the identical codes to the UNCHANGED CPU MI dispatcher below -- preserving selection
    # bit-for-bit. Same size+HW gate as the pair path (``_fe_gpu_discretize_enabled``); any GPU
    # failure falls back to the CPU discretise (never a regression).
    if disc_2d is None:
        try:
            # Route the standalone binning through the DEDICATED binning crossover (2026-06-23): the
            # bit-identical GPU binning is 17-24x faster at n=100k but was wrongly disabled by the full
            # ``fe_gpu_pairs_mi`` sweep's "cpu" verdict at the n<=100k band. The binning has its own gate
            # so the cheap, bit-identical op is no longer held hostage to the full MI path's crossover.
            from ._pairs_core import _fe_gpu_binning_enabled
            if _fe_gpu_binning_enabled(chunk_buffer.shape[0], col):
                from .._gpu_resident_fe import gpu_discretize_codes_host  # type: ignore[attr-defined]  # dynamically re-exported via globals()
                disc_2d = gpu_discretize_codes_host(chunk_buffer[:, :col], int(quantization_nbins), dtype=_code_dtype)
        except Exception:
            logger.debug("FE chunk GPU binning failed; CPU discretise", exc_info=True)
            disc_2d = None
    if disc_2d is None:
        disc_2d = discretize_2d_quantile_batch(
            chunk_buffer[:, :col], n_bins=quantization_nbins,
            dtype=_code_dtype,
            parallel=_fe_use_parallel_kernels(col, serial_main_thread),  # OPT-A
            # ``chunk_buffer[:, :col]`` is NaN-free here on BOTH branches: the njit materialise kernel
            # scrubs inline, and the numpy-fallback path ran the vectorised nan_to_num just above. So the
            # discretiser's per-call ``np.isnan().any()`` scan is guaranteed-False wasted work; skip it.
            assume_finite=True,
        )
    fe_mi_arr = _dispatch_batch_mi_with_noise_gate(
        disc_2d=disc_2d,
        quantization_nbins=quantization_nbins,
        classes_y=classes_y,
        classes_y_safe=classes_y_safe,
        freqs_y=freqs_y,
        npermutations=fe_npermutations,
        min_nonzero_confidence=fe_min_nonzero_confidence,
        use_su=use_su,
        batch_mi_kernel=batch_mi_kernel,
    )
    fe_mi_full = np.asarray(fe_mi_arr, dtype=np.float64)
    for raw_vars_pair in chunk_pairs:
        candidates, local_times = chunk_records[raw_vars_pair]
        out[raw_vars_pair] = (candidates, fe_mi_full, local_times)
    # Residency deferral signal: True iff the GPU FUSED codes path produced disc_2d AND we skipped the
    # float D2H (out_cand=None). The caller then GPU-re-materialises the few buffer columns it reads via
    # the chunk-wide (a_col,b_col,op_code) arrays. False whenever the float buffer WAS filled (CPU
    # materialise / numpy fallback / fused disabled) -> the caller reads the buffer as before.
    out["__float_deferred__"] = bool(_deferred_float and (_gpu_disc_2d is not None))
    out["__defer_meta__"] = (_defer_a_cols, _defer_b_cols, _defer_ops) if out["__float_deferred__"] else None
    return out
