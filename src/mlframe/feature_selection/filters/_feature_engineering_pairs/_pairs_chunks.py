"""Cross-pair (chunk) batching for the FE pair-search: chunk planning + the
one-batched-pass materialise -> discretize_2d -> batch_mi over a whole chunk."""
from __future__ import annotations

import numpy as np

from ._pairs_dispatch import _dispatch_batch_mi_with_noise_gate
from ._pairs_materialise import (
    _fe_use_parallel_kernels,
    _materialise_chunk_njit,
    _materialise_chunk_njit_parallel,
    _narrow_code_dtype,
    _njit_binary_op_codes,
)


# CRITICAL: the hoisted shared buffer at
# ``check_prospective_fe_pairs`` allocates ``(n, max_n_combs * len(binary))``
# float32. With n=4M and the medium preset that's ~17.6 GiB -- production
# MRMR crashed with numpy.core._exceptions._ArrayMemoryError on a real run.
# The hoist landed in Wave Pack G (commit 068acdd) under small-n benchmarks
# and never measured peak RAM on million-row data.
#
# Two-strategy dispatch:
#   Fast path (current): if buffer < ``_FE_BUFFER_RAM_BUDGET_RATIO`` * available
#     RAM, allocate the shared buffer and use the hoist (cheapest if it fits).
#   Recompute fallback: drop the multi-column buffer, scratch into a fresh 1D
#     ``np.empty(n, float32)`` per inner iteration, and rebuild the ~10
#     survivor columns from their (transformations_pair, bin_func_name) metadata
#     after the inner loop. Extra recompute cost: ~K bin_func calls per pair
#     (K = num survivors, typically <= fe_max_pair_features + |leading|);
#     <= 1% of the ~max_combs*|binary| calls already done in the inner loop.
#
# Subsample path remains a separate opt-in (``subsample_n`` parameter); this
# memory dispatcher is the deterministic, accuracy-preserving fallback that
# auto-engages when the shared buffer would OOM.
_FE_BUFFER_RAM_BUDGET_RATIO: float = 0.4

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
        valid = [
            tp for tp in combs
            if (tp[0] in vars_transformations) and (tp[1] in vars_transformations)
        ]
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
                uses_pw = (
                    transformations_pair[0][1] == prewarp_unary
                    or transformations_pair[1][1] == prewarp_unary
                )
                for _opn, bin_func_name in enumerate(_name_list):
                    _a_cols.append(_ai)
                    _b_cols.append(_bi)
                    _ops.append(int(_op_codes_by_name[_opn]))
                    cands.append((transformations_pair, bin_func_name, col, uses_pw))
                    col += 1
            _cands_by_pair[raw_vars_pair] = cands
        _t0 = timer()
        if col > 0:
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
                uses_pw = (
                    transformations_pair[0][1] == prewarp_unary
                    or transformations_pair[1][1] == prewarp_unary
                )
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    for bin_func_name, bin_func in binary_transformations.items():
                        start = timer()
                        try:
                            chunk_buffer[:, col] = bin_func(param_a, param_b)
                        except Exception:
                            logger.error(f"Error when performing {bin_func}")
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

    disc_2d = discretize_2d_quantile_batch(
        chunk_buffer[:, :col], n_bins=quantization_nbins,
        dtype=_narrow_code_dtype(quantization_nbins, quantization_dtype),  # OPT-B narrow codes
        parallel=_fe_use_parallel_kernels(col, serial_main_thread),  # OPT-A
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
    return out
