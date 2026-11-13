"""Recompute-path scoring of the pair candidates, carved out of _pairs_score_steps.

Holds the pending-candidate scheduling, the blocked recompute and the step-three transformation scorer that
the pair scorer calls. The functions are moved verbatim; the head module re-exports the step-three entry point.
"""

from __future__ import annotations

from typing import Any
import logging
from timeit import default_timer as timer
import numpy as np
from ._pairs_code_bins import code_table_bins, scrub_nonfinite
from .._numba_utils import rows_to_columns
from ._pairs_dispatch import _dispatch_batch_mi_with_noise_gate
from ._pairs_materialise import _narrow_code_dtype
from mlframe.utils.log_throttle import log_throttle
from ._pairs_score_helpers import logger


def _columns_of_one_matrix(columns: list, n_rows: int) -> "np.ndarray | None":
    """The ``(n_rows, len(columns))`` matrix whose columns ``columns`` are, in order (each a view ``matrix[:, j]``), or ``None`` when they are not exactly that."""
    if not columns:
        return None
    base = columns[0].base
    if not isinstance(base, np.ndarray) or base.ndim != 2 or base.shape != (n_rows, len(columns)) or not base.flags.c_contiguous:
        return None
    start = base.__array_interface__["data"][0]
    step = base.strides[1]
    for j, col in enumerate(columns):
        if col.base is not base or col.shape != (n_rows,) or col.strides != (base.strides[0],) or col.__array_interface__["data"][0] != start + j * step:
            return None
    matrix: np.ndarray = base
    return matrix


def _score_pending_candidates(
    pending, n_rows, quantization_nbins, quantization_dtype, classes_y, classes_y_safe, freqs_y, fe_npermutations, fe_min_nonzero_confidence,
    use_su_normalization, batch_mi_with_noise_gate, _fe_env_gate, mi_direct,
) -> list:
    """fe_mi of the pending candidates, in order: one batched call per block, per-candidate ``mi_direct`` when the batch kernel is not supplied or fails."""
    if batch_mi_with_noise_gate is not None and len(pending) > 1:
        try:
            _code_dtype: Any = _narrow_code_dtype(quantization_nbins, quantization_dtype)
            _whole = _columns_of_one_matrix([_entry[4] for _entry in pending], n_rows)
            if _whole is not None:
                # The candidates are the columns, in order, of the matrix the block binner returned: one sequential pass narrows it (the column-by-column copy into rows and the transpose back rebuilt this matrix).
                _block = np.ascontiguousarray(_whole, dtype=_code_dtype)
            else:
                _rows = np.empty((len(pending), n_rows), dtype=_code_dtype)
                for _j, _entry in enumerate(pending):
                    _rows[_j] = _entry[4]
                _block = rows_to_columns(_rows)
            _mi = _dispatch_batch_mi_with_noise_gate(
                disc_2d=_block,
                quantization_nbins=quantization_nbins,
                classes_y=classes_y,
                classes_y_safe=classes_y_safe,
                freqs_y=freqs_y,
                npermutations=fe_npermutations,
                min_nonzero_confidence=fe_min_nonzero_confidence,
                use_su=use_su_normalization() if use_su_normalization is not None else False,
                batch_mi_kernel=batch_mi_with_noise_gate,
                env_gate=_fe_env_gate,
            )
            return [float(v) for v in _mi]
        except Exception as e:
            log_throttle(logger, "pairs_score_recompute_batch_mi_failed", logging.WARNING, "Batched candidate MI failed (%s); scoring the candidates one by one.", e)
    return [
        mi_direct(
            _entry[4].reshape(-1, 1),
            x=np.array([0], dtype=np.int64),
            y=None,
            factors_nbins=np.array([code_table_bins(_entry[4], quantization_nbins)], dtype=np.int64),
            classes_y=classes_y,
            classes_y_safe=classes_y_safe,
            freqs_y=freqs_y,
            min_nonzero_confidence=fe_min_nonzero_confidence,
            npermutations=fe_npermutations,
        )[0]
        for _entry in pending
    ]


def _replay_pending_scores(st, pending, scores, fe_print_best_mis_only, verbose, pair_mi) -> None:
    """Replay the best / prewarp / config tracking over the scored pending candidates in the order they were produced, then clear them."""
    for (config, transformations_pair, bin_func_name, _uses_pw, _codes), fe_mi in zip(pending, scores):
        st.var_pairs_perf[config] = fe_mi
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
    pending.clear()


def _bin_held_block(blk, meta, pending, st, need_recompute_map, config_by_i, quantization_nbins, quantization_dtype, quantization_method, discretize_array) -> None:
    """Clean and quantile-bin the ``len(meta)`` columns held in ``blk`` in one call, and queue each (config, codes) entry on ``pending`` in order."""
    _n_held = len(meta)
    _vals = rows_to_columns(blk[:_n_held])  # ``blk`` holds one candidate per row
    scrub_nonfinite(_vals)
    try:
        from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch

        _codes = discretize_2d_quantile_batch(_vals, n_bins=quantization_nbins, dtype=quantization_dtype, parallel=True)
    except Exception as _exc:
        logger.warning("pair-step: blocked binning failed, falling back to per-column binning: %s", _exc)
        _codes = np.stack(
            [discretize_array(arr=_vals[:, c], n_bins=quantization_nbins, method=quantization_method, dtype=quantization_dtype) for c in range(_n_held)],
            axis=1,
        )
    for _c, (_tp, _nm, _uw, _idx) in enumerate(meta):
        _config = (_tp, _nm, _idx)
        _col_codes = _codes[:, _c]
        # Keep the codes this configuration was scored on (the pair-level family gate needs every configuration of the pair).
        st.__dict__.setdefault("var_pairs_codes", {})[_config] = np.asarray(_col_codes).reshape(-1, 1)
        if need_recompute_map:
            assert config_by_i is not None
            config_by_i[_idx] = (_tp[0], _tp[1], _nm)
        pending.append((_config, _tp, _nm, _uw, _col_codes))


def _contiguous_column(cache: dict, matrix: np.ndarray, j: int) -> np.ndarray:
    """Column ``j`` of the row-major ``matrix`` as a contiguous vector (the same values), copied once per call."""
    col = cache.get(j)
    if col is None:
        col = np.ascontiguousarray(matrix[:, j])
        cache[j] = col
    return col


def _score_recompute_blocked(
    st, final_transformed_vals, transformed_vars, vars_transformations, binary_transformations, _PREWARP_UNARY,
    quantization_nbins, quantization_dtype, quantization_method, discretize_array, classes_y, classes_y_safe, freqs_y,
    fe_npermutations, fe_min_nonzero_confidence, use_su_normalization, batch_mi_with_noise_gate, _fe_env_gate,
    _need_recompute_map, _config_by_i, fe_print_best_mis_only, verbose, pair_mi, mi_direct,
):
    """The recompute path, blocked: materialise, bin and score the candidates of a pair in bounded blocks (see the comment in the caller)."""
    _pending: list = []  # (config, transformations_pair, bin_func_name, uses_pw, codes) of the candidates whose MI is not scored yet
    _n_rows = int(transformed_vars.shape[0])
    _block_cols = max(1, min(4096, (64 << 20) // max(1, _n_rows)))  # the block's code matrix stays within ~64 MiB

    # BLOCKED RECOMPUTE: the candidates of the pair are materialised into a bounded float block (about 64 MiB), the block is cleaned and quantile-binned
    # in ONE ``discretize_2d_quantile_batch`` call (bit-identical to the per-column binning) and its MI is scored by the batched noise gate.
    _blk_dtype = st._col_buf_1d.dtype
    _blk_cols = max(1, min(4096, (64 << 20) // max(1, _n_rows * np.dtype(_blk_dtype).itemsize)))
    _blk = np.empty((_blk_cols, _n_rows), dtype=_blk_dtype)  # one candidate per contiguous row; transposed once per block
    _meta: list = []  # (transformations_pair, bin_func_name, uses_pw, config index) of the columns held in _blk
    _contiguous_cols: dict = {}  # column index -> contiguous copy of that column of ``transformed_vars``: every op below reads it, a strided view made each ufunc 2.6x slower

    def _close_block() -> None:
        """Bin the held columns in one call, score their MI and replay the tracking in order."""
        if not _meta:
            return
        _bin_held_block(_blk, _meta, _pending, st, _need_recompute_map, _config_by_i, quantization_nbins, quantization_dtype, quantization_method, discretize_array)
        _meta.clear()
        if _pending:
            _replay_pending_scores(
                st,
                _pending,
                _score_pending_candidates(
                    _pending, _n_rows, quantization_nbins, quantization_dtype, classes_y, classes_y_safe, freqs_y, fe_npermutations,
                    fe_min_nonzero_confidence, use_su_normalization, batch_mi_with_noise_gate, _fe_env_gate, mi_direct,
                ),
                fe_print_best_mis_only,
                verbose,
                pair_mi,
            )

    for transformations_pair in st.combs:
        if (transformations_pair[0] not in vars_transformations) or (transformations_pair[1] not in vars_transformations):
            continue
        param_a = _contiguous_column(_contiguous_cols, transformed_vars, vars_transformations[transformations_pair[0]])
        param_b = _contiguous_column(_contiguous_cols, transformed_vars, vars_transformations[transformations_pair[1]])
        _uses_pw = transformations_pair[0][1] == _PREWARP_UNARY or transformations_pair[1][1] == _PREWARP_UNARY
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for bin_func_name, bin_func in binary_transformations.items():
                start = timer()
                try:
                    _blk[len(_meta)] = bin_func(param_a, param_b)
                except Exception:
                    # Failed transform: no candidate is recorded for this bin_func (its block column is overwritten by the next candidate).
                    log_throttle(logger, "pairs_score_transform_recompute_failed", logging.ERROR, "Error when performing %s", bin_func, exc_info=True)
                    continue
                st._local_times[bin_func_name] = st._local_times.get(bin_func_name, 0.0) + (timer() - start)
                _meta.append((transformations_pair, bin_func_name, _uses_pw, st.i))
                st.i += 1
                if len(_meta) >= _blk_cols:
                    _close_block()
    _close_block()
    return


def _maybe_pair_batch_codes(need_recompute_map, binary_transformations, param_a, param_b, quantization_nbins, quantization_dtype, st):
    """Batch-binned codes of the pair's candidates on the recompute path, or ``None`` when that path does not apply."""
    if need_recompute_map and len(binary_transformations) > 1:
        return _pair_batch_codes(binary_transformations, param_a, param_b, quantization_nbins, quantization_dtype, st._col_buf_1d.dtype)
    return None


def _takes_blocked_recompute(st, final_transformed_vals, quantization_method, batch_mi_with_noise_gate) -> bool:
    """True when a pair's candidates must be scored through the blocked recompute path: no buffered values, quantile binning, a batch kernel and a scratch buffer."""
    return final_transformed_vals is None and quantization_method == "quantile" and batch_mi_with_noise_gate is not None and st._col_buf_1d is not None


def _record_pair_candidate(
    st,
    transformations_pair,
    bin_func_name,
    fe_mi,
    discretized_transformed_values,
    uses_pw,
    need_recompute_map,
    config_by_i,
    fe_print_best_mis_only,
    verbose,
    pair_mi,
):
    """Store one scored candidate of the pair: its configuration's MI and codes, and the running best overall, with prewarp and without it."""
    config = (transformations_pair, bin_func_name, st.i)
    st.var_pairs_perf[config] = fe_mi
    # Keep the codes this configuration was scored on: the pair-level family gate (_pairs_emit.apply_pair_family_gate_from_state)
    # needs every configuration of the pair to test the winner against the maximum over all of them.
    st.__dict__.setdefault("var_pairs_codes", {})[config] = np.asarray(discretized_transformed_values).reshape(-1, 1)
    if need_recompute_map:
        # Map i -> (a_key, b_key, bin_func_name) for downstream
        # rebuild; bin_func is looked up via the original dict.
        assert config_by_i is not None
        config_by_i[st.i] = (transformations_pair[0], transformations_pair[1], bin_func_name)

    if fe_mi > st.best_mi:
        st.best_mi = fe_mi
        st.best_config = config
    # Track best-with-prewarp vs best-without so the alternative
    # uplift gate below can decide whether the prewarp earned its
    # place (``uses_pw`` hoisted above the bin_func loop).
    if uses_pw:
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


def _score_one_pair_step3_step2_transformations_pair_st(
    st,
    vars_transformations,
    transformed_vars,
    _PREWARP_UNARY,
    binary_transformations,
    final_transformed_vals,
    discretize_array,
    quantization_nbins,
    quantization_method,
    quantization_dtype,
    mi_direct,
    classes_y,
    classes_y_safe,
    freqs_y,
    fe_min_nonzero_confidence,
    fe_npermutations,
    _need_recompute_map,
    _config_by_i,
    fe_print_best_mis_only,
    verbose,
    pair_mi,
    batch_mi_with_noise_gate=None,
    _fe_env_gate=None,
    use_su_normalization=None,
):
    """Step 2 of _score_one_pair_step3_resolve_col_treat: lines starting at ``for transformations_pair in st.combs:``.

    The candidates' MI is scored in blocks through the batched noise gate (one call per block, bit-identical to one ``mi_direct`` call per candidate: see
    ``test_pair_candidates_batched_mi_equals_per_candidate``) and the per-candidate tracking is replayed in the original order. This path runs when the
    shared candidate buffer does not fit the free-RAM budget; without the batching it made one ``mi_direct`` call per candidate (about 1200 per pair),
    which is 50 to 100 times slower than the buffered path for a saving of a few hundred MB. Without ``batch_mi_with_noise_gate`` the scoring falls back to
    the per-candidate calls.
    """
    if _takes_blocked_recompute(st, final_transformed_vals, quantization_method, batch_mi_with_noise_gate):
        _score_recompute_blocked(
            st, final_transformed_vals, transformed_vars, vars_transformations, binary_transformations, _PREWARP_UNARY,
            quantization_nbins, quantization_dtype, quantization_method, discretize_array, classes_y, classes_y_safe, freqs_y,
            fe_npermutations, fe_min_nonzero_confidence, use_su_normalization, batch_mi_with_noise_gate, _fe_env_gate,
            _need_recompute_map, _config_by_i, fe_print_best_mis_only, verbose, pair_mi, mi_direct,
        )
        return

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
        # RECOMPUTE PATH (L-24): the per-column quantile binning of this pair's candidates runs as ONE batch over the pair's
        # columns (bit-identical to the per-column discretize_array, parallel over columns), instead of one np.quantile per candidate.
        _pair_codes = _maybe_pair_batch_codes(_need_recompute_map, binary_transformations, param_a, param_b, quantization_nbins, quantization_dtype, st)
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
                    scrub_nonfinite(_col_view)

                    # ``times_spent`` is shared across mrmr.py's
                    # parallel threading dispatch. Accumulate this pair's
                    # per-bin_func timings in a thread-LOCAL dict and merge them
                    # under ``_TIMES_SPENT_LOCK`` once per pair (below); the old
                    # per-inner-iteration lock was a serialization point on the
                    # hot path. Totals are identical.
                    st._local_times[bin_func_name] = st._local_times.get(bin_func_name, 0.0) + (timer() - start)

                    if _pair_codes is not None and bin_func_name in _pair_codes:
                        discretized_transformed_values = _pair_codes[bin_func_name]
                    else:
                        discretized_transformed_values = discretize_array(
                            arr=_col_view, n_bins=quantization_nbins, method=quantization_method, dtype=quantization_dtype
                        )
                    fe_mi, _fe_conf = mi_direct(
                        discretized_transformed_values.reshape(-1, 1),
                        x=np.array([0], dtype=np.int64),
                        y=None,
                        factors_nbins=np.array([code_table_bins(discretized_transformed_values, quantization_nbins)], dtype=np.int64),
                        classes_y=classes_y,
                        classes_y_safe=classes_y_safe,
                        freqs_y=freqs_y,
                        min_nonzero_confidence=fe_min_nonzero_confidence,
                        npermutations=fe_npermutations,
                    )

                    _record_pair_candidate(
                        st, transformations_pair, bin_func_name, fe_mi, discretized_transformed_values, _uses_pw, _need_recompute_map,
                        _config_by_i, fe_print_best_mis_only, verbose, pair_mi,
                    )
                    st.i += 1


def _pair_batch_codes(binary_transformations, param_a, param_b, quantization_nbins, quantization_dtype, scratch_dtype):
    """Quantile codes of every binary transform of one pair, from one batched discretisation (recompute path only).

    Each transform is evaluated as in the per-column loop, the results are written into a block with the scratch buffer's dtype
    (so both paths see the same values), non-finite values are scrubbed the same way, and ``discretize_2d_quantile_batch`` bins all
    columns together (bit-identical to per-column ``discretize_array``). A transform that raises is left out, so the caller's
    per-column path reports it exactly as before. Returns ``{name: codes}``, or None when the batch cannot be built."""
    from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch

    names, cols = [], []
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for bin_func_name, bin_func in binary_transformations.items():
            try:
                col = np.asarray(bin_func(param_a, param_b)).ravel()
            except Exception as _exc:
                logger.debug("pair-step: candidate %r skipped: %s", bin_func_name, _exc)
                continue
            names.append(bin_func_name)
            cols.append(col)
    if not cols:
        return None
    block = np.empty((len(param_a), len(cols)), dtype=scratch_dtype)
    for j, col in enumerate(cols):
        block[:, j] = col
    scrub_nonfinite(block)
    try:
        codes = discretize_2d_quantile_batch(block, n_bins=quantization_nbins, dtype=quantization_dtype, parallel=True)
    except Exception as _exc:
        logger.warning("pair-step: batched recompute failed, falling back to per-transform: %s", _exc)
        return None
    return {nm: codes[:, j] for j, nm in enumerate(names)}
