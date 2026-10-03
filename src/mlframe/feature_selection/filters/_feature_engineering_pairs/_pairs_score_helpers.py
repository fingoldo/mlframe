"""Helpers carved out of ``_pairs_score`` to keep that module under its size budget."""
from __future__ import annotations


import logging
from timeit import default_timer as timer

import numpy as np

from ._pairs_common import _TIMES_SPENT_LOCK
from ._pairs_gates import (
    _FE_MARGINAL_UPLIFT_MIN_JOINT_RATIO,
    _FE_MARGINAL_UPLIFT_MIN_RATIO,
    _FE_MARGINAL_UPLIFT_STRICT_JOINT_RATIO,
    _FE_MARGINAL_UPLIFT_SYNERGY_UPLIFT,
)
from ._pairs_materialise import (
    _fe_use_parallel_kernels,
    _narrow_code_dtype,
)
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

# DEGENERATE-PAIR |corr| threshold. A prospective pair is DEGENERATE when its winning
# composite is numerically ~= a SINGLE one of its own (warped) operands - the other operand's transform
# collapsed to ~constant so the binary op is just a re-wrap of one source and adds NO genuine joint
# information (e.g. ``mul(prewarp(b),prewarp(a__L2))`` where ``prewarp(b)`` is ~constant -> the column is
# essentially ``prewarp(a__L2)`` ~= a**2). Such a pair keeps |corr|~=1.0 with that one operand, clears the
# joint-prevalence gate, and DISPLACES the clean single-source univariate basis (a__L2). A GENUINE 2-var
# pair (a/b, a*b) has |corr| well below this with EITHER single operand, so the bar leaves it untouched.
# 0.999 is intentionally near-1.0: it fires only on a true single-operand re-wrap, never on a real pair.
_DEGENERATE_PAIR_SINGLE_OPERAND_CORR: float = 0.999


logger = logging.getLogger(__name__)


def _should_demote_prewarp(pw_corr, clean_corr) -> bool:
    """Whether a prewarp-form winner is demoted to its MI-equivalent clean form, from the two forms' |corr| with continuous y.

    ``-1.0`` means the clean config is genuinely unrecoverable (nothing to demote to); ``None`` means a correlation could not be measured.
    An unmeasured side resolves toward the simpler clean form: the prewarp is a distorted re-expression that must EARN its place by being
    at least 5% more linearly usable, and a failed measurement is no evidence that it is.
    """
    if clean_corr is None or pw_corr is None:
        return bool(clean_corr is None or clean_corr >= 0.0)
    return bool(clean_corr >= 0.0 and pw_corr < clean_corr * 1.05)


def _score_one_pair_fe_gpu_binning_enabled(_K, _fe_gpu_binning_enabled, final_transformed_vals, quantization_nbins, quantization_dtype, transformed_vars, _a_cols, _b_cols, _ops, _gpu_cands, _name_list, _local_times, _batch_candidates, _disc_2d, _gpu_fused_done):
    """Block of _score_one_pair starting at ``if _K > 0 and _fe_gpu_binning_enabled(final_transformed_vals.shape[0],``."""
    if _K > 0 and _fe_gpu_binning_enabled(final_transformed_vals.shape[0], _K):
        _code_dtype = _narrow_code_dtype(quantization_nbins, quantization_dtype)
        _start = timer()
        from mlframe.feature_selection.filters._gpu_resident_fe import gpu_materialise_discretize_codes_host  # type: ignore[attr-defined]  # dynamically re-exported via globals()
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
    return _batch_candidates, _disc_2d, _gpu_fused_done


def _score_one_pair_gpu_fused_done(_gpu_fused_done, combs, vars_transformations, transformed_vars, _PREWARP_UNARY, binary_transformations, final_transformed_vals, _local_times, _batch_candidates, i):
    """Block of _score_one_pair starting at ``if not _gpu_fused_done:``."""
    if not _gpu_fused_done:
        # The GPU-prep candidate loop above advances ``i`` to size the fused launch; if the GPU-binning gate declines (no device / below crossover) we reach here with NO exception,
        # so the ``except``-path ``i = 0`` reset never ran - restart the column cursor before the CPU materialise, else it overruns ``final_transformed_vals`` (``_batch_candidates``/``_disc_2d`` keep their pre-try init).
        i = 0
        for transformations_pair in combs:
            if (transformations_pair[0] not in vars_transformations) or (transformations_pair[1] not in vars_transformations):
                continue
            param_a = transformed_vars[:, vars_transformations[transformations_pair[0]]]
            param_b = transformed_vars[:, vars_transformations[transformations_pair[1]]]
            _uses_pw = transformations_pair[0][1] == _PREWARP_UNARY or transformations_pair[1][1] == _PREWARP_UNARY
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
                        log_throttle(logger, "pairs_score_transform_final_failed", logging.ERROR, "Error when performing %s", bin_func, exc_info=True)
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
        # SKIPPED on the GPU fused path (the kernel scrubs NaN/inf inline - bit-identical).
        if i > 0:
            np.nan_to_num(final_transformed_vals[:, :i], copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return i


def _score_one_pair_fut_none(_fut, _my_chunk, _mi_cache):
    """Block of _score_one_pair starting at ``if _fut is not None:``."""
    if _fut is not None:
        try:
            _mi_cache = _fut.result()
        except Exception:
            logger.debug("pipelined chunk %d producer failed; inline recompute", _my_chunk, exc_info=True)
            _mi_cache = None
    return _mi_cache


def _score_one_pair_prefetch_next_chunk_into(_ex, _bufs, _my_chunk, _fe_chunks, _produce, chunk_state):
    """Block of _score_one_pair starting at ``if _ex is not None and _bufs is not None and (_my_chunk + 1) < len(_fe``."""
    if _ex is not None and _bufs is not None and (_my_chunk + 1) < len(_fe_chunks):
        try:
            chunk_state.setdefault("pipeline_futures", {})[_my_chunk + 1] = _ex.submit(_produce, _my_chunk + 1, _bufs[(_my_chunk + 1) % 2])
        except Exception:
            logger.debug("chunk prefetch submit failed; falling back to lazy compute", exc_info=True)


def _score_one_pair_transformations_pair_combs(combs, vars_transformations, _PREWARP_UNARY, _name_list, _a_cols, _b_cols, _ops, _op_code_arr, _gpu_cands, i):
    """Block of _score_one_pair starting at ``for transformations_pair in combs:``."""
    for transformations_pair in combs:
        if (transformations_pair[0] not in vars_transformations) or (transformations_pair[1] not in vars_transformations):
            continue
        _ai = vars_transformations[transformations_pair[0]]
        _bi = vars_transformations[transformations_pair[1]]
        _uses_pw = transformations_pair[0][1] == _PREWARP_UNARY or transformations_pair[1][1] == _PREWARP_UNARY
        for _opn, bin_func_name in enumerate(_name_list):
            _a_cols.append(_ai)
            _b_cols.append(_bi)
            _ops.append(int(_op_code_arr[_opn]))
            _gpu_cands.append((transformations_pair, bin_func_name, i, _uses_pw))
            i += 1
    return i


def _score_one_pair_non_analytic_branch_su(_fe_gpu_discretize_enabled, final_transformed_vals, i, quantization_nbins, classes_y, classes_y_safe, freqs_y, fe_npermutations, fe_min_nonzero_confidence, use_su_normalization, _fe_mi_arr):
    """Block of _score_one_pair starting at ``if _fe_gpu_discretize_enabled(final_transformed_vals.shape[0], i):``."""
    if _fe_gpu_discretize_enabled(final_transformed_vals.shape[0], i):
        try:
            from mlframe.feature_selection.filters._gpu_resident_fe import gpu_pairs_fe_mi  # type: ignore[attr-defined]  # dynamically re-exported via globals()
            _fe_mi_arr = gpu_pairs_fe_mi(
                final_transformed_vals[:, :i], int(quantization_nbins),
                classes_y, classes_y_safe, freqs_y,
                fe_npermutations, fe_min_nonzero_confidence, use_su_normalization(),
            )
        except Exception:
            logger.debug("FE GPU pair-MI failed; falling back to CPU", exc_info=True)
            _fe_mi_arr = None
    return _fe_mi_arr


def _score_one_pair_fe_mi_arr_none(_fe_mi_arr, _disc_2d, final_transformed_vals, i, quantization_nbins, _code_dtype, discretize_2d_quantile_batch, serial_main_thread):
    """Block of _score_one_pair starting at ``if _fe_mi_arr is None:``."""
    if _fe_mi_arr is None:
        # ``_disc_2d`` may ALREADY be the codes from the GPU FUSED materialise+bin path
        # (Phase 1 above) - in that case skip re-binning (the fused kernel produced the
        # SAME codes ``gpu_discretize_codes_host`` would). Only bin here when it is None.
        # GPU BINNING: the per-pair Phase-2 binning gets the SAME dedicated
        # binning crossover the chunk path now uses. ``gpu_discretize_codes_host`` is
        # bit-identical to the CPU njit binning (verified maxdiff 0) and 17-24x faster at
        # n=100k; it was previously reachable here ONLY through the full ``gpu_pairs_fe_mi``
        # path (declined on the non-analytic branch -> CPU njit). Any GPU failure falls back
        # to the CPU discretise below (never a regression; selection bit-identical).
        if _disc_2d is None:
            try:
                from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core import _fe_gpu_binning_enabled
                if _fe_gpu_binning_enabled(final_transformed_vals.shape[0], i):
                    from mlframe.feature_selection.filters._gpu_resident_fe import gpu_discretize_codes_host  # type: ignore[attr-defined]  # dynamically re-exported via globals()
                    # defer_host_fill: the codes flow straight into _dispatch_batch_mi_with_noise_gate,
                    # whose resident-CUDA gate consumes the DEVICE codes in place (take_resident_codes)
                    # and only triggers the lazy host fill (ensure_host_codes_filled) on a host-reading
                    # branch. Skips the (n, K) codes D2H - the fit's single largest - whenever the
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
                # OPT-A extension: same main-thread parallel searchsorted
                # gate as the chunk + marginal-uplift discretise - byte-identical
                # column-prange twin when serial_main_thread (no joblib nest).
                parallel=_fe_use_parallel_kernels(i, serial_main_thread),
                # The ``np.nan_to_num(..., copy=False)`` directly above scrubbed this exact
                # buffer slice, so the per-call ``np.isnan().any()`` scan inside the discretiser
                # is guaranteed-False wasted work; skip it (bit-identical on a NaN-free buffer).
                assume_finite=True,
            )
    return _disc_2d


def _score_one_pair_fe_mi_best_mi(fe_mi, best_mi, fe_print_best_mis_only, verbose, bin_func_name, transformations_pair, pair_mi):
    """Block of _score_one_pair starting at ``if fe_mi > best_mi * 0.85:``."""
    if fe_mi > best_mi * 0.85:
        if not fe_print_best_mis_only or (fe_mi == best_mi):
            if verbose > 2:
                logger.debug("MI of transformed pair %s(%s)=%.4f, MI of the plain pair %.4f", bin_func_name, transformations_pair, fe_mi, pair_mi)


def _score_one_pair_iteration_serialization_point_under(_local_times, times_spent):
    """Block of _score_one_pair starting at ``if _local_times:``."""
    if _local_times:
        with _TIMES_SPENT_LOCK:
            for _bf, _dt in _local_times.items():
                times_spent[_bf] += _dt


def _score_one_pair_monotone_equivalent_warp_scores(_pw_corr, _clean_corr, best_nonprewarp_config, best_nonprewarp_mi, var_pairs_perf, _PREWARP_UNARY, verbose, messages, best_config, best_mi):
    """Block of _score_one_pair starting at ``if _should_demote_prewarp(_pw_corr, _clean_corr):``."""
    if _should_demote_prewarp(_pw_corr, _clean_corr):
        best_config, best_mi = best_nonprewarp_config, best_nonprewarp_mi
        # The single-best emission path (below) does NOT read ``best_config``
        # directly - it rebuilds the leaders band from ``var_pairs_perf`` and
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
    return best_config, best_mi


def _score_one_pair_mi_threshold_ratio_fix(fe_mm_debias_prevalence, pair_mi, best_config, classes_y, freqs_y, quantization_nbins, discretize_array, _resolve_col, quantization_method, quantization_dtype, _operand_discretized, raw_vars_pair, best_mi, _gate_ratio):
    """Block of _score_one_pair starting at ``if fe_mm_debias_prevalence and pair_mi > 0.0 and best_config is not No``."""
    if fe_mm_debias_prevalence and pair_mi > 0.0 and best_config is not None:
        from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_gates import _occupied_k, mm_debiased_prevalence_ratio
        _n_rows = len(classes_y)
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
        except Exception as e:
            logger.debug("_occupied_k computation failed, falling back to quantization_nbins: %s", e)
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
    return _gate_ratio


def _score_one_pair_uplift_fires_prewarp_config(_prewarp_active, _passes_joint_gate, best_prewarp_config, best_nonprewarp_mi, best_prewarp_mi, prewarp_uplift_threshold, verbose, messages, pair_mi, fe_min_engineered_mi_prevalence, _prewarp_accept, best_config, best_mi):
    """Block of _score_one_pair starting at ``if (``."""
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
    return _prewarp_accept, best_config, best_mi


def _score_one_pair_warp_spec_round_trip(_passes_joint_gate, _prewarp_accept, best_nonprewarp_config, best_nonprewarp_mi, pair_mi, _operand_marginal_mi, raw_vars_pair, verbose, messages, fe_min_engineered_mi_prevalence, _marginal_uplift_accept, best_config, best_mi):
    """Block of _score_one_pair starting at ``if not _passes_joint_gate and not _prewarp_accept and best_nonprewarp_``."""
    if not _passes_joint_gate and not _prewarp_accept and best_nonprewarp_config is not None and best_nonprewarp_mi > 0.0 and pair_mi > 0.0:
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
        _joint_recovery_ok = _joint_ratio >= _FE_MARGINAL_UPLIFT_STRICT_JOINT_RATIO or (
            _uplift_ratio >= _FE_MARGINAL_UPLIFT_SYNERGY_UPLIFT and _joint_ratio >= _FE_MARGINAL_UPLIFT_MIN_JOINT_RATIO
        )
        if _max_operand_marginal > 0.0 and best_nonprewarp_mi >= _max_operand_marginal * _FE_MARGINAL_UPLIFT_MIN_RATIO and _joint_recovery_ok:
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
    return _marginal_uplift_accept, best_config, best_mi


def _score_one_pair_flips(var_pairs_perf, _resolve_col, _safe_abs_corr, _use_map):
    """Block of _score_one_pair starting at ``for _cfg in var_pairs_perf.keys():``."""
    for _cfg in var_pairs_perf.keys():
        try:
            _cv = _resolve_col(_cfg[2])
            if _cv is not None:
                _use_map[_cfg] = _safe_abs_corr(_cv)
        except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design  # noqa: PERF203 - per-iteration fault isolation is intentional, not a hoisting candidate
            logger.debug("suppressed: %s", e)
            continue


def _score_one_pair_computed_once_per_distinct(_tp, vars_transformations, _op_corr, _transformed_operand_abs_corr, _raw_operand_abs_corr, raw_vars_pair):
    """Block of _score_one_pair starting at ``for _side in (0, 1):``."""
    for _side in (0, 1):
        _opk = _tp[_side] if isinstance(_tp, (tuple, list)) and len(_tp) > _side else None
        if _opk is not None and _opk in vars_transformations:
            _op_corr = max(_op_corr, _transformed_operand_abs_corr(_opk))
        _op_corr = max(_op_corr, _raw_operand_abs_corr(raw_vars_pair[_side]))
    return _op_corr


def _score_one_pair_side(_tp2, vars_transformations, transformed_vars, _abs_corr_zerofill_njit, _win_vals_f64, _max_single_op_corr):
    """Block of _score_one_pair starting at ``for _side in (0, 1):``."""
    for _side in (0, 1):
        _opk = _tp2[_side] if isinstance(_tp2, (tuple, list)) and len(_tp2) > _side else None
        if _opk is not None and _opk in vars_transformations:
            _ov = transformed_vars[:, vars_transformations[_opk]]
            # One-pass njit correlation (bit-equivalent to the previous
            # nan_to_num-then-np.corrcoef - see _abs_corr_zerofill_njit's docstring)
            # instead of 2 full-array nan_to_num allocations + a 2x2 corrcoef matrix.
            _r = _abs_corr_zerofill_njit(_win_vals_f64, np.asarray(_ov, dtype=np.float64))
            _max_single_op_corr = max(_max_single_op_corr, _r)
    return _max_single_op_corr


def _score_one_pair_all_values_were_already(_passes_joint_gate, _prewarp_accept, _marginal_uplift_accept, _usability_accept, fe_min_engineered_mi_prevalence, num_fs_steps, best_config, raw_vars_pair, cols, _gate_ratio, rejection_records, rejection_ledger_out):
    """Block of _score_one_pair starting at ``if not (_passes_joint_gate or _prewarp_accept or _marginal_uplift_acce``."""
    if not (_passes_joint_gate or _prewarp_accept or _marginal_uplift_accept or _usability_accept):
        try:
            _rej_thr = float(fe_min_engineered_mi_prevalence) * (1.0 if num_fs_steps < 1 else 1.025)
            _rej_op = None
            if best_config is not None:
                try:
                    _rej_op = best_config[1]  # binary func name of the best engineered form
                except Exception as e:
                    logger.debug("reading the best-config binary op name failed: %s", e)
                    _rej_op = None
            if not _passes_joint_gate:
                _rej_rec = {
                    "gate": "engineered_mi_prevalence",
                    "candidate": str(raw_vars_pair),
                    "operands": tuple(raw_vars_pair),
                    "operand_names": "(" + ", ".join(str(cols[_i]) for _i in raw_vars_pair) + ")",
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
                    "operand_names": "(" + ", ".join(str(cols[_i]) for _i in raw_vars_pair) + ")",
                    "operator": _rej_op,
                    "observed": float(_gate_ratio),
                    "threshold": _rej_thr,
                    "reason": "marginal_uplift_and_prewarp_declined",
                }
            rejection_records.append(_rej_rec)
            if rejection_ledger_out is not None:
                rejection_ledger_out.append(_rej_rec)
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed: %s", e)
            pass
