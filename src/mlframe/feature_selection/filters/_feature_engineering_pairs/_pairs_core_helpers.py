"""Helpers carved out of ``_pairs_core`` to keep that module under its size budget."""

from __future__ import annotations

from typing import Any

import logging
import os
from itertools import combinations

import numpy as np
import pandas as pd

# A column is near-constant when its std is within this factor of machine epsilon of its largest |value|: its spread is then rounding noise.
_DEGENERATE_REL_TOL = 32.0 * np.finfo(np.float64).eps
from ._pairs_chunks import _FE_CHUNK_MAX_COLS_HARD_CAP
from ._pairs_gates import _GATE_MED_SPECS_RESULT_KEY
from ._pairs_gates import _PREWARP_SPECS_RESULT_KEY
from ._pairs_gates import _prewarp_pair_spec_key


def _short_fe_name(name, maxlen: int = 30) -> str:
    """Truncate a (possibly long engineered) feature expression for live progress-bar
    display, keeping head + tail so both operator and operand stay legible
    (``mul(log(c),sin(d))`` -> ``mul(log(c..sin(d))``). Robust to non-str input."""
    try:
        s = str(name)
    except Exception as e:
        logger.debug("_short_fe_name: str(name) failed for progress-display truncation, showing '?': %s", e)
        return "?"
    if len(s) <= maxlen:
        return s
    head = (maxlen - 2) // 2
    tail = maxlen - 2 - head
    return s[:head] + ".." + s[-tail:]


logger = logging.getLogger(__name__)


def _check_prospective__use_subsample(_use_subsample, _shared_idx, subsample_seed, fe_subsample_stratify, classes_y, subsample_n, _full_n_rows, _sample_idx, _X_full, classes_y_safe, freqs_y, verbose, X):
    """Block of check_prospective_fe_pairs starting at ``if _use_subsample:``."""
    from mlframe.feature_selection.filters.feature_engineering import (
        logger,
    )

    if _use_subsample:
        _sample_idx = _check_prospective__shared_idx_none(_shared_idx, subsample_seed, fe_subsample_stratify, classes_y, subsample_n, _full_n_rows, _sample_idx)
        if isinstance(_X_full, pd.DataFrame):
            X = _X_full.iloc[_sample_idx].reset_index(drop=True)
        else:
            # Polars path - row indexing returns a fresh frame; preserves zero-copy where possible.
            X = _X_full[_sample_idx]
        # Realign per-row target encodings; recompute freqs from the subsampled
        # class labels so MI estimates use the actual subsample distribution
        # rather than the full-n freq table (which would bias the MI estimator
        # toward classes that shrank under the random subset).
        _cy = np.asarray(classes_y)
        _cy_safe = np.asarray(classes_y_safe)
        classes_y = _cy[_sample_idx]
        classes_y_safe = _cy_safe[_sample_idx]
        # Recompute freqs from subsampled class labels. merge_vars returns
        # freqs_y as a FLOAT proportions array (sum=1.0), not raw counts; the
        # subsample needs the same shape. bincount gives counts -> divide by
        # total to get proportions matching the caller's expectation.
        freqs_y = _check_prospective__total_get_proportions_matching(classes_y, freqs_y)
        # else: leave the caller-supplied freqs_y; mi_direct handles its own
        # validation and would crash anyway on a non-integer class table.
        if verbose:
            logger.info(
                "check_prospective_fe_pairs: subsample_n=%d active (full_n=%d, %.1f%% sample); " "MI sweep runs on the subsample, survivor columns rebuilt at full n.",
                int(subsample_n),
                _full_n_rows,
                100.0 * subsample_n / _full_n_rows,
            )
    return X, _sample_idx, classes_y, classes_y_safe, freqs_y


def _check_prospective__behaviour_change_wrapped_experimental(X):
    """Block of check_prospective_fe_pairs starting at ``if fe_matrix_p0_enabled():``."""
    from mlframe.feature_selection.filters._fe_matrix_io import fe_matrix_p0_enabled
    from mlframe.feature_selection.filters.feature_engineering import (
        logger,
    )

    if fe_matrix_p0_enabled():
        try:
            from mlframe.feature_selection.filters._fe_matrix_io import from_feature_matrix, to_feature_matrix

            X = from_feature_matrix(to_feature_matrix(X))
        except Exception:
            logger.warning("FE matrix P-seam round-trip failed; using X unchanged.", exc_info=True)
    return X


def _check_prospective__scores_same_rows_fall(shared_subsample_idx, _full_n_rows, _shared_idx):
    """Block of check_prospective_fe_pairs starting at ``if shared_subsample_idx is not None:``."""
    from mlframe.feature_selection.filters.feature_engineering import (
        logger,
    )

    if shared_subsample_idx is not None:
        try:
            _si = np.asarray(shared_subsample_idx)
            if _si.ndim == 1 and 0 < _si.shape[0] < _full_n_rows and int(_si.max()) < _full_n_rows:
                _shared_idx = _si.astype(np.int64, copy=False)
        except Exception as e:
            logger.debug("shared_subsample_idx validation failed, falling back to no shared subsample: %s", e)
            _shared_idx = None
    return _shared_idx


def _check_prospective__shared_idx_none(_shared_idx, subsample_seed, fe_subsample_stratify, classes_y, subsample_n, _full_n_rows, _sample_idx):
    """Block of check_prospective_fe_pairs starting at ``if _shared_idx is not None:``."""
    if _shared_idx is not None:
        _sample_idx = _shared_idx
    else:
        _rng_sub = np.random.default_rng(int(subsample_seed))
        if fe_subsample_stratify:
            # Stratify on the discretised class codes (classification: balances classes; this path's
            # ``classes_y`` is the discrete target the MI sweep scores against). is_clf=True is correct
            # because ``classes_y`` is always discrete codes here (a continuous y is binned upstream).
            from mlframe.feature_selection.filters._fe_subsample import stratified_subsample_idx

            _sample_idx = stratified_subsample_idx(_rng_sub, np.asarray(classes_y), int(subsample_n), is_clf=True)
        else:
            _sample_idx = np.sort(_rng_sub.choice(_full_n_rows, size=int(subsample_n), replace=False))
    return _sample_idx


def _check_prospective__total_get_proportions_matching(classes_y, freqs_y):
    """Block of check_prospective_fe_pairs starting at ``if classes_y.size > 0 and classes_y.dtype.kind in ("i", "u"):``."""
    if classes_y.size > 0 and classes_y.dtype.kind in ("i", "u"):
        # Without minlength=, a subsample that happens to
        # drop every row of the numerically-highest class label silently under-counts freqs_y.shape[0]
        # (k_y) by the number of missing trailing labels, skewing the MM-debias/tie-band width
        # (fe_mm_debias_prevalence, default OFF) - use the PRE-subsample freqs_y width so k_y never
        # shrinks just because a class got unlucky in the draw.
        _minlength = int(np.asarray(freqs_y).shape[0])
        _counts = np.bincount(classes_y.astype(np.int64), minlength=_minlength)
        _total = _counts.sum()
        if _total > 0:
            freqs_y = _counts.astype(np.float64) / float(_total)
    return freqs_y


def _check_prospective__win_do_re_implement(prospective_pairs, _unary_names_eff, pair_combs, max_n_combs):
    """Block of check_prospective_fe_pairs starting at ``for raw_vars_pair, _ in prospective_pairs.keys():``."""
    for raw_vars_pair, _ in prospective_pairs.keys():
        combs = list(
            combinations(
                [(raw_vars_pair[0], k) for k in _unary_names_eff] + [(raw_vars_pair[1], k) for k in _unary_names_eff],
                2,
            )
        )
        combs = [tp for tp in combs if tp[0][0] != tp[1][0]]
        pair_combs[raw_vars_pair] = combs
        if len(combs) > max_n_combs:
            max_n_combs = len(combs)
    return max_n_combs


def _check_prospective__ignored_thread_multiplication_byte(max_n_combs, X, _n_binary, _n_workers, verbose, _fe_chunk_max_cols, final_transformed_vals_shared):
    """Block of check_prospective_fe_pairs starting at ``if max_n_combs > 0:``."""
    from mlframe.feature_selection.filters.feature_engineering import (
        _FE_BUFFER_RAM_BUDGET_RATIO,
        _can_hoist_shared_buffer,
        _estimate_fe_shared_buffer_bytes,
        _fe_effective_buffer_budget_bytes,
        logger,
    )

    if max_n_combs > 0:
        _buf_bytes = _estimate_fe_shared_buffer_bytes(len(X), max_n_combs, _n_binary)
        _can_hoist, _bb, _avail = _can_hoist_shared_buffer(_buf_bytes, n_workers=_n_workers)
        if _can_hoist:
            try:
                final_transformed_vals_shared = np.empty(
                    shape=(len(X), max_n_combs * _n_binary),
                    dtype=np.float32,
                )
                # Cross-pair chunk width cap: the largest column count whose
                # ``n_rows * cols * 4`` float32 buffer stays inside the overhead+worker-aware
                # envelope (the SUM of the coexisting per-worker buffers fits the SAME 0.4
                # global budget). One pair (``max_n_combs * _n_binary`` cols) already passed,
                # so this is always >= one pair; we pack whole pairs up to it.
                _per_row_bytes = max(1, len(X) * 4)
                _eff_budget_bytes = _fe_effective_buffer_budget_bytes(_avail, n_workers=_n_workers)
                if _eff_budget_bytes >= 0:
                    _fe_chunk_max_cols = int(_eff_budget_bytes // _per_row_bytes)
                else:
                    # No psutil reading -> bound the chunk to the hard cap only.
                    _fe_chunk_max_cols = _FE_CHUNK_MAX_COLS_HARD_CAP
                _fe_chunk_max_cols = max(
                    max_n_combs * _n_binary,
                    min(_fe_chunk_max_cols, _FE_CHUNK_MAX_COLS_HARD_CAP),
                )
            except MemoryError:
                # psutil over-reported available; falling back stays safe.
                final_transformed_vals_shared = None
                if verbose:
                    logger.warning(
                        "check_prospective_fe_pairs: shared buffer (%.1f GiB) allocation raised "
                        "MemoryError despite passing the available-RAM check (%.1f GiB available, "
                        "%.0f%% budget); switching to recompute-from-metadata fallback (~1%% extra "
                        "bin_func calls per pair).",
                        _bb / 2**30,
                        _avail / 2**30 if _avail >= 0 else float("nan"),
                        _FE_BUFFER_RAM_BUDGET_RATIO * 100.0,
                    )
        else:
            if verbose:
                logger.warning(
                    "check_prospective_fe_pairs: shared buffer would need %.1f GiB but only %.1f GiB "
                    "RAM is available (%.0f%% budget = %.1f GiB cap); using recompute-from-metadata "
                    "fallback path (~1%% extra bin_func calls per pair, identical survivors). To force "
                    "the fast path either free RAM or raise _FE_BUFFER_RAM_BUDGET_RATIO; to bound "
                    "compute, pass subsample_n>0 from the MRMR config.",
                    _bb / 2**30,
                    _avail / 2**30 if _avail >= 0 else float("nan"),
                    _FE_BUFFER_RAM_BUDGET_RATIO * 100.0,
                    (_avail * _FE_BUFFER_RAM_BUDGET_RATIO) / 2**30 if _avail >= 0 else float("nan"),
                )
    return _fe_chunk_max_cols, final_transformed_vals_shared


def _check_prospective__classes_codes_still_usable(usability_y_continuous, prewarp_y_continuous, classes_y, _use_subsample, _full_n_rows, _sample_idx, _corr_y_cont, _corr_y_cont_finite):
    """Block of check_prospective_fe_pairs starting at ``try:``."""
    from mlframe.feature_selection.filters.feature_engineering import (
        logger,
    )

    try:
        _cyc_src = usability_y_continuous if usability_y_continuous is not None else (prewarp_y_continuous if prewarp_y_continuous is not None else classes_y)
        _cyc = np.asarray(_cyc_src, dtype=np.float64).ravel()
        if _use_subsample and _cyc.shape[0] == _full_n_rows:
            _cyc = _cyc[_sample_idx]
        if _cyc.shape[0] == len(classes_y) and np.isfinite(_cyc).any() and float(np.nanstd(_cyc)) > 1e-12:
            _corr_y_cont = _cyc
            _corr_y_cont_finite = np.isfinite(_corr_y_cont)
    except Exception as e:
        logger.debug("y-continuous correlation prep failed, skipping that signal: %s", e)
        _corr_y_cont = None
        _corr_y_cont_finite = None
    return _corr_y_cont, _corr_y_cont_finite


def _check_prospective__only_worth_chunking_least(_fe_chunks, _chunk_buf_width, X, _pair_to_chunk, verbose, prospective_pairs, _chunk_buffer, _chunk_global_batch):
    """Block of check_prospective_fe_pairs starting at ``if _fe_chunks and max(len(c) for c in _fe_chunks) > 1 and _chunk_buf_w``."""
    from mlframe.feature_selection.filters.feature_engineering import (
        logger,
    )

    if _fe_chunks and max(len(c) for c in _fe_chunks) > 1 and _chunk_buf_width > 0:
        try:
            _chunk_buffer = np.empty((len(X), _chunk_buf_width), dtype=np.float32)
            for _ci_chunk, _chunk in enumerate(_fe_chunks):
                for _p in _chunk:
                    _pair_to_chunk[_p] = _ci_chunk
            if verbose:
                logger.info(
                    "check_prospective_fe_pairs: cross-pair chunk batching active " "(%d pairs -> %d chunks, buffer %d cols, widest chunk %d pairs).",
                    len(prospective_pairs),
                    len(_fe_chunks),
                    _chunk_buf_width,
                    max(len(c) for c in _fe_chunks),
                )
        except MemoryError:
            _chunk_buffer = None
            _pair_to_chunk = {}
            if verbose:
                logger.warning(
                    "check_prospective_fe_pairs: cross-pair chunk buffer (%d x %d) raised " "MemoryError; using per-pair batching.",
                    len(X),
                    _chunk_buf_width,
                )
    else:
        _chunk_global_batch = False
    return _chunk_buffer, _chunk_global_batch, _pair_to_chunk


def _check_prospective__read_weakref_cache_no(_chunk_global_batch, _chunk_buffer, _fe_chunks, X, _chunk_buf_width, transformed_vars, _chunk_state, verbose):
    """Block of check_prospective_fe_pairs starting at ``if _chunk_global_batch and _chunk_buffer is not None and len(_fe_chunk``."""
    from mlframe.feature_selection.filters.feature_engineering import (
        logger,
    )

    if _chunk_global_batch and _chunk_buffer is not None and len(_fe_chunks) >= 2:
        _pipe_env = os.environ.get("MLFRAME_FE_PIPELINE_CHUNKS", "1").strip().lower() in ("1", "true", "on", "yes")
        _pipe_on = False
        if _pipe_env:
            try:
                from mlframe.feature_selection.filters._fe_gpu_strict import fe_gpu_strict_enabled

                _pipe_on = bool(fe_gpu_strict_enabled(n=len(X), p=int(_chunk_buf_width)))
            except Exception as e:
                logger.debug("fe_gpu_strict_enabled() check failed, defaulting _pipe_on to False: %s", e)
                _pipe_on = False
        if _pipe_on:
            try:
                import cupy as _pl_cp
                from concurrent.futures import ThreadPoolExecutor

                _chunk_buffer2 = np.empty_like(_chunk_buffer)
                from mlframe.feature_selection.filters._gpu_resident_fe import _resident_operand_table  # type: ignore[attr-defined]  # dynamically re-exported via globals()

                _resident_operand_table(_pl_cp, transformed_vars)  # pre-warm: both threads then only read
                _chunk_state["pipeline_buffers"] = [_chunk_buffer, _chunk_buffer2]
                _chunk_state["pipeline_ex"] = ThreadPoolExecutor(max_workers=1)
                _chunk_state["pipeline_futures"] = {}
                if verbose:
                    logger.info("check_prospective_fe_pairs: chunk pipeline active (%d chunks, double buffer).", len(_fe_chunks))
            except Exception:
                logger.debug("chunk pipeline setup failed; synchronous chunk path", exc_info=True)
                _chunk_state.pop("pipeline_buffers", None)
                _ex0 = _chunk_state.pop("pipeline_ex", None)
                if _ex0 is not None:
                    _ex0.shutdown(wait=False)


def _check_prospective__pair_res_entry_none(_pair_res_entry, res, raw_vars_pair, best_config, cols, pair_mi, _rejection_records, rejection_ledger_out):
    """Block of check_prospective_fe_pairs starting at ``if _pair_res_entry is not None:``."""
    from mlframe.feature_selection.filters.feature_engineering import (
        logger,
    )

    if _pair_res_entry is not None:
        res[raw_vars_pair] = _pair_res_entry
    elif best_config is None:
        # A pair that produced NO candidate at all leaves no trace otherwise: the rejection ledger only
        # records candidates that were built and then failed a gate, so a pair whose operator search
        # emits nothing is invisible in the fitted object and can only be found by instrumenting the
        # search by hand. That is the exact shape behind the open FE-recovery findings in this audit -
        # the (c,d) and (x0,x1) pairs each carry the signal, are eligible, and never appear anywhere.
        try:
            _barren = {
                "gate": "pair_candidate_generation",
                "candidate": f"({cols[raw_vars_pair[0]]},{cols[raw_vars_pair[1]]})",
                "operands": tuple(cols[i] for i in raw_vars_pair),
                "operator": "",
                "observed": float(pair_mi) if pair_mi is not None else float("nan"),
                "threshold": float("nan"),
                "reason": "no candidate produced for this pair",
            }
            _rejection_records.append(_barren)
            if rejection_ledger_out is not None:
                rejection_ledger_out.append(_barren)
        except Exception as e:  # nosec B110 - instrumentation must never break the FE search
            logger.debug("barren-pair ledger record failed: %s", e)


def _check_prospective__no_config_nan_edge(verbose, best_mi, best_config, _sweep_best_mi, cols, raw_vars_pair, pair_pbar):
    """Block of check_prospective_fe_pairs starting at ``if verbose:``."""
    _sweep_best_name: Any = None
    from mlframe.feature_selection.filters.feature_engineering import (
        get_new_feature_name,
    )

    if verbose:
        try:
            _bm = float(best_mi)
            if best_config is not None and np.isfinite(_bm) and _bm > _sweep_best_mi:
                _sweep_best_mi = _bm
                _sweep_best_name = get_new_feature_name(fe_tuple=best_config, cols_names=cols)
            _cur_pair = f"{cols[raw_vars_pair[0]]},{cols[raw_vars_pair[1]]}"
            _pf = {"pair": _short_fe_name(_cur_pair, 22)}
            if _sweep_best_name is not None:
                _pf["best"] = f"{_short_fe_name(_sweep_best_name)}={_sweep_best_mi:.4f}"
            pair_pbar.set_postfix(_pf, refresh=False)
        except (TypeError, ValueError, IndexError):
            pass


def _check_prospective__pair_scoped_copies_spec(prospective_pairs, _prewarp_spec_by_var, _fitted_specs):
    """Block of check_prospective_fe_pairs starting at ``for _rvp, _ in prospective_pairs.keys():``."""
    for _rvp, _ in prospective_pairs.keys():
        for _v in _rvp:
            if _prewarp_spec_by_var.get(_v) is not None:
                _fitted_specs[_prewarp_pair_spec_key(_rvp, _v)] = _prewarp_spec_by_var[_v]


def _check_prospective__fitted_specs(_fitted_specs, prewarp_specs_out, res):
    """Block of check_prospective_fe_pairs starting at ``if _fitted_specs:``."""
    if _fitted_specs:
        if prewarp_specs_out is not None:
            prewarp_specs_out.update(_fitted_specs)
        res[_PREWARP_SPECS_RESULT_KEY] = _fitted_specs


def _check_prospective__value_single_float_per(_fitted_medians, gate_med_specs_out, res):
    """Block of check_prospective_fe_pairs starting at ``if _fitted_medians:``."""
    if _fitted_medians:
        if gate_med_specs_out is not None:
            gate_med_specs_out.update(_fitted_medians)
        res[_GATE_MED_SPECS_RESULT_KEY] = _fitted_medians
