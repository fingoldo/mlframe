"""Parallel / serial candidate-scoring branches of ``_confirm_predictor.score_candidates``.

The callables that tests patch on ``_confirm_predictor`` (``evaluate_candidates``, ``evaluate_candidate``, ``handle_best_candidate``,
``_prefill_cond_MIs_gpu``) are looked up there at call time and handed in as arguments, so patching that module keeps working.
"""
from __future__ import annotations

import logging
from timeit import default_timer as timer
from typing import Any, Callable

import numpy as np
from joblib import delayed

from pyutilz.parallel import split_list_into_chunks

from ._internals import sanitize

logger = logging.getLogger(__name__)


def _snapshot_worker_toggles() -> tuple[Any, bool, bool, float, float, float, Any, Any, bool]:
    """Snapshot the main-thread info-theory toggles so the joblib workers receive them as explicit arguments.

    Returns ``(group_mi, use_su, use_jmim, bur_lambda, relaxmrmr_alpha, pid_synergy_bonus, cmi_perm, cpt, mi_miller_madow)``.
    """
    # snapshot main-thread SU/JMIM/BUR
    # toggles. ``threading.local`` storage from info_theory.py does NOT
    # propagate across joblib workers (each worker thread gets its own
    # local namespace), so without explicit forwarding the workers
    # silently read ``False`` / ``0.0`` defaults and those features
    # (SU normalization, JMIM, BUR) become no-ops in the parallel hot
    # path - the common case since backend='threading' is the default.
    from .info_theory import (
        use_su_normalization as _use_su, use_jmim_aggregator as _use_jmim,
        get_bur_lambda as _get_bur, get_relaxmrmr_alpha as _get_relax,
        get_pid_synergy_bonus as _get_pid, get_cmi_perm_stop as _get_cmi,
        use_mi_miller_madow as _use_mm, get_cpt_test as _get_cpt,
    )
    from mlframe.feature_selection.filters.info_theory.shared import get_group_mi as _get_gmi
    return (
        _get_gmi(),
        bool(_use_su()),
        bool(_use_jmim()),
        float(_get_bur()),
        float(_get_relax()),
        float(_get_pid()),
        _get_cmi(),
        _get_cpt(),
        bool(_use_mm()),
    )


def score_candidates_parallel(
    ctx: Any,
    feasible_candidates: list,
    best_gain: float,
    best_candidate: Any,
    expected_gains: np.ndarray,
    evaluate_candidates: Callable[..., Any],
) -> tuple:
    """Score ``feasible_candidates`` on the worker pool and merge the workers' caches / gains back; returns ``(best_gain, best_candidate, run_out_of_time)``."""
    selected_vars = ctx.selected_vars
    n_workers = ctx.n_workers
    use_simple_mode = ctx.use_simple_mode
    cached_MIs = ctx.cached_MIs
    cached_cond_MIs = ctx.cached_cond_MIs
    entropy_cache = ctx.entropy_cache
    workers_pool = ctx.workers_pool
    y = ctx.y
    factors_data = ctx.factors_data
    factors_nbins = ctx.factors_nbins
    factors_names = ctx.factors_names
    classes_y = ctx.classes_y
    freqs_y = ctx.freqs_y
    use_gpu = ctx.use_gpu
    freqs_y_safe = ctx.freqs_y_safe
    partial_gains = ctx.partial_gains
    baseline_npermutations = ctx.baseline_npermutations
    mrmr_relevance_algo = ctx.mrmr_relevance_algo
    mrmr_redundancy_algo = ctx.mrmr_redundancy_algo
    max_veteranes_interactions_order = ctx.max_veteranes_interactions_order
    cached_confident_MIs = ctx.cached_confident_MIs
    max_runtime_mins = ctx.max_runtime_mins
    start_time = ctx.start_time
    min_relevance_gain = ctx.min_relevance_gain
    verbose = ctx.verbose
    ndigits = ctx.ndigits
    random_seed = ctx.random_seed

    run_out_of_time = False

    temp_cached_cond_MIs = sanitize(cached_cond_MIs)
    temp_entropy_cache = sanitize(entropy_cache)
    (
        _gmi_snapshot, _su_snapshot, _jmim_snapshot, _bur_snapshot, _relax_snapshot,
        _pid_snapshot, _cmi_snapshot, _cpt_snapshot, _mm_snapshot,
    ) = _snapshot_worker_toggles()
    res = workers_pool(
        delayed(evaluate_candidates)(
            workload=workload,
            y=y,
            best_gain=best_gain,
            factors_data=factors_data,
            factors_nbins=factors_nbins,
            factors_names=factors_names,
            classes_y=classes_y,
            freqs_y=freqs_y,
            use_gpu=use_gpu,
            freqs_y_safe=freqs_y_safe,
            # Per-worker COPIES of the two dicts the worker writes in-place (unlike cond_MIs/entropy,
            # which each worker re-copies into its own numba dict). Sharing one object let several
            # threading workers __setitem__ it concurrently -> a "dict changed size during iteration"
            # crash at scale. sanitize() is evaluated once per workload so each worker owns its copy;
            # the merge-back below consolidates them (workers own disjoint candidate keys, so the
            # merged result is byte-identical to the old shared-write path).
            partial_gains=sanitize(partial_gains),
            baseline_npermutations=baseline_npermutations,
            mrmr_relevance_algo=mrmr_relevance_algo,
            mrmr_redundancy_algo=mrmr_redundancy_algo,
            max_veteranes_interactions_order=max_veteranes_interactions_order,
            selected_vars=selected_vars,
            cached_MIs=sanitize(cached_MIs),
            cached_confident_MIs=cached_confident_MIs,
            cached_cond_MIs=temp_cached_cond_MIs,
            entropy_cache=temp_entropy_cache,
            max_runtime_mins=max_runtime_mins,
            start_time=start_time,
            min_relevance_gain=min_relevance_gain,
            verbose=verbose,
            ndigits=ndigits,
            use_simple_mode=use_simple_mode,
            use_su=_su_snapshot,
            use_jmim=_jmim_snapshot,
            bur_lambda=_bur_snapshot,
            relaxmrmr_alpha=_relax_snapshot,
            pid_synergy_bonus=_pid_snapshot,
            cmi_perm=_cmi_snapshot,
            cpt=_cpt_snapshot,
            mi_miller_madow=_mm_snapshot,
            group_mi=_gmi_snapshot,
            random_seed=random_seed,
        )
        for workload in split_list_into_chunks(feasible_candidates, max(1, len(feasible_candidates) // n_workers))
    )

    for (
        worker_best_gain,
        worker_best_candidate,
        worker_partial_gains,
        worker_expected_gains,
        worker_cached_MIs,
        worker_cached_cond_MIs,
        worker_entropy_cache,
    ) in res:

        if worker_best_gain > best_gain:
            best_candidate = worker_best_candidate
            best_gain = worker_best_gain

        # sync caches
        for local_storage, global_storage in [
            (worker_expected_gains, expected_gains),
            (worker_cached_MIs, cached_MIs),
            (worker_cached_cond_MIs, cached_cond_MIs),
            (worker_entropy_cache, entropy_cache),
        ]:
            for key, value in local_storage.items():
                global_storage[key] = value

        for cand_idx, (worker_current_gain, worker_z_idx) in worker_partial_gains.items():
            if cand_idx in partial_gains:
                _, z_idx = partial_gains[cand_idx]
            else:
                z_idx = -2
            if worker_z_idx > z_idx:
                partial_gains[cand_idx] = (worker_current_gain, worker_z_idx)

    if use_simple_mode:
        # need to sort all cands by perfs
        pass
    if max_runtime_mins and not run_out_of_time:
        run_out_of_time = (timer() - start_time) > max_runtime_mins * 60
        if run_out_of_time:
            logger.info("Time limit exhausted. Finalizing the search...")

    return best_gain, best_candidate, run_out_of_time


def score_candidates_serial(
    ctx: Any,
    feasible_candidates: list,
    best_gain: float,
    best_candidate: Any,
    expected_gains: np.ndarray,
    cached_jmim_MIs: Any,
    jmim_hit_counter: np.ndarray,
    _prefill_cond_MIs_gpu: Callable[..., Any],
    evaluate_candidate: Callable[..., Any],
    handle_best_candidate: Callable[..., Any],
) -> tuple:
    """Score ``feasible_candidates`` one by one on the calling thread (with the batched-GPU conditional-MI pre-fill); returns ``(best_gain, best_candidate, run_out_of_time)``."""
    selected_vars = ctx.selected_vars
    use_simple_mode = ctx.use_simple_mode
    cached_MIs = ctx.cached_MIs
    cached_cond_MIs = ctx.cached_cond_MIs
    entropy_cache = ctx.entropy_cache
    y = ctx.y
    factors_data = ctx.factors_data
    factors_nbins = ctx.factors_nbins
    factors_names = ctx.factors_names
    classes_y = ctx.classes_y
    classes_y_safe = ctx.classes_y_safe
    freqs_y = ctx.freqs_y
    use_gpu = ctx.use_gpu
    freqs_y_safe = ctx.freqs_y_safe
    partial_gains = ctx.partial_gains
    baseline_npermutations = ctx.baseline_npermutations
    mrmr_relevance_algo = ctx.mrmr_relevance_algo
    mrmr_redundancy_algo = ctx.mrmr_redundancy_algo
    max_veteranes_interactions_order = ctx.max_veteranes_interactions_order
    cached_confident_MIs = ctx.cached_confident_MIs
    max_runtime_mins = ctx.max_runtime_mins
    start_time = ctx.start_time
    min_relevance_gain = ctx.min_relevance_gain
    verbose = ctx.verbose
    ndigits = ctx.ndigits
    random_seed = ctx.random_seed

    run_out_of_time = False

    # Batched-GPU conditional-MI cache pre-fill for the SERIAL path (default ON;
    # env kill-switch MLFRAME_MRMR_GPU_CMI=0). Same win as the parallel wiring in
    # _evaluate_candidates_inner: pre-populate the shared python ``cached_cond_MIs``
    # with batched I(X; Y | Z) so the @njit evaluate_gain loop hits the cache and
    # skips the serial scalar conditional_mi. Unlike the parallel path (which writes
    # a per-worker numba dict to avoid a concurrent "dict changed size" race), the
    # serial loop has NO concurrent iteration, so writing the plain python dict in
    # place BEFORE the loop is race-free. The helper reuses the exact arr2str key
    # format and the same size/HW dispatch + SU/JMIM/order-1 gating; bit-parity
    # kernel => selection unchanged; any failure / non-eligible regime is a silent
    # no-op (scalar path).
    _prefill_cond_MIs_gpu(
        workload=feasible_candidates,
        y=y,
        factors_data=factors_data,
        factors_nbins=factors_nbins,
        selected_vars=selected_vars,
        cached_cond_MIs=cached_cond_MIs,
        use_simple_mode=use_simple_mode,
        mrmr_relevance_algo=mrmr_relevance_algo,
        max_veteranes_interactions_order=max_veteranes_interactions_order,
    )

    for cand_idx, X, nexisting in feasible_candidates:

        current_gain, _sink_reasons = evaluate_candidate(
            cand_idx=cand_idx,
            X=X,
            y=y,
            nexisting=nexisting,
            best_gain=best_gain,
            factors_data=factors_data,
            factors_nbins=factors_nbins,
            factors_names=factors_names,
            classes_y=classes_y,
            classes_y_safe=classes_y_safe,
            freqs_y=freqs_y,
            use_gpu=use_gpu,
            freqs_y_safe=freqs_y_safe,
            partial_gains=partial_gains,
            baseline_npermutations=baseline_npermutations,
            mrmr_relevance_algo=mrmr_relevance_algo,
            mrmr_redundancy_algo=mrmr_redundancy_algo,
            max_veteranes_interactions_order=max_veteranes_interactions_order,
            expected_gains=expected_gains,
            selected_vars=selected_vars,
            cached_MIs=cached_MIs,
            cached_confident_MIs=cached_confident_MIs,
            cached_cond_MIs=cached_cond_MIs,
            cached_jmim_MIs=cached_jmim_MIs,
            jmim_hit_counter=jmim_hit_counter,
            entropy_cache=entropy_cache,
            verbose=verbose,
            ndigits=ndigits,
            use_simple_mode=use_simple_mode,
            random_seed=random_seed,
        )

        best_gain, best_candidate, run_out_of_time = handle_best_candidate(
            current_gain=current_gain,
            best_gain=best_gain,
            X=X,
            best_candidate=best_candidate,
            factors_names=factors_names,
            max_runtime_mins=max_runtime_mins,
            start_time=start_time,
            min_relevance_gain=min_relevance_gain,
            verbose=verbose,
            ndigits=ndigits,
        )

        if run_out_of_time:
            if verbose:
                logger.info("Time limit exhausted. Finalizing the search...")
            break

    return best_gain, best_candidate, run_out_of_time
