"""Serial candidate-scoring branch of ``_confirm_predictor.score_candidates``.

The callables that tests patch on ``_confirm_predictor`` (``evaluate_candidates``, ``evaluate_candidate``, ``handle_best_candidate``,
``_prefill_cond_MIs_gpu``) are looked up there at call time and handed in as arguments, so patching that module keeps working.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


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
