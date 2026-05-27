"""Single-predictor confirmation primitives, carved out of
``mlframe.feature_selection.filters._screen_predictors`` so that file stays well below the
1k-line monolith threshold and the (most-frequently-patched) confirmation math lives in one place:

* :func:`score_candidates`     -- evaluate every feasible candidate's conditional-MI
                                  gain given the current ``selected_vars`` (serial +
                                  joblib-parallel paths), filling ``expected_gains`` /
                                  ``partial_gains`` and returning the running best.
* :func:`confirm_candidate`    -- permutation-confirm a single candidate ``X`` (direct
                                  MI bootstrap + Fleuret conditional recheck), returning
                                  ``(bootstrapped_gain, confidence)``.
* :func:`confirm_one_predictor`-- the full single-predictor confirmation cycle (the
                                  ``while True`` loop with lexsort tiebreak, partial-gain
                                  recompute/retry, and patience accounting), composed from
                                  the two primitives above.

``screen_predictors`` calls :func:`confirm_one_predictor` once per added predictor.

All shared/static state (data arrays, algorithm parameters, the four MI caches, and
the per-interactions-order mutable collections) is bundled in :class:`ScreenContext`;
per-predictor scalars (``best_gain`` / ``best_candidate`` / ``expected_gains`` /
``confidence`` / accumulators) are threaded as call arguments and return values so the
moved bodies stay byte-for-byte identical to the original inline blocks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from timeit import default_timer as timer
from typing import Sequence

import numpy as np
from joblib import delayed

from pyutilz.parallel import split_list_into_chunks

from ._internals import MAX_ITERATIONS_TO_TRACK, NMAX_NONPARALLEL_ITERS, sanitize
from ._numba_utils import count_cand_nbins
from .evaluation import (
    evaluate_candidate,
    evaluate_candidates,
    find_best_partial_gain,
    get_candidate_name,
    handle_best_candidate,
    should_skip_candidate,
)
from .fleuret import get_fleuret_criteria_confidence, get_fleuret_criteria_confidence_parallel
from .gpu import mi_direct_gpu
from .permutation import mi_direct

logger = logging.getLogger(__name__)


@dataclass
class ScreenContext:
    """Bundle of shared/static screening state threaded into the confirmation primitives.

    Static fields (data, algorithm parameters, thresholds) are set once by the orchestrator;
    the four caches are mutated in place across candidates; the per-order fields
    (``candidates`` .. ``failed_candidates``) are reassigned by the caller between predictors
    and interaction orders.
    """

    # --- data / target ---
    factors_data: np.ndarray
    factors_nbins: Sequence[int]
    factors_names: Sequence[str]
    y: Sequence[int]
    data_copy: np.ndarray
    classes_y: np.ndarray
    classes_y_safe: object
    freqs_y: np.ndarray
    freqs_y_safe: object
    # --- algorithm params ---
    mrmr_relevance_algo: str
    mrmr_redundancy_algo: str
    reduce_gain_on_subelement_chosen: bool
    max_veteranes_interactions_order: int
    only_unknown_interactions: bool
    use_gpu: bool
    use_simple_mode: bool
    extra_x_shuffling: bool
    engineered_lineage: object
    # --- performance ---
    n_workers: int
    workers_pool: object
    parallel_kwargs: dict
    # --- confidence / stopping ---
    baseline_npermutations: int
    full_npermutations: int
    min_nonzero_confidence: float
    max_failed: int
    min_relevance_gain: float
    max_consec_unconfirmed: int
    max_runtime_mins: float
    max_confirmation_cand_nbins: int
    random_seed: int
    # --- misc ---
    verbose: int
    ndigits: int
    start_time: float
    num_possible_candidates: int
    # --- shared MI caches (mutated in place) ---
    cached_MIs: dict
    cached_confident_MIs: dict
    cached_cond_MIs: object
    entropy_cache: object
    # --- per-interactions-order / per-node mutable state ---
    candidates: list = None
    interactions_order: int = 1
    selected_vars: list = None
    selected_interactions_vars: list = None
    partial_gains: dict = None
    added_candidates: set = field(default=None)
    failed_candidates: set = field(default=None)


def score_candidates(ctx: ScreenContext, best_gain: float, best_candidate, expected_gains: np.ndarray):
    """Evaluate every feasible candidate's gain given ``ctx.selected_vars``.

    Mutates ``expected_gains`` (dense, keyed by global candidate index), ``ctx.partial_gains``,
    and the MI caches in place. Returns ``(best_gain, best_candidate, run_out_of_time)``. This is
    the verbatim scoring block from the original ``screen_predictors`` confirmation loop.
    """

    # Alias shared/static state so the moved body stays identical to the original inline block.
    candidates = ctx.candidates
    interactions_order = ctx.interactions_order
    only_unknown_interactions = ctx.only_unknown_interactions
    failed_candidates = ctx.failed_candidates
    added_candidates = ctx.added_candidates
    selected_vars = ctx.selected_vars
    selected_interactions_vars = ctx.selected_interactions_vars
    engineered_lineage = ctx.engineered_lineage
    reduce_gain_on_subelement_chosen = ctx.reduce_gain_on_subelement_chosen
    n_workers = ctx.n_workers
    use_simple_mode = ctx.use_simple_mode
    cached_MIs = ctx.cached_MIs
    num_possible_candidates = ctx.num_possible_candidates
    cached_cond_MIs = ctx.cached_cond_MIs
    entropy_cache = ctx.entropy_cache
    workers_pool = ctx.workers_pool
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

    run_out_of_time = False

    if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
        eval_start = timer()

    feasible_candidates = []
    for cand_idx, X in enumerate(candidates):
        skip, nexisting = should_skip_candidate(
            cand_idx=cand_idx,
            X=X,
            interactions_order=interactions_order,
            only_unknown_interactions=only_unknown_interactions,
            failed_candidates=failed_candidates,
            added_candidates=added_candidates,
            expected_gains=expected_gains,
            selected_vars=selected_vars,
            selected_interactions_vars=selected_interactions_vars,
            engineered_lineage=engineered_lineage,
        )
        if skip:
            continue

        feasible_candidates.append((cand_idx, X, nexisting if reduce_gain_on_subelement_chosen else 0))

    if (
        n_workers > 1
        and (not use_simple_mode or len(cached_MIs) < num_possible_candidates)
        and len(feasible_candidates) > NMAX_NONPARALLEL_ITERS
    ):
        temp_cached_cond_MIs = sanitize(cached_cond_MIs)
        temp_entropy_cache = sanitize(entropy_cache)
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
                partial_gains=partial_gains,
                baseline_npermutations=baseline_npermutations,
                mrmr_relevance_algo=mrmr_relevance_algo,
                mrmr_redundancy_algo=mrmr_redundancy_algo,
                max_veteranes_interactions_order=max_veteranes_interactions_order,
                selected_vars=selected_vars,
                cached_MIs=cached_MIs,
                cached_confident_MIs=cached_confident_MIs,
                cached_cond_MIs=temp_cached_cond_MIs,
                entropy_cache=temp_entropy_cache,
                max_runtime_mins=max_runtime_mins,
                start_time=start_time,
                min_relevance_gain=min_relevance_gain,
                verbose=verbose,
                ndigits=ndigits,
                use_simple_mode=use_simple_mode,
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
                    current_gain, z_idx = partial_gains[cand_idx]
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

    else:
        if use_simple_mode and False:
            # No need to check every can out of order: let's just return next best known candidate
            best_gain, best_candidate, run_out_of_time = 1, 1, False
        else:
            for cand_idx, X, nexisting in feasible_candidates:

                current_gain, sink_reasons = evaluate_candidate(
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
                    entropy_cache=entropy_cache,
                    verbose=verbose,
                    ndigits=ndigits,
                    use_simple_mode=use_simple_mode,
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

    if verbose > 2 and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
        logger.info("evaluate_candidates took %.1f sec.", timer() - eval_start)

    return best_gain, best_candidate, run_out_of_time


def confirm_candidate(ctx: ScreenContext, X: tuple, next_best_gain: float):
    """Permutation-confirm a single candidate ``X`` given ``ctx.selected_vars``.

    Direct-MI bootstrap (cached in ``cached_confident_MIs[X]``) plus the optional Fleuret
    conditional recheck. Returns ``(bootstrapped_gain, confidence)``. Verbatim port of the
    original confirmation block.
    """

    # Alias shared/static state.
    full_npermutations = ctx.full_npermutations
    selected_vars = ctx.selected_vars
    use_simple_mode = ctx.use_simple_mode
    cached_confident_MIs = ctx.cached_confident_MIs
    use_gpu = ctx.use_gpu
    factors_data = ctx.factors_data
    factors_nbins = ctx.factors_nbins
    factors_names = ctx.factors_names
    classes_y = ctx.classes_y
    classes_y_safe = ctx.classes_y_safe
    freqs_y = ctx.freqs_y
    freqs_y_safe = ctx.freqs_y_safe
    min_nonzero_confidence = ctx.min_nonzero_confidence
    n_workers = ctx.n_workers
    workers_pool = ctx.workers_pool
    parallel_kwargs = ctx.parallel_kwargs
    max_confirmation_cand_nbins = ctx.max_confirmation_cand_nbins
    data_copy = ctx.data_copy
    y = ctx.y
    max_failed = ctx.max_failed
    mrmr_relevance_algo = ctx.mrmr_relevance_algo
    mrmr_redundancy_algo = ctx.mrmr_redundancy_algo
    max_veteranes_interactions_order = ctx.max_veteranes_interactions_order
    cached_cond_MIs = ctx.cached_cond_MIs
    entropy_cache = ctx.entropy_cache
    extra_x_shuffling = ctx.extra_x_shuffling
    random_seed = ctx.random_seed
    verbose = ctx.verbose
    ndigits = ctx.ndigits

    if full_npermutations:

        if verbose > 2:
            logger.info(
                "confirming candidate %s, next_best_gain=%.*f",
                get_candidate_name(X, factors_names=factors_names), ndigits, next_best_gain,
            )

        if X in cached_confident_MIs:
            bootstrapped_gain, confidence = cached_confident_MIs[X]
        else:
            if use_gpu:
                bootstrapped_gain, confidence = mi_direct_gpu(
                    factors_data,
                    x=X,
                    y=y,
                    factors_nbins=factors_nbins,
                    classes_y=classes_y,
                    freqs_y=freqs_y,
                    freqs_y_safe=freqs_y_safe,
                    classes_y_safe=classes_y_safe,
                    min_nonzero_confidence=min_nonzero_confidence,
                    npermutations=full_npermutations,
                )
            else:
                if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                    eval_start = timer()
                bootstrapped_gain, confidence = mi_direct(
                    factors_data,
                    x=X,
                    y=y,
                    factors_nbins=factors_nbins,
                    classes_y=classes_y,
                    freqs_y=freqs_y,
                    classes_y_safe=classes_y_safe,
                    min_nonzero_confidence=min_nonzero_confidence,
                    npermutations=full_npermutations,
                    n_workers=n_workers,
                    workers_pool=workers_pool,
                    parallel_kwargs=parallel_kwargs,
                )
                if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                    logger.info("mi_direct bootstrapped eval took %.1f sec.", timer() - eval_start)
            cached_confident_MIs[X] = bootstrapped_gain, confidence
    else:
        if X in cached_confident_MIs:
            bootstrapped_gain, confidence = cached_confident_MIs[X]
        else:
            bootstrapped_gain, confidence = next_best_gain, 1.0

    if full_npermutations and bootstrapped_gain > 0 and selected_vars and not use_simple_mode:  # additional check of Fleuret criteria

        if count_cand_nbins(X, factors_nbins) <= max_confirmation_cand_nbins:

            skip_cand = [(subel in selected_vars) for subel in X]
            nexisting = sum(skip_cand)

            if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                eval_start = timer()

            _fleuret_base_seed = int(((int(random_seed or 0) * 2654435761) + len(selected_vars) + 1) & 0xFFFFFFFFFFFFFFFF)
            if n_workers and n_workers > 1 and full_npermutations > NMAX_NONPARALLEL_ITERS:
                bootstrapped_gain, confidence, parallel_entropy_cache = get_fleuret_criteria_confidence_parallel(
                    data_copy=data_copy,
                    factors_nbins=factors_nbins,
                    x=X,
                    y=y,
                    selected_vars=selected_vars,
                    bootstrapped_gain=next_best_gain,
                    npermutations=full_npermutations,
                    max_failed=max_failed,
                    nexisting=nexisting,
                    mrmr_relevance_algo=mrmr_relevance_algo,
                    mrmr_redundancy_algo=mrmr_redundancy_algo,
                    max_veteranes_interactions_order=max_veteranes_interactions_order,
                    cached_cond_MIs=cached_cond_MIs,
                    entropy_cache=entropy_cache,
                    extra_x_shuffling=extra_x_shuffling,
                    n_workers=n_workers,
                    workers_pool=workers_pool,
                    parallel_kwargs=parallel_kwargs,
                    base_seed=_fleuret_base_seed,
                )
                for key, value in parallel_entropy_cache.items():
                    entropy_cache[key] = value
            else:
                nfailed, nchecked = get_fleuret_criteria_confidence(
                    data_copy=data_copy,
                    factors_nbins=factors_nbins,
                    x=X,
                    y=y,
                    selected_vars=selected_vars,
                    bootstrapped_gain=next_best_gain,
                    npermutations=full_npermutations,
                    max_failed=max_failed,
                    nexisting=nexisting,
                    mrmr_relevance_algo=mrmr_relevance_algo,
                    mrmr_redundancy_algo=mrmr_redundancy_algo,
                    max_veteranes_interactions_order=max_veteranes_interactions_order,
                    cached_cond_MIs=cached_cond_MIs,
                    entropy_cache=entropy_cache,
                    extra_x_shuffling=extra_x_shuffling,
                    base_seed=np.uint64(_fleuret_base_seed),
                )
                confidence = 1 - nfailed / nchecked
                if nfailed >= max_failed:
                    bootstrapped_gain = 0.0

            if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                logger.info("get_fleuret_criteria_confidence bootstrapped eval took %.1f sec.", timer() - eval_start)

    return bootstrapped_gain, confidence


def confirm_one_predictor(
    ctx: ScreenContext,
    nconsec_unconfirmed: int,
    total_checked: int,
    total_disproved: int,
    patience_triggered: bool,
):
    """Run one full single-predictor confirmation cycle (the original ``while True`` loop).

    Re-scores all feasible candidates, then permutation-confirms them in expected-gain order
    (with the partial-gain recompute/retry + patience logic), composing :func:`score_candidates`
    and :func:`confirm_candidate`. Returns
    ``(best_candidate, best_gain, confidence, run_out_of_time, nconsec_unconfirmed,
    total_checked, total_disproved, patience_triggered)``.
    """

    candidates = ctx.candidates
    min_relevance_gain = ctx.min_relevance_gain
    selected_vars = ctx.selected_vars
    failed_candidates = ctx.failed_candidates
    added_candidates = ctx.added_candidates
    partial_gains = ctx.partial_gains
    full_npermutations = ctx.full_npermutations
    max_consec_unconfirmed = ctx.max_consec_unconfirmed
    factors_names = ctx.factors_names
    verbose = ctx.verbose
    ndigits = ctx.ndigits

    best_candidate = None
    best_gain = min_relevance_gain - 1
    expected_gains = np.zeros(len(candidates), dtype=np.float64)
    confidence = 1.0
    run_out_of_time = False

    while True:  # confirmation loop (by random permutations)

        best_gain, best_candidate, run_out_of_time = score_candidates(ctx, best_gain, best_candidate, expected_gains)

        if run_out_of_time:
            break

        if best_gain < min_relevance_gain:
            if verbose >= 2:
                logger.info("Minimum expected gain reached or no candidates to check anymore.")
            break  # exit confirmation while loop

        # ---------------------------------------------------------------------------------------------------------------
        # Now need to confirm best expected gain with a permutation test
        # ---------------------------------------------------------------------------------------------------------------

        cand_confirmed = False
        any_cand_considered = False
        for n, next_best_candidate_idx in enumerate(
            np.lexsort((np.arange(len(expected_gains)), -np.asarray(expected_gains)))
        ):
            next_best_gain = expected_gains[next_best_candidate_idx]
            if next_best_gain >= min_relevance_gain:  # only can consider here candidates fully checked against every Z

                X = candidates[next_best_candidate_idx]

                if n > 0:
                    best_partial_gain, best_key = find_best_partial_gain(
                        partial_gains=partial_gains,
                        failed_candidates=failed_candidates,
                        added_candidates=added_candidates,
                        candidates=candidates,
                        selected_vars=selected_vars,
                    )

                    if best_partial_gain > next_best_gain:
                        best_gain = next_best_gain
                        if verbose > 2:
                            print(
                                "Have no best_candidate anymore. Need to recompute partial gains. best_partial_gain of candidate",
                                get_candidate_name(candidates[best_key], factors_names=factors_names),
                                "was",
                                best_partial_gain,
                            )
                        break  # out of best candidates confirmation, to retry all cands evaluation

                any_cand_considered = True

                if full_npermutations:
                    total_checked += 1

                bootstrapped_gain, confidence = confirm_candidate(ctx, X, next_best_gain)

                if bootstrapped_gain > 0:

                    nconsec_unconfirmed = 0

                    next_best_gain = next_best_gain * confidence
                    expected_gains[next_best_candidate_idx] = next_best_gain

                    best_partial_gain, best_key = find_best_partial_gain(
                        partial_gains=partial_gains,
                        failed_candidates=failed_candidates,
                        added_candidates=added_candidates,
                        candidates=candidates,
                        selected_vars=selected_vars,
                        skip_indices=(next_best_candidate_idx,),
                    )
                    if best_partial_gain > next_best_gain:
                        if verbose > 2:
                            logger.info(
                                "\t\tCandidate's lowered confidence %s requires re-checking other candidates, "
                                "as now its expected gain is only %.*f, vs %.*f, of %s",
                                confidence, ndigits, next_best_gain, ndigits, best_partial_gain,
                                get_candidate_name(candidates[best_key], factors_names=factors_names),
                            )
                        break  # out of best candidates confirmation, to retry all cands evaluation
                    else:
                        cand_confirmed = True
                        if full_npermutations:
                            if verbose > 2:
                                logger.info("\t\tconfirmed with confidence %.*f", ndigits, confidence)
                        break  # out of best candidates confirmation, to add candidate to the list, and go to more candidates
                else:
                    expected_gains[next_best_candidate_idx] = 0.0
                    failed_candidates.add(next_best_candidate_idx)
                    if verbose > 2:
                        logger.info("\t\tconfirmation failed with confidence %.*f", ndigits, confidence)

                    nconsec_unconfirmed += 1
                    total_disproved += 1
                    if max_consec_unconfirmed and (nconsec_unconfirmed > max_consec_unconfirmed):
                        patience_triggered = True
                        if verbose:
                            logger.info("Maximum consecutive confirmation failures reached.")
                        break  # out of best candidates confirmation, to finish the level

            else:  # next_best_gain = 0
                break  # nothing wrong, just retry all cands evaluation

        # ---------------------------------------------------------------------------------------------------------------
        # Let's act upon results of the permutation test
        # ---------------------------------------------------------------------------------------------------------------

        if cand_confirmed:
            added_candidates.add(next_best_candidate_idx)  # so it won't be selected again
            best_candidate = X
            best_gain = next_best_gain
            break  # exit confirmation while loop
        else:
            if not any_cand_considered:
                best_gain = min_relevance_gain - 1
                if verbose:
                    logger.info("No more candidates to confirm.")
                break  # exit confirmation while loop
            else:
                best_gain = min_relevance_gain - 1
                if max_consec_unconfirmed and (nconsec_unconfirmed > max_consec_unconfirmed):
                    patience_triggered = True
                    break  # exit confirmation while loop
                else:
                    pass  # retry all cands evaluation

    return (
        best_candidate,
        best_gain,
        confidence,
        run_out_of_time,
        nconsec_unconfirmed,
        total_checked,
        total_disproved,
        patience_triggered,
    )
