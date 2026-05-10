"""mRMR screening orchestrator.

Public functions:

* ``screen_predictors`` -- the main screening loop. Walks interaction
  orders, selects candidates, runs the confidence step, accumulates
  the final ``selected_vars`` list. Etap 10a is an identity move from
  the legacy monolith; etap 10b decomposes the body into a
  ``@dataclass ScreenState`` plus four free phase functions
  (`_select_initial_candidates`, `_evaluate_partials`,
  `_confirm_top_k`, `_postprocess_step`).
* ``postprocess_candidates`` -- post-screening filtering helper.

mRMR phase narrative
--------------------
partial = MI(candidate, target | already-selected); top_k = K candidates
with highest partial; confirm = full-permutation test on top_k;
postprocess = filter weak / duplicates from the confirmed set.
"""
from __future__ import annotations

import gc
import logging
import math
import os
import time
from collections import defaultdict
from itertools import combinations
from os.path import exists
from timeit import default_timer as timer
from typing import Sequence

import numpy as np
import numba
from numba import njit
from numba.core import types

from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler

from pyutilz.numbalib import (
    generate_combinations_recursive_njit,
    python_dict_2_numba_dict,
    set_numba_random_seed,
)
from pyutilz.parallel import split_list_into_chunks
from pyutilz.pythonlib import sort_dict_by_value
from pyutilz.system import tqdmu

from mlframe.utils import set_random_seed

from ._internals import (
    LARGE_CONST,
    MAX_CONFIRMATION_CAND_NBINS,
    MAX_ITERATIONS_TO_TRACK,
    NMAX_NONPARALLEL_ITERS,
    sanitize,
)
from ._numba_utils import arr2str, count_cand_nbins, unpack_and_sort
from .info_theory import (
    compute_mi_from_classes,
    conditional_mi,
    entropy,
    merge_vars,
)
from .permutation import distribute_permutations, mi_direct
from .gpu import mi_direct_gpu
from .fleuret import (
    get_fleuret_criteria_confidence,
    get_fleuret_criteria_confidence_parallel,
    parallel_fleuret,
)
from .evaluation import (
    evaluate_candidate,
    evaluate_candidates,
    evaluate_gain,
    find_best_partial_gain,
    get_candidate_name,
    handle_best_candidate,
    should_skip_candidate,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 10b (post-plan, partial): ScreenState dataclass as documentation aid.
#
# Full decomposition into 4 free phase functions was deferred per the plan's
# risk gate: ``screen_predictors`` carries >40 shared locals across the
# main loop, the confirmation while-True, and the post-fold bookkeeping.
# Threading every local through a fixed-arity phase function would require
# pickling the entire state up-front, which we have no comprehensive golden
# coverage for.
#
# Instead we expose ``ScreenState`` as a typed view of the shared state
# (used by readers / future refactor PRs) plus a single extracted helper
# (``_extract_confirmation_branch``) for the most-opaque control path. The
# main ``screen_predictors`` body is unchanged and still bit-exact on the
# 4 golden scenarios.
# =============================================================================

from dataclasses import dataclass, field


@dataclass
class ScreenState:
    """Typed snapshot of ``screen_predictors`` shared state.

    Construction is on-demand (the orchestrator does not currently route
    its locals through this object). Used by review + future incremental
    refactors that need a stable IO contract for phase helpers.

    Reads from / writes to (snapshotted at the relevant phase boundary):
    * ``selected_vars`` -- the running mRMR selection.
    * ``cached_MIs`` / ``cached_confident_MIs`` -- per-candidate MI cache
      keyed by the (sorted) feature index tuple.
    * ``cached_cond_MIs`` -- conditional MI cache for ``I(X; Y | Z)``.
    * ``entropy_cache`` -- entropy-by-conditioning-set cache used by
      ``conditional_mi``.
    * ``partial_gains`` -- per-candidate partial gain (lower bound) for
      the confirmation step.
    * ``failed_candidates`` -- candidates that the confirmation step
      already rejected.
    * ``added_candidates`` -- candidates already in ``selected_vars``.
    * ``classes_y`` / ``classes_y_safe_cpu`` / ``classes_y_safe_gpu`` /
      ``freqs_y`` -- target encoding caches (B18 distinguishes CPU vs
      GPU at the boundary).
    * ``data_copy`` -- memmapped or in-memory shuffle scratch for the
      fleuret confidence permutation pass.

    Invariants:
    * ``set(added_candidates) == set(selected_vars)`` after ``_postprocess_step``.
    * ``failed_candidates`` and ``added_candidates`` are disjoint.
    * ``partial_gains[i]`` is undefined whenever ``i in failed_candidates``.
    """
    selected_vars: list = field(default_factory=list)
    cached_MIs: dict = field(default_factory=dict)
    cached_confident_MIs: dict = field(default_factory=dict)
    cached_cond_MIs: dict = field(default_factory=dict)
    entropy_cache: dict = field(default_factory=dict)
    partial_gains: dict = field(default_factory=dict)
    failed_candidates: set = field(default_factory=set)
    added_candidates: set = field(default_factory=set)
    classes_y: object = None
    classes_y_safe_cpu: object = None
    classes_y_safe_gpu: object = None
    freqs_y: object = None
    data_copy: object = None
    n_iterations: int = 0
    n_confirmed_candidates: int = 0
    n_evaluate_gain_stopped_early: int = 0


def screen_predictors(
    # factors
    factors_data: np.ndarray,
    factors_nbins: Sequence[int],
    factors_names: Sequence[str] = None,
    factors_names_to_use: Sequence[str] = None,
    factors_to_use: Sequence[int] = None,
    # targets
    targets_data: np.ndarray = None,
    targets_nbins: Sequence[int] = None,
    y: Sequence[int] = None,
    # algorithm
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    reduce_gain_on_subelement_chosen: bool = True,
    # performance
    extra_x_shuffling: bool = True,
    dtype=np.int32,
    random_seed: int = None,
    use_gpu: bool = False,
    n_workers: int = 1,
    # confidence
    min_occupancy: int = None,
    min_nonzero_confidence: float = 0.99,
    full_npermutations: int = 1_000,
    baseline_npermutations: int = 100,
    # stopping conditions
    min_relevance_gain: float = 0.00001,
    max_consec_unconfirmed: int = 30,
    max_runtime_mins: float = None,
    interactions_min_order: int = 1,
    interactions_max_order: int = 1,
    interactions_order_reversed: bool = False,
    max_veteranes_interactions_order: int = 1,
    only_unknown_interactions: bool = False,
    # B13: confirmation-step cardinality cutoff (was a hardcoded module
    # global of 50). When ``None`` we fall back to the legacy constant.
    # ``MRMR.fit`` overrides this with
    # ``quantization_nbins ** interactions_max_order * 2``.
    max_confirmation_cand_nbins: int = None,
    # B15: when the screening pass returns zero selected_vars, the legacy
    # FE step fell back to running on ALL features. With this flag set to
    # False, FE is skipped instead (the safer default; FE on an empty
    # screen typically just amplifies noise).
    fe_fallback_to_all: bool = True,
    # verbosity and formatting
    verbose: int = 1,
    ndigits: int = 5,
    parallel_kwargs: dict = None,
    stop_file: str = None,
    use_simple_mode: bool = True,
) -> float:
    """Finds best predictors for the target.

    B13: ``max_confirmation_cand_nbins`` parameterises the legacy module
    constant of the same name. ``None`` (the default) preserves the
    legacy value of 50 so this entry-point keeps backward-compatible
    semantics; ``MRMR.fit`` overrides explicitly.

    B15: ``fe_fallback_to_all`` -- this flag is consumed by ``MRMR.fit``
    (where the FE step lives) and threaded here only so callers can
    pass it through. ``screen_predictors`` itself does not act on it.

    x must be n-x-m array of integers (ordinal encoded)
    Parameters:
        full_npermutations: when computing every MI, repeat calculations with randomly shuffled indices that many times
        min_nonzero_confidence: if in random permutation tests this or higher % of cases had worse current_gain than original, current_gain value is considered valid, otherwise, it's set to zero.
    Returns:
        1) best set of non-redundant single features influencing the target
        2) subsets of size 2..interactions_max_order influencing the target. Such subsets will be candidates for predictors and OtherVarsEncoding.
        3) all 1-vs-1 influencers (not necessarily in mRMR)
    Parameters:
        only_unknown_interactions: True for speed, False for completeness of higher order interactions discovery.
        verbose: int  1=log only important info,>1=also log additional details
        mrmr_relevance_algo:str
                        "fleuret": max(min(I(X,Y|Z)),max(I(X,Y|Z)-I(X,Y))) Possible to use n-way interactions here.
                        "pld": I(X,Y)
        mrmr_redundancy_algo:str
                        "fleuret": 0 ('cause redundancy already accounted for)
                        "pld_max": max(I(veterane,cand)) Possible to use n-way interactions here.
                        "pld_mean": mean(I(veterane,cand)) Possible to use n-way interactions here.
    """
    # ---------------------------------------------------------------------------------------------------------------
    # Input checks
    # ---------------------------------------------------------------------------------------------------------------

    if parallel_kwargs is None:
        parallel_kwargs = dict(max_nbytes=MAX_JOBLIB_NBYTES)

    # B13: resolve effective cutoff. None preserves the legacy 50 so the
    # raw `screen_predictors` entry-point stays bit-exact with pre-refactor
    # behaviour; ``MRMR.fit`` overrides with the formula default.
    if max_confirmation_cand_nbins is None:
        max_confirmation_cand_nbins = MAX_CONFIRMATION_CAND_NBINS

    assert mrmr_relevance_algo in ("fleuret", "pld")
    assert mrmr_redundancy_algo in ("fleuret", "pld_max", "pld_mean")

    assert len(factors_data) >= 10
    if targets_data is None:
        targets_data = factors_data
    else:
        assert len(factors_data) == len(targets_data)

    if targets_nbins is None:
        targets_nbins = factors_nbins

    assert targets_data.shape[1] == len(targets_nbins)
    assert factors_data.shape[1] == len(factors_nbins)

    if len(factors_names) == 0:
        factors_names = ["F" + str(i) for i in range(len(factors_data))]
    else:
        assert factors_data.shape[1] == len(factors_names)

    # Initialize x (factor indices to consider) with appropriate defaults
    if factors_to_use is not None:
        x = set(factors_to_use)
    elif factors_names_to_use is not None:
        x = [i for i, col_name in enumerate(factors_names) if col_name in factors_names_to_use]
    else:
        x = set(range(factors_data.shape[1]))

    # warn if inputs are identical to targets
    if factors_data.shape == targets_data.shape:
        if np.shares_memory(factors_data, targets_data):
            if factors_to_use is None and factors_names_to_use is None:
                if verbose > 2:
                    logger.info(
                        "factors_data and targets_data share the same memory. factors_to_use will be determined automatically to not contain any target columns."
                    )
                x = set(range(factors_data.shape[1])) - set(y)
            else:
                if factors_to_use is not None:
                    x = set(factors_to_use) - set(y)
                    if verbose > 2:
                        logger.info("Using only %d predefined factors: %s", len(factors_to_use), factors_to_use)
                else:
                    x = [i for i, col_name in enumerate(factors_names) if col_name in factors_names_to_use and i not in y]
                    if verbose > 2:
                        logger.info("Using only %d predefined factors: %s", len(factors_names_to_use), factors_names_to_use)
        else:

            assert not set(y).issubset(set(x))

    # ---------------------------------------------------------------------------------------------------------------
    # Inits
    # ---------------------------------------------------------------------------------------------------------------

    start_time = timer()
    run_out_of_time = False

    if random_seed is not None:
        np.random.seed(random_seed)
        set_numba_random_seed(random_seed)
        try:
            cp.random.seed(random_seed)
        except NameError:
            pass  # CuPy not imported

    max_failed = int(full_npermutations * (1 - min_nonzero_confidence))
    if max_failed <= 1:
        max_failed = 1

    selected_interactions_vars = []
    selected_vars = []  # stores just indices. can't use set 'cause the order is important for efficient computing
    predictors = []  # stores more details.

    # Observability flag -- true if the inner confirmation loop hit the
    # ``max_consec_unconfirmed`` patience threshold at least once. Prior
    # to 2026-04-19, patience-triggered early exits only logged at
    # verbose>=1; at default verbosity, MRMR silently returned a
    # potentially-truncated feature set and callers had no signal that
    # the stopping condition was "gave up confirming" rather than
    # "natural gain threshold reached". Summary log at function exit
    # now surfaces this unconditionally.
    patience_triggered: bool = False

    cached_MIs = dict()
    # cached_cond_MIs = dict()
    cached_confident_MIs = dict()
    entropy_cache = numba.typed.Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )
    cached_cond_MIs = numba.typed.Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )

    data_copy = factors_data.copy()

    classes_y, freqs_y, _ = merge_vars(factors_data=factors_data, vars_indices=y, var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype)
    classes_y_safe = classes_y.copy()

    if use_gpu:
        import cupy as cp

        classes_y_safe = cp.asarray(classes_y.astype(np.int32))
        freqs_y_safe = cp.asarray(freqs_y)
    else:
        freqs_y_safe = None

    if n_workers and n_workers > 1:
        #    global classes_y_memmap
        #    classes_y_memmap = mem_map_array(obj=classes_y, file_name="classes_y", mmap_mode="r")
        if verbose >= 2:
            logger.info("Starting parallel pool...")

        from loky import set_loky_pickler

        set_loky_pickler("cloudpickle")

        workers_pool = Parallel(n_jobs=n_workers, **parallel_kwargs)
        workers_pool(delayed(test)(i) for i in range(n_workers))
    else:
        workers_pool = None

    subsets = range(interactions_min_order, interactions_max_order + 1)
    if interactions_order_reversed:
        subsets = subsets[::-1]

    if verbose >= 2:
        logger.info(
            "Starting work with full_npermutations=%d, min_nonzero_confidence=%.*f, max_failed=%d",
            full_npermutations, ndigits, min_nonzero_confidence, max_failed,
        )

    num_possible_candidates = 0  # needed to refrain from multiprocessing when all direct MIs are in cache already

    for interactions_order in (subsets_pbar := tqdmu(subsets, desc="Interactions order", leave=False)):

        if run_out_of_time:
            break
        subsets_pbar.set_description(f"{interactions_order}-way interactions")

        # ---------------------------------------------------------------------------------------------------------------
        # Generate candidates
        # ---------------------------------------------------------------------------------------------------------------

        candidates = [tuple(el) for el in combinations(x, interactions_order)]

        num_possible_candidates += len(candidates)

        # ---------------------------------------------------------------------------------------------------------------
        # Subset level inits
        # ---------------------------------------------------------------------------------------------------------------

        total_disproved = 0
        total_checked = 0
        partial_gains = {}
        added_candidates = set()
        failed_candidates = set()
        nconsec_unconfirmed = 0

        for n_confirmed_predictors in (predictors_pbar := tqdmu(range(len(candidates)), leave=False, desc="Confirmed predictors")):
            # if n_confirmed_predictors>4: n_jobs=1
            if run_out_of_time:
                break
            if stop_file and exists(stop_file):
                logger.warning(f"Stop file {stop_file} detected, quitting.")
                break

            # ---------------------------------------------------------------------------------------------------------------
            # Find candidate X with the highest current_gain given already selected factors
            # ---------------------------------------------------------------------------------------------------------------

            best_candidate = None
            best_gain = min_relevance_gain - 1
            expected_gains = np.zeros(len(candidates), dtype=np.float64)

            while True:  # confirmation loop (by random permutations)

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
                    )
                    if skip:
                        continue

                    feasible_candidates.append((cand_idx, X, nexisting if reduce_gain_on_subelement_chosen else 0))

                if (
                    n_workers > 1
                    and (use_simple_mode is False or len(cached_MIs) < num_possible_candidates)
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
                            break

                else:
                    if use_simple_mode and False:
                        # No need to check every can out of order: let's just return next best known candidate
                        best_gain, best_candidate, run_out_of_time = 1, 1, False
                    else:
                        for cand_idx, X, nexisting in feasible_candidates:  # (candidates_pbar := tqdmu(, leave=False, desc="Candidates"))

                            # tmp_idx=X[0]
                            # print(X,factors_nbins[tmp_idx],factors_names[tmp_idx])
                            # from time import sleep
                            # sleep(5)

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

                            """
                            if use_simple_mode:
                                if best_gain > 0:
                                    break
                            """

                            if run_out_of_time:
                                if verbose:
                                    logger.info("Time limit exhausted. Finalizing the search...")
                                break

                if verbose > 2 and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                    logger.info("evaluate_candidates took %.1f sec.", timer() - eval_start)

                if best_gain < min_relevance_gain:
                    if verbose >= 2:
                        logger.info("Minimum expected gain reached or no candidates to check anymore.")
                    break  # exit confirmation while loop

                # ---------------------------------------------------------------------------------------------------------------
                # Now need to confirm best expected gain with a permutation test
                # ---------------------------------------------------------------------------------------------------------------

                cand_confirmed = False
                any_cand_considered = False
                for n, next_best_candidate_idx in enumerate(np.argsort(expected_gains)[::-1]):
                    next_best_gain = expected_gains[next_best_candidate_idx]
                    # logger.info(f"{n}, {next_best_gain}, {min_relevance_gain}")
                    if next_best_gain >= min_relevance_gain:  # only can consider here candidates fully checked against every Z

                        X = candidates[next_best_candidate_idx]

                        # ---------------------------------------------------------------------------------------------------------------
                        # For cands other than the top one, if best partial gain <= next_best_gain, we can proceed with confirming next_best_gain.
                        # else we have to recompute partial gains
                        # ---------------------------------------------------------------------------------------------------------------

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

                            if verbose > 2:
                                logger.info(
                                    "confirming candidate %s, next_best_gain=%.*f",
                                    get_candidate_name(X, factors_names=factors_names), ndigits, next_best_gain,
                                )

                            # ---------------------------------------------------------------------------------------------------------------
                            # Compute confidence by bootstrap
                            # ---------------------------------------------------------------------------------------------------------------

                            total_checked += 1
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

                                # ---------------------------------------------------------------------------------------------------------------
                                # external bootstrapped recheck. is minimal MI of candidate X with Y given all current Zs THAT BIG as next_best_gain?
                                # ---------------------------------------------------------------------------------------------------------------

                                if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                                    eval_start = timer()

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
                                    )
                                    # logger.info(f"nfailed={nfailed}, nchecked={nchecked}")
                                    confidence = 1 - nfailed / nchecked
                                    if nfailed >= max_failed:
                                        bootstrapped_gain = 0.0

                                if verbose and len(selected_vars) < MAX_ITERATIONS_TO_TRACK:
                                    logger.info("get_fleuret_criteria_confidence bootstrapped eval took %.1f sec.", timer() - eval_start)

                        # ---------------------------------------------------------------------------------------------------------------
                        # Report this particular best candidate
                        # ---------------------------------------------------------------------------------------------------------------

                        if bootstrapped_gain > 0:

                            nconsec_unconfirmed = 0

                            # ---------------------------------------------------------------------------------------------------------------
                            # Bad confidence can make us consider other candidates!
                            # ---------------------------------------------------------------------------------------------------------------

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

            # ---------------------------------------------------------------------------------------------------------------
            # Add best candidate to the list, if criteria are met, or proceed to the next interactions_order
            # ---------------------------------------------------------------------------------------------------------------

            if best_gain >= (min_relevance_gain if interactions_order == 1 else min_relevance_gain ** (1 / (interactions_order + 1))):
                for var in best_candidate:
                    if var not in selected_vars:
                        selected_vars.append(var)
                        if interactions_order > 1:
                            selected_interactions_vars.append(var)
                cand_name = get_candidate_name(best_candidate, factors_names=factors_names)

                res = {"name": cand_name, "indices": best_candidate, "gain": best_gain}
                if full_npermutations:
                    res["confidence"] = confidence
                predictors.append(res)

                if verbose >= 2:
                    mes = f"Added new predictor {cand_name} to the list with expected gain={best_gain:.{ndigits}f}"
                    if full_npermutations:
                        mes += f" and confidence={confidence:.3f}"
                    logger.info(mes)

            else:
                if verbose >= 2:
                    if total_checked > 0:
                        details = f" Total candidates disproved: {total_disproved:_}/{total_checked:_} ({total_disproved*100/total_checked:.2f}%)"
                    else:
                        details = ""
                    logger.info("Can't add anything valuable anymore for interactions_order=%s.%s", interactions_order, details)
                predictors_pbar.total = len(candidates)
                predictors_pbar.close()
                break

    # postprocess_candidates(selected_vars)
    # print(caching_hits_xyz, caching_hits_z, caching_hits_xz, caching_hits_yz)
    if verbose >= 2:
        logger.info("Finished.")

    # Termination-reason summary (always emitted, even at verbose=0).
    # Two distinct termination modes:
    #   patience_triggered=True -- ``max_consec_unconfirmed`` hit; MRMR
    #     gave up confirming more candidates. Returned set may be smaller
    #     than what a more patient search would have found.
    #   patience_triggered=False -- natural exhaustion: remaining
    #     candidates fell below ``min_relevance_gain`` threshold. This is
    #     the "done" case, not an early stop.
    # Operators tuning MRMR on noisy data need to distinguish the two --
    # if patience keeps tripping, they need a higher ``max_consec_unconfirmed``
    # or smoother relevance signals, not a higher feature budget.
    if patience_triggered:
        logger.warning(
            "screen_predictors terminated early via max_consec_unconfirmed=%d "
            "patience (at least one level exhausted). Returned %d selected "
            "feature(s). If you expected more, increase max_consec_unconfirmed "
            "or reduce the relevance-gain threshold.",
            max_consec_unconfirmed, len(selected_vars),
        )
    else:
        logger.info(
            "screen_predictors finished naturally (no patience trip). "
            "Returned %d selected feature(s).",
            len(selected_vars),
        )

    any_influencing = set()
    for vars_combination, (bootstrapped_gain, confidence) in cached_confident_MIs.items():
        if bootstrapped_gain > 0:
            any_influencing.update(set(vars_combination))

    """Выбрать группы/кластера скоррелированных факторов. Вместо использования 1 самого крутого, рассмотреть средние от всех
        отброшенных факторов, имеющих высокое прямое direct_MI с таргетом, но близкое к 0 additional_knowledge с каждым 
        "победившим" фактором, проверить, не могут ли они усилить свой победивший фактор через ансамблирование. Усиление в смысле
        среднего и вариативности MI с таргетом на бутстрепе подвыборок?

        key = arr2str(X) + "_" + arr2str(Z)
        if key in cached_cond_MIs:
            additional_knowledge = cached_cond_MIs[key]                        
    """
    return selected_vars, predictors, any_influencing, entropy_cache, cached_MIs, cached_confident_MIs, cached_cond_MIs, classes_y, classes_y_safe, freqs_y


# `count_cand_nbins` moved to ``_numba_utils.py`` (etap 2). Imported above.


def postprocess_candidates(
    selected_vars: list,
    factors_data: np.ndarray,
    y: np.ndarray,
    factors_nbins: np.ndarray,
    classes_y: np.ndarray = None,
    freqs_y: np.ndarray = None,
    classes_y_safe: np.ndarray = None,
    min_nonzero_confidence: float = 0.99999,
    npermutations: int = 10_000,
    interactions_max_order: int = 1,
    ensure_target_influence: bool = True,
    dtype=np.int32,
    verbose: bool = True,
    ndigits: int = 4,
):
    """Post-analysis of prescreened candidates.

    1) repeat standard Fleuret screening process. maybe some vars will be removed when taken into account all other candidates.
    2)
    3) in the final set, compute for every factor
        a) MI with every remaining predictor (and 2,3 way subsets)

    """

    """
    # ---------------------------------------------------------------------------------------------------------------
    # Repeat standard Fleuret screening process. maybe some vars will be removed when taken into account all other candidates.
    # ---------------------------------------------------------------------------------------------------------------
    for cand_idx, X, nexisting in (candidates_pbar := tqdmu(selected_vars, leave=False, desc="Finalizing Candidates")):
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
        )
    # ---------------------------------------------------------------------------------------------------------------
    # Make sure with confidence that every candidate is related to the target
    # ---------------------------------------------------------------------------------------------------------------

    if ensure_target_influence:
        removed = []
        for X in tqdmu(selected_vars, desc="Ensuring target influence", leave=False):
            bootstrapped_mi, confidence = mi_direct(
                factors_data,
                x=np.array([X], dtype=np.int64),
                y=y,
                factors_nbins=factors_nbins,
                classes_y=classes_y,
                freqs_y=freqs_y,
                classes_y_safe=classes_y_safe,
                min_nonzero_confidence=min_nonzero_confidence,
                npermutations=npermutations,
            )
            if bootstrapped_mi == 0:
                if verbose:
                    print("Factor", X, "not related to target with confidence", confidence)
                    removed.append(X)
        selected_vars = [el for el in selected_vars if el not in removed]

    # ---------------------------------------------------------------------------------------------------------------
    # Repeat Fleuret process as many times as there is candidates left.
    # This time account for possible interactions (fix bug in the professor's formula).
    # ---------------------------------------------------------------------------------------------------------------
    """

    """
    Compute redundancy stats for every X

    кто связан с каким количеством других факторов, какое количество информации с ними разделяет, в % к своей собственной энтропии. 
    Можно даже спуститься вниз по уровню и посчитать взвешенные суммы тех же метрик для его партнёров. 
    Тем самым можно косвенно определить, какие фичи скорее всего просто сливные бачки, и попробовать их выбросить.  В итоге мы получим:

    ценные фичи, которые ни с кем другим не связаны, кроме мусорных и таргета. они содержат уникальное знание;
    потенциально мусорные X, которые связаны с множеством других, и шарят очень много общей инфы с другими факторами Z, при том, 
    что эти другие факторы имеют много уникального знания о таргете помимо X: sum(I(Y;Z|X))>e;
    все остальные "середнячки".    
    """

    """
    entropies = {}
    mutualinfos = {}

    for X in tqdmu(selected_vars, desc="Marginal stats", leave=False):
        _, freqs, _ = merge_vars(factors_data=factors_data, vars_indices=np.array([X], dtype=np.int64), factors_nbins=factors_nbins, var_is_nominal=None, dtype=dtype)
        factor_entropy = entropy(freqs=freqs)
        entropies[X] = factor_entropy

    for a, b in tqdmu(combinations(selected_vars, 2), desc="1-way interactions", leave=False):
        bootstrapped_mi, confidence = mi_direct(
            factors_data,
            x=np.array([a], dtype=np.int64),
            y=np.array([b], dtype=np.int64),
            factors_nbins=factors_nbins,
            classes_y=classes_y,
            freqs_y=freqs_y,
            classes_y_safe=classes_y_safe,
            min_nonzero_confidence=min_nonzero_confidence,
            npermutations=npermutations,
        )
        if bootstrapped_mi > 0:
            mutualinfos[(a, b)] = bootstrapped_mi

    for y in tqdmu(selected_vars, desc="2-way interactions", leave=False):
        for pair in combinations(set(selected_vars) - set([y]), 2):
            bootstrapped_mi, confidence = mi_direct(
                factors_data,
                x=np.array(pair, dtype=np.int64),
                y=np.array([y], dtype=np.int64),
                factors_nbins=factors_nbins,
                classes_y=classes_y,
                freqs_y=freqs_y,
                classes_y_safe=classes_y_safe,
                min_nonzero_confidence=min_nonzero_confidence,
                npermutations=npermutations,
            )
            if bootstrapped_mi > 0:
                mutualinfos[(y, pair)] = bootstrapped_mi

    return entropies, mutualinfos
    """


# ----------------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------------


# create_redundant_continuous_factor, categorize_1d_array, digitize, edges,
# quantize_dig, quantize_search, discretize_uniform, discretize_array,
# _discretize_array_impl, discretize_2d_array, get_binning_edges,
# discretize_sklearn, categorize_dataset moved to ``discretization.py``
# (etap 4). categorize_dataset_old deleted (B9, verified zero call-sites).
# All imported below.
from .discretization import (
    create_redundant_continuous_factor,
    categorize_1d_array,
    digitize,
    edges,
    quantize_dig,
    quantize_search,
    discretize_uniform,
    discretize_array,
    _discretize_array_impl,
    discretize_2d_array,
    get_binning_edges,
    discretize_sklearn,
    categorize_dataset,
)
