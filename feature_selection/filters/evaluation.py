"""Per-candidate evaluation: gain computation, confirmation checks.

See ``screen.py`` for the screening orchestrator that calls these functions.
"""
from __future__ import annotations

import gc
import logging
import math
import time
from timeit import default_timer as timer
from typing import Sequence

import numba
import numpy as np
from numba import njit
from numba.core import types

from pyutilz.numbalib import generate_combinations_recursive_njit, python_dict_2_numba_dict
# Module-level import so cloudpickle can resolve tqdmu when the function ships to a joblib worker.
from pyutilz.system import tqdmu

from ._internals import LARGE_CONST, MAX_CONFIRMATION_CAND_NBINS, MAX_ITERATIONS_TO_TRACK
from ._numba_utils import arr2str, count_cand_nbins, unpack_and_sort
from .gpu import mi_direct_gpu
from .info_theory import compute_mi_from_classes, conditional_mi, entropy, merge_vars
from .permutation import mi_direct

logger = logging.getLogger(__name__)


def get_candidate_name(candidate_indices: list, factors_names: list) -> str:
    cand_name = "-".join([factors_names[el] for el in candidate_indices])
    return cand_name


def should_skip_candidate(
    cand_idx: int,
    X: tuple,
    interactions_order: int,
    failed_candidates: set,
    added_candidates: set,
    expected_gains: np.ndarray,
    selected_vars: list,
    selected_interactions_vars: list,
    only_unknown_interactions: bool = True,
    engineered_lineage: dict = None,
) -> tuple:
    """Decide if current candidate for predictors should be skipped (already accepted, failed, or computed).

    ``engineered_lineage``: optional ``{engineered_idx -> frozenset(parent_indices)}`` from the cat-FE step. When set, a k-way candidate is skipped if it
    combines an engineered column with one of its own parents (conditional MI degenerates and confidence gates waste budget). Legacy/numeric path leaves it ``None``.
    """

    nexisting = 0

    if (cand_idx in failed_candidates) or (cand_idx in added_candidates) or expected_gains[cand_idx]:
        return True, nexisting

    if interactions_order > 1:  # disabled for single predictors 'cause Fleuret formula won't detect pairs predictors

        # Lineage filter: skip k-way candidates that combine an engineered column with one of its own parent columns.
        if engineered_lineage:
            X_set = set(X)
            for subel in X:
                parents = engineered_lineage.get(subel)
                if parents is not None and not parents.isdisjoint(X_set):
                    return True, nexisting

        # Check if any sub-element is already selected at this stage.
        skip_cand = False
        for subel in X:
            if subel in selected_interactions_vars:
                skip_cand = True
                break
        if skip_cand:
            return True, nexisting

        # Or all selected at the lower stages.
        skip_cand = [(subel in selected_vars) for subel in X]
        nexisting = sum(skip_cand)
        if (only_unknown_interactions and any(skip_cand)) or all(skip_cand):
            return True, nexisting

    return False, nexisting


def evaluate_candidates(
    workload: list,
    y: Sequence[int],
    best_gain: float,
    factors_data: np.ndarray,
    factors_nbins: np.ndarray,
    factors_names: Sequence[str],
    partial_gains: dict,
    selected_vars: list,
    baseline_npermutations: int,
    classes_y: np.ndarray = None,
    freqs_y: np.ndarray = None,
    freqs_y_safe: np.ndarray = None,
    use_gpu: bool = True,
    cached_MIs: dict = None,
    cached_confident_MIs: dict = None,
    cached_cond_MIs: dict = None,
    entropy_cache: dict = None,
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    max_veteranes_interactions_order: int = 1,
    dtype=np.int32,
    max_runtime_mins: float = None,
    start_time: float = None,
    min_relevance_gain: float = None,
    verbose: int = 1,
    ndigits: int = 5,
    use_simple_mode: bool = True,
) -> None:

    best_gain = -LARGE_CONST
    best_candidate = None
    expected_gains = {}

    entropy_cache_dict = numba.typed.Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )
    python_dict_2_numba_dict(python_dict=entropy_cache, numba_dict=entropy_cache_dict)

    cached_cond_MIs_dict = numba.typed.Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )
    python_dict_2_numba_dict(python_dict=cached_cond_MIs, numba_dict=cached_cond_MIs_dict)

    classes_y_safe = classes_y.copy()

    for cand_idx, X, nexisting in tqdmu(workload, leave=False, desc="Thread Candidates"):

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
            cached_cond_MIs=cached_cond_MIs_dict,
            entropy_cache=entropy_cache_dict,
            verbose=verbose,
            ndigits=ndigits,
            dtype=dtype,
            use_simple_mode=use_simple_mode,
        )

        best_gain, best_candidate, run_out_of_time = handle_best_candidate(
            current_gain=current_gain,
            best_gain=best_gain,
            X=X,
            best_candidate=best_candidate,
            factors_names=factors_names,
            verbose=verbose,
            ndigits=ndigits,
            max_runtime_mins=max_runtime_mins,
            start_time=start_time,
            min_relevance_gain=min_relevance_gain,
        )

        if run_out_of_time:
            break

    entropy_cache = dict(entropy_cache_dict)
    cached_cond_MIs = dict(cached_cond_MIs_dict)

    return best_gain, best_candidate, partial_gains, expected_gains, cached_MIs, cached_cond_MIs, entropy_cache


def handle_best_candidate(
    current_gain: float,
    best_gain: float,
    X: Sequence,
    best_candidate: Sequence,
    factors_names: list,
    verbose: int = 1,
    ndigits: int = 5,
    max_runtime_mins: float = None,
    start_time: float = None,
    min_relevance_gain: float = None,
):
    # Save best known candidate, to enable early stopping.
    run_out_of_time = False

    if current_gain > best_gain:
        best_candidate = X
        best_gain = current_gain
        if verbose > 2:
            logger.info(
                "\t%s is so far the best candidate with best_gain=%.*f",
                get_candidate_name(best_candidate, factors_names=factors_names), ndigits, best_gain,
            )
    else:
        if min_relevance_gain and verbose > 2 and current_gain > min_relevance_gain:
            logger.info("\t\t%s current_gain=%.*f", get_candidate_name(X, factors_names=factors_names), ndigits, current_gain)

    if max_runtime_mins and not run_out_of_time:
        run_out_of_time = (timer() - start_time) > max_runtime_mins * 60

    return best_gain, best_candidate, run_out_of_time


@njit()
def evaluate_gain(
    current_gain: float,
    last_checked_k: int,
    direct_gain: float,
    X: np.ndarray,
    y: np.ndarray,
    nexisting: int,
    best_gain: float,
    factors_data: np.ndarray,
    factors_nbins: np.ndarray,
    selected_vars: np.ndarray,
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    max_veteranes_interactions_order: int = 2,
    extra_knowledge_multipler: float = -1.0,
    sink_threshold: float = -1.0,
    entropy_cache: dict = None,
    cached_cond_MIs: dict = None,
    can_use_x_cache=False,
    can_use_y_cache=False,
    dtype=np.int32,
    confidence_mode: bool = False,
    max_confirmation_cand_nbins: int = MAX_CONFIRMATION_CAND_NBINS,
) -> tuple:
    """``max_confirmation_cand_nbins`` is a parameter (not a baked-in constant). Default mirrors the legacy value; ``MRMR.fit`` overrides with ``quantization_nbins ** interactions_max_order * 2`` unless the user pins it explicitly."""

    positive_mode = False
    stopped_early = False
    sink_reasons = None

    k = 0
    for interactions_order in range(max_veteranes_interactions_order):
        combs = generate_combinations_recursive_njit(np.array(selected_vars, dtype=np.int32), interactions_order + 1)[::-1]

        for Z in combs:

            if k > last_checked_k:
                if confidence_mode and count_cand_nbins(Z, factors_nbins) > max_confirmation_cand_nbins:
                    additional_knowledge = 0.0  # this is needed to skip checking agains hi cardinality approved factors
                else:
                    if mrmr_relevance_algo == "fleuret":
                        # additional_knowledge = I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z); I(X, Z) = entropy_x + entropy_z - entropy_xz.
                        key_found = False
                        if not confidence_mode:
                            key = arr2str(X) + "_" + arr2str(Z)
                            if key in cached_cond_MIs:
                                additional_knowledge = cached_cond_MIs[key]
                                key_found = True

                        if not key_found:

                            additional_knowledge = conditional_mi(
                                factors_data=factors_data,
                                x=X,
                                y=y,
                                z=Z,
                                var_is_nominal=None,
                                factors_nbins=factors_nbins,
                                entropy_cache=entropy_cache,
                                can_use_x_cache=can_use_x_cache,
                                can_use_y_cache=can_use_y_cache,
                                dtype=dtype,
                            )

                            if nexisting > 0:
                                additional_knowledge = additional_knowledge ** (nexisting + 1)

                            if not confidence_mode:
                                cached_cond_MIs[key] = additional_knowledge

                    # Account for possible extra knowledge from conditioning on Z; must update best_gain globally and log such cases. Order of discovery is
                    # not guaranteed, but cases are too precious to ignore. Also enables skipping higher-order interactions containing all approved candidates.
                    if extra_knowledge_multipler > 0 and additional_knowledge > direct_gain * extra_knowledge_multipler:
                        if not positive_mode:
                            current_gain = additional_knowledge
                            positive_mode = True
                        else:
                            # rare chance that a candidate has many excellent relationships
                            if additional_knowledge > current_gain:
                                current_gain = additional_knowledge

                if not positive_mode and (additional_knowledge < current_gain):

                    current_gain = additional_knowledge

                    if best_gain is not None and current_gain <= best_gain:
                        # No point checking other Zs -- current_gain already won't beat best_gain (assuming best_gain was estimated confidently; checked at end).
                        # Also fix what Z caused X (the most) to sink.
                        if sink_threshold > -1 and current_gain < sink_threshold:
                            sink_reasons = Z

                        stopped_early = True
                        return stopped_early, current_gain, k, sink_reasons
            k += 1

    return stopped_early, current_gain, k - 1, sink_reasons


def evaluate_candidate(
    cand_idx: int,
    X: Sequence[int],
    y: Sequence[int],
    nexisting: int,
    best_gain: float,
    factors_data: np.ndarray,
    factors_nbins: np.ndarray,
    factors_names: Sequence[str],
    expected_gains: np.ndarray,
    partial_gains: dict,
    selected_vars: list,
    baseline_npermutations: int,
    classes_y: np.ndarray = None,
    classes_y_safe: np.ndarray = None,
    freqs_y: np.ndarray = None,
    freqs_y_safe: np.ndarray = None,
    use_gpu: bool = True,
    cached_MIs: dict = None,
    cached_confident_MIs: dict = None,
    cached_cond_MIs: dict = None,
    entropy_cache: dict = None,
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    max_veteranes_interactions_order: int = 2,
    extra_knowledge_multipler: float = -1.0,
    sink_threshold: float = -1.0,
    dtype=np.int32,
    verbose: int = 1,
    ndigits: int = 5,
    use_simple_mode: bool = True,
) -> None:
    sink_reasons = set()

    # Is this candidate any good for target 1-vs-1?
    if X in cached_confident_MIs:  # cached_confident_MIs first -- more reliable (but don't fill them here).
        direct_gain = cached_confident_MIs[X]
    else:
        if X in cached_MIs:
            direct_gain = cached_MIs[X]
        else:
            if use_gpu:
                direct_gain, _ = mi_direct_gpu(
                    factors_data,
                    x=X,
                    y=y,
                    factors_nbins=factors_nbins,
                    classes_y=classes_y,
                    classes_y_safe=classes_y_safe,
                    freqs_y=freqs_y,
                    freqs_y_safe=freqs_y_safe,
                    min_nonzero_confidence=1.0,
                    npermutations=baseline_npermutations,
                    dtype=dtype,
                )
            else:
                direct_gain, _ = mi_direct(
                    factors_data,
                    x=X,
                    y=y,
                    factors_nbins=factors_nbins,
                    classes_y=classes_y,
                    classes_y_safe=classes_y_safe,
                    freqs_y=freqs_y,
                    min_nonzero_confidence=1.0,
                    npermutations=baseline_npermutations,
                    dtype=dtype,
                )
            cached_MIs[X] = direct_gain

    if direct_gain > 0:
        if selected_vars and not use_simple_mode:
            # Some factors already selected. Best gain from including X is min I(X; Y | Z) over Z in selected_vars. But a variable correlated to every real
            # predictor plus noise has real value zero -- I(X; Y | Z) alone still gives significant impact. Summing I(X, Z) over Zs reveals it shares all
            # knowledge with the rest and has no value alone, but that requires all real factors already in S; otherwise 'connected-to-all' trash dominates.
            # Solution: compute sum(X, Z) not only when adding Z, but repeat for all Zs once a new X is added -- some Zs may become useless after.
            if cand_idx in partial_gains:
                current_gain, last_checked_k = partial_gains[cand_idx]
                if best_gain is not None and (current_gain <= best_gain):
                    return current_gain, sink_reasons
            else:
                current_gain = LARGE_CONST
                last_checked_k = -1

            stopped_early, current_gain, k, sink_reasons = evaluate_gain(
                current_gain=current_gain,
                last_checked_k=last_checked_k,
                direct_gain=direct_gain,
                X=X,
                y=y,
                nexisting=nexisting,
                best_gain=best_gain,
                factors_data=factors_data,
                factors_nbins=factors_nbins,
                selected_vars=selected_vars,
                mrmr_relevance_algo=mrmr_relevance_algo,
                mrmr_redundancy_algo=mrmr_redundancy_algo,
                max_veteranes_interactions_order=max_veteranes_interactions_order,
                extra_knowledge_multipler=extra_knowledge_multipler,
                sink_threshold=sink_threshold,
                entropy_cache=entropy_cache,
                cached_cond_MIs=cached_cond_MIs,
                can_use_x_cache=True,
                can_use_y_cache=True,
            )

            partial_gains[cand_idx] = current_gain, k
            if not stopped_early:  # no break -- current_gain computed fully.
                expected_gains[cand_idx] = current_gain
        else:
            # No factors selected yet -- current_gain is just direct_gain.
            current_gain = direct_gain
            expected_gains[cand_idx] = current_gain
    else:
        current_gain = 0

    return current_gain, sink_reasons


def find_best_partial_gain(
    partial_gains: dict, failed_candidates: set, added_candidates: set, candidates: list, selected_vars: list, skip_indices: tuple = ()
) -> float:
    best_partial_gain = -LARGE_CONST
    best_key = None
    for key, value in partial_gains.items():
        if (key not in failed_candidates) and (key not in added_candidates) and (key not in skip_indices):
            skip_cand = False
            for subel in candidates[key]:
                if subel in selected_vars:
                    skip_cand = True  # the sub-element or var itself is already selected.
                    break
            if skip_cand:
                continue
            partial_gain, _ = value
            if partial_gain > best_partial_gain:
                best_partial_gain = partial_gain
                best_key = key
    return best_partial_gain, best_key

