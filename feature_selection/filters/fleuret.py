"""Fleuret-criterion permutation-confidence step. Three layers: ``get_fleuret_criteria_confidence_parallel`` (joblib pool spawner) ->
``parallel_fleuret`` (joblib worker) -> ``get_fleuret_criteria_confidence`` (``@njit`` core looping over the permutation budget).

Known weakness (TODO -- separate research PR): the Fleuret formulation ``gain = I(X; Y) - max_k I(X; Y | S_k)`` rejects synergistic features (a candidate
uninformative on its own but critical in combination with an already-selected variable scores ``gain < 0``). Mitigations to investigate (none implemented):
CMIM (Brown 2012) replaces ``max_k`` with ``min_k``; JMI uses ``sum_k I(X, S_k; Y)`` for 3-way interactions; HOmRMR enumerates pair/triple interactions
explicitly. Plan: extend ``mrmr_relevance_algo`` (currently ``"fleuret"``, ``"pld"``) with ``"cmim"`` / ``"jmi"`` and benchmark on the 4 golden scenarios.
"""
from __future__ import annotations

import gc

import numpy as np
import numba
from joblib import Parallel, delayed
from numba import njit
from numba.core import types

from pyutilz.numbalib import python_dict_2_numba_dict

from .permutation import distribute_permutations
from ._internals import LARGE_CONST
from .evaluation import evaluate_gain


def get_fleuret_criteria_confidence_parallel(
    data_copy: np.ndarray,
    factors_nbins: np.ndarray,
    x: tuple,
    y: tuple,
    selected_vars: list,
    bootstrapped_gain: float,
    npermutations: int,
    max_failed: int,
    nexisting: int,
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    max_veteranes_interactions_order: int = 1,
    extra_knowledge_multipler: float = -1.0,
    sink_threshold: float = -1.0,
    cached_cond_MIs: dict = None,
    n_workers: int = 1,
    workers_pool: object = None,
    parallel_kwargs: dict = None,
    entropy_cache: dict = None,
    extra_x_shuffling: bool = True,
    dtype=np.int32,
) -> tuple:
    if parallel_kwargs is None:
        parallel_kwargs = {}
    nfailed = 0

    if workers_pool is None:
        workers_pool = Parallel(n_jobs=n_workers, **parallel_kwargs)

    gc.collect()
    res = workers_pool(
        delayed(parallel_fleuret)(
            data=data_copy,
            factors_nbins=factors_nbins,
            x=x,
            y=y,
            selected_vars=selected_vars,
            npermutations=worker_npermutations,
            bootstrapped_gain=bootstrapped_gain,
            max_failed=max_failed,
            nexisting=nexisting,
            mrmr_relevance_algo=mrmr_relevance_algo,
            mrmr_redundancy_algo=mrmr_redundancy_algo,
            max_veteranes_interactions_order=max_veteranes_interactions_order,
            extra_knowledge_multipler=extra_knowledge_multipler,
            sink_threshold=sink_threshold,
            cached_cond_MIs=dict(cached_cond_MIs),
            entropy_cache=dict(entropy_cache),
            extra_x_shuffling=extra_x_shuffling,
            dtype=dtype,
        )
        for worker_npermutations in distribute_permutations(npermutations=npermutations, n_workers=n_workers)
    )

    nchecked = 0
    for worker_nfailed, worker_i, entropy_cache_dict in res:
        nfailed += worker_nfailed
        nchecked += worker_i
        for key, value in entropy_cache_dict.items():
            entropy_cache[key] = value

    if nfailed >= max_failed:
        bootstrapped_gain = 0.0

    # Guard against empty result aggregation (nchecked == 0).
    confidence = (1 - nfailed / nchecked) if nchecked > 0 else 0.0

    return bootstrapped_gain, confidence, entropy_cache


def parallel_fleuret(
    data: np.ndarray,
    factors_nbins: np.ndarray,
    x: tuple,
    y: tuple,
    selected_vars: list,
    npermutations: int,
    bootstrapped_gain: float,
    max_failed: int,
    nexisting: int,
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    max_veteranes_interactions_order: int = 1,
    extra_knowledge_multipler: float = -1.0,
    sink_threshold: float = -1.0,
    cached_cond_MIs: dict = None,
    entropy_cache: dict = None,
    extra_x_shuffling: bool = True,
    dtype=np.int32,
):
    """Joblib worker: rebuild numba.typed.Dict from pickled Python dicts, run the njit core, return a Python dict for the parent's union."""
    data_copy = data.copy()

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

    nfailed, i = get_fleuret_criteria_confidence(
        data_copy=data_copy,
        factors_nbins=factors_nbins,
        x=x,
        y=y,
        selected_vars=selected_vars,
        npermutations=npermutations,
        bootstrapped_gain=bootstrapped_gain,
        max_failed=max_failed,
        nexisting=nexisting,
        mrmr_relevance_algo=mrmr_relevance_algo,
        mrmr_redundancy_algo=mrmr_redundancy_algo,
        max_veteranes_interactions_order=max_veteranes_interactions_order,
        cached_cond_MIs=cached_cond_MIs_dict,
        entropy_cache=entropy_cache_dict,
        extra_x_shuffling=extra_x_shuffling,
        dtype=dtype,
    )

    return nfailed, i, dict(entropy_cache_dict)


@njit()
def get_fleuret_criteria_confidence(
    data_copy: np.ndarray,
    factors_nbins: np.ndarray,
    x: tuple,
    y: tuple,
    selected_vars: list,
    npermutations: int,
    bootstrapped_gain: float,
    max_failed: int,
    nexisting: int,
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    max_veteranes_interactions_order: int = 1,
    extra_knowledge_multipler: float = -1.0,
    sink_threshold: float = -1.0,
    cached_cond_MIs: dict = None,
    entropy_cache: dict = None,
    extra_x_shuffling: bool = True,
    dtype=np.int32,
) -> tuple:
    """Sub to njit work with random shuffling as well.

    Zero-permutation guard returns ``(0, 0)`` cleanly. The legacy code raised ``UnboundLocalError`` on ``i`` because the for-loop never assigned it.
    """
    if npermutations == 0:
        return 0, 0

    nfailed = 0
    _i = 0
    for _i in range(npermutations):

        for idx in y:
            np.random.shuffle(data_copy[:, idx])

        if extra_x_shuffling:
            for idx in x:
                np.random.shuffle(data_copy[:, idx])

        stopped_early, current_gain, k, sink_reasons = evaluate_gain(
            current_gain=LARGE_CONST,
            last_checked_k=-1,
            X=x,
            y=y,
            best_gain=None,
            factors_data=data_copy,
            factors_nbins=factors_nbins,
            selected_vars=selected_vars,
            nexisting=nexisting,
            direct_gain=bootstrapped_gain,
            mrmr_relevance_algo=mrmr_relevance_algo,
            mrmr_redundancy_algo=mrmr_redundancy_algo,
            max_veteranes_interactions_order=max_veteranes_interactions_order,
            extra_knowledge_multipler=extra_knowledge_multipler,
            sink_threshold=sink_threshold,
            cached_cond_MIs=cached_cond_MIs,
            entropy_cache=entropy_cache,
            can_use_x_cache=not extra_x_shuffling,
            can_use_y_cache=False,
            confidence_mode=True,
        )

        if current_gain >= bootstrapped_gain:
            nfailed += 1
            if nfailed >= max_failed:
                break

    return nfailed, _i + 1
