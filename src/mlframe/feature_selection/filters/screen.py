"""mRMR screening orchestrator.

Public functions:

* ``screen_predictors`` -- main screening loop. Walks interaction orders, selects candidates, runs the confidence step, accumulates ``selected_vars``.
* ``postprocess_candidates`` -- post-screening filtering helper.

mRMR phase narrative: partial = MI(candidate, target | already-selected); top_k = K candidates with highest partial; confirm = full-permutation test on top_k;
postprocess = filter weak / duplicates from the confirmed set.
"""
from __future__ import annotations

import gc
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from os.path import exists
from timeit import default_timer as timer
from typing import Sequence

import numba
import numpy as np
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from numba import njit
from numba.core import types

from pyutilz.numbalib import (
    generate_combinations_recursive_njit,
    python_dict_2_numba_dict,
    set_numba_random_seed,
)
from pyutilz.parallel import split_list_into_chunks
from pyutilz.pythonlib import sort_dict_by_value
from pyutilz.system import tqdmu

from mlframe.utils.misc import set_random_seed

from ._internals import (
    LARGE_CONST,
    MAX_CONFIRMATION_CAND_NBINS,
    MAX_ITERATIONS_TO_TRACK,
    MAX_JOBLIB_NBYTES,
    NMAX_NONPARALLEL_ITERS,
    sanitize,
)
from ._numba_utils import arr2str, count_cand_nbins, unpack_and_sort
from .evaluation import (
    evaluate_candidate,
    evaluate_candidates,
    evaluate_gain,
    find_best_partial_gain,
    get_candidate_name,
    handle_best_candidate,
    should_skip_candidate,
)
from .fleuret import (
    get_fleuret_criteria_confidence,
    get_fleuret_criteria_confidence_parallel,
    parallel_fleuret,
)
from .gpu import mi_direct_gpu
from .info_theory import (
    compute_mi_from_classes,
    conditional_mi,
    entropy,
    merge_vars,
)
from .permutation import distribute_permutations, mi_direct

logger = logging.getLogger(__name__)


def _pool_warmup_noop(i):
    """Top-level no-op for joblib worker pool warmup. Must be module-level (not a closure) so cloudpickle can serialise it for worker transmission."""
    return i


from contextlib import contextmanager


@contextmanager
def _preserve_global_numpy_rng_state(seed: int | None):
    """Snapshot and restore ``np.random``'s global MT19937 state around a block.

    ``screen_predictors`` historically reseeded the process-global RNG to make permutation-based confidence
    checks deterministic; that bled state into the rest of the caller's process. The snapshot+restore pattern
    keeps the inner reseed (downstream ``np.random.shuffle`` consumers depend on it) while leaving the global
    state byte-identical from the caller's POV. No-op when ``seed`` is ``None``.
    """
    if seed is None:
        yield
        return
    snapshot = np.random.get_state()
    # Wave 49 (2026-05-20): capture entropy-derived restoration seeds for numba +
    # cupy too -- those exposed no portable get_state and were not previously
    # restored on exit, leaving the caller's downstream numba/cupy stream shifted.
    import os as _os, struct as _struct
    _numba_restore_seed = _struct.unpack("<Q", _os.urandom(8))[0]
    _cp_restore_seed = _struct.unpack("<Q", _os.urandom(8))[0]
    _cp_module = None
    try:
        np.random.seed(seed)
        set_numba_random_seed(seed)
        try:
            import cupy as _cp  # local import to avoid hard dep
            _cp.random.seed(seed)
            _cp_module = _cp
        except ImportError:
            pass
        yield
    finally:
        np.random.set_state(snapshot)
        try:
            set_numba_random_seed(int(_numba_restore_seed))
        except Exception:
            pass
        if _cp_module is not None:
            try:
                _cp_module.random.seed(int(_cp_restore_seed))
            except Exception:
                pass


@dataclass
class ScreenState:
    """Typed snapshot of ``screen_predictors`` shared state. Not currently routed through by the orchestrator; kept as a stable IO contract for phase helpers.

    Fields: running selection, MI caches (direct / confident / conditional), entropy cache, partial gains, failed / added candidate sets, target encoding caches
    (CPU vs GPU distinguished at boundary), shuffle scratch buffer for the fleuret confidence pass.

    Invariants:
    * ``set(added_candidates) == set(selected_vars)`` after the postprocess step.
    * ``failed_candidates`` and ``added_candidates`` are disjoint.
    * ``partial_gains[i]`` is undefined whenever ``i in failed_candidates``.
    """
    selected_vars: list = field(default_factory=list)
    # MI cache attributes: name kept mixedCase to match the kwarg-name contract threaded through evaluate_candidate / mi_direct / mi_direct_gpu (renaming would cascade through ~40 call sites in screen/evaluation/gpu/permutation).
    cached_MIs: dict = field(default_factory=dict)  # noqa: N815
    cached_confident_MIs: dict = field(default_factory=dict)  # noqa: N815
    cached_cond_MIs: dict = field(default_factory=dict)  # noqa: N815
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


# Re-exports for backwards compatibility (discretisation helpers live in their own module).
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


# ----------------------------------------------------------------------
# Sibling-module re-export. The 842-LOC ``screen_predictors`` body lives
# in ``_screen_predictors.py`` so this file stays below the 1k-LOC
# monolith threshold.
# ----------------------------------------------------------------------
from ._screen_predictors import screen_predictors  # noqa: E402,F401
