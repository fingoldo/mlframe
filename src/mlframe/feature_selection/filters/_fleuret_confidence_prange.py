"""Thread-parallel twin of ``_numba_utils.get_fleuret_criteria_confidence`` for the compacted inputs of the Fleuret conditional recheck.

The serial kernel shuffles ``y`` (and ``x``) of one shared array per permutation and stops at the permutation where the failures reach ``max_failed``.
Every permutation is seeded on its own (``base_seed``, ``perm_offset + i + 1``), so here each thread shuffles its OWN copy of the (few, compacted)
columns, with its own entropy / conditional-MI caches (they hold entropies of unshuffled columns and conditional MIs: a fresh cache recomputes the same
values), and records whether permutation ``i`` failed. The flags are then counted in permutation order with the serial stop rule, so ``(nfailed,
nchecked)`` equal the serial kernel's; the cost is at most one block of extra permutations past the stop. ``screen_predictors`` ran the serial kernel on
one core (the joblib twin needs ``n_jobs > 1``): 1.2 s of a warm 100k fit at 20% CPU.
"""

from __future__ import annotations

import numba
import numpy as np
from numba import njit, prange, types

from mlframe._numba_parallel_guard import parallel_kernel_entry

from ._numba_utils import LARGE_CONST, _fleuret_shuffle_col_lcg, evaluate_gain
from typing import Any


@njit(cache=True, parallel=True)
def _block_flags(data, factors_nbins, x, y, selected_vars_arr, b0, b1, bootstrapped_gain, nexisting, relevance_algo, redundancy_algo, max_order,
                 extra_knowledge_multipler, sink_threshold, extra_x_shuffling, base_seed, perm_offset, use_su, use_mm, use_jmim, jmim_discount_only):
    """``flags[i - b0]`` = permutation ``i`` of ``[b0, b1)`` reached ``bootstrapped_gain`` (a failure), each on a private copy of the columns."""
    flags = np.zeros(b1 - b0, dtype=np.bool_)
    for k in prange(b1 - b0):
        i = b0 + k
        own = data.copy()  # loop-invariant-ok: each permutation shuffles its own private copy of the columns
        ent = numba.typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)  # Intentional float64: numba typed dict of entropy values, float64 like the serial kernel
        cond = numba.typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
        state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(perm_offset + i + 1)
        for idx in y:
            state = np.uint64(_fleuret_shuffle_col_lcg(own[:, idx], state))
        if extra_x_shuffling:
            for idx in x:
                state = np.uint64(_fleuret_shuffle_col_lcg(own[:, idx], state))
        _stopped, gain, _k, _reasons = evaluate_gain(
            current_gain=LARGE_CONST,
            last_checked_k=-1,
            X=x,
            y=y,
            best_gain=None,
            factors_data=own,
            factors_nbins=factors_nbins,
            selected_vars=selected_vars_arr,
            nexisting=nexisting,
            direct_gain=bootstrapped_gain,
            mrmr_relevance_algo=relevance_algo,
            mrmr_redundancy_algo=redundancy_algo,
            max_veteranes_interactions_order=max_order,
            extra_knowledge_multipler=extra_knowledge_multipler,
            sink_threshold=sink_threshold,
            cached_cond_MIs=cond,
            entropy_cache=ent,
            can_use_x_cache=not extra_x_shuffling,
            can_use_y_cache=False,
            confidence_mode=True,
            use_su=use_su,
            use_mm=use_mm,
            use_jmim=use_jmim,
            jmim_discount_only=jmim_discount_only,
        )
        flags[k] = gain >= bootstrapped_gain
    return flags


def fleuret_confidence_prange(data_copy: Any, factors_nbins: Any, x: Any, y: Any, selected_vars: Any, npermutations: Any, bootstrapped_gain: Any, max_failed: Any, nexisting: Any,
                              mrmr_relevance_algo: str="fleuret", mrmr_redundancy_algo: str="fleuret", max_veteranes_interactions_order: int=1,
                              extra_knowledge_multipler: float=-1.0, sink_threshold: float=-1.0, extra_x_shuffling: bool=True, base_seed: int=0, perm_offset: int=0,
                              use_su: bool=False, use_mm: bool=False, use_jmim: bool=False, jmim_discount_only: bool=False, block: Any=None) -> "tuple[int, int]":
    """``(nfailed, nchecked)`` of ``get_fleuret_criteria_confidence`` on the same inputs, the permutations of a block run on all threads."""
    if npermutations == 0:
        return 0, 0
    block = int(block) if block else numba.get_num_threads()
    sel = np.asarray(list(selected_vars), dtype=np.int64)
    xs = np.asarray(x, dtype=np.int64)
    ys = np.asarray(y, dtype=np.int64)
    nfailed = 0
    for b0 in range(0, int(npermutations), block):
        b1 = min(int(npermutations), b0 + block)
        # The prange block runs on bootstrap-stability threads when MRMR fits concurrently: enter it through the guard.
        with parallel_kernel_entry():
            flags = _block_flags(data_copy, factors_nbins, xs, ys, sel, b0, b1, float(bootstrapped_gain), int(nexisting), str(mrmr_relevance_algo),
                                 str(mrmr_redundancy_algo), int(max_veteranes_interactions_order), float(extra_knowledge_multipler), float(sink_threshold),
                                 bool(extra_x_shuffling), np.uint64(base_seed), int(perm_offset), bool(use_su), bool(use_mm), bool(use_jmim),
                                 bool(jmim_discount_only))
        for k in range(b1 - b0):
            if flags[k]:
                nfailed += 1
                if nfailed >= max_failed:
                    return nfailed, b0 + k + 1
    return nfailed, int(npermutations)
