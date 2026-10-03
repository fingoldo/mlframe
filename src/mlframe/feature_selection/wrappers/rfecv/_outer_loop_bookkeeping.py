"""Per-iteration bookkeeping of the RFECV outer loop: best-so-far tracking and the wall-clock budget check.

Split from ``_fit_outer_loop`` so ``run_outer_loop_iteration`` keeps within its length and complexity ceilings.
"""
from __future__ import annotations

from mlframe.utils.budgets import active_budget

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.wrappers.rfecv")


def update_best_and_noimprove(self: Any, state: Any, final_score: float, n_features: int, was_stored: bool) -> None:
    """Record ``final_score`` as the new best or advance the no-improvement counter.

    Runs before any stop check: a stop used to return first, so the final iteration's subset never reached best_nfeatures/best_score
    (the SFFS swap pass seeds from them) and the checkpoint persisted a best one iteration stale.

    The no-improve counter only advances when the optimizer proposed something it had not seen before (``was_stored``) or when
    ``noimprove_counts_revisit`` is set: revisits of the same N with a worse subset used to trip ``max_noimproving_iters`` prematurely.
    """
    if final_score > state.best_score:
        state.best_score = final_score
        state.best_iter = state.nsteps
        state.best_nfeatures = n_features
        state.n_noimproving_iters = 0
    elif was_stored or getattr(self, "noimprove_counts_revisit", False):
        state.n_noimproving_iters += 1


def runtime_budget_exhausted(state: Any, max_runtime_mins: Any, start_time: float, iter_t0: float, now: float, verbose: Any) -> bool:
    """Record this iteration's duration (``now`` - ``iter_t0``; the caller owns the clock) and report whether ``max_runtime_mins`` is spent or the next iteration would overrun it.

    Sets ``state.stop_reason`` / ``state.ran_out_of_time`` when it returns True.
    """
    state.iter_durations.append(now - iter_t0)
    if active_budget(max_runtime_mins) is None or state.ran_out_of_time:
        return False
    budget_s = max_runtime_mins * 60
    elapsed_s = now - start_time
    mean_iter_s = float(np.mean(state.iter_durations))
    if elapsed_s > budget_s:
        state.stop_reason = f"max_runtime_mins={max_runtime_mins:_.1f} reached ({elapsed_s / 60:_.1f} min elapsed)"
    elif elapsed_s + mean_iter_s > budget_s:
        state.stop_reason = (
            f"max_runtime_mins={max_runtime_mins:_.1f}: another iteration (mean {mean_iter_s / 60:_.1f} min) "
            f"would end past the budget ({elapsed_s / 60:_.1f} min elapsed)"
        )
    state.ran_out_of_time = state.stop_reason is not None
    if state.ran_out_of_time and verbose:
        logger.info("RFECV: stopping, %s.", state.stop_reason)
    return bool(state.ran_out_of_time)
