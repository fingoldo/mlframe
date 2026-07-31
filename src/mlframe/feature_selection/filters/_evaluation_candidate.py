"""Per-candidate bookkeeping helpers for the MRMR greedy screen.

Carved out of ``evaluation.py`` (keeping that file under the 1k-LOC gate). These three functions are pure,
self-contained leaf helpers (the only cross-module dependency is a lazy in-body import of the DCD prune-mask
check, avoided at import time to keep this a dependency-free leaf); ``evaluation.py`` re-imports them so every
existing ``from .evaluation import should_skip_candidate`` / ``handle_best_candidate`` / ``get_candidate_name``
call site is unchanged.
"""
from __future__ import annotations

import logging
from timeit import default_timer as timer
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from ._dynamic_cluster_discovery import DCDState

logger = logging.getLogger(__name__)


def get_candidate_name(candidate_indices: Sequence, factors_names: Sequence[str]) -> str:
    """Render a candidate (single index or k-way interaction tuple) as a human-readable ``"-"``-joined name for logging, resolving each factor index against ``factors_names``."""
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
    engineered_lineage: dict | None = None,
    dcd_state: "DCDState | None" = None,
) -> tuple:
    """Decide if current candidate for predictors should be skipped (already accepted, failed, or computed).

    ``engineered_lineage``: optional ``{engineered_idx -> frozenset(parent_indices)}`` from the cat-FE step. When set, a k-way candidate is skipped if it
    combines an engineered column with one of its own parents (conditional MI degenerates and confidence gates waste budget). Legacy/numeric path leaves it ``None``.

    ``dcd_state``: optional ``DCDState`` reference. When provided, a
    candidate (single index OR k-way tuple) is skipped if its ``pool_pruned_mask``
    bit is set per ``should_be_pruned`` semantics (a tuple of indices is
    skipped iff ALL components pruned). Bit-stable when ``None``.
    """

    nexisting = 0

    if (cand_idx in failed_candidates) or (cand_idx in added_candidates) or expected_gains[cand_idx]:
        return True, nexisting

    # DCD prune-mask short-circuit: consult the mask instead of mutating the candidates list.
    if dcd_state is not None:
        try:
            from ._dynamic_cluster_discovery import should_be_pruned as _should_be_pruned
            target = X if interactions_order > 1 else int(cand_idx)
            if _should_be_pruned(dcd_state, target):
                return True, nexisting
        except Exception as _dcd_exc:  # nosec B110 - non-trivial body
            # DCD is best-effort; never break candidate evaluation.
            logger.debug("should_skip_candidate: DCD prune-mask lookup failed (%s); skipping DCD short-circuit.", _dcd_exc)

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
        skip_cand_flags = [(subel in selected_vars) for subel in X]
        nexisting = sum(skip_cand_flags)
        if (only_unknown_interactions and any(skip_cand_flags)) or all(skip_cand_flags):
            return True, nexisting

    return False, nexisting


def handle_best_candidate(
    current_gain: float,
    best_gain: float,
    X: Sequence,
    best_candidate: Sequence,
    factors_names: list,
    verbose: int = 1,
    ndigits: int = 5,
    max_runtime_mins: float | None = None,
    start_time: float | None = None,
    min_relevance_gain: float | None = None,
) -> tuple:
    """Update the running best-candidate/best-gain tracker for the current MRMR search iteration, logging progress when verbose, and signal whether the ``max_runtime_mins`` budget has been exceeded (early-stop check). Returns ``(best_gain, best_candidate, run_out_of_time)``."""
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

    if max_runtime_mins and start_time is not None and not run_out_of_time:
        run_out_of_time = (timer() - start_time) > max_runtime_mins * 60

    return best_gain, best_candidate, run_out_of_time
