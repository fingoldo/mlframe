"""The confirmation loop's redirect target: the best already-scored candidate that is still eligible.

Carved out of ``evaluation.py`` (1k-LOC budget); ``evaluation.find_best_partial_gain`` re-exports it.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

from ._internals import LARGE_CONST

logger = logging.getLogger(__name__)


def find_best_partial_gain(
    partial_gains: dict, failed_candidates: set, added_candidates: set, candidates: list, selected_vars: list, skip_indices: tuple = (),
    dcd_state=None,
) -> Tuple[float, Any]:
    """Find the highest-scoring already-evaluated-but-not-yet-confirmed candidate in ``partial_gains`` (used to redirect the confirmation loop to the next-best option when the current top candidate fails confirmation), excluding failed/added/skip_indices candidates and any candidate DCD has since pruned."""
    # a DCD-pruned candidate must NOT be returned as a
    # redirect target. ``partial_gains`` persists across the confirmation
    # ``while`` retries within one interactions-order; when DCD prunes a
    # candidate AFTER it was scored (``discover_cluster_members`` sets
    # ``pool_pruned_mask`` once a same-cluster member is selected), the
    # candidate is skipped from RE-scoring (``should_skip_candidate``) but its
    # now-STALE high partial gain stays in the dict. Pre-fix
    # ``find_best_partial_gain`` had no view of the prune mask, so it kept
    # returning that pruned candidate's stale gain as "the best other option",
    # the confirmation loop redirected to it forever (it can never be confirmed
    # - it is skipped), and the genuinely-good candidate that DID confirm was
    # never committed -> the screen stopped early and dropped real signal
    # (sensor-mesh: 6 features -> 2, -4% downstream AUC). Skipping pruned
    # candidates here closes the redirect loop. ``None`` dcd_state is the
    # legacy/bit-stable path (no DCD).
    _should_be_pruned: Any = None
    if dcd_state is not None:
        try:
            from ._dynamic_cluster_discovery import should_be_pruned as _should_be_pruned
        except Exception as _dcd_import_exc:
            logger.debug("find_best_partial_gain: DCD prune-mask import failed (%s); DCD short-circuit disabled for this call.", _dcd_import_exc)
            _should_be_pruned = None
    best_partial_gain = -LARGE_CONST
    best_key = None
    # Hoist selected_vars to a set: the inner ``subel in selected_vars`` membership is O(len) on the list, and it runs
    # per sub-element per candidate per confirmation-retry -> an O(1) set lookup is ~1.6x on a wide candidate pool
    # (bit-identical - same membership test). selected_vars is small so building the set once is negligible.
    _selected_set = set(selected_vars)
    for key, value in partial_gains.items():
        if (key not in failed_candidates) and (key not in added_candidates) and (key not in skip_indices):
            skip_cand = False
            for subel in candidates[key]:
                if subel in _selected_set:
                    skip_cand = True  # the sub-element or var itself is already selected.
                    break
            if skip_cand:
                continue
            if _should_be_pruned is not None and _should_be_pruned(dcd_state, candidates[key]):
                continue  # DCD-pruned: out of contention, never a valid redirect target.
            partial_gain, _ = value
            if partial_gain > best_partial_gain:
                best_partial_gain = partial_gain
                best_key = key
    return best_partial_gain, best_key
