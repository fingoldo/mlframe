"""Helpers shared by the dense and the classification GPU-resident usability greedy searches."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _shrink_shortlist_to_ram_budget(K, P, n, shortlist):
    """Shrink the candidate buffer width K (and the shortlist with it) when a K-wide float64 copy of the rows would not fit the RAM budget."""
    try:
        from mlframe.feature_selection.filters.feature_engineering import _can_hoist_shared_buffer, _fe_effective_buffer_budget_bytes
        _k_eff = max(1, min(int(K), P))
        _can, _need, _avail = _can_hoist_shared_buffer(n * _k_eff * 8, n_workers=1)
        if (not _can) and _avail > 0:
            _budget = _fe_effective_buffer_budget_bytes(_avail, n_workers=1)
            _k_fit = int(_budget // (n * 8)) if _budget > 0 else 1
            if _k_fit < _k_eff:
                K = max(1, _k_fit)
                shortlist = min(int(shortlist), max(int(K), 1))
    except Exception as e:  # nosec B110 - best-effort path
        logger.debug("shortlist auto-sizing failed, keeping the caller-provided shortlist: %s", e)
    return K, shortlist
