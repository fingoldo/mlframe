"""One splitter factory for every discovery and ensemble CV: groups and time order are honoured the same way everywhere.

The tiny rerank used GroupKFold / TimeSeriesSplit because shuffled folds let per-group memorisation and future-to-past
leakage look like skill, while the WAIC tie-break, the auto-chain "beats both singles" decision and other CVs scored the same
specs with shuffled KFold, so they rewarded exactly what the rerank was fixed to stop rewarding. Every CV now asks this
module for its splitter.

Precedence: groups (when at least ``n_splits`` distinct groups exist) over time order over shuffled KFold, matching the
rerank. ``contiguous=True`` asks for unshuffled contiguous blocks, for components whose state runs along the rows (a
recurrent transform's EWMA / rolling window has no meaning over a scattered fold).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def make_discovery_splitter(n_splits: int, *, groups: Any = None, time_aware: bool = False, random_state: int = 0, contiguous: bool = False) -> tuple[Any, Any]:
    """``(splitter, groups_for_split)``: GroupKFold with the groups, TimeSeriesSplit, contiguous KFold, or shuffled KFold.

    ``groups_for_split`` is the array to hand to ``splitter.split(X, groups=...)``, or ``None``. Fewer distinct groups than
    folds cannot be grouped; that falls through to the next rule with a WARNING rather than silently.
    """
    from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

    n_splits = int(n_splits)
    if groups is not None:
        g = np.asarray(groups)
        n_groups = int(np.unique(g).size)
        if n_groups >= n_splits:
            return GroupKFold(n_splits=n_splits), g
        logger.warning("make_discovery_splitter: only %d distinct group(s) for %d folds; group separation is NOT enforced for this CV.", n_groups, n_splits)
    if time_aware:
        return TimeSeriesSplit(n_splits=n_splits), None
    if contiguous:
        return KFold(n_splits=n_splits, shuffle=False), None
    return KFold(n_splits=n_splits, shuffle=True, random_state=int(random_state)), None


def discovery_splits(n_rows: int, n_splits: int, *, groups: Any = None, time_aware: bool = False, random_state: int = 0, contiguous: bool = False) -> list:
    """The ``(train_idx, val_idx)`` folds of ``make_discovery_splitter`` over ``n_rows`` rows (groups aligned to those rows)."""
    splitter, g = make_discovery_splitter(n_splits, groups=groups, time_aware=time_aware, random_state=random_state, contiguous=contiguous)
    placeholder = np.zeros((int(n_rows), 1))
    return list(splitter.split(placeholder, groups=g)) if g is not None else list(splitter.split(placeholder))
