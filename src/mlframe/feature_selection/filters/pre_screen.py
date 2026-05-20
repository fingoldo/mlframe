"""Backward-compat shim. The canonical home for the unsupervised pre-screen filter
moved to ``mlframe.feature_selection.pre_screen`` (one level up) so importing it
does not trigger ``filters/__init__.py``'s star-import of ``_legacy``, which in turn
cascades into ``_numba_utils`` and 32 @njit decorator inits at module load (~0.8s on
cold-start measured 2026-05-20 on Windows; numba ``caching.py`` stat-check storm).

Old import path still works:

    from mlframe.feature_selection.filters.pre_screen import compute_unsupervised_drops, apply_drops

but new code should prefer the shorter:

    from mlframe.feature_selection.pre_screen import compute_unsupervised_drops, apply_drops
"""
from __future__ import annotations

from mlframe.feature_selection.pre_screen import (  # noqa: F401
    apply_drops,
    compute_unsupervised_drops,
)

__all__ = ["apply_drops", "compute_unsupervised_drops"]
