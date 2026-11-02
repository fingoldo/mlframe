"""Per-fold score list that keeps fold order when worker threads finish out of order."""

from __future__ import annotations

from bisect import bisect_right
from typing import Any, List


class FoldOrderedScores(list):
    """A list whose :meth:`add` places each score at its fold's position, whichever fold finishes first; plain ``append`` still works."""

    def __init__(self) -> None:
        super().__init__()
        self._folds: List[Any] = []

    def add(self, nfold: Any, value: float) -> None:
        """Insert ``value`` so the list stays sorted by fold index."""
        pos = bisect_right(self._folds, nfold)
        self._folds.insert(pos, nfold)
        self.insert(pos, value)


def add_fold_score(scores: list, nfold: Any, value: float) -> None:
    """Store a fold score by fold position when ``scores`` is order-aware, else append."""
    if isinstance(scores, FoldOrderedScores):
        scores.add(nfold, value)
    else:
        scores.append(value)
