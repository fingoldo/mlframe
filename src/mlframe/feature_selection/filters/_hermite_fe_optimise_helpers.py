"""Helpers carved out of ``_hermite_fe_optimise`` to keep that module under its size budget."""
from __future__ import annotations

import logging

logger = logging.getLogger("mlframe.feature_selection.filters.hermite_fe")


def _compact_finite_candidate_rows(P, KBF, finite, nc, X_batch, col_meta):
    """Move the finite candidate rows of X_batch to the front, recording each one's (pair, basis-function) index; returns the new row count."""
    for r_row in range(P * KBF):
        if finite[r_row]:
            if r_row != nc:
                X_batch[nc] = X_batch[r_row]
            col_meta.append((r_row // KBF, r_row % KBF))
            nc += 1
    return nc
