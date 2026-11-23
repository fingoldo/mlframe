"""Metadata step of the target-transform search: ``MRMR.target_transform_`` (PLAN_target_transform_in_mrmr, step 3).

Off by default (``target_transform_search=False``): the attribute is ``None``. On, for a continuous regression target and at least one numeric column, the search of
:mod:`mlframe.feature_selection.filters._target_transform_search` runs on at most ``_MAX_ROWS`` rows of the raw numeric columns and the verdict is stored; the target the estimator was given is never changed and
the selection is the same with the flag on or off. A failure is logged and leaves ``None``: it must not break a fit.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

__all__ = ["record_target_transform"]

_MAX_ROWS = 20000
_MIN_DISTINCT_TARGET = 20  # fewer distinct values than this is a class label or a count, not a continuous target


def record_target_transform(estimator: Any, X: Any, y: Any) -> None:
    """Set ``estimator.target_transform_`` to the search verdict, or ``None`` when the flag is off or the target is not continuous."""
    estimator.target_transform_ = None
    if not bool(getattr(estimator, "target_transform_search", False)):
        return
    try:
        y_raw = np.asarray(y)
        if y_raw.dtype.kind not in "fiu":
            return
        y_arr = y_raw.astype(np.float64).ravel()
        if np.unique(y_arr[np.isfinite(y_arr)]).size < _MIN_DISTINCT_TARGET:
            return
        names = [str(c) for c in getattr(estimator, "feature_names_in_", [])]
        cols = [c for c in names if c in getattr(X, "columns", []) and np.asarray(X[c]).dtype.kind in "fiu"]
        if not cols:
            return
        seed = int(getattr(estimator, "random_seed", 0) or 0)
        n = len(y_arr)
        idx = np.random.default_rng(seed).choice(n, size=_MAX_ROWS, replace=False) if n > _MAX_ROWS else np.arange(n)
        mat = np.column_stack([np.asarray(X[c], dtype=np.float64)[idx] for c in cols])
        from ._target_transform_search import default_transforms, propose_target_transforms, search_target_transform

        pool = default_transforms()
        chosen, gates = search_target_transform(mat, y_arr[idx], pool, seed=seed)
        estimator.target_transform_ = {
            "name": chosen.name,
            "accepted": chosen.name != "identity",
            "proposals": propose_target_transforms(mat, y_arr[idx], pool),
            "gates": [{"name": g.name, "mean_loss_difference": g.mean_loss_difference, "upper_limit": g.upper_limit, "accepted": g.accepted} for g in gates],
            "n_rows_used": len(idx),
            "n_columns_used": len(cols),
        }
    except Exception as exc:  # metadata only
        estimator.target_transform_ = None
        log_throttle(logger, "target_transform.record", logging.WARNING, "target_transform_search: the search failed (%r); target_transform_ is None", exc)
