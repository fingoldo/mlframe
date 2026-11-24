"""The target-transform search inside MRMR: ``MRMR.target_transform_`` (PLAN_target_transform_in_mrmr, step 3) and the composite credit.

Off by default (``target_transform_search=False``): the attribute is ``None``. On, for a continuous regression target and at least one numeric column, the recommender of
:mod:`mlframe.feature_selection.filters._target_transform_ll` (Jacobian log-likelihood ranking of the inverse links of the admissible unary forms, a sign-flip null, a held-out MAE gate) runs ONCE per fit on at most
``_MAX_ROWS`` rows of the selected raw numeric columns with an ADDITIVE-SPLINE head, and the verdict is stored; the target the estimator was given is never changed and the selection is the same with the flag on or off.

The spline head is what lets a single-index target ``y = U(sum_i u_i(x_i))`` show its link: ``exp(0.5 (sin(x1) + log(x2)))`` has no link on the raw columns with a linear head (the sine and the log are in the way) and
the link ``log`` with additive splines. When a link is accepted, the unary credit (``_fe_unary_credit``) is run on the transformed target ``g(y)`` (:func:`link_target`): the forms it names are the ones a linear head
on ``g(y)`` needs (``sin(x1)``, ``log(x2)``), not the approximations a multiplicative ``y`` gives (``sqrt(x2)``, products). A failure is logged and leaves ``None``: it must not break a fit.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

__all__ = ["find_target_transform", "link_target", "record_target_transform"]

_MAX_ROWS = 6000  # a seeded subsample of this size keeps the search near one second
_MAX_COLUMNS = 30  # the selected raw numeric columns the search sees (the first ones in selection order)
_MIN_DISTINCT_TARGET = 20  # fewer distinct values than this is a class label or a count, not a continuous target


def find_target_transform(estimator: Any, X: Any, y: Any, names: "list[str]") -> "Optional[dict]":
    """The verdict of the search for this fit (computed once, kept on ``estimator._target_transform_cache_``), or ``None`` when the flag is off or the target is not continuous.

    ``names`` are the raw columns to search on, in selection order; the numeric ones are used (at most ``_MAX_COLUMNS``)."""
    if not bool(getattr(estimator, "target_transform_search", False)):
        return None
    cached = getattr(estimator, "_target_transform_cache_", None)
    if cached is not None:
        return cached or None
    estimator._target_transform_cache_ = {}  # an empty dict marks "searched, nothing to report": a second call in the same fit does not search again
    try:
        y_raw = np.asarray(y)
        if y_raw.dtype.kind not in "fiu":
            return None
        y_arr = y_raw.astype(np.float64).ravel()
        if np.unique(y_arr[np.isfinite(y_arr)]).size < _MIN_DISTINCT_TARGET:
            return None
        cols = [c for c in dict.fromkeys(str(n) for n in names) if c in getattr(X, "columns", []) and np.asarray(X[c]).dtype.kind in "fiu"][:_MAX_COLUMNS]
        if not cols:
            return None
        seed = int(getattr(estimator, "random_seed", 0) or 0)
        n = len(y_arr)
        idx = np.random.default_rng(seed).choice(n, size=_MAX_ROWS, replace=False) if n > _MAX_ROWS else np.arange(n)
        mat = np.column_stack([np.asarray(X[c], dtype=np.float64)[idx] for c in cols])
        from ._target_transform_ll import recommend_target_transform

        choice = recommend_target_transform(mat, y_arr[idx], seed=seed, max_rows=_MAX_ROWS, head="spline")
        verdict = {
            "name": choice.name,
            "accepted": not choice.abstained,
            "params": dict(choice.params),
            "ll_gain": choice.ll_gain,
            "null_threshold": choice.null_threshold,
            "p_value": choice.p_value,
            "metric": choice.metric,
            "head": "spline",
            "err_identity": choice.err_identity,
            "err_chosen": choice.err_chosen,
            "reason": choice.reason,
            "best_candidate": choice.best_candidate,
            "gains": dict(choice.gains),
            "transform": choice.transform,
            "n_rows_used": len(idx),
            "n_columns_used": len(cols),
            "columns_used": cols,
        }
        estimator._target_transform_cache_ = verdict
        return verdict
    except Exception as exc:  # metadata only
        log_throttle(logger, "target_transform.record", logging.WARNING, "target_transform_search: the search failed (%r); target_transform_ is None", exc)
        return None


def link_target(verdict: "Optional[dict]", y: Any) -> Any:
    """``g(y)`` for an accepted verdict (finite values only), else ``y`` unchanged: the target the unary credit is run on when a link was found."""
    if not verdict or not verdict.get("accepted") or verdict.get("transform") is None:
        return y
    values = np.asarray(y, dtype=np.float64).ravel()
    with np.errstate(all="ignore"):
        linked = np.asarray(verdict["transform"].forward(values), dtype=np.float64)
    return linked if linked.shape == values.shape and np.isfinite(linked).all() else y


def record_target_transform(estimator: Any, X: Any, y: Any) -> None:
    """Set ``estimator.target_transform_`` to the search verdict (computing it if the credit did not), or ``None`` when the flag is off or the target is not continuous."""
    estimator.target_transform_ = None
    if not bool(getattr(estimator, "target_transform_search", False)):
        return
    names = [str(c) for c in getattr(estimator, "feature_names_in_", [])]
    support = [int(i) for i in (getattr(estimator, "support_", None) if getattr(estimator, "support_", None) is not None else [])]
    selected = [names[i] for i in support if 0 <= i < len(names)]
    verdict = find_target_transform(estimator, X, y, selected + [c for c in names if c not in selected])
    estimator.target_transform_ = dict(verdict) if verdict else None
