"""The target-transform search inside MRMR: ``MRMR.target_transform_`` (PLAN_target_transform_in_mrmr, step 3) and the composite credit.

On by default (``target_transform_search=True``; ``False`` leaves the attribute ``None``). For a continuous regression target and at least one numeric column, the recommender of
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

__all__ = ["choose_credit_target", "find_target_transform", "link_target", "record_target_transform"]

_MAX_ROWS = 6000  # a seeded subsample of this size keeps the search near one second
_MAX_COLUMNS = 30  # the selected raw numeric columns the search sees (the first ones in selection order)
_MIN_DISTINCT_TARGET = 20  # fewer distinct values than this is a class label or a count, not a continuous target


def _log_verdict(verdict: dict) -> None:
    """Write the verdict of the search to the log at INFO: the transform and the held-out errors when one is accepted, the reason when the identity stays."""
    if verdict["accepted"]:
        logger.info(
            "MRMR target transform: %s%s accepted (held-out %s of the original-scale prediction %.4g against %.4g for the identity, log-likelihood gain %.4f, p %.3f, %d rows, %d columns, additive-spline head); "
            "a linear or additive head predicts better on %s(y); the target given to MRMR is unchanged (target_transform_['transform'] holds the forward and the inverse)",
            verdict["name"],
            f" {verdict['params']}" if verdict["params"] else "",
            verdict["metric"].upper(),
            verdict["err_chosen"],
            verdict["err_identity"],
            verdict["ll_gain"],
            verdict["p_value"],
            verdict["n_rows_used"],
            verdict["n_columns_used"],
            verdict["name"],
        )
    else:
        logger.info("MRMR target transform: identity kept (%s); held-out %s %.4g, best candidate %s", verdict["reason"] or "no transform predicts better", verdict["metric"].upper(), verdict["err_identity"], verdict["best_candidate"])


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
        # Intentional float64: held-out error statistics of the link verdict
        y_arr = y_raw.astype(np.float64).ravel()
        if np.unique(y_arr[np.isfinite(y_arr)]).size < _MIN_DISTINCT_TARGET:
            return None
        cols = [c for c in dict.fromkeys(str(n) for n in names) if c in getattr(X, "columns", []) and np.asarray(X[c]).dtype.kind in "fiu"][:_MAX_COLUMNS]
        if not cols:
            return None
        seed = int(getattr(estimator, "random_seed", 0) or 0)
        n = len(y_arr)
        idx = np.random.default_rng(seed).choice(n, size=_MAX_ROWS, replace=False) if n > _MAX_ROWS else np.arange(n)
        # Intentional float64: held-out error statistics of the link verdict
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
        _log_verdict(verdict)
        return verdict
    except Exception as exc:  # metadata only
        log_throttle(logger, "target_transform.record", logging.WARNING, "target_transform_search: the search failed (%r); target_transform_ is None", exc)
        return None


#: The unary credit is run on the linked target only when the forms it names there predict the ORIGINAL target at least this much better (held-out MAE of a ridge on the selected raw columns and the credited columns,
#: the prediction inverted through the link) than the forms named on the target itself. Measured: for ``y = exp(0.5 (sin(x1) + log(x2) + ..))`` the link ``log`` leaves 92 percent of the identity's
#: MAE and its forms (``sin``, ``log``) predict with a fraction of the error of the forms the raw target gives (``sin``, ``sqrt``), whereas for a SUM or PRODUCT of forms the link the spline head accepts (Yeo-Johnson,
#: asinh, 91 to 93 percent) is spurious and its forms (``cbrt``, ``log`` where the inputs were ``sqrt``, ``recip``) predict no better. The MAE ratio of the link alone cannot tell the two apart; the utility of the forms can.
LINK_CREDIT_MIN_GAIN = 0.1


def link_target(verdict: "Optional[dict]", y: Any) -> Any:
    """``g(y)`` for an accepted verdict (finite values only), else ``y`` unchanged: the CANDIDATE target of the unary credit when a link was found (see :func:`choose_credit_target`)."""
    if not verdict or not verdict.get("accepted") or verdict.get("transform") is None:
        return y
    # Intentional float64: held-out error statistics of the link verdict
    values = np.asarray(y, dtype=np.float64).ravel()
    with np.errstate(all="ignore"):
        # Intentional float64: held-out error statistics of the link verdict
        linked = np.asarray(verdict["transform"].forward(values), dtype=np.float64)
    return linked if linked.shape == values.shape and np.isfinite(linked).all() else y


def _replay(apply_recipe: Any, recipe: Any, frame: Any) -> "Optional[np.ndarray]":
    """The column of ``recipe`` on ``frame``, or ``None`` when it cannot be replayed (such a recipe is simply not a regressor)."""
    try:
        # Intentional float64: held-out error statistics of the link verdict
        return np.asarray(apply_recipe(recipe, frame), dtype=np.float64).ravel()
    except Exception as exc:
        log_throttle(logger, "target_transform.forms_mae", logging.WARNING, "target_transform_search: recipe %r not replayed (%r)", getattr(recipe, "name", "?"), exc)
        return None


def _forms_mae(mrmr: Any, X: Any, y_original: np.ndarray, y_fit: np.ndarray, names: "list[str]", inverse: Any, seed: int) -> float:
    """Held-out MAE on the original scale of a ridge on ``names`` and the credited columns that a credit run on ``y_fit`` produces (5 folds, at most ``_MAX_ROWS`` rows); ``inf`` when nothing can be fitted."""
    from types import SimpleNamespace

    from ._fe_unary_credit import credit_unary_forms
    from .engineered_recipes.shared import apply_recipe

    # no name is "already taken" in the dry run: a form that the credit of an earlier pass or another family (``sin(x1)`` of the MI-greedy transforms) already made is part of the design of its variant, so both
    # variants are judged on all the forms the credit names on their own target. The recipes of the credit itself are left out so they are regenerated.
    credit_names = {str(r.name) for r in (getattr(mrmr, "_engineered_recipes_", None) or []) if getattr(r, "kind", "") in ("unary_credit", "unary_credit_product")}
    scratch = SimpleNamespace(
        _engineered_features_=[],
        _engineered_recipes_=[r for r in (getattr(mrmr, "_engineered_recipes_", None) or []) if str(r.name) not in credit_names],
        feature_names_in_=getattr(mrmr, "feature_names_in_", []),
        unary_forms_={},
        unary_forms_tied_={},
        _unary_credit_soft_=set(),
    )
    extra = credit_unary_forms(scratch, X, y_fit, names, seed=seed)
    n = len(y_fit)
    idx = np.random.default_rng(seed).choice(n, size=_MAX_ROWS, replace=False) if n > _MAX_ROWS else np.arange(n)
    sub = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    # Intentional float64: held-out error statistics of the link verdict
    cols = [np.asarray(sub[c], dtype=np.float64) for c in names if c in getattr(sub, "columns", [])]
    replayed = (_replay(apply_recipe, recipe, sub) for recipe, _name in extra)
    cols.extend(values for values in replayed if values is not None)
    if not cols:
        return float("inf")
    D = np.column_stack(cols)
    D = (D - D.mean(0)) / np.where(D.std(0) > 0, D.std(0), 1.0)
    # Intentional float64: held-out error statistics of the link verdict
    z, y0 = np.asarray(y_fit, dtype=np.float64)[idx], np.asarray(y_original, dtype=np.float64)[idx]
    fold = np.arange(len(idx)) % 5
    err = 0.0
    fallback = float(np.median(y0))
    for k in range(5):
        tr, te = fold != k, fold == k
        A = np.column_stack([np.ones(int(tr.sum())), D[tr]])
        coef = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ z[tr])
        pred = np.column_stack([np.ones(int(te.sum())), D[te]]) @ coef
        pred = inverse(pred) if inverse is not None else pred
        err += float(np.abs(y0[te] - np.where(np.isfinite(pred), pred, fallback)).sum())
    return err / len(idx)


def choose_credit_target(mrmr: Any, X: Any, y: Any, verdict: "Optional[dict]", names: "list[str]", seed: int = 0) -> Any:
    """The target the unary credit should run on: ``g(y)`` when the credited forms of the linked target predict ``y`` at least ``LINK_CREDIT_MIN_GAIN`` better than the forms of ``y`` itself, else ``y``.

    The decision is logged at INFO and recorded in the verdict (``credit_on_link``). Never raises: any failure keeps ``y``."""
    linked = link_target(verdict, y)
    if linked is y or verdict is None:
        return y
    if verdict.get("credit_on_link") is not None:  # decided by an earlier credit pass of this fit
        return linked if verdict["credit_on_link"] else y
    try:
        # Intentional float64: held-out error statistics of the link verdict
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        transform = verdict["transform"]
        numeric = [c for c in names if c in getattr(X, "columns", []) and np.asarray(X[c]).dtype.kind in "fiu"][:_MAX_COLUMNS]
        mae_plain = _forms_mae(mrmr, X, y_arr, y_arr, numeric, None, seed)
        # Intentional float64: held-out error statistics of the link verdict
        mae_link = _forms_mae(mrmr, X, y_arr, np.asarray(linked, dtype=np.float64).ravel(), numeric, transform.inverse, seed)
        use = bool(np.isfinite(mae_link) and mae_link <= (1.0 - LINK_CREDIT_MIN_GAIN) * mae_plain)
        verdict["credit_on_link"] = use
        verdict["credit_mae_original_target"] = float(mae_plain)
        verdict["credit_mae_linked_target"] = float(mae_link)
        logger.info(
            "MRMR target transform: the unary credit runs on %s (held-out MAE of the credited forms on the original scale %.4g on the target itself, %.4g through the %s link)",
            f"{verdict['name']}(y)" if use else "the target itself",
            mae_plain,
            mae_link,
            verdict["name"],
        )
        return linked if use else y
    except Exception as exc:  # the optional coupling never breaks a fit
        log_throttle(logger, "target_transform.choose", logging.WARNING, "target_transform_search: the credit target could not be chosen (%r); the credit runs on the target itself", exc)
        return y


def record_target_transform(estimator: Any, X: Any, y: Any) -> None:
    """Set ``estimator.target_transform_`` to the search verdict (computing it if the credit did not), or ``None`` when the flag is off or the target is not continuous."""
    estimator.target_transform_ = None
    if not bool(getattr(estimator, "target_transform_search", False)):
        return
    names = [str(c) for c in getattr(estimator, "feature_names_in_", [])]
    support_attr = getattr(estimator, "support_", None)
    support = [int(i) for i in (support_attr if support_attr is not None else [])]
    selected = [names[i] for i in support if 0 <= i < len(names)]
    verdict = find_target_transform(estimator, X, y, selected + [c for c in names if c not in selected])
    estimator.target_transform_ = dict(verdict) if verdict else None
