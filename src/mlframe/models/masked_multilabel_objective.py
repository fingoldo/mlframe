"""``masked_multilabel_logloss_objective``/``flatten_masked_multilabel``: sentinel-masked multi-label loss.

Source: 4th_santander-product-recommendation.md -- implemented a LambdaRank/MAP-style objective but exposed
via the simpler multiclass-classification interface: flattened ``row_num * n_classes`` labels with a
"don't care" SENTINEL value (>1.5, outside the valid ``{0, 1}`` label range) marking previously-owned products
that must be excluded from the loss entirely (neither a positive nor a negative example), avoiding the 24x
row explosion XGBoost's native ``rank:map``/group-based ranking API would otherwise require.

Generic beyond recommendation: any masked-multilabel scenario (partially-labeled multi-target problems where
some (row, label) pairs are known-inapplicable rather than genuinely negative) can reuse this convention --
encode "ignore in loss" via a sentinel label value, and the custom XGBoost objective below zeroes both
gradient and hessian for those entries so they exert no update pressure on the trees.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

_DEFAULT_SENTINEL = 2.0


def flatten_masked_multilabel(y_multilabel: np.ndarray, dont_care_mask: np.ndarray, sentinel: float = _DEFAULT_SENTINEL) -> np.ndarray:
    """Flatten a ``(n_rows, n_labels)`` binary matrix to ``(n_rows * n_labels,)``, sentinel-marking don't-care cells.

    Parameters
    ----------
    y_multilabel
        ``(n_rows, n_labels)`` binary (0/1) ground truth.
    dont_care_mask
        ``(n_rows, n_labels)`` boolean -- True marks a cell to EXCLUDE from the loss (e.g. an already-owned
        product: neither a valid positive nor negative training example for "will newly acquire").
    sentinel
        Value written at don't-care cells; must be outside ``{0, 1}`` (default ``2.0``, matching the source's
        own ``>1.5`` convention).

    Returns
    -------
    np.ndarray
        ``(n_rows * n_labels,)`` flattened label array, row-major (matches XGBoost's own multiclass label
        flattening convention), with don't-care cells set to ``sentinel``.
    """
    if sentinel in (0.0, 1.0):
        raise ValueError(f"flatten_masked_multilabel: sentinel must be outside {{0, 1}}, got {sentinel!r}.")
    y_arr = np.asarray(y_multilabel, dtype=np.float64).copy()
    mask_arr = np.asarray(dont_care_mask, dtype=bool)
    y_arr[mask_arr] = sentinel
    return np.asarray(y_arr.ravel())


def masked_multilabel_logloss_objective(sentinel: float = _DEFAULT_SENTINEL) -> Callable[[np.ndarray, object], Tuple[np.ndarray, np.ndarray]]:
    """Return an XGBoost-compatible custom objective for sentinel-masked binary log-loss.

    Parameters
    ----------
    sentinel
        The don't-care label value produced by :func:`flatten_masked_multilabel` -- entries equal to it get
        zero gradient AND zero hessian (no contribution to any tree split), the standard XGBoost convention
        for "this sample doesn't participate in this objective's loss."

    Returns
    -------
    callable
        ``objective(y_pred_margin, dtrain) -> (grad, hess)`` matching XGBoost's ``obj=`` custom-objective
        signature exactly (``dtrain`` is the ``xgboost.DMatrix`` being trained on; its ``.get_label()``
        supplies the sentinel-masked label array) -- pass directly as ``xgb.train(..., obj=this_callable)``.
    """

    def objective(y_pred_margin: np.ndarray, dtrain: object) -> Tuple[np.ndarray, np.ndarray]:
        y_pred_margin_arr = np.asarray(y_pred_margin, dtype=np.float64)
        y_true_arr = np.asarray(dtrain.get_label(), dtype=np.float64)  # type: ignore[attr-defined]

        care_mask = y_true_arr != sentinel
        prob = 1.0 / (1.0 + np.exp(-y_pred_margin_arr))

        grad = np.where(care_mask, prob - y_true_arr, 0.0)
        hess = np.where(care_mask, prob * (1.0 - prob), 0.0)
        return grad, hess

    return objective


__all__ = ["flatten_masked_multilabel", "masked_multilabel_logloss_objective"]
