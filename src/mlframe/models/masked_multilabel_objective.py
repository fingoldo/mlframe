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

from typing import Callable, Optional, Tuple

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


def masked_multilabel_logloss_objective(
    sentinel: float = _DEFAULT_SENTINEL, use_sample_weight: bool = False
) -> Callable[[np.ndarray, object], Tuple[np.ndarray, np.ndarray]]:
    """Return an XGBoost-compatible custom objective for sentinel-masked binary log-loss.

    Parameters
    ----------
    sentinel
        The don't-care label value produced by :func:`flatten_masked_multilabel` -- entries equal to it get
        zero gradient AND zero hessian (no contribution to any tree split), the standard XGBoost convention
        for "this sample doesn't participate in this objective's loss."
    use_sample_weight
        Opt-in (default ``False``, matching prior uniform-weight behavior bit-for-bit). When ``True``, the raw
        grad/hess are additionally scaled by ``dtrain.get_weight()`` -- XGBoost does NOT auto-apply DMatrix
        weights to custom objectives (only to built-in ones), so this multiplication must happen here.
        Combine with :func:`flatten_masked_multilabel_class_weights` (e.g. seeded by
        :func:`compute_inverse_frequency_class_weights`) to upweight rare label classes instead of treating
        every unmasked (row, label) cell as equally important. Raises ``ValueError`` at call time if the
        DMatrix carries no weight vector (or one of the wrong length) -- silently falling back to uniform
        weighting would defeat the purpose of asking for weighting.

    Returns
    -------
    callable
        ``objective(y_pred_margin, dtrain) -> (grad, hess)`` matching XGBoost's ``obj=`` custom-objective
        signature exactly (``dtrain`` is the ``xgboost.DMatrix`` being trained on; its ``.get_label()``
        supplies the sentinel-masked label array) -- pass directly as ``xgb.train(..., obj=this_callable)``.
    """

    def objective(y_pred_margin: np.ndarray, dtrain: object) -> Tuple[np.ndarray, np.ndarray]:
        """Compute sentinel-masked, optionally sample-weighted logloss gradient and hessian for XGBoost."""
        y_pred_margin_arr = np.asarray(y_pred_margin, dtype=np.float64)
        y_true_arr = np.asarray(dtrain.get_label(), dtype=np.float64)  # type: ignore[attr-defined]

        care_mask = y_true_arr != sentinel
        prob = 1.0 / (1.0 + np.exp(-y_pred_margin_arr))

        grad = np.where(care_mask, prob - y_true_arr, 0.0)
        hess = np.where(care_mask, prob * (1.0 - prob), 0.0)

        if use_sample_weight:
            weight_arr = np.asarray(dtrain.get_weight(), dtype=np.float64)  # type: ignore[attr-defined]
            if weight_arr.shape != y_true_arr.shape:
                raise ValueError(
                    "masked_multilabel_logloss_objective: use_sample_weight=True requires dtrain to carry a "
                    f"weight vector matching the label shape ({y_true_arr.shape}), got {weight_arr.shape}. "
                    "Pass weight= to xgb.DMatrix, e.g. via flatten_masked_multilabel_class_weights()."
                )
            grad = grad * weight_arr
            hess = hess * weight_arr

        return grad, hess

    return objective


def flatten_masked_multilabel_class_weights(y_multilabel: np.ndarray, dont_care_mask: np.ndarray, class_weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Flatten per-class weights to align with :func:`flatten_masked_multilabel`'s row-major label layout.

    Parameters
    ----------
    y_multilabel
        ``(n_rows, n_labels)`` ground truth -- used only for its shape.
    dont_care_mask
        ``(n_rows, n_labels)`` boolean mask; don't-care cells get weight ``0.0`` (their grad/hess are already
        zeroed by the objective's sentinel check regardless, so this just keeps the weight vector's semantics
        consistent with "excluded from the loss").
    class_weights
        ``(n_labels,)`` per-class weight, e.g. inverse label frequency to upweight rare classes. ``None``
        (default) gives every care cell weight ``1.0`` -- the uniform-weight baseline.

    Returns
    -------
    np.ndarray
        ``(n_rows * n_labels,)`` flattened weight array, row-major, matching ``flatten_masked_multilabel``'s
        layout -- pass as ``xgb.DMatrix(..., weight=this_array)`` together with
        ``masked_multilabel_logloss_objective(use_sample_weight=True)``.
    """
    y_arr = np.asarray(y_multilabel)
    n_rows, n_labels = y_arr.shape
    if class_weights is None:
        weights = np.ones(n_labels, dtype=np.float64)
    else:
        weights = np.asarray(class_weights, dtype=np.float64)
        if weights.shape != (n_labels,):
            raise ValueError(f"flatten_masked_multilabel_class_weights: class_weights must have shape ({n_labels},), got {weights.shape}.")
    weight_matrix = np.tile(weights, (n_rows, 1))
    mask_arr = np.asarray(dont_care_mask, dtype=bool)
    weight_matrix[mask_arr] = 0.0
    return np.asarray(weight_matrix.ravel())


def compute_inverse_frequency_class_weights(y_multilabel: np.ndarray, dont_care_mask: np.ndarray) -> np.ndarray:
    """Per-class weights inversely proportional to positive-label rate among care (unmasked) cells.

    Rare positive classes (few positives among unmasked cells) get upweighted proportionally to
    ``1 / positive_rate``, then normalized so the weights' mean is ``1.0`` -- keeps the overall gradient/hessian
    scale comparable to the unweighted objective (avoids implicitly changing the effective learning rate).

    Parameters
    ----------
    y_multilabel
        ``(n_rows, n_labels)`` binary ground truth.
    dont_care_mask
        ``(n_rows, n_labels)`` boolean mask; masked cells are excluded from the frequency estimate.

    Returns
    -------
    np.ndarray
        ``(n_labels,)`` weight vector, feedable straight into :func:`flatten_masked_multilabel_class_weights`.
    """
    y_arr = np.asarray(y_multilabel, dtype=np.float64)
    mask_arr = np.asarray(dont_care_mask, dtype=bool)
    care = ~mask_arr
    n_labels = y_arr.shape[1]
    weights = np.ones(n_labels, dtype=np.float64)
    for j in range(n_labels):
        care_j = care[:, j]
        n_care = int(care_j.sum())
        if n_care == 0:
            continue
        pos_rate = max(float(y_arr[care_j, j].mean()), 1e-6)
        weights[j] = 1.0 / pos_rate
    mean_weight = float(weights.mean())
    if mean_weight > 0:
        weights = weights / mean_weight
    return weights


__all__ = [
    "flatten_masked_multilabel",
    "masked_multilabel_logloss_objective",
    "flatten_masked_multilabel_class_weights",
    "compute_inverse_frequency_class_weights",
]
