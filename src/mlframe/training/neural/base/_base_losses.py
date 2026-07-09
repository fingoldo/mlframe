"""Loss-builder + input-validation leaves carved out of ``neural.base``."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import torch


def _make_binary_focal_loss(gamma: float, alpha: float):
    """Build a callable focal-loss for binary classification.

    Uses torchvision.ops.sigmoid_focal_loss when available (one fused
    kernel, well-tested). Falls back to a pure-PyTorch implementation
    otherwise (torchvision is optional). Both paths accept the standard
    ``(predictions, targets, reduction='mean'|'sum'|'none')`` signature
    so they compose with the estimator's _compute_weighted_loss /
    _loss_unreduced shape handling.

    Lin et al. 2017 (RetinaNet): FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t).
    Default alpha=0.25 weights the minority class (class 1) higher;
    gamma=2.0 is the original paper's recommended starting point.
    """
    try:
        from torchvision.ops import sigmoid_focal_loss as _tv_focal
        _has_torchvision = True
    except ImportError:
        _has_torchvision = False

    def _focal(input, target, reduction: str = "mean"):  # noqa: A002 -- matches torch.nn.functional loss-fn convention (input, target)
        """Compute the focal loss between ``input`` logits and ``target`` labels, dispatching to ``torchvision``'s fused kernel when available and otherwise a pure-PyTorch fallback; both honor the ``mean``/``sum``/``none`` reduction contract."""
        # The estimator passes labels as float for the BCE-replacement
        # path (labels_dtype was set to float32). Cast defensively in case
        # a custom carrier slips a Long through.
        if target.dtype != input.dtype:
            target = target.to(input.dtype)
        # Shape alignment: squeeze either side's (N, 1) -> (N,) so the
        # focal kernel sees matching ranks (BCE-shaped path).
        if input.dim() == 2 and input.shape[-1] == 1 and target.dim() == 1:
            input = input.squeeze(-1)  # noqa: A001
        elif target.dim() == 2 and target.shape[-1] == 1 and input.dim() == 1:
            target = target.squeeze(-1)
        if _has_torchvision:
            return _tv_focal(input, target, alpha=alpha, gamma=gamma, reduction=reduction)
        # Pure-PyTorch fallback: mirror torchvision's implementation.
        p = torch.sigmoid(input)
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input, target, reduction="none",
        )
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * target + (1 - alpha) * (1 - target)
            loss = alpha_t * loss
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        return loss  # reduction == "none"

    return _focal


def _validate_no_nan_inf(arg_name: str, data, allow_object_dtype: bool = False) -> None:
    """Reject NaN / inf in features or labels at fit() entry with a clear,
    actionable error. Pre-fix NaN propagated silently through the network
    producing all-NaN predictions.

    ``allow_object_dtype=True`` short-circuits the check for object-dtype
    targets (string / Python labels), which the LabelEncoder block will
    reject downstream with its own message if invalid.
    """
    if data is None:
        return
    # Normalise to np.ndarray for the check. Avoid copy when possible.
    if isinstance(data, pd.DataFrame):
        arr = data.to_numpy()
    elif isinstance(data, pd.Series):
        arr = data.to_numpy()
    elif isinstance(data, pl.DataFrame):
        arr = data.to_numpy()
    elif isinstance(data, np.ndarray):
        arr = data
    elif isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data)
    # Skip the finite-check on non-numeric dtypes; np.isnan raises on
    # object/str arrays. LabelEncoder will reject string labels for
    # regressors downstream; this guard is for numeric data only.
    if arr.dtype.kind not in ("f", "i", "u", "b"):
        if allow_object_dtype:
            return
        raise ValueError(
            f"{arg_name} has dtype {arr.dtype!r}; PytorchLightningEstimator "
            "requires numeric dtype (float / int / bool). Convert via "
            "pd.get_dummies, sklearn OrdinalEncoder, or similar."
        )
    if arr.dtype.kind == "f":
        # Only float arrays can carry NaN / inf; int / bool arrays can't.
        if not np.isfinite(arr).all():
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            raise ValueError(
                f"{arg_name} contains {n_nan} NaN and {n_inf} inf values. "
                "PytorchLightningEstimator does NOT impute internally because "
                "NaN propagates through the network -> all-NaN predictions. "
                "Pre-process with sklearn.impute.SimpleImputer / "
                "IterativeImputer, drop the offending rows via "
                f"{arg_name}.dropna(), or wrap the estimator in a sklearn "
                "Pipeline whose first step handles missing values."
            )
