"""Per-row lag routing on top of a component the AR(1) failsafe val cross-check chose over lag_predict.

Both routers, like the veto itself, choose on the val split, so the val metric later stamped for the deployed predictor is
not a held-out estimate; this module records that on ``cross_target_ensemble_metrics`` (``val_selection_biased``) so the
suite-end verdict can say so.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _rows(y_full: Any, idx: Any) -> np.ndarray | None:
    """``y_full[idx]`` as float64, or None when either is missing."""
    return np.asarray(y_full)[idx].astype(np.float64) if (y_full is not None and idx is not None) else None


def attach_val_selected_lag_routers(
    deployed: Any, lag_model: Any, y_full: Any, train_idx: Any, val_idx: Any, val_df: Any, ctx: Any, cfg: Any,
    metadata: dict, target_type: Any, target_name: str,
) -> Any:
    """``deployed`` wrapped in the OOD-lag and then the volatility-lag router where each improves the val RMSE.

    Marks the target's CT-ensemble metrics as val-selected, since the veto that chose ``deployed`` and both routers decide
    on val. A router that cannot be built is skipped (routing is advisory) and the model built so far is kept.
    """
    metadata.setdefault("cross_target_ensemble_metrics", {}).setdefault(str(target_type), {}).setdefault(target_name, {})["val_selection_biased"] = True
    # Per-row OOD-lag routing on top: the trained model wins overall but still extrapolates on unseen
    # groups whose target level is out of the train range; route those rows (lag out of range) to lag,
    # but only when it improves the honest val RMSE. Transferable (train-range rule, not group-id).
    try:
        from .._ood_lag_router import build_ood_lag_router

        deployed = build_ood_lag_router(deployed, lag_model, _rows(y_full, train_idx), val_df, _rows(y_full, val_idx), cfg)
    except Exception as err:  # -- routing is advisory; keep the trained model
        logger.info("[CompositeCrossTargetEnsemble] target='%s' OOD-lag routing skipped (%s).", target_name, err)
    # Per-row VOLATILITY-lag routing: on a strong-AR target the lag-wins groups are IN-range but locally
    # SMOOTH, which the range rule above cannot catch. Route rows whose MD-local target volatility is low
    # (lag near-perfect) to lag, only when it improves the honest val RMSE. Needs group_column + a MD
    # order column (time_column) on the frame -- ordering is explicit, never a frame-row-order guess.
    try:
        from .._volatility_lag_router import build_volatility_lag_router

        ctx_groups = getattr(ctx, "group_ids", None) if ctx is not None else None
        groups_val = np.asarray(ctx_groups)[val_idx] if (ctx_groups is not None and val_idx is not None) else None
        deployed = build_volatility_lag_router(
            deployed, lag_model, groups_val, val_df, _rows(y_full, val_idx),
            getattr(cfg, "group_column", None), getattr(cfg, "time_column", None), cfg,
        )
    except Exception as err:  # -- advisory; keep whatever we have
        logger.info("[CompositeCrossTargetEnsemble] target='%s' volatility-lag routing skipped (%s).", target_name, err)
    return deployed
