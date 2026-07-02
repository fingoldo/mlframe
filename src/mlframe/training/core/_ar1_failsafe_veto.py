"""Val cross-check that vetoes the AR(1) lag_predict failsafe when the OOF tie is a group-K-fold pessimism artifact.

The cross-target ensemble deploys the zero-parameter ``lag_predict`` baseline over a trained stack whenever lag's OOF RMSE
is within ``lag_predict_failsafe_tolerance`` of the best trained component's OOF RMSE ("cannot overfit on test"). But the
OOF RMSE is a group-K-fold estimate: each fold trains on a SUBSET of the groups, so it systematically UNDERESTIMATES the
full-data trained model's generalisation. On a strong-AR wellbore target this produced a lag<->trained OOF tie (~13.64
each) that shipped lag_predict (test RMSE 12.29) even though the trained LGBM scored 9.31 on BOTH the group-disjoint val
AND test splits -- a 32% worse deployment chosen from a pessimistic proxy.

The group-disjoint VAL split is the SAME honest-holdout regime as test (whole groups held out). When a trained component
beats lag on VAL by more than the failsafe tolerance, the OOF tie is spurious: deploy the trained component, not lag.
Conservative by construction -- it only ever PREVENTS a lag deployment in favour of a val-confirmed-better trained model,
never the reverse, and no-ops when val data is absent.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


def decide_ar1_failsafe_val_veto(
    oof_names: Sequence[str],
    oof_rmses: Sequence[float],
    lag_failsafe_tol: float,
    val_rmse_of: Callable[[int], float],
    *,
    lag_name: str = "lag_predict",
) -> Optional[int]:
    """Return the trained-component index to deploy INSTEAD of lag when the val cross-check vetoes the AR(1) failsafe.

    Returns ``None`` when the failsafe would not fire, when there is no trained (non-lag) component, or when no trained
    component beats lag on the val split by more than ``lag_failsafe_tol``. When it returns an index the caller should
    deploy ``components[idx]`` and skip the OOF-only failsafe / gate (which would otherwise re-select lag).

    ``val_rmse_of(i)`` returns component ``i``'s RMSE on the group-disjoint val split (``nan`` when unmeasurable). It is
    called at most twice (lag + the best-by-OOF trained candidate), so the caller's val predicts stay cheap.
    """
    if lag_failsafe_tol <= 0 or lag_name not in oof_names:
        return None
    rmses = np.asarray(oof_rmses, dtype=np.float64)
    if not np.isfinite(rmses).any():
        return None
    lp_idx = list(oof_names).index(lag_name)
    lp_oof = float(rmses[lp_idx])
    best_single_oof = float(np.nanmin(rmses))
    # The failsafe only fires when lag's OOF RMSE is within tolerance of the best (any) component's OOF RMSE.
    if not (np.isfinite(lp_oof) and lp_oof <= (1.0 + lag_failsafe_tol) * best_single_oof):
        return None
    nonlag = [i for i in range(len(oof_names)) if i != lp_idx and np.isfinite(rmses[i])]
    if not nonlag:
        return None
    best_trained_idx = min(nonlag, key=lambda i: float(rmses[i]))
    lp_val = float(val_rmse_of(lp_idx))
    bt_val = float(val_rmse_of(best_trained_idx))
    if not (np.isfinite(lp_val) and np.isfinite(bt_val)):
        return None
    # Veto only when the trained component beats lag on VAL by MORE than the same tolerance -- a large, honest margin
    # that early-stopping optimism (val is used for ES) cannot manufacture, so the OOF tie is confirmed spurious.
    if bt_val < lp_val / (1.0 + lag_failsafe_tol):
        return best_trained_idx
    return None


def compute_val_veto(
    oof_names: Sequence[str],
    oof_rmses: Sequence[float],
    oof_components: Sequence,
    filtered_val_df,
    filtered_val_idx,
    oof_y_full,
    lag_failsafe_tol: float,
    config,
) -> Optional[int]:
    """Call-site wrapper: build the per-component val-RMSE probe and run :func:`decide_ar1_failsafe_val_veto`.

    Returns the trained-component index to deploy instead of lag, or ``None`` (disabled / no val data / no veto). Each
    component's val prediction is computed at most once (cached); the underlying decision calls the probe at most twice.
    Never raises -- any failure degrades to ``None`` (the OOF-only failsafe then runs unchanged).
    """
    if not bool(getattr(config, "ar1_failsafe_val_crosscheck", True)):
        return None
    if ("lag_predict" not in oof_names or filtered_val_df is None
            or filtered_val_idx is None or oof_y_full is None):
        return None
    try:
        yv = np.asarray(oof_y_full)[filtered_val_idx].astype(np.float64)
    except Exception:  # noqa: BLE001
        return None
    cache: dict[int, float] = {}

    def _val_rmse_of(ci: int) -> float:
        if ci in cache:
            return cache[ci]
        try:
            pv = np.asarray(oof_components[ci].predict(filtered_val_df), dtype=np.float64)
            f = np.isfinite(pv) & np.isfinite(yv)
            r = float(np.sqrt(np.mean((pv[f] - yv[f]) ** 2))) if int(f.sum()) >= 50 else float("nan")
        except Exception:  # noqa: BLE001 -- a component that cannot predict on val yields no veto signal
            r = float("nan")
        cache[ci] = r
        return r

    try:
        return decide_ar1_failsafe_val_veto(oof_names, oof_rmses, lag_failsafe_tol, _val_rmse_of)
    except Exception:  # noqa: BLE001
        return None
