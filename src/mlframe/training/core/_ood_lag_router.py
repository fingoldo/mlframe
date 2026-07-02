"""Per-row OOD-lag router: deploy the trained model where the lag is in the train target range, else the lag baseline.

After the AR(1) val cross-check deploys the trained model over lag (it wins overall), the composite value report still
shows a subset of unseen groups where lag beats the trained model badly -- e.g. TVT prod: 21/71 groups, group 17 raw RMSE
22.4 vs lag 2.58. Those are groups whose target LEVEL sits OUTSIDE the train range: a tree can only clamp/extrapolate its
leaf values there, so the trained prediction is catastrophically wrong, while lag_predict (y_hat = the row's own previous
target value) is exact. Per-group-ID routing cannot fix this -- the split is group-disjoint, so test groups are entirely
unseen. But the SIGNAL is transferable: route a row to lag whenever its lag value lies outside the train target range
(the model never saw that level -> its prediction is an extrapolation), else keep the trained model.

Safety: :func:`build_ood_lag_router` only returns a router when it strictly improves RMSE on the group-disjoint val split
(the same honest regime as test) over the trained model alone; otherwise it returns the trained component unchanged. The
lo/hi bounds come from the TRAIN target range (fixed), so the routing rule transfers to any split by construction.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class OODLagRouter:
    """Deployable predictor: ``y_hat = lag_pred`` where the lag is outside ``[lo, hi]`` (out-of-train-range -> the trained
    model extrapolates), else ``trained.predict``. Picklable (holds two picklable component predictors + two floats)."""

    def __init__(self, trained: Any, lag_component: Any, lo: float, hi: float) -> None:
        self.trained = trained
        self.lag_component = lag_component
        self.lo = float(lo)
        self.hi = float(hi)

    def predict(self, X: Any) -> np.ndarray:
        raw = np.asarray(self.trained.predict(X), dtype=np.float64)
        lag = np.asarray(self.lag_component.predict(X), dtype=np.float64)
        out = raw.copy()
        # Route OOD rows (lag outside the train target range) to the lag baseline -- but only where lag is finite, so a
        # missing lag never overwrites a valid trained prediction with NaN.
        ood = np.isfinite(lag) & ((lag < self.lo) | (lag > self.hi))
        out[ood] = lag[ood]
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"OODLagRouter(lo={self.lo:.4g}, hi={self.hi:.4g})"


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(f.sum()) < 50:
        return float("nan")
    d = y_true[f] - y_pred[f]
    return float(np.sqrt(np.mean(d * d)))


def build_ood_lag_router(
    trained: Any,
    lag_component: Any,
    y_train: Optional[np.ndarray],
    filtered_val_df: Any,
    y_val: Optional[np.ndarray],
    config: Any,
) -> Optional[Any]:
    """Return an :class:`OODLagRouter` wrapping ``trained`` when routing out-of-train-range rows to ``lag_component``
    strictly improves RMSE on the group-disjoint val split; otherwise return ``trained`` unchanged (never worse).

    ``y_train`` sets the train target range ``[lo, hi]`` (optionally widened by ``ood_lag_router_margin_frac``). The
    decision is made on ``filtered_val_df`` / ``y_val`` (the honest holdout). ``None`` on any missing input.
    """
    if not bool(getattr(config, "ood_lag_routing_enabled", True)):
        return trained
    if y_train is None or filtered_val_df is None or y_val is None:
        return trained
    yt = np.asarray(y_train, dtype=np.float64)
    yt = yt[np.isfinite(yt)]
    yv = np.asarray(y_val, dtype=np.float64)
    if yt.size < 50 or yv.size < 50:
        return trained
    lo, hi = float(np.min(yt)), float(np.max(yt))
    margin = float(getattr(config, "ood_lag_router_margin_frac", 0.0))
    if margin > 0 and hi > lo:
        span = hi - lo
        lo -= margin * span
        hi += margin * span

    try:
        raw_val = np.asarray(trained.predict(filtered_val_df), dtype=np.float64)
        lag_val = np.asarray(lag_component.predict(filtered_val_df), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 -- a component that cannot predict on val -> no routing
        logger.info("[OODLagRouter] val predict failed (%s); routing skipped.", exc)
        return trained
    if raw_val.shape != yv.shape or lag_val.shape != yv.shape:
        return trained

    ood = np.isfinite(lag_val) & ((lag_val < lo) | (lag_val > hi))
    n_ood = int(ood.sum())
    if n_ood < 50:
        return trained  # too few out-of-range rows to matter -> keep the trained model
    routed_val = raw_val.copy()
    routed_val[ood] = lag_val[ood]
    rmse_raw = _rmse(yv, raw_val)
    rmse_routed = _rmse(yv, routed_val)
    if not (np.isfinite(rmse_raw) and np.isfinite(rmse_routed)) or rmse_routed >= rmse_raw:
        return trained  # routing does not help on the honest val split -> deploy the trained model alone

    logger.warning(
        "[OODLagRouter] routing %d/%d out-of-train-range val row(s) (lag outside [%.4g, %.4g]) to lag_predict improves "
        "val RMSE %.4g -> %.4g; deploying the per-row OOD router instead of the trained model alone.",
        n_ood, yv.size, lo, hi, rmse_raw, rmse_routed,
    )
    return OODLagRouter(trained, lag_component, lo, hi)
