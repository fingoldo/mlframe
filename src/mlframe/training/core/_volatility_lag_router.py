"""Per-row VOLATILITY-lag router: deploy lag_predict on rows where the target series is locally SMOOTH, else the model.

The OOD-lag router (``_ood_lag_router``) only fires on out-of-train-range extrapolation. On a strong-AR target (a
wellbore log) the groups where lag beats the trained model are instead IN-range but LOCALLY SMOOTH: within such a well
consecutive target values barely move, so the previous value (lag_predict) is near-perfect while the model's features
add noise (prod TVT: 21/71 groups, all in-range, e.g. group 15 raw RMSE 29 vs lag 15). The transferable signal is the
LOCAL VOLATILITY of the (shifted) target series -- how fast the target moves per unit depth within a well. It is a
mechanical per-row property (a flat local trajectory) that generalises across unseen wells, unlike a per-group-id rule
which cannot cross the group-disjoint split.

Ordering is EXPLICIT by the depth / order column (``time_column``, e.g. MD): the volatility is ``|lag(next-in-well) -
lag(this)|`` where "next" is the MD-adjacent row IN THE WELL, computed via a ``lexsort`` on ``(group, order_key)``. It
does NOT assume the frame is row-sorted (prod logs warn it is not). When no order column is configured, or the group /
order column is missing on the predict frame, the router degrades to the plain trained prediction -- never a guess.

Safety: :func:`build_volatility_lag_router` sweeps a threshold and deploys the router ONLY when routing the
low-volatility rows to lag strictly improves RMSE on the group-disjoint val split; otherwise it returns the trained
model unchanged.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _group_local_volatility(group_ids: np.ndarray, values: np.ndarray, order_key: np.ndarray) -> np.ndarray:
    """``|values(next-in-group) - values(this)|`` where rows are ordered WITHIN each group by ``order_key`` (e.g. MD).

    Uses ``lexsort((order_key, group))`` so groups need not be contiguous and rows need not be pre-sorted. Rows with no
    in-group MD-successor (the last row of a well / a lone row) get ``nan`` -> the caller does NOT route them. A local
    volatility is thus the depth-forward step of the target, a well-defined per-row quantity independent of frame order.
    """
    g = np.asarray(group_ids)
    v = np.asarray(values, dtype=np.float64)
    ok = np.asarray(order_key, dtype=np.float64)
    n = v.size
    if n == 0 or g.shape[0] != n or ok.shape[0] != n:
        return np.full(n, np.nan, dtype=np.float64)
    order = np.lexsort((ok, g))          # sort by group, then by order_key within group
    gs = g[order]
    vs = v[order]
    # Forward step within group: diff to the NEXT row when it is the same group; last-in-group -> nan.
    same_next = np.zeros(n, dtype=bool)
    same_next[:-1] = gs[1:] == gs[:-1]
    step = np.full(n, np.nan, dtype=np.float64)
    step[:-1] = np.abs(vs[1:] - vs[:-1])
    step[~same_next] = np.nan
    step[~np.isfinite(vs)] = np.nan
    vol = np.empty(n, dtype=np.float64)
    vol[order] = step
    return vol


def _extract_column(X: Any, name: str) -> Optional[np.ndarray]:
    try:
        if hasattr(X, "get_column"):  # polars
            return np.asarray(X.get_column(name).to_numpy())
        cols = getattr(X, "columns", None)
        if cols is not None and name in cols:  # pandas
            return np.asarray(X[name].to_numpy())
    except Exception:  # noqa: BLE001
        return None
    return None


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(f.sum()) < 50:
        return float("nan")
    d = y_true[f] - y_pred[f]
    return float(np.sqrt(np.mean(d * d)))


class VolatilityLagRouter:
    """Deploy ``lag_component`` on rows whose local target volatility ``<= threshold`` (smooth trajectory -> lag is
    near-perfect), else ``trained.predict``. Ordering is explicit by ``order_column`` (MD). Picklable."""

    def __init__(self, trained: Any, lag_component: Any, group_column: str, order_column: str, threshold: float) -> None:
        self.trained = trained
        self.lag_component = lag_component
        self.group_column = str(group_column)
        self.order_column = str(order_column)
        self.threshold = float(threshold)

    def predict(self, X: Any) -> np.ndarray:
        raw = np.asarray(self.trained.predict(X), dtype=np.float64)
        lag = np.asarray(self.lag_component.predict(X), dtype=np.float64)
        gids = _extract_column(X, self.group_column)
        okey = _extract_column(X, self.order_column)
        if gids is None or okey is None or gids.shape[0] != raw.shape[0] or okey.shape[0] != raw.shape[0]:
            return raw  # cannot order within-well -> plain trained prediction (never a guess)
        vol = _group_local_volatility(gids, lag, okey)
        route = np.isfinite(vol) & (vol <= self.threshold) & np.isfinite(lag)
        out = raw.copy()
        out[route] = lag[route]
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"VolatilityLagRouter(group={self.group_column!r}, order={self.order_column!r}, "
                f"threshold={self.threshold:.4g})")


def build_volatility_lag_router(
    trained: Any,
    lag_component: Any,
    filtered_val_df: Any,
    y_val: Optional[np.ndarray],
    group_column: Optional[str],
    order_column: Optional[str],
    config: Any,
) -> Optional[Any]:
    """Return a :class:`VolatilityLagRouter` when routing low-local-volatility rows to lag strictly improves the honest
    group-disjoint val RMSE; otherwise return ``trained`` unchanged. Requires an EXPLICIT order column (MD): a missing
    group / order column on the val frame -> no routing (never a row-order guess). The group + order arrays are pulled
    FROM ``filtered_val_df`` -- the same extraction the router uses at predict -- so fit and deploy see identical keys.
    """
    if not bool(getattr(config, "volatility_lag_routing_enabled", True)):
        return trained
    if filtered_val_df is None or y_val is None or not group_column or not order_column:
        return trained
    yv = np.asarray(y_val, dtype=np.float64)
    gv = _extract_column(filtered_val_df, group_column)
    ov = _extract_column(filtered_val_df, order_column)
    if gv is None or ov is None:
        logger.info("[VolatilityLagRouter] no routing: group column %r or order column %r not on the val frame.", group_column, order_column)
        return trained
    ov = np.asarray(ov, dtype=np.float64)
    if yv.size < 100 or gv.shape[0] != yv.size or ov.shape[0] != yv.size:
        return trained
    try:
        raw_val = np.asarray(trained.predict(filtered_val_df), dtype=np.float64)
        lag_val = np.asarray(lag_component.predict(filtered_val_df), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001
        logger.info("[VolatilityLagRouter] val predict failed (%s); routing skipped.", exc)
        return trained
    if raw_val.shape != yv.shape or lag_val.shape != yv.shape:
        return trained

    vol = _group_local_volatility(gv, lag_val, ov)
    finite = np.isfinite(vol)
    if int(finite.sum()) < 100:
        logger.info("[VolatilityLagRouter] no routing: too few rows with a defined MD-forward local volatility (%d).", int(finite.sum()))
        return trained

    rmse_raw = _rmse(yv, raw_val)
    if not np.isfinite(rmse_raw):
        return trained
    qs = np.quantile(vol[finite], [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    best_t, best_rmse, best_n = None, rmse_raw, 0
    for t in np.unique(qs):
        route = finite & (vol <= t) & np.isfinite(lag_val)
        if int(route.sum()) < 50:
            continue
        routed = raw_val.copy()
        routed[route] = lag_val[route]
        r = _rmse(yv, routed)
        if np.isfinite(r) and r < best_rmse:
            best_t, best_rmse, best_n = float(t), r, int(route.sum())

    if best_t is None:
        logger.info(
            "[VolatilityLagRouter] no routing: routing low-volatility rows to lag never beats the trained model on val "
            "(raw RMSE %.4g); local smoothness does not transfer here -> keeping the trained model.",
            rmse_raw,
        )
        return trained
    logger.warning(
        "[VolatilityLagRouter] routing %d val row(s) with MD-local target volatility <= %.4g to lag_predict improves "
        "val RMSE %.4g -> %.4g; deploying the volatility router instead of the trained model alone.",
        best_n, best_t, rmse_raw, best_rmse,
    )
    return VolatilityLagRouter(trained, lag_component, group_column, order_column, best_t)
