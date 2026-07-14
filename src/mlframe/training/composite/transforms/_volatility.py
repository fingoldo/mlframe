"""``volatility_normalized_residual`` composite transform.

``T = (y - EWMA_k(base)) / max(EWMA_k(|base - EWMA_k(base)|), floor)``: the EWMA-level residual of ``ewma_residual`` additionally normalised by a
recency-weighted volatility estimate of the base series, so a downstream model sees a unit-variance-ish target across calm and turbulent regimes.
The volatility is computed from the BASE series (its absolute deviation from its own EWMA), not from ``y``: the inverse only has ``T_hat`` and
``base`` at predict time, so a volatility that read ``y`` would not be invertible. Inverse: ``y_hat = T_hat * vol + EWMA_k(base)``.

Time-recurrent (``recurrent=True``): both EWMA traces carry state across the row sequence exactly like ``ewma_residual``; the caller is responsible
for chronological row order at fit and predict. Anchors follow the ``ewma_residual`` convention (train-mean seed by default, train-tail seed when the
estimator opted into recurrence continuation).
"""
from __future__ import annotations

from typing import Any

import numpy as np

_VNR_DEFAULT_K: int = 7
"""Default EWMA span, matching ``_EWMA_RESIDUAL_DEFAULT_K`` (kept module-local so this leaf module never top-level-imports the parent and stays out of the transforms import SCC)."""

_VOL_FLOOR_FRAC: float = 1e-3
"""The volatility floor is ``max(_VOL_FLOOR_FRAC * scale(vol_train), 1e-12)``: keeps the division meaningful on a locally-constant base (vol -> 0 would blow T up) without biasing normal-regime rows."""


def _vol_traces(base_f: np.ndarray, k: int, anchor: float, vol_anchor: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the (level EWMA, volatility EWMA) traces of ``base_f`` with the given seeds."""
    from .nonlinear import _ewma_compute  # lazy: nonlinear imports the parent at top, a top-level import here would cycle
    level = _ewma_compute(base_f, k, anchor)
    absdev = np.abs(base_f - level)
    vol = _ewma_compute(absdev, k, vol_anchor)
    return level, vol


def _volatility_normalized_residual_fit(
    y: np.ndarray, base: np.ndarray, k: int = _VNR_DEFAULT_K,
    _finite_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit stores the span ``k``, the mean/tail anchors for both EWMA traces, and the train-scale volatility floor; the traces themselves are recomputed at forward / inverse time (stateless, JSON-serialisable params)."""
    k = max(1, int(k))
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    finite = _finite_mask if _finite_mask is not None else np.isfinite(base_f)
    anchor = float(np.mean(base_f[finite])) if finite.any() else 0.0
    if finite.any():
        from .nonlinear import _ewma_compute
        level = _ewma_compute(base_f, k, anchor)
        absdev = np.abs(base_f - level)
        vol_anchor = float(np.mean(absdev[np.isfinite(absdev)])) if np.isfinite(absdev).any() else 0.0
        vol = _ewma_compute(np.where(np.isfinite(absdev), absdev, np.nan), k, vol_anchor)
        vol_finite = vol[np.isfinite(vol)]
        scale = float(np.median(vol_finite)) if vol_finite.size else 0.0
        level_f = level[np.isfinite(level)]
        tail_anchor = float(level_f[-1]) if level_f.size else anchor
        vol_tail_anchor = float(vol_finite[-1]) if vol_finite.size else vol_anchor
    else:
        vol_anchor = 0.0
        scale = 0.0
        tail_anchor = anchor
        vol_tail_anchor = vol_anchor
    floor = max(scale * _VOL_FLOOR_FRAC, 1e-12)
    return {
        "k": k, "anchor": anchor, "vol_anchor": vol_anchor, "floor": floor,
        "tail_anchor": tail_anchor, "vol_tail_anchor": vol_tail_anchor,
    }


def _vnr_anchors(params: dict[str, Any]) -> tuple[float, float]:
    """Mean anchors by default; train-tail anchors when the estimator opted into recurrence-continuation seeding (mirrors ``_ewma_anchor``)."""
    if params.get("recurrence_continuation") and "tail_anchor" in params:
        return float(params["tail_anchor"]), float(params.get("vol_tail_anchor", params["vol_anchor"]))
    return float(params["anchor"]), float(params["vol_anchor"])


def _volatility_normalized_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Apply ``T = (y - EWMA_k(base)) / max(vol, floor)``."""
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    level, vol = _vol_traces(base_f, int(params["k"]), float(params["anchor"]), float(params["vol_anchor"]))
    v = np.maximum(vol, float(params["floor"]))
    return np.asarray((np.asarray(y, dtype=np.float64).reshape(-1) - level) / v)


def _volatility_normalized_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Undo the transform: ``y = T_hat * max(vol, floor) + EWMA_k(base)`` with the same floored volatility as the forward."""
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    anchor, vol_anchor = _vnr_anchors(params)
    level, vol = _vol_traces(base_f, int(params["k"]), anchor, vol_anchor)
    v = np.maximum(vol, float(params["floor"]))
    return np.asarray(np.asarray(t_hat, dtype=np.float64).reshape(-1) * v + level)


def _volatility_normalized_residual_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    """Finite ``base`` (and finite ``y`` when provided), matching ``ewma_residual``."""
    base_ok = np.isfinite(np.asarray(base, dtype=np.float64).reshape(-1))
    if y is None:
        return base_ok
    return np.asarray(base_ok & np.isfinite(np.asarray(y, dtype=np.float64).reshape(-1)))
