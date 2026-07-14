"""``nadaraya_watson_residual`` composite transform.

``T = y - g(base)`` where ``g`` is a Gaussian-kernel Nadaraya-Watson regression of ``y`` on ``base``: ``g(x) = sum_j K((x - x_j)/h) * y_j / sum_j K(...)``
over a set of train knots. Unlike ``monotonic_residual`` (monotone PCHIP) and ``smoothing_spline_residual`` (global smooth spline), NW is a purely local
average -- it tracks arbitrary non-monotone dependence without knot-median coarsening, degrading gracefully to the nearest-knot value far from support.

Bandwidth: Silverman's rule of thumb on the train base, ``h = 0.9 * min(std, IQR/1.34) * n^(-1/5)``. Knots: the fitted ``(base, y)`` pairs subsampled to
at most ``_NW_MAX_KNOTS`` points taken evenly along the base-sorted order (deterministic, preserves support coverage), so predict is O(n * m) with a
bounded m. Inverse: ``y_hat = T_hat + g(base)`` (pure additive; exact round-trip).
"""
from __future__ import annotations

from typing import Any

import numpy as np

_NW_MAX_KNOTS: int = 2000
"""Cap on stored (base, y) knots: bounds the O(n * m) kernel-weight matrix at predict time. 2000 evenly-spaced-in-rank knots reproduce g to well below the kernel smoothing error on practical train sizes."""

_NW_ROW_CHUNK_ELEMS: int = 4_000_000
"""Predict-time (rows x knots) weight blocks are built in row chunks capped at this many float64 elements (~32 MB) so a 1M-row predict never materialises an n x m matrix."""


def _nw_g(base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Evaluate the fitted Nadaraya-Watson regression at ``base``. Per-row max-weight normalisation keeps the softmax-style weights underflow-free, so far-from-support rows converge to the nearest knot's y instead of 0/0."""
    knots_x = np.asarray(params["knots_x"], dtype=np.float64)
    knots_y = np.asarray(params["knots_y"], dtype=np.float64)
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    if knots_x.size == 0:
        return np.full(base_f.shape, float(params.get("y_train_mean", 0.0)), dtype=np.float64)
    h = float(params["bandwidth"])
    out = np.empty(base_f.size, dtype=np.float64)
    m = knots_x.size
    chunk = max(1, _NW_ROW_CHUNK_ELEMS // m)
    for lo in range(0, base_f.size, chunk):
        x = base_f[lo : lo + chunk]
        d2 = 0.5 * ((x[:, None] - knots_x[None, :]) / h) ** 2
        d2 -= d2.min(axis=1, keepdims=True)
        w = np.exp(-d2)
        out[lo : lo + chunk] = (w @ knots_y) / w.sum(axis=1)
    return out


def _nadaraya_watson_residual_fit(
    y: np.ndarray, base: np.ndarray,
    _finite_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Store subsampled (base, y) knots + Silverman bandwidth + the train-y mean fallback."""
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    finite = _finite_mask if _finite_mask is not None else (np.isfinite(y_f) & np.isfinite(base_f))
    if finite.sum() < 5:
        y_mean = float(np.mean(y_f[finite])) if finite.any() else 0.0
        return {
            "knots_x": np.array([], dtype=np.float64), "knots_y": np.array([], dtype=np.float64),
            "bandwidth": 1.0, "y_train_mean": y_mean,
        }
    yc = y_f[finite]
    bc = base_f[finite]
    n = bc.size
    order = np.argsort(bc, kind="stable")
    if n > _NW_MAX_KNOTS:
        # Evenly spaced along the base-sorted order: deterministic + full-support coverage.
        pick = order[np.linspace(0, n - 1, _NW_MAX_KNOTS).astype(np.int64)]
    else:
        pick = order
    knots_x = bc[pick]
    knots_y = yc[pick]
    std = float(np.std(bc))
    iqr = float(np.subtract(*np.percentile(bc, [75, 25])))
    spread = min(std, iqr / 1.34) if iqr > 0 else std
    if spread <= 0:
        # Constant base: bandwidth is irrelevant (g == mean everywhere); use a positive placeholder.
        spread = max(abs(float(bc[0])), 1.0)
    h = 0.9 * spread * n ** (-0.2)
    h = max(h, 1e-12)
    return {
        "knots_x": knots_x, "knots_y": knots_y,
        "bandwidth": float(h), "y_train_mean": float(np.mean(yc)),
    }


def _nadaraya_watson_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Apply ``T = y - g(base)`` with the fitted NW regression ``g``."""
    return np.asarray(np.asarray(y, dtype=np.float64).reshape(-1) - _nw_g(base, params))


def _nadaraya_watson_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Undo the transform: ``y = T_hat + g(base)``."""
    return np.asarray(np.asarray(t_hat, dtype=np.float64).reshape(-1) + _nw_g(base, params))


def _nadaraya_watson_residual_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    """Finite ``base`` (and finite ``y`` when provided)."""
    base_ok = np.isfinite(np.asarray(base, dtype=np.float64).reshape(-1))
    if y is None:
        return base_ok
    return np.asarray(base_ok & np.isfinite(np.asarray(y, dtype=np.float64).reshape(-1)))
