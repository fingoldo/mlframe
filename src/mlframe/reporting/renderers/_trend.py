"""Robust fit-line helper shared by both renderers.

Returns two endpoint coordinates for a robust line (Theil-Sen or Huber) so a renderer can draw it with a single
two-point ``plot`` regardless of point count. Theil-Sen (median of pairwise slopes) and Huber (M-estimator)
resist the heavy-tailed outliers a pred-vs-actual cloud carries, where an OLS line would tilt toward the worst
points -- exactly the residual structure these overlays are meant to expose against the y=x reference.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def robust_fit_endpoints(
    x: np.ndarray, y: np.ndarray, method: str
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Fit a robust line ``y ~ x`` and return ((x_lo, y_lo), (x_hi, y_hi)) at the x extremes.

    Returns ``None`` when the fit is undefined (fewer than 2 finite points, or all x identical).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape or x.size < 2:
        return None
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return None
    x = x[finite]
    y = y[finite]
    x_lo, x_hi = float(np.min(x)), float(np.max(x))
    if x_hi <= x_lo:
        return None

    method = method.lower()
    if method == "theil-sen":
        from sklearn.linear_model import TheilSenRegressor
        model = TheilSenRegressor(random_state=0)
    elif method == "huber":
        from sklearn.linear_model import HuberRegressor
        model = HuberRegressor()
    else:
        raise ValueError(f"unknown trend_line method {method!r}; use 'theil-sen' or 'huber'")

    model.fit(x.reshape(-1, 1), y)
    y_lo, y_hi = model.predict(np.array([[x_lo], [x_hi]]))
    return (x_lo, float(y_lo)), (x_hi, float(y_hi))


__all__ = ["robust_fit_endpoints"]
