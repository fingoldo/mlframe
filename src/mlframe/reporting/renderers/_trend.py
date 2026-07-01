"""Robust fit-line helper shared by both renderers.

Returns two endpoint coordinates for a robust line (Theil-Sen or Huber) so a renderer can draw it with a single
two-point ``plot`` regardless of point count. Theil-Sen (median of pairwise slopes) and Huber (M-estimator)
resist the heavy-tailed outliers a pred-vs-actual cloud carries, where an OLS line would tilt toward the worst
points -- exactly the residual structure these overlays are meant to expose against the y=x reference.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Theil-Sen pairwise-slope cost grows with n and Huber iterates over every point; a robust line is a visual
# guide, so fit on at most this many points (x extremes always kept so the endpoints anchor the true range).
# Bounds the default-ON hexbin pred-vs-actual overlay on multi-million-row clouds.
#
# Set to 3000 (was 20000): Theil-Sen's slope is dominated by its ``max_subpopulation=1000`` stochastic pair sample,
# NOT by how many rows it draws those pairs from, so the fitted line is essentially cap-insensitive above ~2-3k points.
# Measured across 25 diverse pred-vs-actual clouds (heteroscedastic + up to 5% outliers), dropping the cap 20000 -> 3000
# shifts the drawn endpoints by <=1.7% of the y-range -- within Theil-Sen's own run-to-run sampling variance at the old
# cap -- while cutting the per-fit cost ~5.5x (820ms -> 149ms at n=60k). The overlay is an explicit visual guide, so a
# sub-2% endpoint shift on an already-stochastic robust line is imperceptible; the ~5.5x pays back directly on the
# default-ON regression pred-vs-actual panel (this fit was ~786ms/call, one of the largest single reporting costs).
_TREND_FIT_CAP = 3_000


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

    if x.size > _TREND_FIT_CAP:
        rng = np.random.default_rng(0)
        keep = rng.choice(x.size, size=_TREND_FIT_CAP - 2, replace=False)
        # Always retain the x extremes so the fit spans the full range it will be drawn across.
        keep = np.concatenate([keep, [int(np.argmin(x)), int(np.argmax(x))]])
        x = x[keep]
        y = y[keep]

    method = method.lower()
    if method == "theil-sen":
        from sklearn.linear_model import TheilSenRegressor
        # Bound the pairwise-slope subpopulation: a fixed 1000-pair sample recovers the slope to ~1e-3 in ~1s
        # vs ~6.5s at the sklearn default 1e4, and the result is a visual guide, not a published estimate.
        model = TheilSenRegressor(random_state=0, max_subpopulation=1000)
    elif method == "huber":
        from sklearn.linear_model import HuberRegressor
        model = HuberRegressor()
    else:
        raise ValueError(f"unknown trend_line method {method!r}; use 'theil-sen' or 'huber'")

    model.fit(x.reshape(-1, 1), y)
    y_lo, y_hi = model.predict(np.array([[x_lo], [x_hi]]))
    return (x_lo, float(y_lo)), (x_hi, float(y_hi))


__all__ = ["robust_fit_endpoints"]
