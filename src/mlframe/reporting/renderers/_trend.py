"""Robust fit-line helper shared by both renderers.

Returns two endpoint coordinates for a robust line (Theil-Sen or Huber) so a renderer can draw it with a single
two-point ``plot`` regardless of point count. Theil-Sen (median of pairwise slopes) and Huber (M-estimator)
resist the heavy-tailed outliers a pred-vs-actual cloud carries, where an OLS line would tilt toward the worst
points -- exactly the residual structure these overlays are meant to expose against the y=x reference.
"""

from __future__ import annotations

import threading
import weakref
from typing import Dict, Optional, Tuple, Union

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


# The default ``plot_outputs`` renders BOTH backends from the same frozen FigureSpec, so every trend panel
# was fitted twice from the identical arrays for the identical answer -- 0.24 s and 0.25 s of the two
# renders of one regression figure, and ~1.6 s per figure before ``_TREND_FIT_CAP`` was cut.
#
# Keyed on array IDENTITY rather than content: hashing two 2M-row arrays to avoid a 0.25 s fit would cost
# more than the fit. Identity alone is not safe on its own -- a freed array's id is reused -- so each entry
# holds WEAK references and a hit is confirmed by checking they still resolve to these very arrays. Weak,
# not strong, because pinning a 2M-row cloud to keep a cache entry warm is the trade this package's memory
# rules exist to prevent. Guarded because ``render_and_save`` renders the two backends concurrently.
_FIT_CACHE_MAX = 8
_FIT_CACHE: Dict[tuple, tuple] = {}
_FIT_CACHE_LOCK = threading.Lock()


_Endpoints = Optional[Tuple[Tuple[float, float], Tuple[float, float]]]


def _cached_fit(x: np.ndarray, y: np.ndarray, method: str) -> Union[_Endpoints, "_Miss"]:
    """The memoised endpoints for these exact arrays, or ``_MISS``."""
    key = (id(x), id(y), method)
    with _FIT_CACHE_LOCK:
        entry = _FIT_CACHE.get(key)
    if entry is None:
        return _MISS
    x_ref, y_ref, result = entry
    return result if (x_ref() is x and y_ref() is y) else _MISS


def _store_fit(x: np.ndarray, y: np.ndarray, method: str, result: _Endpoints) -> None:
    """Remember ``result`` for these arrays, dropping the oldest entry once the cache is full."""
    try:
        entry = (weakref.ref(x), weakref.ref(y), result)
    except TypeError:
        return  # not weak-referenceable (a plain memoryview / subclass without __weakref__): skip the cache
    with _FIT_CACHE_LOCK:
        if len(_FIT_CACHE) >= _FIT_CACHE_MAX:
            _FIT_CACHE.pop(next(iter(_FIT_CACHE)), None)
        _FIT_CACHE[(id(x), id(y), method)] = entry


class _Miss:
    """Sentinel distinguishing "not cached" from a cached ``None`` (an undefined fit is worth remembering too)."""


_MISS = _Miss()


def robust_fit_endpoints(x: np.ndarray, y: np.ndarray, method: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Fit a robust line ``y ~ x`` and return ((x_lo, y_lo), (x_hi, y_hi)) at the x extremes.

    Returns ``None`` when the fit is undefined (fewer than 2 finite points, or all x identical).

    Memoised on the identity of the input arrays: the same FigureSpec is rendered by both backends.
    """
    cached = _cached_fit(x, y, method)
    if not isinstance(cached, _Miss):
        return cached
    result = _fit_endpoints_uncached(x, y, method)
    _store_fit(x, y, method, result)
    return result


def _fit_endpoints_uncached(x: np.ndarray, y: np.ndarray, method: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """The fit itself, with no memoisation -- the body ``robust_fit_endpoints`` wraps."""
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
        # Hardcoded seed (not caller-configurable) is deliberate: this is a visual overlay, and re-rendering
        # the same (x, y) should draw the identical trend line rather than jittering with process entropy.
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
