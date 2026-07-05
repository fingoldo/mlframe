"""Sampling helpers for plot panel builders.

The violin / scatter / density panel renderers all run through
``gaussian_kde`` (matplotlib's ``ax.violinplot``) or per-point
matplotlib primitives that scale poorly past ~10k points. On a 1M-row
production suite the un-sampled panels dominate chart wall-time:

  * multiclass ``_prob_dist_panel``  — violin per class, 333k points / group
  * LTR        ``_score_by_rel_panel`` — violin per relevance bin, 250k / quartile
  * regression scatter                — sampled to 500 already in evaluation.py

This module centralizes sampling so every chart builder picks the same
cap with the same deterministic RNG, and a future bump (or per-panel
override) lives in one place.

Cap rationale: ``gaussian_kde`` bandwidth selection via Scott's /
Silverman's rule converges within <1 % on 5000 samples for unimodal /
bimodal distributions (vs the population-bandwidth ground truth). The
visible violin shape diff vs an un-sampled rendering is below pixel
precision at typical chart sizes. 5000 also keeps the per-violin KDE
cost under 50 ms on a single core.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

# Default cap chosen so gaussian_kde converges (Scott's rule plateau)
# and per-violin KDE cost stays <50 ms on a single core. Raise via
# ``mlframe.reporting.charts._sampling.DEFAULT_VIOLIN_SAMPLE_CAP = N``
# at import time if a downstream callsite needs more resolution.
DEFAULT_VIOLIN_SAMPLE_CAP: int = 5_000


def subsample_for_density(
    arr: np.ndarray,
    *,
    cap: int = DEFAULT_VIOLIN_SAMPLE_CAP,
    seed: int = 0,
) -> np.ndarray:
    """Return ``arr`` unchanged when it fits in ``cap``; else uniformly-
    sampled view of length ``cap`` via a fixed-seed RNG.

    Use this immediately before passing a per-group array to
    ``ax.violinplot`` / KDE / per-point matplotlib primitives. Empty /
    near-empty arrays pass through unchanged so degenerate-group
    placeholders (``np.array([0.0])``) keep their semantic.
    """
    if arr is None or len(arr) <= cap:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), size=cap, replace=False)
    return arr[idx]


def _finite_argmin_argmax(arr: np.ndarray) -> list:
    """argmin/argmax indices of ``arr`` ignoring NaN; empty list when nothing finite or non-numeric.

    For floats the plain ``argmin``/``argmax`` are tried first (the common no-NaN case): they avoid the
    full-array copy ``nanargmin``/``nanargmax`` pay via ``_replace_nan``. Only when the result lands on a NaN
    (which happens iff any NaN is present, since NaN propagates to the plain argmin/argmax) do we fall back to
    the nan-aware reductions. ~6.7x on a 2M no-NaN array; bit-identical on both branches.
    """
    if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
        return []
    if np.issubdtype(arr.dtype, np.floating):
        i_min = int(np.argmin(arr))
        i_max = int(np.argmax(arr))
        if not (np.isnan(arr[i_min]) or np.isnan(arr[i_max])):
            return [i_min, i_max]
        try:
            return [int(np.nanargmin(arr)), int(np.nanargmax(arr))]
        except ValueError:  # all-NaN
            return []
    return [int(np.argmin(arr)), int(np.argmax(arr))]


def subsample_preserving_extremes(
    *arrays: np.ndarray,
    sample_size: int,
    extreme_values: Optional[np.ndarray] = None,
    k_extremes: int = 10,
    rng: Union[None, int, np.random.Generator] = None,
) -> np.ndarray:
    """Indices for a uniform random subsample that ALWAYS retains the extreme points.

    Uniform random sampling of large scatters silently drops the very points headline metrics quote
    (e.g. the MaxError row of a regression scatter, or the range endpoints that anchor the axes).
    The returned index set is the union of:

    * a uniform random draw (fixed-seed deterministic unless ``rng`` is supplied),
    * the ``k_extremes`` indices with the largest ``|extreme_values|`` (when given), and
    * the finite argmin/argmax of every positional array.

    Total length is capped at ``sample_size`` (extremes are budgeted first, the random draw fills the
    rest), and indices come back sorted ascending for cache-friendly gathers. O(n) via ``np.argpartition``.

    Parameters
    ----------
    *arrays : 1-D arrays of equal length n (e.g. x, y of a scatter). Their argmin/argmax are kept.
    sample_size : max number of indices to return; when ``n <= sample_size`` all indices come back.
    extreme_values : optional length-n score (e.g. residuals); the k largest-|.| indices are kept.
    k_extremes : how many top-|extreme_values| indices to force-include.
    rng : None (deterministic seed 0), an int seed, or a ``np.random.Generator``.
    """
    if not arrays:
        raise ValueError("subsample_preserving_extremes needs at least one array")
    n = len(arrays[0])
    for a in arrays[1:]:
        if len(a) != n:
            raise ValueError("subsample_preserving_extremes: all arrays must have the same length")
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}")
    if n <= sample_size:
        return np.arange(n, dtype=np.int64)
    if rng is None or isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(0 if rng is None else int(rng))

    forced: list = []
    for a in arrays:
        forced.extend(_finite_argmin_argmax(np.asarray(a)))
    if extreme_values is not None and k_extremes > 0:
        ev = np.abs(np.asarray(extreme_values, dtype=np.float64))
        if len(ev) != n:
            raise ValueError("extreme_values must have the same length as the positional arrays")
        # NaN would sort to the top of argpartition; demote it below every real value (inf stays a legit extreme).
        ev[np.isnan(ev)] = -np.inf
        k = min(int(k_extremes), n)
        forced.extend(np.argpartition(ev, n - k)[n - k :].tolist())

    forced_idx = np.unique(np.asarray(forced, dtype=np.int64)) if forced else np.empty(0, dtype=np.int64)
    budget = sample_size - len(forced_idx)
    if budget <= 0:
        return forced_idx[:sample_size]
    random_idx = rng.choice(n, size=budget, replace=False)
    return np.union1d(forced_idx, random_idx.astype(np.int64))


def prebin_histogram(values: np.ndarray, bins: int, density: bool):
    """Bin raw values once with numpy so renderers can draw bars instead of shipping/re-binning raw n.

    Returns ``(heights, centers, width)`` or ``(None, None, None)`` when the input can't be binned
    (non-numeric dtype, or nothing finite) — callers then fall back to the backend-native histogram.
    Non-finite values are dropped (mirrors what browser-side plotly binning does with NaN).
    """
    vals = np.asarray(values)
    if vals.size == 0 or vals.dtype.kind not in "fiub":
        return None, None, None
    if vals.dtype.kind == "f":
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None, None, None
    heights, edges = np.histogram(vals, bins=bins, density=density)
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = float(edges[1] - edges[0]) if len(edges) > 1 else 1.0
    return heights, centers, width
