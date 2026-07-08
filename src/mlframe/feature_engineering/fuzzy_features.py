"""Fuzzification: encode a numeric column as fuzzy-partition membership features (PZAD fuzzy-set theory).

Dyakonov's lecture frames a fuzzy partition as a POSP (полное ортогональное семантическое пространство, slide 51): a
family of membership functions ``{mu_j}`` that are non-negative, cover the range, and are ORTHOGONAL in the fuzzy sense
``sum_j mu_j(u) = 1`` everywhere (a Ruspini partition). Realized as a feature transformer, this turns one numeric
column into ``n_sets`` interpretable soft-membership columns ("low / below-average / average / above-average / high")
whose rows sum to 1 - a smooth alternative to hard one-hot binning (`KBinsDiscretizer`): where hard binning is
piecewise-CONSTANT and jumps at bin edges, a fuzzy partition interpolates smoothly across neighbouring sets, so a linear
model on the memberships fits a smooth nonlinear target with far lower error at the same number of bins.

Two membership families:
- ``triangular`` - overlapping tents at the set centres with open shoulders at the ends; an EXACT Ruspini partition by
  construction (at any x only the two bracketing sets are active and their weights sum to 1). This coincides with a
  degree-1 B-spline basis (`sklearn.preprocessing.SplineTransformer(degree=1)`); provided here for the interpretable
  membership framing and the lightweight njit recipe path.
- ``gaussian`` - normalized RBFs at the centres (each raw membership ``exp(-0.5((x-c)/sigma)^2)`` divided by the row
  sum). Smooth everywhere with infinite support; a softmax-like fuzzy partition NOT available in sklearn.

`fuzzy_partition_fit` picks the set centres from quantiles (default) or a uniform grid; `fuzzy_partition_transform`
applies a fitted recipe to new data (leakage-safe: centres come from train only); `fuzzy_partition_encode` does both.
"""

from __future__ import annotations

import logging

import numpy as np

try:
    import numba

    _HAS_NUMBA = True
except Exception:  # numba is an optional accelerator
    _HAS_NUMBA = False

logger = logging.getLogger(__name__)

__all__ = [
    "fuzzy_partition_fit",
    "fuzzy_partition_transform",
    "fuzzy_partition_encode",
    "fuzzy_partition_names",
    "FUZZY_MEMBERSHIP_KINDS",
]

FUZZY_MEMBERSHIP_KINDS = ("triangular", "gaussian")


def _triangular_impl(x, centers, out):
    """Exact Ruspini tent partition: only the two bracketing sets active, weights sum to 1; NaN -> all-zero row."""
    n = x.shape[0]
    m = centers.shape[0]
    for i in range(n):
        v = x[i]
        if np.isnan(v):
            continue
        if v <= centers[0]:
            out[i, 0] = 1.0
        elif v >= centers[m - 1]:
            out[i, m - 1] = 1.0
        else:
            for j in range(m - 1):
                if centers[j] <= v < centers[j + 1]:
                    span = centers[j + 1] - centers[j]
                    w = (v - centers[j]) / span if span > 0 else 0.0
                    out[i, j] = 1.0 - w
                    out[i, j + 1] = w
                    break
    return out


def _gaussian_impl(x, centers, sigmas, out):
    """Normalized-RBF fuzzy partition: row-normalized gaussians at the centres; NaN -> all-zero row."""
    n = x.shape[0]
    m = centers.shape[0]
    for i in range(n):
        v = x[i]
        if np.isnan(v):
            continue
        s = 0.0
        for j in range(m):
            d = (v - centers[j]) / sigmas[j]
            e = np.exp(-0.5 * d * d)
            out[i, j] = e
            s += e
        if s > 0.0:
            for j in range(m):
                out[i, j] /= s
    return out


if _HAS_NUMBA:
    _triangular_impl = numba.njit(cache=True, nogil=True)(_triangular_impl)
    _gaussian_impl = numba.njit(cache=True, nogil=True)(_gaussian_impl)


def fuzzy_partition_names(prefix: str, n_sets: int) -> list[str]:
    """Interpretable names for the membership columns of an ``n_sets`` fuzzy partition."""
    if n_sets == 3:
        labels = ["low", "medium", "high"]
    elif n_sets == 5:
        labels = ["low", "below_avg", "average", "above_avg", "high"]
    else:
        labels = [f"set{j}" for j in range(n_sets)]
    return [f"{prefix}_fuzzy_{lab}" for lab in labels]


def fuzzy_partition_fit(
    x,
    *,
    n_sets: int = 5,
    strategy: str = "quantile",
    kind: str = "triangular",
    gaussian_width: float = 1.0,
) -> dict:
    """Choose the fuzzy-set centres for a numeric column, returning a reusable recipe (leakage-safe: fit on train only).

    Parameters
    ----------
    x : 1-D numeric array (NaNs ignored when choosing centres).
    n_sets : number of fuzzy sets (membership columns). >= 2.
    strategy : ``'quantile'`` (centres at equally-spaced quantiles - adapts to the distribution) or ``'uniform'``
        (centres on a uniform grid between the min and max).
    kind : ``'triangular'`` or ``'gaussian'``.
    gaussian_width : multiplier on the median centre spacing used for the gaussian sigmas (ignored for triangular).
    """
    if kind not in FUZZY_MEMBERSHIP_KINDS:
        raise ValueError(f"kind must be one of {FUZZY_MEMBERSHIP_KINDS}, got {kind!r}.")
    if n_sets < 2:
        raise ValueError("fuzzy_partition_fit: require n_sets >= 2.")
    xx = np.asarray(x, dtype=np.float64).ravel()
    finite = xx[np.isfinite(xx)]
    if finite.size == 0:
        raise ValueError("fuzzy_partition_fit: no finite values to fit centres on.")
    if strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_sets)
        centers = np.quantile(finite, qs)
    elif strategy == "uniform":
        centers = np.linspace(finite.min(), finite.max(), n_sets)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'.")
    centers = np.unique(centers)  # collapse duplicate quantiles (degenerate/low-cardinality columns)
    if centers.size < 2:  # all values identical: two centres at +-eps so the partition is well-defined
        c = float(finite[0])
        eps = abs(c) * 1e-6 + 1e-6
        centers = np.array([c - eps, c + eps], dtype=np.float64)
    spacing = np.median(np.diff(centers)) if centers.size > 1 else 1.0
    sigmas = np.full(centers.shape, gaussian_width * max(spacing, 1e-12), dtype=np.float64)
    return {"centers": centers, "sigmas": sigmas, "kind": kind, "n_sets": int(centers.size)}


def fuzzy_partition_transform(x, recipe: dict) -> np.ndarray:
    """Apply a fitted fuzzy-partition recipe to a numeric column, returning the ``(n, n_sets)`` membership matrix."""
    xx = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
    centers = np.ascontiguousarray(recipe["centers"], dtype=np.float64)
    out = np.zeros((xx.shape[0], centers.shape[0]), dtype=np.float64)
    if recipe["kind"] == "triangular":
        return np.asarray(_triangular_impl(xx, centers, out))
    sigmas = np.ascontiguousarray(recipe["sigmas"], dtype=np.float64)
    return np.asarray(_gaussian_impl(xx, centers, sigmas, out))


def fuzzy_partition_encode(
    x,
    *,
    n_sets: int = 5,
    strategy: str = "quantile",
    kind: str = "triangular",
    gaussian_width: float = 1.0,
):
    """Fit + transform in one call. Returns ``(memberships (n, n_sets), recipe)``."""
    recipe = fuzzy_partition_fit(x, n_sets=n_sets, strategy=strategy, kind=kind, gaussian_width=gaussian_width)
    return fuzzy_partition_transform(x, recipe), recipe
