"""Shared out-of-fold nonparametric warp service for operators B (2-D cell table E[y|x,z]) and C (1-D warp E[y|x]).

API (all numpy, recipes are JSON-serialisable dicts that never contain ``y``):

* ``fit_warp1d(x, y, nb, m)`` / ``fit_cell2d(x, z, y, K, m, prior)``: one table fitted on the given rows.
* ``oof_fit(kind, X, y, src, k, seed, **params)``: k-fold cross-fitting; returns the out-of-fold training column and the recipe (full-train table plus the k fold tables).
* ``replay(recipe, X)``: transform-time function; ``mode='full'`` applies the full-train table, ``mode='foldavg'`` averages the k fold tables (closer in distribution to the
  out-of-fold training column because each fold table saw 1 - 1/k of the rows).

Leak safety: training rows get a value from a table that never saw them (edges and cell means are refitted inside each fold); new rows get values from tables fitted on training rows only.
NaN source values map to the marginal mean of the training target (stored as ``fill``); the output is clipped to the range of the out-of-fold training column (stored as ``lo`` / ``hi``).
"""

from __future__ import annotations

import numpy as np

__all__ = ["fit_warp1d", "fit_cell2d", "oof_fit", "replay", "make_recipe"]


def _edges(x: np.ndarray, k: int) -> np.ndarray:
    """Unique interior quantile edges of ``x`` giving up to ``k`` bins (NaN ignored)."""
    xf = x[np.isfinite(x)]
    return np.unique(np.quantile(xf, np.linspace(0.0, 1.0, k + 1)[1:-1]))


def fit_warp1d(x: np.ndarray, y: np.ndarray, nb: int = 20, m: float = 10.0) -> dict:
    """Binned E[y|x]: shrunk per-bin target means (``m`` pseudo-counts toward the global mean) at the bin mean-x, linearly interpolated at apply time."""
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    e = _edges(x, nb)
    i = np.searchsorted(e, x, side="right")
    nbin = len(e) + 1
    cnt = np.bincount(i, minlength=nbin).astype(float)
    mu = float(y.mean())
    cy = (np.bincount(i, y, nbin) + m * mu) / (cnt + m)
    cx = np.bincount(i, x, nbin) / np.maximum(cnt, 1.0)
    keep = cnt > 0
    return {"type": "warp1d", "cx": cx[keep].tolist(), "cy": cy[keep].tolist(), "fill": mu}


def fit_cell2d(x: np.ndarray, z: np.ndarray, y: np.ndarray, K: int = 10, m: float = 20.0, prior: str = "marginal") -> dict:
    """Binned E[y|x,z] on a K x K quantile grid. Cell mean shrunk with ``m`` pseudo-counts toward ``prior``: ``'marginal'`` (row mean + column mean - grand mean,
    i.e. the additive fit, so a cell only moves away from additivity when it has evidence) or ``'global'`` (grand mean)."""
    ok = np.isfinite(x) & np.isfinite(z)
    x, z, y = x[ok], z[ok], y[ok]
    ex, ez = _edges(x, K), _edges(z, K)
    ix, iz = np.searchsorted(ex, x, side="right"), np.searchsorted(ez, z, side="right")
    nx, nz = len(ex) + 1, len(ez) + 1
    mu = float(y.mean())
    cnt = np.bincount(ix * nz + iz, minlength=nx * nz).astype(float).reshape(nx, nz)
    sm = np.bincount(ix * nz + iz, y, nx * nz).reshape(nx, nz)
    if prior == "marginal":
        rx = (np.bincount(ix, y, nx) + m * mu) / (np.bincount(ix, minlength=nx) + m)
        rz = (np.bincount(iz, y, nz) + m * mu) / (np.bincount(iz, minlength=nz) + m)
        pri = rx[:, None] + rz[None, :] - mu
    else:
        pri = np.full((nx, nz), mu)
    tab = np.where(cnt + m > 0, (sm + m * pri) / np.maximum(cnt + m, 1e-12), pri)
    return {"type": "cell2d", "ex": ex.tolist(), "ez": ez.tolist(), "tab": tab.tolist(), "fill": mu}


def _apply_one(st: dict, cols: list) -> np.ndarray:
    """Apply one fitted table to the source column(s)."""
    if st["type"] == "warp1d":
        x = cols[0]
        out = np.interp(np.where(np.isfinite(x), x, 0.0), st["cx"], st["cy"])
        return np.where(np.isfinite(x), out, st["fill"])
    ix = np.searchsorted(np.asarray(st["ex"]), np.where(np.isfinite(cols[0]), cols[0], 0.0), side="right")
    iz = np.searchsorted(np.asarray(st["ez"]), np.where(np.isfinite(cols[1]), cols[1], 0.0), side="right")
    out = np.asarray(st["tab"])[ix, iz]
    return np.where(np.isfinite(cols[0]) & np.isfinite(cols[1]), out, st["fill"])


def _get_cols(X, src: list) -> list:
    """Source columns of ``X`` as float64 arrays (DataFrame: by name; ndarray: by integer position)."""
    if hasattr(X, "columns"):
        return [np.asarray(X[s], dtype=np.float64) for s in src]
    return [np.asarray(X)[:, s].astype(np.float64) for s in src]


def make_recipe(kind: str, src: list, full: dict, folds: list, lo: float, hi: float, mode: str = "full") -> dict:
    """Frozen, JSON-serialisable recipe: source columns, full-train table, k fold tables, transform mode and output clip."""
    return {"kind": kind, "src": list(src), "full": full, "folds": folds, "mode": mode, "lo": lo, "hi": hi}


def replay(recipe: dict, X, mode: str | None = None) -> np.ndarray:
    """Transform-time function of the recipe and the source columns only (no target, no fit-time arrays)."""
    cols = _get_cols(X, recipe["src"])
    md = mode or recipe["mode"]
    if md == "full":
        out = _apply_one(recipe["full"], cols)
    else:
        out = np.mean([_apply_one(f, cols) for f in recipe["folds"]], axis=0)
    return np.clip(out, recipe["lo"], recipe["hi"])


def oof_fit(kind: str, X: np.ndarray, y: np.ndarray, src: list, k: int = 5, seed: int = 0, **params) -> tuple:
    """Cross-fit ``kind`` (``'warp1d'`` or ``'cell2d'``) on the source columns ``src`` of ``X``: returns (out-of-fold training column, recipe).

    Fold scheme: one random permutation (``seed``) cut into ``k`` disjoint folds; each fold's value comes from a table fitted on the other ``k - 1`` folds
    (edges recomputed inside the fold). The recipe stores the full-train table and the ``k`` fold tables.
    """
    cols = _get_cols(X, src)
    if kind == "warp1d":

        def fit(idx):
            """Fit the 1-D table on rows ``idx``."""
            return fit_warp1d(cols[0][idx], y[idx], **params)

    else:

        def fit(idx):
            """Fit the 2-D table on rows ``idx``."""
            return fit_cell2d(cols[0][idx], cols[1][idx], y[idx], **params)

    n = len(y)
    perm = np.random.default_rng(seed).permutation(n)
    out = np.empty(n)
    folds = []
    for f in range(k):
        te = perm[f::k]
        tr = np.ones(n, bool)
        tr[te] = False
        st = fit(np.where(tr)[0])
        folds.append(st)
        out[te] = _apply_one(st, [c[te] for c in cols])
    full = fit(np.arange(n))
    return out, make_recipe("oof_" + kind, src, full, folds, float(out.min()), float(out.max()))
