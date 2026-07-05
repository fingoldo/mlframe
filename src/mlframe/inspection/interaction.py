"""Friedman & Popescu H-statistic: model-agnostic pairwise feature-interaction strength (PZAD interpretability).

The interpretability lecture (Дьяконов 2018, slide 34) presents Friedman & Popescu's H-statistic as THE way to
measure how much a trained black box's dependence on two features is a genuine INTERACTION versus the sum of their
separate (additive) effects. For a fitted model `f`, using partial-dependence functions PD, the pairwise statistic is

    H²_ij = Σ_k [ PD_ij(x_i^k, x_j^k) − PD_i(x_i^k) − PD_j(x_j^k) ]²  /  Σ_k PD_ij(x_i^k, x_j^k)²

(all PD functions centred to mean 0 over the evaluation points). H² ≈ 0 means the joint effect is purely additive
(``PD_ij = PD_i + PD_j``); H² ≈ 1 means the effect is almost entirely interaction (e.g. an XOR / product target with
no main effects). H = sqrt(H²) ∈ [0, 1].

This is genuinely absent from sklearn (which has `partial_dependence` but no interaction statistic) and is only in
niche packages (`artemis`, `sklearn-gbmi`); it is built here directly on the suite's existing
`mlframe.reporting.charts.pdp_ice.compute_pdp` / `compute_pdp_2d` (so it inherits their efficient
``grid``-predict-calls-independent-of-n contract). Distinct from mlframe's MI-based FE pair search (which finds raw
feature pairs carrying joint signal about y): this quantifies interaction in an ALREADY-TRAINED model, and high-H²
pairs are exactly the ones worth giving an explicit engineered interaction feature.
"""

from __future__ import annotations

import logging

import numpy as np

from mlframe.reporting.charts.pdp_ice import (
    DEFAULT_PDP_GRID,
    DEFAULT_PDP_SAMPLE,
    _as_2d,
    _resolve_feature_index,
    _subsample_idx,
    compute_pdp,
    compute_pdp_2d,
)

logger = logging.getLogger(__name__)

__all__ = ["friedman_h_statistic", "pairwise_interaction_strength"]


def _centered_interp_1d(grid, pdp, x):
    """Interpolate a 1-D PDP to points ``x`` (dedup grid defensively), then centre to mean 0 over ``x``."""
    gu, first = np.unique(np.asarray(grid, dtype=np.float64), return_index=True)
    if gu.shape[0] < 2:
        return np.zeros_like(x, dtype=np.float64)  # constant feature -> no dependence
    vals = np.interp(x, gu, np.asarray(pdp, dtype=np.float64)[first])
    return vals - vals.mean()


def _centered_interp_2d(grid0, grid1, surface, xi, xj):
    """Bilinear-interpolate a 2-D PD surface to points ``(xi, xj)``, centre to mean 0. Returns None if degenerate."""
    from scipy.interpolate import RegularGridInterpolator

    g0, f0 = np.unique(np.asarray(grid0, dtype=np.float64), return_index=True)
    g1, f1 = np.unique(np.asarray(grid1, dtype=np.float64), return_index=True)
    if g0.shape[0] < 2 or g1.shape[0] < 2:
        return None
    surf = np.asarray(surface, dtype=np.float64)[np.ix_(f0, f1)]
    interp = RegularGridInterpolator((g0, g1), surf, bounds_error=False, fill_value=None)
    pts = np.column_stack([np.clip(xi, g0[0], g0[-1]), np.clip(xj, g1[0], g1[-1])])
    vals = interp(pts)
    return vals - vals.mean()


def friedman_h_statistic(
    model,
    X,
    features,
    *,
    grid: int = DEFAULT_PDP_GRID,
    sample: int = DEFAULT_PDP_SAMPLE,
    seed: int = 0,
) -> float:
    """Friedman & Popescu H-statistic for one feature PAIR: interaction strength in ``[0, 1]``.

    Parameters
    ----------
    model : fitted estimator with ``predict`` (or ``predict_proba`` for a binary classifier).
    X : the data the model was trained/evaluated on (numpy array, pandas or polars frame).
    features : ``(f0, f1)`` two feature indices or names.
    grid, sample, seed : forwarded to the underlying PDP computation (grid resolution, row subsample, RNG seed).

    Returns ``H = sqrt(H²)`` where ``H²`` is the fraction of the pair's joint centred partial-dependence variance that
    is non-additive interaction. ``0`` for an additive effect, near ``1`` for a pure interaction (XOR / product).
    """
    f0, f1 = features
    vals, _, names = _as_2d(X)
    n, n_cols = vals.shape
    i0 = _resolve_feature_index(f0, names, n_cols)
    i1 = _resolve_feature_index(f1, names, n_cols)
    if i0 == i1:
        raise ValueError("friedman_h_statistic: the two features must differ.")

    p0 = compute_pdp(model, X, i0, grid=grid, sample=sample, ice=False, seed=seed)
    p1 = compute_pdp(model, X, i1, grid=grid, sample=sample, ice=False, seed=seed)

    idx = _subsample_idx(n, sample, seed)  # same subsample the PDPs used -> consistent evaluation points
    xi = vals[idx, i0].astype(np.float64)
    xj = vals[idx, i1].astype(np.float64)
    return _h_from_1d_pdps(model, X, i0, i1, p0, p1, xi, xj, grid=grid, sample=sample, seed=seed)


def _h_from_1d_pdps(model, X, i0, i1, p0, p1, xi, xj, *, grid, sample, seed) -> float:
    """Core H-statistic given the two features' already-computed 1-D PDPs; only the 2-D surface is computed here."""
    p2 = compute_pdp_2d(model, X, (i0, i1), grid=grid, sample=sample, seed=seed)
    c_ij = _centered_interp_2d(p2["grid0"], p2["grid1"], p2["surface"], xi, xj)
    if c_ij is None:
        return 0.0  # a constant feature has no interaction
    c_i = _centered_interp_1d(p0["grid"], p0["pdp"], xi)
    c_j = _centered_interp_1d(p1["grid"], p1["pdp"], xj)

    denom = float(np.sum(c_ij**2))
    if denom <= 1e-12:
        return 0.0  # joint PD is flat -> model ignores the pair
    numer = float(np.sum((c_ij - c_i - c_j) ** 2))
    h2 = min(max(numer / denom, 0.0), 1.0)
    return float(np.sqrt(h2))


def pairwise_interaction_strength(
    model,
    X,
    features,
    *,
    grid: int = DEFAULT_PDP_GRID,
    sample: int = DEFAULT_PDP_SAMPLE,
    seed: int = 0,
) -> np.ndarray:
    """Symmetric ``(F, F)`` matrix of H-statistics over a list of ``F`` features (diagonal 0).

    ``M[a, b]`` is :func:`friedman_h_statistic` for ``(features[a], features[b])`` - the pairs worth engineering an
    explicit interaction feature for are the largest off-diagonal entries. Each feature's 1-D PDP is computed ONCE and
    reused across all its pairs (only the ``F(F-1)/2`` 2-D surfaces are recomputed), so this is materially cheaper than
    calling :func:`friedman_h_statistic` on every pair.
    """
    f = list(features)
    F = len(f)
    vals, _, names = _as_2d(X)
    n, n_cols = vals.shape
    cols = [_resolve_feature_index(x, names, n_cols) for x in f]
    idx = _subsample_idx(n, sample, seed)
    pdps = [compute_pdp(model, X, c, grid=grid, sample=sample, ice=False, seed=seed) for c in cols]
    xs = [vals[idx, c].astype(np.float64) for c in cols]

    M = np.zeros((F, F), dtype=np.float64)
    for a in range(F):
        for b in range(a + 1, F):
            if cols[a] == cols[b]:
                continue  # duplicate feature reference -> no interaction with itself
            h = _h_from_1d_pdps(model, X, cols[a], cols[b], pdps[a], pdps[b], xs[a], xs[b], grid=grid, sample=sample, seed=seed)
            M[a, b] = M[b, a] = h
    return M
