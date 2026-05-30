"""Per-feature adaptive bin-count selection for MRMR (2026-05-28).

Pre-fix: ``MRMR(quantization_method='quantile', quantization_nbins=10)`` binned
EVERY column to the same fixed n_bins=10 regardless of distribution shape,
sample size, or signal strength. That's a one-size-fits-all compromise that:

* under-resolves a binary signal (only 2 bins needed, gets 10)
* under-resolves a continuous fine-structure signal (10 bins lose the cuts)
* equalises post-binning cardinality, washing out the MI cardinality bias
  (so SU normalisation has no effect -- see test_biz_val_mrmr_symmetric_uncertainty)

This module ships per-feature adaptive bin selection via six methods:

* **'sturges'**     -- `ceil(1 + log2(n))`. Simplest formula; assumes Gaussian-ish.
* **'freedman_diaconis'** (= `'fd'`, DEFAULT for auto) -- `ceil((max-min) / (2*IQR/n^(1/3)))`.
  Robust to outliers/skew; the recommended default for MI estimation
  on natural data per Freedman & Diaconis 1981.
* **'knuth'**       -- Bayesian-posterior optimum over M in [1, sqrt(N)*4].
  Native impl in ``discretization.py``. Returns 1 bin for featureless uniform.
* **'blocks'**      -- Bayesian Blocks (Scargle 2013). Variable-width edges, no n_bins.
* **'fayyad_irani'** (= `'mdlp'`)  -- SUPERVISED MDLP from Fayyad & Irani 1993.
  Returns 3-8 bins typically, signal-aware. **Target-leak-safe via CV-fold splits**.
* **'optimal_joint'** (= `'cv'`) -- CV-based: train binning on fold-train, score
  MI on fold-val, average across folds. Most expensive but no overfitting risk.
* **'auto'** -- alias for 'freedman_diaconis'.

The dispatcher ``per_feature_nbins`` returns bin EDGES per column (jagged list since
each feature may have a different bin count). MRMR's downstream ``np.searchsorted``
chain handles variable edges transparently.

Performance notes:
* Sturges / FD: O(1) formula per column, vectorisable
* Knuth: O(N * M_max) per column, M_max ~ sqrt(N)*4
* Blocks: O(N²) per column - cap by sample or downsample
* Fayyad-Irani: O(N log N * depth) per column with depth=8
* OptimalJoint: K-fold CV * any base method
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Union

import numpy as np
from numba import njit

from .discretization import _knuth_bin_edges, _bayesian_blocks_bin_edges
from .supervised_binning import mdlp_bin_edges

logger = logging.getLogger(__name__)


__all__ = [
    "sturges_nbins", "freedman_diaconis_nbins", "qs_nbins",
    "edges_quantile", "edges_uniform",
    "edges_sturges", "edges_freedman_diaconis", "edges_qs",
    "edges_knuth", "edges_bayesian_blocks",
    "edges_fayyad_irani", "edges_mah",
    "edges_optimal_joint",
    "per_feature_edges", "AUTO_METHOD_DEFAULT",
]


AUTO_METHOD_DEFAULT = "freedman_diaconis"


# -----------------------------------------------------------------------------
# Simple closed-form rules
# -----------------------------------------------------------------------------


def sturges_nbins(n: int) -> int:
    """Sturges (1926): n_bins = ceil(1 + log2(n)). Assumes Gaussian-ish data; under-bins skewed data."""
    if n < 2:
        return 1
    return max(1, int(math.ceil(1.0 + math.log2(n))))


def freedman_diaconis_nbins(x: np.ndarray) -> int:
    """Freedman-Diaconis (1981): n_bins = ceil((max-min) / h)  where h = 2*IQR(x) / n^(1/3).

    Robust to outliers via IQR; falls back to Sturges when IQR is 0 (constant /
    very-discrete data). Floor at 1; cap at sqrt(N)*4 to avoid bin-per-sample on
    heavy-tail data.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return max(1, sturges_nbins(n))
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    if iqr <= 0:
        return sturges_nbins(n)
    h = 2.0 * iqr / (n ** (1.0 / 3.0))
    span = float(x.max() - x.min())
    if h <= 0 or span <= 0:
        return sturges_nbins(n)
    nbins = int(math.ceil(span / h))
    cap = max(2, int(math.sqrt(n) * 4))
    return max(1, min(nbins, cap))


# -----------------------------------------------------------------------------
# Edge-builders. Each returns a SORTED ndarray of bin edges (inner cuts) suitable
# for ``np.searchsorted(edges, x, side='right')`` -- matching the legacy contract.
# -----------------------------------------------------------------------------


def _edges_from_quantiles(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile binning helper. Returns INNER edges only (n_bins - 1 values).

    2026-05-30 Wave 9.1 fix (loop iter 6): de-duplicate via ``np.unique``
    before returning. ``np.nanpercentile`` on a constant column emits
    ``n_bins + 1`` identical values; on a heavily skewed / sparse column
    (e.g. 99% zeros) it emits a run of identical inner edges. Both cases
    inflate downstream ``K_x = len(edges) + 1`` for ``searchsorted``,
    silently over-correcting Miller-Madow MI bias by
    ``(K_x_phantom - 1) * (K_y - 1) / (2N)`` and inflating ``log(K_x)``
    in the SU normaliser denominator. ``edges_qs`` already does this -
    bring ``_edges_from_quantiles`` in line so every caller funneling
    through ``edges_quantile`` / ``edges_freedman_diaconis`` /
    ``edges_sturges`` / ``edges_optimal_joint`` inherits the fix.
    """
    if n_bins < 2:
        return np.array([], dtype=np.float64)
    quantiles = np.linspace(0.0, 100.0, n_bins + 1)
    full_edges = np.nanpercentile(np.asarray(x, dtype=np.float64).ravel(), quantiles)
    full_edges = np.unique(full_edges)
    if full_edges.size <= 2:
        return np.array([], dtype=np.float64)
    return np.asarray(full_edges[1:-1], dtype=np.float64)


def _edges_from_uniform(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Uniform binning helper. Returns INNER edges only (n_bins - 1 values)."""
    if n_bins < 2:
        return np.array([], dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).ravel()
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    if xmax <= xmin:
        return np.array([], dtype=np.float64)
    full_edges = np.linspace(xmin, xmax, n_bins + 1)
    return np.asarray(full_edges[1:-1], dtype=np.float64)


def qs_nbins(n: int, alpha: float = 0.30) -> int:
    """Quantile Spacing nbins (Gupta 2021): ``n_q = round(alpha * N)``, clamped [3, 64].

    Gupta et al. *Entropy* 23(6):740, 2021 (https://arxiv.org/abs/2102.12675) show
    that alpha in [0.25, 0.35] is insensitive to sample size and distribution.
    Default alpha=0.30 picks the middle of that band. The clamp prevents
    bin-per-sample at large N (which would inflate plug-in MI bias) and prevents
    1-bin degeneracy at tiny N.
    """
    if n < 2:
        return 1
    raw = int(round(alpha * n))
    return max(3, min(64, raw))


def edges_qs(x: np.ndarray, alpha: float = 0.30) -> np.ndarray:
    """Quantile Spacing (QS) bin edges from Gupta et al. *Entropy* 2021.

    Picks ``n_q = qs_nbins(N, alpha)`` equiprobable bins; returns the (n_q - 1)
    INNER quantile edges. The key property: under independence the joint
    histogram is uniform by construction (each marginal is forced flat), so
    plug-in MI collapses to ~0 with O(1/sqrt(N)) noise instead of the
    bias-inflated O(M/N) of equal-width estimators. The trade-off: marginally
    underestimates true MI on signal-bearing data (Gupta 2021 Section 3).

    Args:
        x: 1-D continuous data.
        alpha: Sample-fraction parameter in ``[0.25, 0.35]``; data-independent
            per Gupta 2021 empirics.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    n_q = qs_nbins(n, alpha)
    if n_q < 2:
        return np.array([], dtype=np.float64)
    q_pos = np.linspace(0.0, 100.0, n_q + 1)
    full_edges = np.nanpercentile(x, q_pos)
    # De-duplicate (ties / near-constant columns collapse).
    full_edges = np.unique(full_edges)
    if full_edges.size <= 2:
        return np.array([], dtype=np.float64)
    return np.asarray(full_edges[1:-1], dtype=np.float64)


def edges_quantile(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Fixed-nbins quantile edges. Legacy MRMR default behaviour."""
    return _edges_from_quantiles(x, n_bins)


def edges_uniform(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Fixed-nbins uniform-width edges."""
    return _edges_from_uniform(x, n_bins)


def edges_sturges(x: np.ndarray, base: str = "quantile") -> np.ndarray:
    """Sturges nbins + quantile/uniform edges."""
    n = np.isfinite(np.asarray(x, dtype=np.float64).ravel()).sum()
    n_bins = sturges_nbins(int(n))
    return _edges_from_quantiles(x, n_bins) if base == "quantile" else _edges_from_uniform(x, n_bins)


def edges_freedman_diaconis(x: np.ndarray, base: str = "quantile") -> np.ndarray:
    """Freedman-Diaconis nbins + quantile/uniform edges. Default for ``auto``."""
    n_bins = freedman_diaconis_nbins(x)
    return _edges_from_quantiles(x, n_bins) if base == "quantile" else _edges_from_uniform(x, n_bins)


def edges_knuth(x: np.ndarray, edge_type: str = "uniform",
                m_max_cap: int = 500) -> np.ndarray:
    """Knuth (2006) Bayesian-optimal nbins. Flags forwarded to ``_knuth_bin_edges``:

    Args:
        edge_type: ``'uniform'`` (legacy faithful) | ``'quantile'`` (audit fix).
        m_max_cap: ``500`` legacy | ``64`` audit recommendation to stay in
            low plug-in bias regime on small val-folds.
    """
    full_edges = _knuth_bin_edges(np.asarray(x), edge_type=edge_type,
                                   m_max_cap=int(m_max_cap))
    if full_edges.size <= 2:
        return np.array([], dtype=np.float64)
    return np.asarray(full_edges[1:-1], dtype=np.float64)


def edges_bayesian_blocks(x: np.ndarray, p0: float = 0.05,
                           edge_placement: str = "start",
                           subsample_threshold: int = 0) -> np.ndarray:
    """Bayesian Blocks (Scargle 2013) variable-width edges. Flags forwarded:

    Args:
        p0: ``0.05`` legacy | ``0.10`` audit recommendation for continuous data.
        edge_placement: ``'start'`` legacy | ``'midpoint'`` Scargle/astropy convention fix.
        subsample_threshold: ``0`` disabled | ``1000`` audit fast path.
    """
    full_edges = _bayesian_blocks_bin_edges(np.asarray(x), p0=p0,
                                             edge_placement=edge_placement,
                                             subsample_threshold=int(subsample_threshold))
    if full_edges.size <= 2:
        return np.array([], dtype=np.float64)
    return np.asarray(full_edges[1:-1], dtype=np.float64)


def edges_mah(x: np.ndarray, y: np.ndarray, *, initial_k: int = 16) -> np.ndarray:
    """MAH/SCI (Marx 2021) supervised bin edges. Greedy SCI-guided merge
    starting from K equal-frequency quantile bins; the X-axis merges that
    don't reduce the joint NML code length get reverted into kept edges.

    Default initial_k=16 matches the paper's reported sweep.
    """
    from ._mah import mah_bin_edges
    inner = mah_bin_edges(np.asarray(x), np.asarray(y), initial_k=int(initial_k))
    inner = np.asarray(inner, dtype=np.float64)
    return inner[np.isfinite(inner)] if inner.size else inner


def edges_fayyad_irani(x: np.ndarray, y: np.ndarray, *, max_depth: int = 8,
                        min_split_size: int = 5, backend: str = "njit",
                        scaled_min_split: bool = False) -> np.ndarray:
    """Fayyad-Irani MDLP supervised edges. Flags forwarded:

    Args:
        backend: ``'njit'`` (default; audit-recommended 10-30x speedup over
            the legacy pure-Python path) | ``'python'`` (legacy fallback
            kept for A/B testing). Sibling ``mdlp_bin_edges`` already
            defaults to ``'njit'``; this wrapper now matches that default
            so callers that go through the wrapper benefit too.
            c0022_9f2cf625 @500k profile (2026-05-30): the python-backend
            path consumed 1566 s of a 1700 s suite (88 % of wall) before
            this fix because ``_mdlp_recurse`` calls ``_entropy_from_labels``
            268 345 times, each doing ``np.unique`` + ``np.sort`` on the
            label slice (4 ms per call at n=500k). The njit kernel
            ``_mdlp_recurse_njit`` maintains running per-class counts
            across the candidate-scan, so entropy is O(K_y) per candidate
            instead of O(N log N + N) per candidate.
        scaled_min_split: ``False`` legacy | ``True`` audit fix
            (``max(5, 0.02*N)``).
    """
    full_edges = mdlp_bin_edges(np.asarray(x), np.asarray(y),
                                 max_depth=max_depth, min_split_size=min_split_size,
                                 backend=backend, scaled_min_split=scaled_min_split)
    if full_edges.size <= 2:
        return np.array([], dtype=np.float64)
    inner = full_edges[1:-1]
    inner = inner[np.isfinite(inner)]
    return np.asarray(inner, dtype=np.float64)


def edges_optimal_joint(
    x: np.ndarray, y: np.ndarray,
    *,
    candidates: tuple = (4, 8, 16, 32),
    n_splits: int = 3,
    base: str = "quantile",
    random_state: int = 0,
) -> np.ndarray:
    """CV-folded MI maximisation across candidate nbins.

    For each n_bins in ``candidates``:
      * Split (x, y) into ``n_splits`` folds.
      * For each fold: compute quantile edges on TRAIN x, bin VAL x with those edges,
        estimate I(X_val; y_val) via the standard plug-in histogram MI.
      * Score = mean fold MI.
    Pick the n_bins with highest score, return its edges built on the FULL data.

    This is the OptimalJoint / "wrapper-style" method recommended for MRMR when
    compute is not a bottleneck. ~K_candidates * n_splits times more expensive
    than a single Freedman-Diaconis call.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y).ravel()
    mask = np.isfinite(x)
    x = x[mask]
    y = y[mask]
    n = x.size
    if n < n_splits * 4:
        # Fall back to FD on small samples.
        return edges_freedman_diaconis(x, base=base)
    rng = np.random.default_rng(random_state)
    fold_idx = rng.permutation(n) % n_splits
    best_score = -np.inf
    best_M = candidates[0]
    for M in candidates:
        if M < 2:
            continue
        fold_scores = []
        for k in range(n_splits):
            train_mask = fold_idx != k
            val_mask = ~train_mask
            if train_mask.sum() < M or val_mask.sum() < 4:
                continue
            train_x, train_y = x[train_mask], y[train_mask]
            val_x, val_y = x[val_mask], y[val_mask]
            edges = _edges_from_quantiles(train_x, M) if base == "quantile" \
                else _edges_from_uniform(train_x, M)
            if edges.size == 0:
                continue
            binned_val_x = np.searchsorted(edges, val_x, side="right")
            # Plug-in MI estimation on val fold.
            mi = _plug_in_mi(binned_val_x.astype(np.int64), val_y)
            fold_scores.append(mi)
        if fold_scores:
            mean_mi = float(np.mean(fold_scores))
            if mean_mi > best_score:
                best_score = mean_mi
                best_M = M
    # Return edges built on full data at the winning M.
    return _edges_from_quantiles(x, best_M) if base == "quantile" \
        else _edges_from_uniform(x, best_M)


@njit(nogil=True, cache=True)
def _plug_in_mi_njit(x_binned: np.ndarray, y_b: np.ndarray, K_x: int, K_y: int,
                     miller_madow: bool) -> float:
    """njit core: plug-in MI from already-integer-encoded bins, optional Miller-Madow.

    Miller-Madow (Miller 1955; Madow 1948) bias term: ``(K_x - 1) * (K_y - 1) / (2 * N)``.
    Subtracted post-accumulation, floored at 0 so honest no-signal MI is exactly 0.
    """
    n = x_binned.shape[0]
    if n == 0:
        return 0.0
    joint = np.zeros((K_x, K_y), dtype=np.float64)
    for i in range(n):
        joint[x_binned[i], y_b[i]] += 1.0
    n_f = float(n)
    Px = np.zeros(K_x, dtype=np.float64)
    Py = np.zeros(K_y, dtype=np.float64)
    for i in range(K_x):
        for j in range(K_y):
            v = joint[i, j]
            Px[i] += v
            Py[j] += v
    mi = 0.0
    for i in range(K_x):
        if Px[i] <= 0.0:
            continue
        for j in range(K_y):
            v = joint[i, j]
            if v <= 0.0 or Py[j] <= 0.0:
                continue
            p = v / n_f
            mi += p * math.log(p * n_f / (Px[i] * Py[j] / n_f))
    if miller_madow:
        bias = (K_x - 1) * (K_y - 1) / (2.0 * n_f)
        mi -= bias
        if mi < 0.0:
            mi = 0.0
    return mi


def _plug_in_mi(x_binned: np.ndarray, y: np.ndarray, miller_madow: bool = False) -> float:
    """Plug-in MI estimator: I(X; Y) = H(X) + H(Y) - H(X, Y) on counts.

    Args:
        x_binned: 1-D ``int`` array of bin indices for X.
        y: 1-D array of target values (int classes or float -> regression-binned to 10 quantiles).
        miller_madow: When ``True``, subtract ``(K_x - 1) * (K_y - 1) / (2 * N)``
            bias-correction term (Miller 1955; Madow 1948). Restores honest noise
            floor at higher M values (FD's no-signal inflation collapses from
            0.077 -> ~0.02). Floor at zero. Default ``False`` preserves the
            pre-2026-05-29 leaderboard baseline; opt-in via flag.
    """
    if x_binned.size == 0:
        return 0.0
    if y.dtype.kind not in "iub":
        q = np.quantile(y.astype(np.float64), np.linspace(0, 1, 11))
        q = np.unique(q)
        if q.size < 2:
            return 0.0
        y_b = np.searchsorted(q[1:-1], y.astype(np.float64), side="right").astype(np.int64)
    else:
        y_b = y.astype(np.int64)
    x_b = x_binned.astype(np.int64)
    if x_b.size == 0 or y_b.size == 0:
        return 0.0
    K_y = int(y_b.max()) + 1 if y_b.size else 1
    K_x = int(x_b.max()) + 1 if x_b.size else 1
    if K_x < 1 or K_y < 1:
        return 0.0
    return float(_plug_in_mi_njit(x_b, y_b, K_x, K_y, miller_madow))


# -----------------------------------------------------------------------------
# Public dispatcher
# -----------------------------------------------------------------------------


_METHOD_ALIASES = {
    "auto": AUTO_METHOD_DEFAULT,
    "fd": "freedman_diaconis",
    "freedman-diaconis": "freedman_diaconis",
    "freedman_diaconis": "freedman_diaconis",
    "sturges": "sturges",
    "knuth": "knuth",
    "blocks": "bayesian_blocks",
    "bayesian_blocks": "bayesian_blocks",
    "bb": "bayesian_blocks",
    "fayyad_irani": "fayyad_irani",
    "fayyad-irani": "fayyad_irani",
    "mdlp": "fayyad_irani",
    "optimal_joint": "optimal_joint",
    "cv": "optimal_joint",
    "qs": "qs",
    "quantile_spacing": "qs",
    "gupta": "qs",
    "mah": "mah",
    "mah_sci": "mah",
    "sci": "mah",
    "marx": "mah",
}


def per_feature_edges(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    method: str = "auto",
    base: str = "quantile",
    **kwargs,
) -> list:
    """Return list-of-arrays of bin edges per feature column.

    Args:
        X: 2-D ndarray of shape (n_samples, n_features). May contain NaN.
        y: Target array, required for supervised methods ('fayyad_irani', 'optimal_joint').
        method: One of 'auto' (= 'freedman_diaconis'), 'sturges', 'freedman_diaconis' / 'fd',
            'knuth', 'bayesian_blocks' / 'blocks' / 'bb', 'fayyad_irani' / 'mdlp',
            'optimal_joint' / 'cv'.
        base: For 'sturges' / 'freedman_diaconis' / 'optimal_joint' — 'quantile' (default)
            or 'uniform'.
        **kwargs: Forwarded to the underlying edge-builder
            (e.g. ``max_depth`` for fayyad_irani, ``candidates`` for optimal_joint,
            ``p0`` for bayesian_blocks).

    Returns:
        list of length n_features; each entry is a 1-D ndarray of INNER bin edges
        (i.e. ``n_bins - 1`` values, suitable for ``np.searchsorted(edges, x, side='right')``).
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"per_feature_edges: X must be 2-D; got shape {X.shape}")
    n_features = X.shape[1]
    method_resolved = _METHOD_ALIASES.get(method.lower() if isinstance(method, str) else method)
    if method_resolved is None:
        raise ValueError(
            f"per_feature_edges: unknown method={method!r}. Expected one of "
            f"{sorted(set(_METHOD_ALIASES.values()))}."
        )
    needs_y = method_resolved in ("fayyad_irani", "optimal_joint", "mah")
    if needs_y and y is None:
        raise ValueError(
            f"per_feature_edges: method={method_resolved!r} is supervised and requires y."
        )
    if y is not None:
        y = np.asarray(y).ravel()
    edges_list: list = []
    for j in range(n_features):
        col = X[:, j].astype(np.float64, copy=False)
        if method_resolved == "sturges":
            edges = edges_sturges(col, base=base)
        elif method_resolved == "freedman_diaconis":
            edges = edges_freedman_diaconis(col, base=base)
        elif method_resolved == "qs":
            edges = edges_qs(col, alpha=kwargs.get("qs_alpha", 0.30))
        elif method_resolved == "knuth":
            edges = edges_knuth(
                col,
                edge_type=kwargs.get("knuth_edge_type", "uniform"),
                m_max_cap=kwargs.get("knuth_m_max_cap", 500),
            )
        elif method_resolved == "bayesian_blocks":
            edges = edges_bayesian_blocks(
                col,
                p0=kwargs.get("p0", 0.05),
                edge_placement=kwargs.get("bb_edge_placement", "start"),
                subsample_threshold=kwargs.get("bb_subsample_threshold", 0),
            )
        elif method_resolved == "fayyad_irani":
            edges = edges_fayyad_irani(
                col, y,
                max_depth=kwargs.get("max_depth", 8),
                min_split_size=kwargs.get("min_split_size", 5),
                # Match edges_fayyad_irani / mdlp_bin_edges default ('njit').
                # The legacy 'python' default here re-introduced the
                # 1566 s / 1700 s @500 k regression that the iter570 fix
                # otherwise resolved -- this kwargs.get path is the
                # production caller from categorize_dataset, so the
                # default flip is the actual gating change.
                backend=kwargs.get("mdlp_backend", "njit"),
                scaled_min_split=kwargs.get("mdlp_scaled_min_split", False),
            )
        elif method_resolved == "mah":
            edges = edges_mah(
                col, y,
                initial_k=int(kwargs.get("mah_initial_k", 16)),
            )
        elif method_resolved == "optimal_joint":
            edges = edges_optimal_joint(
                col, y,
                candidates=kwargs.get("candidates", (4, 8, 16, 32)),
                n_splits=kwargs.get("n_splits", 3),
                base=base,
                random_state=kwargs.get("random_state", 0),
            )
        else:
            raise NotImplementedError(method_resolved)
        edges_list.append(edges)
    return edges_list
