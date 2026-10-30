"""Per-feature adaptive bin-count selection for MRMR.

Pre-fix: ``MRMR(quantization_method='quantile', quantization_nbins=10)`` binned
EVERY column to the same fixed n_bins=10 regardless of distribution shape,
sample size, or signal strength. That's a one-size-fits-all compromise that:

* under-resolves a binary signal (only 2 bins needed, gets 10)
* under-resolves a continuous fine-structure signal (10 bins lose the cuts)
* equalises post-binning cardinality, washing out the MI cardinality bias
  (so SU normalisation has no effect - see test_biz_val_mrmr_symmetric_uncertainty)

This module ships per-feature adaptive bin selection via six methods:

* **'sturges'**  - `ceil(1 + log2(n))`. Simplest formula; assumes Gaussian-ish.
* **'freedman_diaconis'** (= `'fd'`, DEFAULT for auto) - `ceil((max-min) / (2*IQR/n^(1/3)))`.
  Robust to outliers/skew; the recommended default for MI estimation
  on natural data per Freedman & Diaconis 1981.
* **'knuth'**  - Bayesian-posterior optimum over M in [1, sqrt(N)*4].
  Native impl in ``discretization.py``. Returns 1 bin for featureless uniform.
* **'blocks'**  - Bayesian Blocks (Scargle 2013). Variable-width edges, no n_bins.
* **'fayyad_irani'** (= `'mdlp'`)  - SUPERVISED MDLP from Fayyad & Irani 1993.
  Returns 3-8 bins typically, signal-aware. **Target-leak-safe via CV-fold splits**.
* **'optimal_joint'** (= `'cv'`) - CV-based: train binning on fold-train, score
  MI on fold-val, average across folds. Most expensive but no overfitting risk.
* **'auto'** - alias for 'freedman_diaconis'.

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
import threading
from typing import Optional, Any

import numpy as np
import numba
from numba import njit, prange

from mlframe.utils.log_throttle import log_throttle
from mlframe.feature_selection.filters.discretization.shared import (
    knuth_bin_edges as _knuth_bin_edges, bayesian_blocks_bin_edges as _bayesian_blocks_bin_edges,
)
from .supervised_binning import mdlp_bin_edges
from ._mdlp_validated_split import edges_fayyad_irani_validated  # noqa: F401  # resolved by _adaptive_nbins_columns via this module

# Shared per-column bin-count ceiling for every adaptive strategy whose own formula has no natural
# upper bound (knuth, bayesian_blocks). MDLP already lands here implicitly via max_depth=8 -> up to
# 2**8=256 leaves (supervised_binning.py); freedman_diaconis's own sqrt(N)*4 cap is additionally
# clamped to this same ceiling below. Kept as one named constant so every strategy answers "how many
# bins can a single column produce" the same way - divergent per-method caps (found live: knuth
# defaulted to 500, bayesian_blocks had NO cap at all) let a single real-data column reach thousands
# of bins, blowing the downstream joint-cardinality (nbins_a * nbins_b) past both the CUDA
# shared-memory budget and the row-chunked global-memory fallback's launch-count budget, forcing a
# multi-thousand-second CPU njit fallback per column pair (found live, 50k-row wellbore GT sweep).
MAX_ADAPTIVE_NBINS = 256

# Default cap for the Bayesian-Blocks DP at the ``per_feature_edges`` dispatch layer. BB is O(N^2)-per-column;
# without a subsample cap ``nbins_strategy='blocks'`` pays unbounded quadratic cost on ordinary column sizes.
# When N exceeds this, the BB DP fits on a uniform sub-sample of this many rows (the block EDGES it returns
# still apply to the full column). Direct low-level ``edges_bayesian_blocks`` callers keep the 0-means-exact
# contract; only this auto dispatch layer defaults a safe bound (per CLAUDE.md "enable corrective mechanisms
# by default"). Callers can override via ``bb_subsample_threshold`` (0 restores the exact full-N DP).
_BB_DEFAULT_SUBSAMPLE_THRESHOLD = 5000

logger = logging.getLogger(__name__)

# Minimum number of cache-MISS columns before per_feature_edges engages the thread pool. The
# GIL-releasing njit MDLP kernels make threading pay off far earlier than the previous threshold
# (128) assumed: 2M-row cProfile (combo c0037_c314bb14, master-seed 2026_04_29) found
# per_feature_edges/_compute_col_edges costing 78s wall on a fayyad_irani fit with only ~30 feature
# columns -- well under the old 128-column gate, so the thread pool never engaged in practice on any
# realistic combo width. Re-benchmarked fresh (n=300k, fayyad_irani, this host): ncols=2 -> 1.24x,
# ncols=4 -> 3.71x, ncols=8 -> 4.44x, ncols=16 -> 4.44x, ncols=30 -> 7.18x -- consistently faster
# threaded from the smallest tested width, no regression at any point measured. The old "p=50: parallel
# ties serial" comment this replaced does not reproduce; trusting the fresh, reproducible A/B over the
# stale claim. Edges stay BIT-IDENTICAL to the serial path regardless of n_jobs/thread scheduling (by
# construction: each column's edges are computed independently, no shared mutable state) at any
# threshold, so this is a pure wall-time change.
_PARALLEL_EDGES_MIN_COLS = 2


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
    return max(1, math.ceil(1.0 + math.log2(n)))


def freedman_diaconis_nbins(x: np.ndarray, max_bins: int = MAX_ADAPTIVE_NBINS) -> int:
    """Freedman-Diaconis (1981): n_bins = ceil((max-min) / h)  where h = 2*IQR(x) / n^(1/3).

    Robust to outliers via IQR; falls back to Sturges when IQR is 0 (constant /
    very-discrete data). Floor at 1; cap at min(sqrt(N)*4, max_bins) to avoid
    bin-per-sample on heavy-tail data AND to bound joint-cardinality cost on
    large real datasets (sqrt(N)*4 alone is unbounded as N grows - e.g. ~895
    bins at N=50k, which is what MAX_ADAPTIVE_NBINS additionally clamps).
    """
    x = np.asarray(x, dtype=np.float64).ravel()  # Intentional float64: bin-edge computation input; edges compared at full precision
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
    nbins = math.ceil(span / h)
    cap = min(max(2, int(math.sqrt(n) * 4)), int(max_bins))
    return int(max(1, min(nbins, cap)))


# -----------------------------------------------------------------------------
# Edge-builders. Each returns a SORTED ndarray of bin edges (inner cuts) suitable
# for ``np.searchsorted(edges, x, side='right')`` - matching the legacy contract.
# -----------------------------------------------------------------------------


def _edges_from_quantiles(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile binning helper. Returns INNER edges only (n_bins - 1 values).

    de-duplicate via ``np.unique``
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
        return np.array([], dtype=np.float64)  # Intentional float64: empty edge vector keeps the float64 edge contract
    x = np.asarray(x, dtype=np.float64).ravel()  # Intentional float64: bin-edge computation input; edges compared at full precision
    x = x[np.isfinite(x)]  # drop nan/+-inf: nanpercentile ignores nan but not inf, so an all/mostly-inf tail leaks an inf inner edge into searchsorted (phantom bin -> inflated K_x / MM bias)
    if x.size == 0:
        return np.array([], dtype=np.float64)  # Intentional float64: empty edge vector keeps the float64 edge contract
    quantiles = np.linspace(0.0, 100.0, n_bins + 1)
    full_edges = np.percentile(x, quantiles)
    full_edges = np.unique(full_edges)
    if full_edges.size <= 2:
        return np.array([], dtype=np.float64)  # Intentional float64: empty edge vector keeps the float64 edge contract
    return np.asarray(full_edges[1:-1], dtype=np.float64)  # Intentional float64: bin-edge output contract


def _edges_from_uniform(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Uniform binning helper. Returns INNER edges only (n_bins - 1 values)."""
    if n_bins < 2:
        return np.array([], dtype=np.float64)  # Intentional float64: empty edge vector keeps the float64 edge contract
    x = np.asarray(x, dtype=np.float64).ravel()  # Intentional float64: bin-edge computation input; edges compared at full precision
    x = x[np.isfinite(x)]  # nanmin/nanmax ignore nan but not inf: a single inf -> xmin/xmax +-inf -> linspace of all-NaN edges -> NaN in searchsorted = garbage codes
    if x.size == 0:
        return np.array([], dtype=np.float64)  # Intentional float64: empty edge vector keeps the float64 edge contract
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax <= xmin:
        return np.array([], dtype=np.float64)  # Intentional float64: empty edge vector keeps the float64 edge contract
    full_edges = np.linspace(xmin, xmax, n_bins + 1)
    return np.asarray(full_edges[1:-1], dtype=np.float64)  # Intentional float64: bin-edge output contract


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
    raw = round(alpha * n)
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

    Returns:
        1-D double array of the inner quantile edges; empty when fewer than two distinct bins result.
    """
    x = np.asarray(x, dtype=np.float64).ravel()  # Intentional float64: bin-edge computation input; edges compared at full precision
    x = x[np.isfinite(x)]
    n = x.size
    n_q = qs_nbins(n, alpha)
    if n_q < 2:
        return np.array([], dtype=np.float64)  # Intentional float64: empty edge vector keeps the float64 edge contract
    q_pos = np.linspace(0.0, 100.0, n_q + 1)
    full_edges = np.nanpercentile(x, q_pos)
    # De-duplicate (ties / near-constant columns collapse).
    full_edges = np.unique(full_edges)
    if full_edges.size <= 2:
        return np.array([], dtype=np.float64)  # Intentional float64: empty edge vector keeps the float64 edge contract
    return np.asarray(full_edges[1:-1], dtype=np.float64)  # Intentional float64: bin-edge output contract


def edges_quantile(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Fixed-nbins quantile edges. Legacy MRMR default behaviour."""
    return _edges_from_quantiles(x, n_bins)


def edges_uniform(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Fixed-nbins uniform-width edges."""
    return _edges_from_uniform(x, n_bins)


def edges_sturges(x: np.ndarray, base: str = "quantile") -> np.ndarray:
    """Sturges nbins + quantile/uniform edges."""
    n = np.isfinite(np.asarray(x, dtype=np.float64).ravel()).sum()  # Intentional float64: finite-count statistic
    n_bins = sturges_nbins(int(n))
    return _edges_from_quantiles(x, n_bins) if base == "quantile" else _edges_from_uniform(x, n_bins)


def edges_freedman_diaconis(x: np.ndarray, base: str = "quantile", max_bins: int = MAX_ADAPTIVE_NBINS) -> np.ndarray:
    """Freedman-Diaconis nbins + quantile/uniform edges. Default for ``auto``."""
    n_bins = freedman_diaconis_nbins(x, max_bins=max_bins)
    return _edges_from_quantiles(x, n_bins) if base == "quantile" else _edges_from_uniform(x, n_bins)


def edges_knuth(x: np.ndarray, edge_type: str = "uniform", m_max_cap: int = MAX_ADAPTIVE_NBINS) -> np.ndarray:
    """Knuth (2006) Bayesian-optimal nbins. Flags forwarded to ``_knuth_bin_edges``:

    Args:
        x: the 1-D numeric column to bin.
        edge_type: ``'uniform'`` (legacy faithful) | ``'quantile'`` (audit fix).
        m_max_cap: ``MAX_ADAPTIVE_NBINS`` (256) shared adaptive-bin-count ceiling by default | ``64``
            audit recommendation to stay in low plug-in bias regime on small val-folds | ``500``
            legacy pre-unification default.

    Returns:
        1-D double array of the inner bin edges (outer boundaries dropped); empty when no interior edge exists.
    """
    full_edges = _knuth_bin_edges(np.asarray(x), edge_type=edge_type, m_max_cap=int(m_max_cap))
    if full_edges.size <= 2:
        return np.array([], dtype=np.float64)  # Intentional float64: empty edge vector keeps the float64 edge contract
    return np.asarray(full_edges[1:-1], dtype=np.float64)  # Intentional float64: bin-edge output contract


def edges_bayesian_blocks(
    x: np.ndarray, p0: float = 0.05, edge_placement: str = "start", subsample_threshold: int = 0, m_max_cap: int = MAX_ADAPTIVE_NBINS,
) -> np.ndarray:
    """Bayesian Blocks (Scargle 2013) variable-width edges. Flags forwarded:

    Args:
        x: the 1-D numeric column to bin.
        p0: ``0.05`` legacy | ``0.10`` audit recommendation for continuous data.
        edge_placement: ``'start'`` legacy | ``'midpoint'`` Scargle/astropy convention fix.
        subsample_threshold: ``0`` disabled | ``1000`` audit fast path.
        m_max_cap: bound on returned block count, default ``MAX_ADAPTIVE_NBINS`` (256) - see
            ``_bayesian_blocks_bin_edges`` docstring (unbounded DP output on near-continuous real
            data blows up downstream pairwise-MI cost).

    Returns:
        1-D double array of the inner block edges (outer boundaries dropped); empty when no interior edge exists.
    """
    full_edges = _bayesian_blocks_bin_edges(
        np.asarray(x), p0=p0, edge_placement=edge_placement, subsample_threshold=int(subsample_threshold), m_max_cap=int(m_max_cap),
    )
    if full_edges.size <= 2:
        return np.array([], dtype=np.float64)  # Intentional float64: empty edge vector keeps the float64 edge contract
    return np.asarray(full_edges[1:-1], dtype=np.float64)  # Intentional float64: bin-edge output contract


def edges_mah(x: np.ndarray, y: np.ndarray, *, initial_k: int = 16) -> np.ndarray:
    """MAH/SCI (Marx 2021) supervised bin edges. Greedy SCI-guided merge
    starting from K equal-frequency quantile bins; the X-axis merges that
    don't reduce the joint NML code length get reverted into kept edges.

    Default initial_k=16 matches the paper's reported sweep.
    """
    from ._mah import mah_bin_edges
    inner = mah_bin_edges(np.asarray(x), np.asarray(y), initial_k=int(initial_k))
    inner = np.asarray(inner, dtype=np.float64)  # Intentional float64: inner cut-point vector; edges compared at full precision
    return inner[np.isfinite(inner)] if inner.size else inner


def edges_fayyad_irani(
    x: np.ndarray, y: np.ndarray, *, max_depth: int = 8, min_split_size: int = 5, backend: str = "njit", scaled_min_split: bool = False,
    max_y_classes: int = 64, fast_mode: bool = False, alpha: float = 0.05, n_permutations: int = 15, bonferroni: bool = False,
    validated_seed: int = 0, y_pseudo_classes: int = 16,
) -> np.ndarray:
    """Fayyad-Irani MDLP supervised edges. Flags forwarded:

    Args:
        x: the 1-D numeric column to bin.
        y: the discrete target used to supervise the split search.
        max_depth: maximum recursive split depth.
        min_split_size: minimum samples on either side of a candidate split.
        backend: ``'njit'`` (default; audit-recommended 10-30x speedup over
            the legacy pure-Python path) | ``'python'`` (legacy fallback
            kept for A/B testing). Sibling ``mdlp_bin_edges`` already
            defaults to ``'njit'``; this wrapper now matches that default
            so callers that go through the wrapper benefit too.
            c0022_9f2cf625 @500k profile: the python-backend
            path consumed 1566 s of a 1700 s suite (88 % of wall) before
            this fix because ``_mdlp_recurse`` calls ``_entropy_from_labels``
            268 345 times, each doing ``np.unique`` + ``np.sort`` on the
            label slice (4 ms per call at n=500k). The njit kernel
            ``_mdlp_recurse_njit`` maintains running per-class counts
            across the candidate-scan, so entropy is O(K_y) per candidate
            instead of O(N log N + N) per candidate. Only consulted when
            ``fast_mode=True``; the default validated-splitting path always
            uses its own njit kernel regardless of ``backend``.
        scaled_min_split: ``False`` legacy | ``True`` audit fix
            (``max(5, 0.02*N)``).
        max_y_classes: cardinality cap on ``y`` before the target is treated as too wide to bin against.
        fast_mode: ``False`` DEFAULT (2026-07-19 user decision, accuracy over speed) -
            significance-gated validated splitting; ``True`` - classic in-sample MDL
            threshold + depth cap, 20-80x cheaper. See ``mdlp_bin_edges`` docstring.
        alpha: forwarded to ``mdlp_bin_edges``'s validated-splitting significance test (ignored when ``fast_mode=True``).
        n_permutations: permutation-null draws for the validated-splitting significance test.
        bonferroni: forwarded to ``mdlp_bin_edges``'s validated-splitting path (ignored when ``fast_mode=True``).
        validated_seed: forwarded to ``mdlp_bin_edges``'s validated-splitting path (ignored when ``fast_mode=True``).
        y_pseudo_classes: pseudo-class count for a target above ``max_y_classes`` distinct values (default 16; see ``mdlp_bin_edges``).

    Returns:
        1-D double array of the finite inner cut points (outer boundaries dropped); empty when no split was accepted.
    """
    full_edges = mdlp_bin_edges(
        np.asarray(x), np.asarray(y), max_depth=max_depth, min_split_size=min_split_size, backend=backend, scaled_min_split=scaled_min_split,
        max_y_classes=max_y_classes, fast_mode=fast_mode, alpha=alpha, n_permutations=n_permutations, bonferroni=bonferroni,
        validated_seed=validated_seed, y_pseudo_classes=y_pseudo_classes,
    )
    if full_edges.size <= 2:
        return np.array([], dtype=np.float64)  # Intentional float64: empty edge vector keeps the float64 edge contract
    inner = full_edges[1:-1]
    inner = inner[np.isfinite(inner)]
    return np.asarray(inner, dtype=np.float64)  # Intentional float64: inner cut-point output contract


@njit(nogil=True, cache=True)
def _plug_in_mi_njit(x_binned: np.ndarray, y_b: np.ndarray, K_x: int, K_y: int, miller_madow: bool) -> float:
    """njit core: plug-in MI from already-integer-encoded bins, optional Miller-Madow.

    Miller-Madow (Miller 1955; Madow 1948) bias term: ``(K_x - 1) * (K_y - 1) / (2 * N)``.
    Subtracted post-accumulation, floored at 0 so honest no-signal MI is exactly 0.
    """
    n = x_binned.shape[0]
    if n == 0:
        return 0.0
    joint = np.zeros((K_x, K_y), dtype=np.float64)  # Intentional float64: joint-count accumulator for the MI statistic
    for i in range(n):
        joint[x_binned[i], y_b[i]] += 1.0
    n_f = float(n)
    Px = np.zeros(K_x, dtype=np.float64)  # Intentional float64: marginal-count accumulator for the MI statistic
    Py = np.zeros(K_y, dtype=np.float64)  # Intentional float64: marginal-count accumulator for the MI statistic
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


@njit(nogil=True, cache=True, parallel=True)
def _oj_pick_candidates(xb: np.ndarray, kx: np.ndarray, nv: np.ndarray, vy: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """Index of the candidate bin count with the highest mean fold MI, per column (``-1`` when no fold scored any), parallel over columns.

    ``xb[f, j, m, :nv[f]]`` are the binned validation values of column ``j`` at candidate ``m`` in fold ``f`` (``kx[f, j, m]`` classes, 0 when that cell does not exist), ``vy[f, :nv[f]]`` the
    binned validation target with ``ky[f]`` classes. The first candidate wins a tie, as in the per-column search."""
    n_f, p, n_m = kx.shape
    pick = np.full(p, -1, dtype=np.int64)
    for j in prange(p):
        best = -np.inf
        best_m = -1
        for m in range(n_m):
            total = 0.0
            cells = 0
            for f in range(n_f):
                k_x = kx[f, j, m]
                if k_x == 0:
                    continue
                mi = 0.0
                if ky[f] >= 1:
                    mi = _plug_in_mi_njit(xb[f, j, m, : nv[f]], vy[f, : nv[f]], k_x, ky[f], False)
                total += mi
                cells += 1
            if cells > 0 and total / cells > best:
                best = total / cells
                best_m = m
        pick[j] = best_m
    return pick


_DISTINCT_BINCOUNT_MAX_RANGE = 1 << 22  # widest integer range counted with a bincount (a 32 MB int64 count vector); wider columns are sorted


def _distinct_exceeds(y: np.ndarray, cap: int) -> bool:
    """``np.unique(y).size > cap`` for an integer / bool vector without the sort when the value range is narrow (class codes): one bincount pass instead."""
    if y.dtype.kind == "b":
        return 2 > cap
    if y.size == 0:
        return False
    lo, hi = int(y.min()), int(y.max())
    if hi - lo < _DISTINCT_BINCOUNT_MAX_RANGE:
        return int(np.count_nonzero(np.bincount((y - lo).astype(np.intp, copy=False)))) > cap
    return int(np.unique(y).size) > cap


def _bin_y_for_mi(y: np.ndarray, max_y_classes: int = 64) -> "tuple[np.ndarray, int]":
    """Quantize ``y`` to int class codes the way :func:`_plug_in_mi` does internally (10-quantile bins
    for a non-integer/bool dtype, pass-through int codes otherwise). Factored out so a caller looping
    the SAME ``y`` over multiple candidate x-binnings (:func:`edges_optimal_joint`) can quantize once
    and reuse, instead of paying the ``np.quantile`` re-sort on every candidate.

    Args:
        y: the target to quantize into MI-scoring class codes.
        max_y_classes: Cardinality guard for int/bool-dtype ``y``. A continuous regression target that
            happens to be int-typed (timestamps, sensor counters, a float column upstream-cast to int)
            was previously treated as a discrete class label with NO cardinality check: ``K_y =
            y.max()+1`` could reach millions, and ``_plug_in_mi_njit`` allocates a dense ``(K_x, K_y)``
            joint-count matrix - confirmed to SEGFAULT the process on an int64 target with ~50k unique
            values (n=50000, K_y ~ 3.2e9 range) via an oversized ``np.zeros`` allocation. This mirrors the
            exact bug class found in MDLP (``mdlp_bin_edges``'s ``max_y_classes``): a high-cardinality
            target silently blows up an internal per-class computation. Fix: int/bool ``y`` with more
            unique values than this cap is treated as continuous and routed through the same 10-quantile
            regression-binning as float ``y``, instead of one class per distinct integer.

    Returns:
        Tuple ``(y_codes, K_y)`` of int64 class codes and the number of classes; an empty array and 0 when the quantile
        binning of a continuous ``y`` degenerates to a single value.
    """
    if y.dtype.kind not in "iub" or _distinct_exceeds(y, max_y_classes):
        q = np.quantile(y.astype(np.float64), np.linspace(0, 1, 11))  # Intentional float64: quantile edges of the target for the statistic
        q = np.unique(q)
        if q.size < 2:
            return np.zeros(0, dtype=np.int64), 0
        y_b = np.searchsorted(q[1:-1], y.astype(np.float64), side="right").astype(np.int64)  # Intentional float64: target bin search for the statistic
    else:
        # ascontiguousarray (not astype) so an already-contiguous int64 code array is NOT re-copied every call.
        y_b = np.ascontiguousarray(y, np.int64)
    K_y = int(y_b.max()) + 1 if y_b.size else 1
    return y_b, K_y


def _plug_in_mi(x_binned: np.ndarray, y: np.ndarray, miller_madow: bool = False, max_y_classes: int = 64) -> float:
    """Plug-in MI estimator: I(X; Y) = H(X) + H(Y) - H(X, Y) on counts.

    Args:
        x_binned: 1-D ``int`` array of bin indices for X.
        y: 1-D array of target values (int classes or float -> regression-binned to 10 quantiles).
        miller_madow: When ``True``, subtract ``(K_x - 1) * (K_y - 1) / (2 * N)``
            bias-correction term (Miller 1955; Madow 1948). Restores honest noise
            floor at higher M values (FD's no-signal inflation collapses from
            0.077 -> ~0.02). Floor at zero. Default ``False`` preserves the
            pre-2026-05-29 leaderboard baseline; opt-in via flag.
        max_y_classes: Forwarded to :func:`_bin_y_for_mi` - caps int/bool ``y`` cardinality before it
            is treated as class labels (see that function's docstring for the segfault this guards).

    Returns:
        Plug-in mutual information in nats (floored at zero); 0.0 for empty or degenerate input.
    """
    if x_binned.size == 0:
        return 0.0
    y_b, K_y = _bin_y_for_mi(y, max_y_classes=max_y_classes)
    x_b = np.ascontiguousarray(x_binned, np.int64)
    if x_b.size == 0 or y_b.size == 0:
        return 0.0
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
    "fayyad_irani_validated": "fayyad_irani_validated",
    "fayyad-irani-validated": "fayyad_irani_validated",
    "mdlp_validated": "fayyad_irani_validated",
    "optimal_joint": "optimal_joint",
    "cv": "optimal_joint",
    "qs": "qs",
    "quantile": "qs",
    "quantile_spacing": "qs",
    "gupta": "qs",
    "uniform": "uniform",
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
    cache_dir: Optional[str] = None,
    n_jobs: int = -1,
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
        cache_dir: Optional directory for a per-column content-addressable disk cache. ``None``
            (default) disables. Each column's edge array is keyed by ``(col-summary-hash,
            method, base, kwargs, y-summary-if-supervised)``; cross-call hits skip the per-column
            edge-builder. For supervised methods (MDLP / mah / optimal_joint), keys include the
            y-summary so refits with the same X but different labels do not collide.
        n_jobs: Worker count for the per-column edge loop. The columns are independent and the
            heavy supervised kernels (MDLP / mah njit nogil) release the GIL, so a THREAD pool
            gives real wall-time parallelism on wide frames. ``-1`` (default) = physical CPU
            count; ``1`` = exact serial path; values are clamped to ``[1, n_features]``. Only
            engaged above ``_PARALLEL_EDGES_MIN_COLS`` columns (thread-pool overhead does not pay
            on narrow frames). Edges are BIT-IDENTICAL to the serial path regardless of n_jobs
            and thread scheduling - column order is preserved in the output list.
        **kwargs: Forwarded to the underlying edge-builder
            (e.g. ``max_depth`` for fayyad_irani, ``candidates`` for optimal_joint,
            ``p0`` for bayesian_blocks).

    Returns:
        list of length n_features; each entry is a 1-D ndarray of INNER bin edges
        (i.e. ``n_bins - 1`` values, suitable for ``np.searchsorted(edges, x, side='right')``).
    """
    _kw_key: Any = None
    X, method_resolved, n_features, needs_y, y = _per_feature_edges_np_asarray(X, method, y)

    # Per-column content-addressable disk cache. Each column's edge computation is independent;
    # caching keys by (column-summary, method, base, kwargs, y-summary-when-supervised) lets a
    # second call on the same X+y skip the per-column edge-builder. Cache failures are non-fatal:
    # we fall back to the un-cached path on any exception so a broken cache cannot break a fit.
    _cache = None
    _y_key: str | None = None
    if cache_dir is not None:
        try:
            from mlframe.utils.disk_cache import DiskCache, hash_array_content, hash_object

            _cache = DiskCache(cache_dir)
            _y_key = hash_array_content(y) if (needs_y and y is not None) else "no_y"
            _kw_key = hash_object({
                "method": method_resolved,
                "base": str(base),
                "kwargs": kwargs,
            })
        except Exception as exc:
            logger.warning("per_feature_edges: cache disabled, computing without it (%s)", exc)
            _cache = None

    # If a column has few unique finite values (e.g. binary target, small
    # categorical, ordinal already pre-encoded), quantile-based
    # binning collapses to 1-bin because ``_edges_from_quantiles``
    # returns empty edges after ``np.unique`` dedup. Detect these
    # columns up front and produce midpoint-edges between consecutive
    # unique values - this preserves the column's natural cardinality
    # without going through quantile / supervised logic that doesn't
    # apply.
    _low_card_cap = int(kwargs.get("low_card_cap", 32))

    from ._adaptive_nbins_columns import compute_col_edges

    def _compute_col_edges(col: np.ndarray):
        """Per-column edge computation for the cache loop below; touches no shared state, so it is safe under the thread pool."""
        return compute_col_edges(col, method_resolved, base, kwargs, y, _low_card_cap)

    # ---- Phase 1 (serial): per-column cache GET. Records misses to compute. ----
    # Cache lookup happens BEFORE the low-card branch so even cheap midpoint edges
    # hit the cache on repeat fits; the code path stays uniform. Doing all GETs (and
    # later all PUTs) serially on the main thread keeps the DiskCache single-threaded
    # - no lock needed, hit/miss behavior bit-identical to the historical loop.
    edges_list: list[Any] = _per_feature_edges_no_lock_needed_hit(n_features, X, _cache, _y_key, _kw_key, n_jobs, _compute_col_edges)

    return edges_list


def _per_feature_edges_np_asarray(X, method, y):
    """Block of per_feature_edges starting at ``X = np.asarray(X)``."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"per_feature_edges: X must be 2-D; got shape {X.shape}")
    n_features = X.shape[1]
    method_resolved = _METHOD_ALIASES.get(method.lower() if isinstance(method, str) else method)
    if method_resolved is None:
        raise ValueError(f"per_feature_edges: unknown method={method!r}. Expected one of " f"{sorted(set(_METHOD_ALIASES.values()))}.")
    needs_y = method_resolved in ("fayyad_irani", "fayyad_irani_validated", "optimal_joint", "mah")
    if needs_y and y is None:
        raise ValueError(f"per_feature_edges: method={method_resolved!r} is supervised and requires y.")
    if y is not None:
        y = np.asarray(y).ravel()
    return X, method_resolved, n_features, needs_y, y


_EDGES_WARMED = False  # set once a threaded call has finished: the njit kernels of the edge builders are compiled in this process
_EDGE_POOL: Any = None
_EDGE_POOL_WORKERS = 0
_EDGE_POOL_LOCK = threading.Lock()


def _edge_pool(workers: int) -> Any:
    """The process-wide thread pool of the per-column edge computation, grown (never shrunk) to ``workers`` threads. Its tasks are leaf computations that never wait on the pool, so callers
    sharing it cannot deadlock."""
    global _EDGE_POOL, _EDGE_POOL_WORKERS
    with _EDGE_POOL_LOCK:
        if _EDGE_POOL is None or _EDGE_POOL_WORKERS < workers:
            from concurrent.futures import ThreadPoolExecutor

            if _EDGE_POOL is not None:
                _EDGE_POOL.shutdown(wait=False)
            _EDGE_POOL = ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="edges")
            _EDGE_POOL_WORKERS = int(workers)
        return _EDGE_POOL


def _resolve_edge_jobs(n_jobs: Any, n_miss: int) -> int:
    """Worker-thread count for the edge computation: the physical core count when ``n_jobs`` is unset, capped at the number of columns that need computing."""
    if n_jobs is None or int(n_jobs) <= 0:
        try:
            import psutil

            _resolved_jobs = psutil.cpu_count(logical=False) or 1
        except Exception as e:
            logger.warning("psutil.cpu_count(logical=False) failed, falling back to os.cpu_count(): %s", e)
            import os

            _resolved_jobs = os.cpu_count() or 1
    else:
        _resolved_jobs = int(n_jobs)
    _resolved_jobs = max(1, min(_resolved_jobs, max(1, n_miss)))
    return _resolved_jobs


def _cache_put_misses(_cache: Any, _miss_cols: list, _miss_keys: list, edges_list: list) -> None:
    """Store the freshly computed edges of the missed columns; a failing put is logged at debug level and skipped."""
    for j in _miss_cols:
        _key = _miss_keys[j]
        edges = edges_list[j]
        if _key is not None and edges is not None:
            try:
                _cache.put(_key, edges)
            except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed: %s", e)
                pass


def _per_feature_edges_no_lock_needed_hit(n_features, X, _cache, _y_key, _kw_key, n_jobs, _compute_col_edges):
    """Block of per_feature_edges starting at ``edges_list: list = [None] * n_features``."""
    edges_list: list = [None] * n_features
    _miss_keys: list = [None] * n_features  # cache key per missed col (None => don't cache)
    _miss_cols: list = []  # indices needing compute, in ascending order
    _cols: list = [None] * n_features  # double view per column (reused by phase 2)
    for j in range(n_features):
        col = X[:, j].astype(np.float64, copy=False)  # Intentional float64: per-column float64 view consumed by the statistic loop
        _cols[j] = col
        _col_cache_key = None
        if _cache is not None:
            try:
                from mlframe.utils.disk_cache import hash_array_content, compose_key

                _col_summary = hash_array_content(col)
                assert _y_key is not None  # set together with _cache in the same cache_dir-guarded try block above
                _col_cache_key = "nbin_" + compose_key(_col_summary, _y_key, _kw_key)
                _hit = _cache.get(_col_cache_key)
                if _hit is not None:
                    edges_list[j] = _hit
                    continue
            except Exception as exc:
                log_throttle(logger, "adaptive_nbins.cache_get_failed", logging.WARNING, "per_feature_edges: cache get failed col=%d, computing this column without it (%s)", j, exc)
                _col_cache_key = None
        _miss_keys[j] = _col_cache_key
        _miss_cols.append(j)

    # ---- Phase 2: compute edges for the misses (serial or threaded). ----
    # Parallelism via a THREAD pool: the heavy supervised kernels (MDLP / mah) are
    # njit(nogil=True) so they release the GIL and threads get real cores. Output
    # order is preserved (results written by column index), so edges are deterministic
    # and bit-identical regardless of n_jobs / thread scheduling.
    #
    # Bench (MDLP, n=20000, default njit backend, this machine):
    #   p=500 : serial 2.45s -> parallel 0.78s  (3.14x)
    #   p=2000: serial 10.10s -> parallel 3.20s (3.16x)
    #   p=50  : serial 0.28s -> parallel 0.22s  (gated to serial, no regression)
    # (see test_per_feature_edges_parallel.py; numbers refreshed if hardware changes.)
    _resolved_jobs = _resolve_edge_jobs(n_jobs, len(_miss_cols))

    # Bit-identity above assumes njit(nogil=True) kernels: numba's nopython-mode functions get their
    # own thread-isolated internal RNG state, so concurrent threads never see each other's draws even
    # though they all call np.random.seed()/np.random.randint() with the same-looking call pattern.
    # Under NUMBA_DISABLE_JIT=1 those calls run as plain Python and mutate numpy's single PROCESS-WIDE
    # global RNG (numpy.random's legacy module-level state) instead -- concurrent GIL-interleaved
    # threads race on that one shared generator, so the MDLP significance test's seeded permutation
    # draws can interleave across threads and silently produce DIFFERENT (still plausible-looking, not
    # obviously wrong) edges than the serial path -- caught live via numba-coverage-nightly
    # (test_cache_thread_safety_and_hit_behavior: same data/seed, serial vs 4-thread edges diverged
    # structurally on one column) and is also why a permutation-heavy MRMR gate test crashed a worker
    # outright in the same run (a genuine native race on numpy's C-level RandomState buffer, not just a
    # wrong-value race). This is a latent correctness bug independent of this test suite: ANY caller
    # running without numba installed (the plain-Python njit no-op fallback several modules define) and
    # requesting n_jobs>1 would hit the exact same silent divergence in production. Falls back to serial
    # whenever JIT is not actually compiling, regardless of n_jobs, since the threaded path buys zero
    # real parallelism over GIL-bound pure-Python code anyway (test_speedup_mdlp's already-documented
    # rationale) -- there is no performance cost to this safety gate, only a latent bug it closes.
    _jit_active = not numba.config.DISABLE_JIT
    if _jit_active and _resolved_jobs > 1 and len(_miss_cols) >= _PARALLEL_EDGES_MIN_COLS:
        # The first column is computed on THIS thread, before any worker runs. The njit kernels reached
        # from ``_compute_col_edges`` (``_mdlp_recurse_validated_bfs`` and friends, all ``cache=True``)
        # are compiled on first call, and several threads entering numba's compiler at once aborts the
        # process on macOS: mlframe's first macOS CI run crashed 150 xdist workers across 10 shards with
        # ``Fatal Python error: Aborted``, and every faulthandler dump showed two or more threads inside
        # ``_mdlp_recurse_validated_bfs`` -> ``dispatcher.compile``. Warming serially costs one column's
        # worth of parallelism on a cold cache and nothing at all on a warm one, and it leaves the pool
        # executing already-compiled code, which is what the measured 1.2x-7.2x speedup was ever about.
        # Once a call has completed in this process the kernels are compiled (and the pool exists), so later calls skip the serial column.
        global _EDGES_WARMED
        if _EDGES_WARMED:
            _warm, _rest = None, list(_miss_cols)
        else:
            _warm, _rest = _miss_cols[0], _miss_cols[1:]
            _warm_edges, _ = _compute_col_edges(_cols[_warm])
            edges_list[_warm] = _warm_edges
        if _rest:
            # One task per worker (strided so that neighbouring, similarly expensive columns spread out) on a pool that lives as long as the process: a call that bins a dozen
            # cheap columns used to spend most of its time creating threads and waiting for a future per column.
            _workers = min(_resolved_jobs, len(_rest))
            _chunks = [_rest[i::_workers] for i in range(_workers)]

            def _run_chunk(chunk):
                """Edges of the columns in ``chunk``, as ``(column index, edges)`` pairs."""
                return [(j, _compute_col_edges(_cols[j])[0]) for j in chunk]

            for _pairs in _edge_pool(_resolved_jobs).map(_run_chunk, _chunks):
                for j, edges in _pairs:
                    edges_list[j] = edges
        _EDGES_WARMED = True
    else:
        for j in _miss_cols:
            edges, _was_lc = _compute_col_edges(_cols[j])
            edges_list[j] = edges

    # ---- Phase 3 (serial): cache PUT for the misses. ----
    if _cache is not None:
        _cache_put_misses(_cache, _miss_cols, _miss_keys, edges_list)
    return edges_list


from ._optimal_joint_edges import (  # noqa: F401  -- the optimal-joint search lives in its own module; re-exported so importers keep one path
    OptimalJointPlanCache,
    _optimal_joint_best_m,
    _optimal_joint_final_edges,
    _optimal_joint_plan,
    edges_optimal_joint,
)
