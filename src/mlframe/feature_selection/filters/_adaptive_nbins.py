"""Per-feature adaptive bin-count selection for MRMR (2026-05-28).

Pre-fix: ``MRMR(quantization_method='quantile', quantization_nbins=10)`` binned
EVERY column to the same fixed n_bins=10 regardless of distribution shape,
sample size, or signal strength. That's a one-size-fits-all compromise that:

* under-resolves a binary signal (only 2 bins needed, gets 10)
* under-resolves a continuous fine-structure signal (10 bins lose the cuts)
* equalises post-binning cardinality, washing out the MI cardinality bias
  (so SU normalisation has no effect -- see test_biz_val_mrmr_symmetric_uncertainty)

This module ships per-feature adaptive bin selection via six methods:

* **'sturges'**  -- `ceil(1 + log2(n))`. Simplest formula; assumes Gaussian-ish.
* **'freedman_diaconis'** (= `'fd'`, DEFAULT for auto) -- `ceil((max-min) / (2*IQR/n^(1/3)))`.
  Robust to outliers/skew; the recommended default for MI estimation
  on natural data per Freedman & Diaconis 1981.
* **'knuth'**  -- Bayesian-posterior optimum over M in [1, sqrt(N)*4].
  Native impl in ``discretization.py``. Returns 1 bin for featureless uniform.
* **'blocks'**  -- Bayesian Blocks (Scargle 2013). Variable-width edges, no n_bins.
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
from typing import Optional

import numpy as np
from numba import njit

from .discretization import _knuth_bin_edges, _bayesian_blocks_bin_edges
from .supervised_binning import mdlp_bin_edges
from ._mdlp_validated_split import edges_fayyad_irani_validated

# Shared per-column bin-count ceiling for every adaptive strategy whose own formula has no natural
# upper bound (knuth, bayesian_blocks). MDLP already lands here implicitly via max_depth=8 -> up to
# 2**8=256 leaves (supervised_binning.py); freedman_diaconis's own sqrt(N)*4 cap is additionally
# clamped to this same ceiling below. Kept as one named constant so every strategy answers "how many
# bins can a single column produce" the same way -- divergent per-method caps (found live: knuth
# defaulted to 500, bayesian_blocks had NO cap at all) let a single real-data column reach thousands
# of bins, blowing the downstream joint-cardinality (nbins_a * nbins_b) past both the CUDA
# shared-memory budget and the row-chunked global-memory fallback's launch-count budget, forcing a
# multi-thousand-second CPU njit fallback per column pair (found live, 50k-row wellbore GT sweep).
MAX_ADAPTIVE_NBINS = 256

logger = logging.getLogger(__name__)

# Minimum number of cache-MISS columns before per_feature_edges engages the thread
# pool. Below this, thread-pool spawn + dispatch overhead outweighs the per-column
# compute (verified on p=50: parallel ties serial). Wide frames (p>=128) are where
# the GIL-releasing njit MDLP kernels yield the ~3x wall-time win.
_PARALLEL_EDGES_MIN_COLS = 128


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
    large real datasets (sqrt(N)*4 alone is unbounded as N grows -- e.g. ~895
    bins at N=50k, which is what MAX_ADAPTIVE_NBINS additionally clamps).
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
    nbins = math.ceil(span / h)
    cap = min(max(2, int(math.sqrt(n) * 4)), int(max_bins))
    return int(max(1, min(nbins, cap)))


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
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]  # drop nan/+-inf: nanpercentile ignores nan but not inf, so an all/mostly-inf tail leaks an inf inner edge into searchsorted (phantom bin -> inflated K_x / MM bias)
    if x.size == 0:
        return np.array([], dtype=np.float64)
    quantiles = np.linspace(0.0, 100.0, n_bins + 1)
    full_edges = np.percentile(x, quantiles)
    full_edges = np.unique(full_edges)
    if full_edges.size <= 2:
        return np.array([], dtype=np.float64)
    return np.asarray(full_edges[1:-1], dtype=np.float64)


def _edges_from_uniform(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Uniform binning helper. Returns INNER edges only (n_bins - 1 values)."""
    if n_bins < 2:
        return np.array([], dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]  # nanmin/nanmax ignore nan but not inf: a single inf -> xmin/xmax +-inf -> linspace of all-NaN edges -> NaN in searchsorted = garbage codes
    if x.size == 0:
        return np.array([], dtype=np.float64)
    xmin, xmax = float(np.min(x)), float(np.max(x))
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


def edges_freedman_diaconis(x: np.ndarray, base: str = "quantile", max_bins: int = MAX_ADAPTIVE_NBINS) -> np.ndarray:
    """Freedman-Diaconis nbins + quantile/uniform edges. Default for ``auto``."""
    n_bins = freedman_diaconis_nbins(x, max_bins=max_bins)
    return _edges_from_quantiles(x, n_bins) if base == "quantile" else _edges_from_uniform(x, n_bins)


def edges_knuth(x: np.ndarray, edge_type: str = "uniform", m_max_cap: int = MAX_ADAPTIVE_NBINS) -> np.ndarray:
    """Knuth (2006) Bayesian-optimal nbins. Flags forwarded to ``_knuth_bin_edges``:

    Args:
        edge_type: ``'uniform'`` (legacy faithful) | ``'quantile'`` (audit fix).
        m_max_cap: ``MAX_ADAPTIVE_NBINS`` (256) shared adaptive-bin-count ceiling by default | ``64``
            audit recommendation to stay in low plug-in bias regime on small val-folds | ``500``
            legacy pre-unification default.
    """
    full_edges = _knuth_bin_edges(np.asarray(x), edge_type=edge_type, m_max_cap=int(m_max_cap))
    if full_edges.size <= 2:
        return np.array([], dtype=np.float64)
    return np.asarray(full_edges[1:-1], dtype=np.float64)


def edges_bayesian_blocks(
    x: np.ndarray, p0: float = 0.05, edge_placement: str = "start", subsample_threshold: int = 0, m_max_cap: int = MAX_ADAPTIVE_NBINS,
) -> np.ndarray:
    """Bayesian Blocks (Scargle 2013) variable-width edges. Flags forwarded:

    Args:
        p0: ``0.05`` legacy | ``0.10`` audit recommendation for continuous data.
        edge_placement: ``'start'`` legacy | ``'midpoint'`` Scargle/astropy convention fix.
        subsample_threshold: ``0`` disabled | ``1000`` audit fast path.
        m_max_cap: bound on returned block count, default ``MAX_ADAPTIVE_NBINS`` (256) -- see
            ``_bayesian_blocks_bin_edges`` docstring (unbounded DP output on near-continuous real
            data blows up downstream pairwise-MI cost).
    """
    full_edges = _bayesian_blocks_bin_edges(
        np.asarray(x), p0=p0, edge_placement=edge_placement, subsample_threshold=int(subsample_threshold), m_max_cap=int(m_max_cap),
    )
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


def edges_fayyad_irani(
    x: np.ndarray, y: np.ndarray, *, max_depth: int = 8, min_split_size: int = 5, backend: str = "njit", scaled_min_split: bool = False,
    max_y_classes: int = 64, fast_mode: bool = False, alpha: float = 0.05, n_permutations: int = 30, bonferroni: bool = False,
    validated_seed: int = 0,
) -> np.ndarray:
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
            instead of O(N log N + N) per candidate. Only consulted when
            ``fast_mode=True``; the default validated-splitting path always
            uses its own njit kernel regardless of ``backend``.
        scaled_min_split: ``False`` legacy | ``True`` audit fix
            (``max(5, 0.02*N)``).
        fast_mode: ``False`` DEFAULT (2026-07-19 user decision, accuracy over speed) --
            significance-gated validated splitting; ``True`` -- classic in-sample MDL
            threshold + depth cap, 20-80x cheaper. See ``mdlp_bin_edges`` docstring.
        alpha, n_permutations, bonferroni, validated_seed: forwarded to
            ``mdlp_bin_edges``'s validated-splitting path (ignored when ``fast_mode=True``).
    """
    full_edges = mdlp_bin_edges(
        np.asarray(x), np.asarray(y), max_depth=max_depth, min_split_size=min_split_size, backend=backend, scaled_min_split=scaled_min_split,
        max_y_classes=max_y_classes, fast_mode=fast_mode, alpha=alpha, n_permutations=n_permutations, bonferroni=bonferroni,
        validated_seed=validated_seed,
    )
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
    max_y_classes: int = 64,
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

    Args:
        max_y_classes: Cardinality cap forwarded to :func:`_bin_y_for_mi` for the internal
            fold-scoring MI. Without this cap, an int/bool-dtype ``y`` with high cardinality (a
            continuous target mistakenly int-typed -- e.g. a timestamp or counter column) was
            treated as one discrete class per distinct value: confirmed to SEGFAULT the process
            (oversized ``(K_x, K_y)`` dense joint-count allocation in ``_plug_in_mi_njit``) at
            n=50000 with ~50k unique int64 values. Same bug class as MDLP's ``max_y_classes``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y).ravel()
    mask = np.isfinite(x)
    # DISCRETIZATION-6 fix (mrmr_audit_2026-07-22): the mask used to check only x's finiteness -- a
    # NaN-y row with finite x survived and could propagate NaN into that fold's quantile edges via
    # _bin_y_for_mi's np.quantile call. Fold y's finiteness into the mask too (only meaningful when y is
    # float-dtype; an int/bool y is never NaN-capable).
    if y.dtype.kind == "f":
        mask &= np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    if n < n_splits * 4:
        # Fall back to FD on small samples.
        return edges_freedman_diaconis(x, base=base)
    rng = np.random.default_rng(random_state)
    fold_idx = rng.permutation(n) % n_splits
    # LOOP ORDER (outer k, inner M): the pre-fix ``for M: for k:`` nesting recomputed the SAME
    # train/val fold slices and re-quantized the SAME val_y (via _plug_in_mi's np.quantile call)
    # once per (M, k) pair, even though neither depends on M. Swapping to outer-k makes the fold
    # slice and val_y quantization happen ONCE per k and get reused across every candidate M, with
    # the exact same per-(M, k) skip conditions and running per-M fold-score aggregation as before
    # (only the ORDER the (M, k) cells are visited in changes, not which cells are skipped/scored).
    fold_scores_by_M: "dict[int, list]" = {M: [] for M in candidates if M >= 2}
    for k in range(n_splits):
        train_mask = fold_idx != k
        val_mask = ~train_mask
        if val_mask.sum() < 4:
            continue  # val-fold-too-small check does not depend on M -> applies to every M for this k
        train_x = x[train_mask]
        val_x, val_y = x[val_mask], y[val_mask]
        n_train = int(train_mask.sum())
        # val_y quantization (_plug_in_mi's np.quantile-based 10-bin regression target) depends only
        # on val_y, not on M -> computed once per k and reused for every candidate M below.
        val_y_b, K_y = _bin_y_for_mi(val_y, max_y_classes=max_y_classes)
        for M in candidates:
            if M < 2:
                continue
            if n_train < M:
                continue
            edges = _edges_from_quantiles(train_x, M) if base == "quantile" else _edges_from_uniform(train_x, M)
            if edges.size == 0:
                continue
            binned_val_x = np.searchsorted(edges, val_x, side="right")
            x_b = np.ascontiguousarray(binned_val_x.astype(np.int64), dtype=np.int64)
            if x_b.size == 0 or val_y_b.size == 0:
                mi = 0.0
            else:
                K_x = int(x_b.max()) + 1 if x_b.size else 1
                mi = 0.0 if (K_x < 1 or K_y < 1) else float(_plug_in_mi_njit(x_b, val_y_b, K_x, K_y, False))
            fold_scores_by_M[M].append(mi)
    best_score = -np.inf
    best_M = None
    for M in candidates:
        if M < 2:
            continue
        fold_scores = fold_scores_by_M.get(M, [])
        if fold_scores:
            mean_mi = float(np.mean(fold_scores))
            if mean_mi > best_score:
                best_score = mean_mi
                best_M = M
    if best_M is None:
        # No candidate M was ever scored by ANY fold -- e.g. every candidate in ``candidates`` exceeds
        # every fold's train size, or every fold/candidate combination produced degenerate (empty) train
        # edges. The pre-fix code silently fell back to ``candidates[0]`` here UNVALIDATED by the CV
        # search -- when ``candidates[0] < 2`` (a caller passing e.g. ``candidates=(1, 1000)`` to probe a
        # single-bin option) this returned EMPTY edges with no signal that the CV search never ran at
        # all, exactly the MDLP silent-empty-output bug class. Fall back to the same Freedman-Diaconis
        # path used for the too-small-n guard above -- at least one principled, unconditional binning
        # instead of an untested/possibly-invalid M.
        return edges_freedman_diaconis(x, base=base)
    # Return edges built on full data at the winning M.
    return _edges_from_quantiles(x, best_M) if base == "quantile" else _edges_from_uniform(x, best_M)


@njit(nogil=True, cache=True)
def _plug_in_mi_njit(x_binned: np.ndarray, y_b: np.ndarray, K_x: int, K_y: int, miller_madow: bool) -> float:
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


def _bin_y_for_mi(y: np.ndarray, max_y_classes: int = 64) -> "tuple[np.ndarray, int]":
    """Quantize ``y`` to int class codes the way :func:`_plug_in_mi` does internally (10-quantile bins
    for a non-integer/bool dtype, pass-through int codes otherwise). Factored out so a caller looping
    the SAME ``y`` over multiple candidate x-binnings (:func:`edges_optimal_joint`) can quantize once
    and reuse, instead of paying the ``np.quantile`` re-sort on every candidate.

    Args:
        max_y_classes: Cardinality guard for int/bool-dtype ``y``. A continuous regression target that
            happens to be int-typed (timestamps, sensor counters, a float column upstream-cast to int)
            was previously treated as a discrete class label with NO cardinality check: ``K_y =
            y.max()+1`` could reach millions, and ``_plug_in_mi_njit`` allocates a dense ``(K_x, K_y)``
            joint-count matrix -- confirmed to SEGFAULT the process on an int64 target with ~50k unique
            values (n=50000, K_y ~ 3.2e9 range) via an oversized ``np.zeros`` allocation. This mirrors the
            exact bug class found in MDLP (``mdlp_bin_edges``'s ``max_y_classes``): a high-cardinality
            target silently blows up an internal per-class computation. Fix: int/bool ``y`` with more
            unique values than this cap is treated as continuous and routed through the same 10-quantile
            regression-binning as float ``y``, instead of one class per distinct integer.
    """
    if y.dtype.kind not in "iub" or np.unique(y).size > max_y_classes:
        q = np.quantile(y.astype(np.float64), np.linspace(0, 1, 11))
        q = np.unique(q)
        if q.size < 2:
            return np.zeros(0, dtype=np.int64), 0
        y_b = np.searchsorted(q[1:-1], y.astype(np.float64), side="right").astype(np.int64)
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
        max_y_classes: Forwarded to :func:`_bin_y_for_mi` -- caps int/bool ``y`` cardinality before it
            is treated as class labels (see that function's docstring for the segfault this guards).
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
            and thread scheduling -- column order is preserved in the output list.
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
        raise ValueError(f"per_feature_edges: unknown method={method!r}. Expected one of " f"{sorted(set(_METHOD_ALIASES.values()))}.")
    needs_y = method_resolved in ("fayyad_irani", "fayyad_irani_validated", "optimal_joint", "mah")
    if needs_y and y is None:
        raise ValueError(f"per_feature_edges: method={method_resolved!r} is supervised and requires y.")
    if y is not None:
        y = np.asarray(y).ravel()

    # Per-column content-addressable disk cache. Each column's edge computation is independent;
    # caching keys by (column-summary, method, base, kwargs, y-summary-when-supervised) lets a
    # second call on the same X+y skip the per-column edge-builder. Cache failures are non-fatal:
    # we fall back to the un-cached path on any exception so a broken cache cannot break a fit.
    _cache = None
    _y_key: str | None = None
    if cache_dir is not None:
        try:
            from mlframe.utils.disk_cache import DiskCache, compose_key, hash_array_summary, hash_object

            _cache = DiskCache(cache_dir)
            _y_key = hash_array_summary(y) if (needs_y and y is not None) else "no_y"
            _kw_key = hash_object({
                "method": method_resolved,
                "base": str(base),
                "kwargs": kwargs,
            })
        except Exception as exc:
            logger.debug("per_feature_edges: cache disabled (%s)", exc)
            _cache = None

    # 2026-05-30 Wave 9.1 fix (synergy-detection regression): if a
    # column has few unique finite values (e.g. binary target, small
    # categorical, ordinal already pre-encoded), quantile-based
    # binning collapses to 1-bin because ``_edges_from_quantiles``
    # returns empty edges after ``np.unique`` dedup. Detect these
    # columns up front and produce midpoint-edges between consecutive
    # unique values - this preserves the column's natural cardinality
    # without going through quantile / supervised logic that doesn't
    # apply.
    _low_card_cap = int(kwargs.get("low_card_cap", 32))

    def _compute_col_edges(col: np.ndarray):
        """Pure per-column edge computation (no cache I/O). Returns (edges, was_low_card).

        Identical math to the historical serial loop body; factored out so the heavy
        path can run under a thread pool. Touches no shared state -> thread-safe.
        """
        _finite = col[np.isfinite(col)]
        # return_counts=True up front so the rare sparse-dominance fallback below (which needs the
        # per-value counts) reuses THIS array instead of re-running np.unique a second time.
        if _finite.size > 0:
            _uniq, _uniq_counts = np.unique(_finite, return_counts=True)
        else:
            _uniq = np.empty(0, dtype=np.float64)
            _uniq_counts = np.empty(0, dtype=np.int64)
        if 1 < _uniq.size <= _low_card_cap:
            # Use midpoints between consecutive uniques as edges so
            # each unique value lands in its own bin.
            return 0.5 * (_uniq[:-1] + _uniq[1:]), True
        if method_resolved == "sturges":
            edges = edges_sturges(col, base=base)
        elif method_resolved == "freedman_diaconis":
            edges = edges_freedman_diaconis(col, base=base, max_bins=kwargs.get("max_adaptive_nbins", MAX_ADAPTIVE_NBINS))
        elif method_resolved == "qs":
            edges = edges_qs(col, alpha=kwargs.get("qs_alpha", 0.30))
        elif method_resolved == "knuth":
            edges = edges_knuth(
                col,
                edge_type=kwargs.get("knuth_edge_type", "uniform"),
                m_max_cap=kwargs.get("knuth_m_max_cap", kwargs.get("max_adaptive_nbins", MAX_ADAPTIVE_NBINS)),
            )
        elif method_resolved == "bayesian_blocks":
            edges = edges_bayesian_blocks(
                col,
                p0=kwargs.get("p0", 0.05),
                edge_placement=kwargs.get("bb_edge_placement", "start"),
                subsample_threshold=kwargs.get("bb_subsample_threshold", 0),
                m_max_cap=kwargs.get("bb_m_max_cap", kwargs.get("max_adaptive_nbins", MAX_ADAPTIVE_NBINS)),
            )
        elif method_resolved == "fayyad_irani":
            assert y is not None  # needs_y guard above raises for this method when y is None
            edges = edges_fayyad_irani(
                col, y,
                # MDLP's own leaf count is bounded by 2**max_depth -- deriving the default from the
                # same max_adaptive_nbins ceiling used by knuth/bayesian_blocks/freedman_diaconis
                # (instead of a hardcoded 8) keeps "how many bins can one column produce" answered
                # the same way across every strategy. An explicit max_depth still wins.
                max_depth=kwargs.get("max_depth", max(1, int(math.log2(kwargs.get("max_adaptive_nbins", MAX_ADAPTIVE_NBINS))))),
                min_split_size=kwargs.get("min_split_size", 5),
                # Match edges_fayyad_irani / mdlp_bin_edges default ('njit').
                # The legacy 'python' default here re-introduced the
                # 1566 s / 1700 s @500 k regression that the iter570 fix
                # otherwise resolved -- this kwargs.get path is the
                # production caller from categorize_dataset, so the
                # default flip is the actual gating change.
                backend=kwargs.get("mdlp_backend", "njit"),
                scaled_min_split=kwargs.get("mdlp_scaled_min_split", False),
                max_y_classes=kwargs.get("mdlp_max_y_classes", 64),
                # 2026-07-19: validated (significance-gated) splitting is now the DEFAULT
                # (accuracy over speed per project convention -- see supervised_binning.py's
                # mdlp_bin_edges docstring for the full A/B). Pass mdlp_fast_mode=True (e.g.
                # MRMR(nbins_strategy_kwargs={"mdlp_fast_mode": True})) to opt back into the
                # cheap depth-capped classic path for a specific run.
                fast_mode=kwargs.get("mdlp_fast_mode", False),
                alpha=kwargs.get("mdlp_alpha", 0.05),
                n_permutations=kwargs.get("mdlp_n_permutations", 30),
                bonferroni=kwargs.get("mdlp_bonferroni", False),
                validated_seed=kwargs.get("mdlp_validated_seed", 0),
            )
        elif method_resolved == "fayyad_irani_validated":
            assert y is not None  # needs_y guard above raises for this method when y is None
            edges = edges_fayyad_irani_validated(
                col, y,
                val_frac=kwargs.get("mdlp_val_frac", 0.3),
                max_depth=kwargs.get("max_depth", max(1, int(math.log2(kwargs.get("max_adaptive_nbins", MAX_ADAPTIVE_NBINS))))),
                min_split_size=kwargs.get("min_split_size", 5),
                val_min_split_size=kwargs.get("mdlp_val_min_split_size", 5),
                random_state=kwargs.get("random_state", 0),
            )
        elif method_resolved == "uniform":
            edges = edges_uniform(col, n_bins=freedman_diaconis_nbins(col, max_bins=kwargs.get("max_adaptive_nbins", MAX_ADAPTIVE_NBINS)))
        elif method_resolved == "mah":
            assert y is not None  # needs_y guard above raises for this method when y is None
            edges = edges_mah(
                col, y,
                initial_k=int(kwargs.get("mah_initial_k", 16)),
            )
        elif method_resolved == "optimal_joint":
            assert y is not None  # needs_y guard above raises for this method when y is None
            edges = edges_optimal_joint(
                col, y,
                candidates=kwargs.get("candidates", (4, 8, 16, 32)),
                n_splits=kwargs.get("n_splits", 3),
                base=base,
                random_state=kwargs.get("random_state", 0),
                max_y_classes=kwargs.get("optimal_joint_max_y_classes", 64),
            )
        else:
            raise NotImplementedError(method_resolved)
        # 2026-05-30 Wave 9.1 fix (synergy-detection regression): when a
        # supervised binning method (MDLP / Mah / optimal_joint /
        # fayyad_irani) returns zero inner edges - meaning the feature
        # was collapsed to a single bin because individually it has no
        # MI with y - the joint MI on any tuple containing this feature
        # is identically 0 (1-cell joint). This silently DESTROYS
        # synergy detection for XOR-family targets (y = sign(x1*x2),
        # boolean conjunctions, etc.) where individual components are
        # independent of y but their joint perfectly predicts y.
        # Fall back to UNSUPERVISED binning (the requested ``base``)
        # for collapsed columns so synergy tuples still have signal at
        # the joint level. The single-feature MDLP signal is already
        # gone (it returned no splits), so the unsupervised fallback
        # can only improve detection power, never hurt.
        if method_resolved in ("fayyad_irani", "fayyad_irani_validated", "optimal_joint", "mah") and (edges is None or (hasattr(edges, "size") and edges.size == 0)):
            _fallback_nb = int(kwargs.get("collapsed_fallback_nbins", 5))
            if base == "quantile":
                edges = _edges_from_quantiles(col, _fallback_nb)
            else:
                edges = _edges_from_uniform(col, _fallback_nb)
        # 2026-05-31: SPARSE-AWARE secondary fallback. For TF-IDF /
        # one-hot / bag-of-words style columns (>50% mass at a single
        # value, e.g. zero for sparse tokens) the unsupervised quantile
        # fallback ALSO collapses: every quantile lands at the dominant
        # value, np.unique dedups to 1-2 edges, and the resulting 1-bin
        # column produces MI=0 with y. This silently kills sparse-token
        # signal (Layer 20 finding: nbins_strategy='mdlp' default + 95%-
        # zero token columns -> screening returns fallback_used_=True
        # with support=['tok_0']). Detect the sparse-dominance pattern
        # explicitly and split into a separate-bin for the dominant
        # value + quantile bins on the non-dominant subset.
        if _finite.size >= 4 and (edges is None or (hasattr(edges, "size") and edges.size <= 1)):
            _vals_sp, _counts_sp = _uniq, _uniq_counts
            if _vals_sp.size >= 2:
                _max_idx = int(_counts_sp.argmax())
                _dom_frac = float(_counts_sp[_max_idx]) / float(_finite.size)
                if _dom_frac > 0.5:
                    _dom_val = float(_vals_sp[_max_idx])
                    _non_dom_mask = _finite != _dom_val
                    _non_dom = _finite[_non_dom_mask]
                    _sparse_nb = int(kwargs.get("sparse_separate_fallback_nbins", 4))
                    if _non_dom.size >= _sparse_nb:
                        _qs = np.linspace(0.0, 1.0, _sparse_nb + 1)[1:-1]
                        _sub = np.quantile(_non_dom, _qs)
                        _sub = np.unique(_sub)
                    else:
                        _sub = np.unique(_non_dom)
                    # Boundary between dominant value and non-dominant range.
                    _non_dom_min = float(_non_dom.min())
                    _non_dom_max = float(_non_dom.max())
                    if _dom_val <= _non_dom_min:
                        _boundary = 0.5 * (_dom_val + _non_dom_min)
                        _new_edges = np.concatenate([[_boundary], _sub])
                    elif _dom_val >= _non_dom_max:
                        _boundary = 0.5 * (_non_dom_max + _dom_val)
                        _new_edges = np.concatenate([_sub, [_boundary]])
                    else:
                        _lower_dom = float(_finite[_finite < _dom_val].max())
                        _upper_dom = float(_finite[_finite > _dom_val].min())
                        _new_edges = np.concatenate([
                            _sub[_sub < _dom_val],
                            [0.5 * (_lower_dom + _dom_val), 0.5 * (_dom_val + _upper_dom)],
                            _sub[_sub > _dom_val],
                        ])
                    edges = np.unique(_new_edges)
        # Systemic silent-degenerate-fallback guardrail (2026-07-19): EVERY method funnels through this
        # single return point, so this is the ONE place that can catch "binning method returned empty/
        # near-empty edges despite the column having real variance" for ALL strategies (qs/mah/sturges/fd/
        # knuth/blocks/fayyad_irani/optimal_joint), not just the three with a dedicated collapse-fallback
        # above. This is exactly the bug class an MDLP overflow (3.0**n_classes -> inf, acceptance check
        # always False, empty edges returned with no signal) slipped through undetected -- a column with
        # >1 distinct finite value that still ends up with 0 usable edges collapses to a single degenerate
        # bin (all rows get the same code) with NO observable signal anywhere. Log so it is diagnosable
        # from a production run's logs alone, without per-strategy vigilance.
        if (edges is None or (hasattr(edges, "size") and edges.size == 0)) and _uniq.size > 1:
            logger.warning(
                "per_feature_edges: method=%r produced EMPTY bin edges for a column with %d distinct finite "
                "values (real variance) -- this column will silently collapse to a single degenerate bin, "
                "destroying its MI signal. If this is unexpected, investigate the binning method on this "
                "column's distribution (extreme skew/cardinality/scale can silently degrade some strategies).",
                method_resolved, int(_uniq.size),
            )
        return edges, False

    # ---- Phase 1 (serial): per-column cache GET. Records misses to compute. ----
    # Cache lookup happens BEFORE the low-card branch so even cheap midpoint edges
    # hit the cache on repeat fits; the code path stays uniform. Doing all GETs (and
    # later all PUTs) serially on the main thread keeps the DiskCache single-threaded
    # -- no lock needed, hit/miss behavior bit-identical to the historical loop.
    edges_list: list = [None] * n_features
    _miss_keys: list = [None] * n_features  # cache key per missed col (None => don't cache)
    _miss_cols: list = []  # indices needing compute, in ascending order
    _cols: list = [None] * n_features  # float64 view per column (reused by phase 2)
    for j in range(n_features):
        col = X[:, j].astype(np.float64, copy=False)
        _cols[j] = col
        _col_cache_key = None
        if _cache is not None:
            try:
                from mlframe.utils.disk_cache import hash_array_summary, compose_key

                _col_summary = hash_array_summary(col)
                assert _y_key is not None  # set together with _cache in the same cache_dir-guarded try block above
                _col_cache_key = "nbin_" + compose_key(_col_summary, _y_key, _kw_key)
                _hit = _cache.get(_col_cache_key)
                if _hit is not None:
                    edges_list[j] = _hit
                    continue
            except Exception as exc:
                logger.debug("per_feature_edges: cache get failed col=%d (%s)", j, exc)
                _col_cache_key = None
        _miss_keys[j] = _col_cache_key
        _miss_cols.append(j)

    # ---- Phase 2: compute edges for the misses (serial or threaded). ----
    # Parallelism via a THREAD pool: the heavy supervised kernels (MDLP / mah) are
    # njit(nogil=True) so they release the GIL and threads get real cores. Output
    # order is preserved (results written by column index), so edges are deterministic
    # and bit-identical regardless of n_jobs / thread scheduling.
    #
    # Bench (MDLP, n=20000, default njit backend, 2026-06-19, this machine):
    #   p=500 : serial 2.45s -> parallel 0.78s  (3.14x)
    #   p=2000: serial 10.10s -> parallel 3.20s (3.16x)
    #   p=50  : serial 0.28s -> parallel 0.22s  (gated to serial, no regression)
    # (see test_per_feature_edges_parallel.py; numbers refreshed if hardware changes.)
    if n_jobs is None or int(n_jobs) <= 0:
        try:
            import psutil

            _resolved_jobs = psutil.cpu_count(logical=False) or 1
        except Exception:
            import os

            _resolved_jobs = os.cpu_count() or 1
    else:
        _resolved_jobs = int(n_jobs)
    _resolved_jobs = max(1, min(_resolved_jobs, max(1, len(_miss_cols))))

    if _resolved_jobs > 1 and len(_miss_cols) >= _PARALLEL_EDGES_MIN_COLS:
        from concurrent.futures import ThreadPoolExecutor

        def _one(j):
            """Column-index-preserving wrapper around ``_compute_col_edges`` so ``ThreadPoolExecutor.map`` results can be scattered back into ``edges_list`` by their original column position."""
            return j, _compute_col_edges(_cols[j])

        with ThreadPoolExecutor(max_workers=_resolved_jobs) as _ex:
            for j, (edges, _was_lc) in _ex.map(_one, _miss_cols):
                edges_list[j] = edges
    else:
        for j in _miss_cols:
            edges, _was_lc = _compute_col_edges(_cols[j])
            edges_list[j] = edges

    # ---- Phase 3 (serial): cache PUT for the misses. ----
    if _cache is not None:
        for j in _miss_cols:
            _key = _miss_keys[j]
            edges = edges_list[j]
            if _key is not None and edges is not None:
                try:
                    _cache.put(_key, edges)
                except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                    logger.debug("suppressed in _adaptive_nbins.py:753: %s", e)
                    pass

    return edges_list
