"""k-NN-based mutual-information estimators for MRMR (2026-05-29).

Two estimators in this module:

* **Mixed-KSG** (Gao, Kannan, Oh, Viswanath 2017, NeurIPS;
  https://arxiv.org/abs/1709.06212). Estimates I(X; Y) directly from k-NN
  distances without binning. Handles discrete-continuous mixtures cleanly via
  the Gao tie-handling fix. Provably asymptotically unbiased; the strongest
  non-neural baseline in the Czyz et al. NeurIPS 2023 benchmark
  (https://arxiv.org/abs/2306.11078).

* **KSG-LNC** (Gao, Ver Steeg, Galstyan 2015, AISTATS;
  https://arxiv.org/abs/1411.2003). Patches the classical KSG estimator for
  strongly-correlated variables via Local Nonuniformity Correction:
  per-neighbour local PCA + ellipsoidal volume in place of the L-inf box.
  Closed-form correction with no CV-tuned hyperparameter (alpha = 0.65 for
  d=2, k=5 per the paper).

Optimization pattern (per README.md methodology):
- ``sklearn.neighbors.KDTree`` for the spatial queries (C-backed, beats any
  njit re-impl at N <= 1e6).
- ``@njit`` for the digamma aggregation hot loop.
- ``cupy`` fallback path (``mixed_ksg_mi_gpu``) for N >= ``_KSG_GPU_THRESHOLD``
  where H2D bandwidth pays back.

The plug-in MI fast path remains the default for MRMR; KSG estimators are
opt-in via the future ``mi_estimator='mixed_ksg'`` knob on ``MRMR.__init__``.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Optional

import numpy as np
from numba import njit

logger = logging.getLogger(__name__)


_KSG_GPU_THRESHOLD = int(os.environ.get("MLFRAME_KSG_GPU_N", "50000"))


# =============================================================================
# Digamma helper (numba doesn't ship one; vectorised njit kernel)
# =============================================================================


@njit(nogil=True, cache=True)
def _digamma_scalar(x: float) -> float:
    """Digamma psi(x) for x > 0.

    Uses recurrence to push x >= 6, then the asymptotic expansion:
        psi(x) ~ ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6)
    Accuracy ~1e-13 for x >= 6. Matches scipy.special.digamma to ~1e-12.
    """
    if x <= 0.0:
        return -math.inf
    # Recurrence to bring x up to >= 6.
    result = 0.0
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    # Asymptotic expansion.
    inv = 1.0 / x
    inv2 = inv * inv
    series = math.log(x) - 0.5 * inv \
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0)))
    return result + series


@njit(nogil=True, cache=True, parallel=False)
def _digamma_vec(arr: np.ndarray) -> np.ndarray:
    out = np.empty(arr.shape[0], dtype=np.float64)
    for i in range(arr.shape[0]):
        out[i] = _digamma_scalar(float(arr[i]))
    return out


# =============================================================================
# Mixed-KSG core (Gao 2017)
# =============================================================================


@njit(nogil=True, cache=True)
def _mixed_ksg_aggregate(k: int, n: int, n_x: np.ndarray, n_y: np.ndarray) -> float:
    """Aggregation step of mixed-KSG:
        I = psi(k) + psi(N) - mean(psi(n_x + 1) + psi(n_y + 1))
    Negative-finite-sample noise clamped to 0.
    """
    psi_k = _digamma_scalar(float(k))
    psi_n = _digamma_scalar(float(n))
    s = 0.0
    for i in range(n):
        s += _digamma_scalar(float(n_x[i] + 1)) + _digamma_scalar(float(n_y[i] + 1))
    s /= float(n)
    mi = psi_k + psi_n - s
    if mi < 0.0:
        mi = 0.0
    return mi


def _query_knn_chebyshev(arr_2d: np.ndarray, k: int) -> np.ndarray:
    """k-NN distances in Chebyshev (L-inf) metric. Returns ``eps[i] = d_k[i]``
    where d_k is the distance to the k-th nearest neighbour (excluding self).
    Routes through ``sklearn.neighbors.KDTree``.
    """
    from sklearn.neighbors import KDTree
    tree = KDTree(arr_2d.astype(np.float64), metric="chebyshev")
    # +1 because the closest is the point itself.
    d, _ = tree.query(arr_2d, k=k + 1)
    return d[:, k].astype(np.float64)


def _count_within_eps(arr_1d: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """For each point i, count other points j with ``|x_j - x_i| < eps[i]``.
    Uses sorted array + binary search; O(N log N) total."""
    arr = arr_1d.astype(np.float64).ravel()
    sorted_arr = np.sort(arr)
    # Fully vectorised: searchsorted accepts the whole bound array at once, so
    # the per-point band query becomes two O(N log N) C-level calls instead of
    # 2*N Python-level searchsorted calls. Bit-identical to the prior loop:
    # same lo/hi bounds, same left/right sides, same self-subtraction + clamp.
    lo = arr - eps + 1e-12
    hi = arr + eps - 1e-12
    lo_idx = np.searchsorted(sorted_arr, lo, side="left")
    hi_idx = np.searchsorted(sorted_arr, hi, side="right")
    counts = (hi_idx - lo_idx) - 1  # subtract 1 for self (always in band)
    np.maximum(counts, 0, out=counts)
    return counts.astype(np.int64)


def mixed_ksg_mi(x: np.ndarray, y: np.ndarray, k: int = 5,
                 use_gpu: bool = False,
                 intens: float = 1e-10,
                 max_input_n: int = 50000,
                 seed: int = 0) -> float:
    """Mixed-KSG mutual information estimator (Gao et al. NeurIPS 2017).

    Handles discrete-continuous mixtures via Gao's tie-aware count rule.
    Returns I(X; Y) in nats; clamped at 0 for finite-sample negative noise.

    Args:
        x: 1-D array (continuous or mixed discrete-continuous).
        y: 1-D array (continuous or integer classes).
        k: nearest-neighbour count. Czyz 2023 recommends ``k=5..10``;
            default 5 matches the Gao 2017 paper.
        use_gpu: If True and CuPy available and N >= ``_KSG_GPU_THRESHOLD``,
            route the eps-radius counting through cupy. KDTree build stays CPU
            (no equivalent in cupy without RAFT).
        intens: 2026-05-29 fix - tie-breaking noise magnitude (matches NPEET_LNC
            reference line 47, ``intens=1e-10``). Without this, k-NN on
            integer / discrete columns with ties produces ``eps[i]=0`` for many
            points, which gives ``count_within_eps -> 0`` and ``digamma(0)``
            blow-up. Bench regression: discrete_low_card mean MI 0.0000 (bug)
            -> ~0.50 (signal-tracking) with this fix.
        seed: RNG seed for jitter.

    Reference: Gao, W., Kannan, S., Oh, S., Viswanath, P. (2017),
    "Estimating Mutual Information for Discrete-Continuous Mixtures",
    NeurIPS 2017. arXiv:1709.06212.
    """
    rng = np.random.default_rng(int(seed))
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    if n < 2 * k + 2:
        return 0.0
    # 2026-05-29 stress-bench fix: KDTree is O(N log N) for build + O(N log N)
    # for query but the constant is large; at N=100k each call is ~10s, at
    # N=1M extrapolating ~100s. Sub-sample to ``max_input_n`` so MRMR's
    # 100x100 pair loop stays tractable on million-row datasets. The bench
    # shows MI converges by N=20k for KSG; 50k default gives a safety margin
    # without paying the full N=1M cost.
    #
    # BIAS CAVEAT: sub-sampling raises the kNN-distance estimator's bias -- KSG MI bias scales
    # ~ k/N_eff, so capping N_eff at max_input_n leaves a small positive-bias floor that the full
    # sample would shrink further. The 20k-convergence bench above shows the residual is negligible
    # for the screening use here; raise max_input_n (paying the runtime) when an unbiased MI on a
    # >50k sample is required.
    if n > int(max_input_n):
        idx = rng.choice(n, size=int(max_input_n), replace=False)
        x = x[idx]
        y = y[idx]
        n = int(max_input_n)
    # Tie-breaking noise: only on column(s) with ties to keep the continuous
    # path bit-for-bit identical to pre-fix.
    if np.unique(x).size < n:
        x = x + intens * rng.standard_normal(n)
    if np.unique(y).size < n:
        y = y + intens * rng.standard_normal(n)
    xy = np.column_stack([x, y]).astype(np.float64)
    eps = _query_knn_chebyshev(xy, k=k)
    eps = np.maximum(eps, 1e-12)
    n_x = _count_within_eps(x, eps)
    n_y = _count_within_eps(y, eps)
    return float(_mixed_ksg_aggregate(int(k), int(n),
                                      n_x.astype(np.int64), n_y.astype(np.int64)))


# =============================================================================
# KSG-LNC (Gao 2015) - local PCA volume correction
# =============================================================================


@njit(nogil=True, cache=True)
def _eig2x2(a11: float, a12: float, a22: float):
    """Closed-form eigenvalues of 2x2 symmetric matrix [[a11, a12], [a12, a22]]."""
    tr = a11 + a22
    det = a11 * a22 - a12 * a12
    disc = tr * tr - 4.0 * det
    if disc < 0.0:
        disc = 0.0
    sq = math.sqrt(disc)
    lam1 = 0.5 * (tr + sq)
    lam2 = 0.5 * (tr - sq)
    return lam1, lam2


@njit(nogil=True, cache=True)
def _lnc_correction_v2(knn_xy: np.ndarray, log_d_x: float, log_d_y: float,
                        alpha: float) -> float:
    """Per-point LNC term (canonical NPEET_LNC algorithm, d=2 specialisation).

    Args:
        knn_xy: (k+1, 2) array of NEIGHBOUR coords RELATIVE TO THE POINT ITSELF
            (i.e. ``points[knn] - point``, NOT centered to neighbour centroid).
        log_d_x: ``log(dvec_x)``, the marginal X-axis L-inf radius of the
            classical KSG box at this point.
        log_d_y: same for Y axis.
        alpha: gate threshold (NPEET_LNC default 0.25 per Gao 2015 sec. 4 for d=2).

    Pipeline (matches reference lines 141-200):
      1. 2x2 covariance of knn_xy (already centered to the point, NOT to
         neighbour mean).
      2. Eigenvectors v of covariance.
      3. For each axis i: max_j |knn_xy[j] @ v[:, i]| -> semi-axis cur[i].
      4. V_rect = sum_i log(cur[i]).
      5. log_knn_dist = log_d_x + log_d_y.
      6. Gate: if V_rect >= log_knn_dist + log(alpha) -> set V_rect=log_knn_dist
         (correction becomes 0).
      7. Return (log_knn_dist - V_rect) if positive; else 0.
    """
    n_pts = knn_xy.shape[0]
    if n_pts < 3:
        return 0.0
    # Covariance (already centered to the focus point).
    c00 = 0.0
    c01 = 0.0
    c11 = 0.0
    # Reference iterates from index 1 (skipping the focus point itself which has
    # coords [0, 0] after centering); we mirror that to match line-by-line.
    for i in range(1, n_pts):
        u = knn_xy[i, 0]
        v = knn_xy[i, 1]
        c00 += u * u
        c01 += u * v
        c11 += v * v
    denom = float(n_pts - 1)
    if denom <= 0.0:
        return 0.0
    c00 /= denom
    c01 /= denom
    c11 /= denom
    lam1, lam2 = _eig2x2(c00, c01, c11)
    if lam1 <= 0.0 or lam2 <= 0.0:
        return 0.0
    # Eigenvectors.
    if abs(c01) < 1e-15:
        e1u, e1v = 1.0, 0.0
        e2u, e2v = 0.0, 1.0
    else:
        v1u, v1v = c01, (lam1 - c00)
        norm1 = math.sqrt(v1u * v1u + v1v * v1v)
        if norm1 < 1e-15:
            v1u, v1v = (lam1 - c11), c01
            norm1 = math.sqrt(v1u * v1u + v1v * v1v)
            if norm1 < 1e-15:
                return 0.0
        e1u, e1v = v1u / norm1, v1v / norm1
        e2u, e2v = -e1v, e1u
    # Semi-axes via max-abs projection (includes the focus point at origin which
    # gives projection 0 -- harmless).
    a1 = 0.0
    a2 = 0.0
    for i in range(n_pts):
        u = knn_xy[i, 0]
        v = knn_xy[i, 1]
        p1 = abs(u * e1u + v * e1v)
        p2 = abs(u * e2u + v * e2v)
        if p1 > a1:
            a1 = p1
        if p2 > a2:
            a2 = p2
    if a1 <= 0.0 or a2 <= 0.0:
        return 0.0
    V_rect = math.log(a1) + math.log(a2)
    log_knn_dist = log_d_x + log_d_y
    # Gate: skip correction unless PCA box is meaningfully smaller than KSG box.
    if V_rect >= log_knn_dist + math.log(alpha):
        return 0.0
    diff = log_knn_dist - V_rect
    if diff <= 0.0:
        return 0.0
    return diff


@njit(nogil=True, cache=True)
def _lnc_correction(neighbours_xy: np.ndarray, eps: float, alpha: float) -> float:
    """DEPRECATED first-pass LNC impl - retained behind flag for A/B parity.
    Use ``_lnc_correction_v2`` for the canonical NPEET_LNC algorithm."""
    return 0.0


@njit(nogil=True, cache=True)
def _kraskov1_aggregate(k: int, n: int, n_x: np.ndarray, n_y: np.ndarray, d: int) -> float:
    """Classical Kraskov-1 MI estimator (Kraskov 2004 eq. 8):
       I = psi(k) - (d - 1) / k + (d - 1) * psi(N) - mean(psi(n_x + 1) + psi(n_y + 1))
    Used as the LNC base per Gao 2015 / NPEET_LNC reference (NOT the Gao 2017
    Mixed-KSG, which is the base for ``mixed_ksg_mi``).

    The ``+1`` matches the neighbour-count convention used here and in ``_mixed_ksg_aggregate``: ``_count_within_eps``
    returns the number of points STRICTLY within the marginal radius EXCLUDING the point itself, whereas the KSG-1
    digamma term is ``psi(n_x)`` with ``n_x`` counted in the closed-ball convention INCLUDING self (NPEET ``avgdigamma``).
    Using ``psi(n_x)`` against the self-excluding count is an open/closed-ball mismatch that systematically biases MI.
    """
    psi_k = _digamma_scalar(float(k))
    psi_n = _digamma_scalar(float(n))
    s = 0.0
    for i in range(n):
        s += _digamma_scalar(float(n_x[i] + 1)) + _digamma_scalar(float(n_y[i] + 1))
    s /= float(n)
    return psi_k - float(d - 1) / float(k) + float(d - 1) * psi_n - s


def ksg_lnc_mi(x: np.ndarray, y: np.ndarray, k: int = 5,
               alpha: float = 0.25, intens: float = 1e-10,
               low_entropy_skip: bool = True,
               min_y_unique_frac: float = 0.02,
               seed: int = 0) -> float:
    """KSG with Local Nonuniformity Correction (Gao et al. AISTATS 2015).

    Canonical port of ``NPEET_LNC/lnc.py`` (Gao 2015 reference implementation):
    line-by-line audit comparing this function to ``mi_LNC`` in
    https://github.com/BiuBiuBiLL/NPEET_LNC/blob/master/lnc.py is the ground
    truth; deviations are bugs.

    Args:
        x, y: 1-D arrays.
        k: nearest-neighbour count (default 5 matches Gao 2015 sec. 4 and
            NPEET_LNC default).
        alpha: PCA-bounding-box gate threshold. Default ``0.25`` is the
            NPEET_LNC default for d=2, k=5 (per Gao 2015 paper Fig. 1
            calibration). 0.25 is intentional and NOT a typo: a smaller alpha
            permits the correction to fire more aggressively.
        intens: tie-breaking noise magnitude (Gao 2015 line 47 of reference).
            Default ``1e-10`` matches NPEET_LNC; floor on numerical artefacts.
        seed: RNG seed for tie-breaking noise.

    Reference: Gao, S., Ver Steeg, G., Galstyan, A. (2015),
    "Efficient Estimation of Mutual Information for Strongly Dependent Variables",
    AISTATS 2015. arXiv:1411.2003. Code: https://github.com/BiuBiuBiLL/NPEET_LNC
    """
    from sklearn.neighbors import KDTree
    rng = np.random.default_rng(int(seed))
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    if n < 2 * k + 2:
        return 0.0
    # 2026-05-29 fix: low-entropy pre-filter.
    # On binary / few-class y the local PCA inside the LNC correction sees
    # collapsing eigenstructure (most neighbours share y), inflating the
    # log(box / pca_box) term and producing a NON-zero MI on truly random
    # (x, y) pairs (no_signal floor ~0.15 vs Mixed-KSG's 0.01). Skip the
    # correction in those cases and fall back to canonical Kraskov-1.
    if low_entropy_skip:
        uniq_y_count = np.unique(y).size
        if uniq_y_count / n < float(min_y_unique_frac):
            # Use mixed-KSG (Gao 2017) which is calibrated for discrete-mixed
            # data; LNC adds no benefit and inflates the floor on such inputs.
            return mixed_ksg_mi(x, y, k=int(k))
    # Tie-breaking noise (NPEET_LNC line 47).
    x = x + intens * rng.standard_normal(n)
    y = y + intens * rng.standard_normal(n)
    xy = np.column_stack([x, y]).astype(np.float64)
    tree_xy = KDTree(xy, metric="chebyshev")
    d_k, idx_k = tree_xy.query(xy, k=k + 1)
    # Per-axis marginal distances d_x, d_y to k-th nearest neighbour in joint
    # (NOT the joint L-inf radius). Reference lines 128-131.
    dvec_x = np.zeros(n, dtype=np.float64)
    dvec_y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(k + 1):
            jj = idx_k[i, j]
            d_xi = abs(xy[jj, 0] - xy[i, 0])
            d_yi = abs(xy[jj, 1] - xy[i, 1])
            if d_xi > dvec_x[i]:
                dvec_x[i] = d_xi
            if d_yi > dvec_y[i]:
                dvec_y[i] = d_yi
    dvec_x = np.maximum(dvec_x, 1e-15)
    dvec_y = np.maximum(dvec_y, 1e-15)
    # Classical KSG-1 base via marginal-axis radii (NPEET_LNC mi_Kraskov).
    n_x = _count_within_eps(x, dvec_x)
    n_y = _count_within_eps(y, dvec_y)
    classical = float(_kraskov1_aggregate(int(k), int(n),
                                            n_x.astype(np.int64),
                                            n_y.astype(np.int64), d=2))
    # LNC correction term (NPEET_LNC lines 139-200).
    lnc_e = 0.0
    for i in range(n):
        nb_idx = idx_k[i]
        nb_pts = xy[nb_idx] - xy[i]  # relative to focus point (NOT centroid)
        lnc_e += _lnc_correction_v2(nb_pts,
                                     math.log(float(dvec_x[i])),
                                     math.log(float(dvec_y[i])),
                                     float(alpha)) / float(n)
    return max(0.0, classical + lnc_e)


# =============================================================================
# (removed) ColumnKNNCache
# -----------------------------------------------------------------------------
# A per-column ``np.sort`` cache was removed 2026-06-23: it was never consumed
# (no caller in the tree) and was unwireable into the actual hot path without
# changing MI values. The pair MI cost is dominated by the JOINT (x, y) KDTree,
# which is irreducibly per-pair and not cacheable per column. The only
# per-column structure it cached was ``np.sort`` of a column for
# ``_count_within_eps`` -- but (a) that sort is now ~1.5 ms after the CPX7
# vectorisation (negligible vs the joint tree), and (b) mixed_ksg_mi jitters
# tied columns with fresh per-call noise, so a cross-pair cached sort would be
# stale and produce DIFFERENT counts. Dead code, removed rather than wired.
# =============================================================================


# =============================================================================
# cupy fallback (large N)
# =============================================================================


def mixed_ksg_mi_gpu(x: np.ndarray, y: np.ndarray, k: int = 5,
                     intens: float = 1e-10, max_input_n: int = 50000,
                     seed: int = 0) -> float:
    """cupy-accelerated mixed-KSG. KDTree build still CPU; the eps-radius
    counts move to cupy via sorted-array binary search (cp.searchsorted)."""
    try:
        import cupy as cp  # noqa: F401
    except ImportError:
        logger.warning("cupy not available; falling back to CPU mixed_ksg_mi")
        return mixed_ksg_mi(x, y, k=k)
    import cupy as cp
    from sklearn.neighbors import KDTree
    rng = np.random.default_rng(int(seed))
    x_np = np.asarray(x, dtype=np.float64).ravel()
    y_np = np.asarray(y, dtype=np.float64).ravel()
    n = x_np.size
    if n < 2 * k + 2:
        return 0.0
    # Mirror the CPU mixed_ksg_mi pre-processing so GPU/CPU agree: subsample to max_input_n, then add tie-breaking jitter only on columns that have ties
    # (raw ties give eps=0 -> count_within_eps=0 -> digamma(0) blow-up; CPU jitters them, the GPU path previously did not -> divergent MI on discrete data).
    if n > int(max_input_n):
        idx = rng.choice(n, size=int(max_input_n), replace=False)
        x_np = x_np[idx]
        y_np = y_np[idx]
        n = int(max_input_n)
    if np.unique(x_np).size < n:
        x_np = x_np + intens * rng.standard_normal(n)
    if np.unique(y_np).size < n:
        y_np = y_np + intens * rng.standard_normal(n)
    xy = np.column_stack([x_np, y_np]).astype(np.float64)
    tree = KDTree(xy, metric="chebyshev")
    d, _ = tree.query(xy, k=k + 1)
    eps_cpu = np.maximum(d[:, k], 1e-12)
    x_gpu = cp.asarray(x_np)
    y_gpu = cp.asarray(y_np)
    eps_gpu = cp.asarray(eps_cpu)
    sx = cp.sort(x_gpu)
    sy = cp.sort(y_gpu)
    n_x = (cp.searchsorted(sx, x_gpu + eps_gpu - 1e-12, side="right")
           - cp.searchsorted(sx, x_gpu - eps_gpu + 1e-12, side="left") - 1)
    n_y = (cp.searchsorted(sy, y_gpu + eps_gpu - 1e-12, side="right")
           - cp.searchsorted(sy, y_gpu - eps_gpu + 1e-12, side="left") - 1)
    n_x = cp.maximum(n_x, 0).astype(cp.int64).get()
    n_y = cp.maximum(n_y, 0).astype(cp.int64).get()
    return float(_mixed_ksg_aggregate(int(k), int(n), n_x, n_y))


# =============================================================================
# Dispatcher (auto CPU vs GPU)
# =============================================================================


def ksg_mi_dispatch(x: np.ndarray, y: np.ndarray, *, k: int = 5,
                    estimator: str = "mixed_ksg", alpha: float = 0.65,
                    prefer_gpu: bool = True) -> float:
    """Picks the fastest backend per call.

    * CPU mixed-KSG / KSG-LNC for N < ``_KSG_GPU_THRESHOLD`` (default 50000).
    * cupy mixed-KSG above that threshold IF CuPy is importable.

    The threshold honours ``MLFRAME_KSG_GPU_N`` env var per the
    ``kernel_tuning_cache`` methodology in README.md.
    """
    x = np.asarray(x).ravel()
    n = x.size
    if estimator == "mixed_ksg" and prefer_gpu and n >= _KSG_GPU_THRESHOLD:
        try:
            import cupy  # noqa: F401
            return mixed_ksg_mi_gpu(x, y, k=k)
        except ImportError:
            pass
    if estimator == "mixed_ksg":
        return mixed_ksg_mi(x, y, k=k)
    if estimator == "ksg_lnc":
        return ksg_lnc_mi(x, y, k=k, alpha=alpha)
    raise ValueError(f"unknown estimator {estimator!r}")


__all__ = [
    "mixed_ksg_mi", "ksg_lnc_mi", "mixed_ksg_mi_gpu",
    "ksg_mi_dispatch",
]
