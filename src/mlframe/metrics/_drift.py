"""Distributional / drift metrics for the mlframe training suite.

Used to compare prediction (or feature) distributions across train,
val, test - the canonical sanity check that "the model in val is
the same kind of model that lands in test".

Public API (re-exported from ``mlframe.metrics.core``):
    * ``population_stability_index``  - PSI (banking standard)
    * ``kl_divergence``               - KL(P || Q)
    * ``js_divergence``               - JS divergence in [0, log 2]
    * ``wasserstein_1d``              - 1-D Wasserstein-1 (earth mover)
    * ``ks_distribution_distance``    - 2-sample KS statistic
"""
from __future__ import annotations

from math import log
from typing import Tuple

import numpy as np
import numba

from ._numba_params import NUMBA_NJIT_PARAMS


def _safe_quantile_bins(
    p_sample: np.ndarray, nbins: int,
) -> np.ndarray:
    """Quantile-based bin edges from a reference sample.

    Returns ``nbins+1`` edges; left/right are -inf / +inf so out-of-range
    drift samples still bin. Falls back to equal-width edges when the
    reference is too constant for quantile binning (otherwise the
    pd.qcut-style "Bin edges must be unique" failure mode would surface).
    """
    finite = p_sample[np.isfinite(p_sample)]
    if finite.size == 0:
        # Degenerate reference - no signal to bin against.
        return np.array([-np.inf, np.inf])
    qs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(finite, qs)
    edges[0] = -np.inf
    edges[-1] = np.inf
    # Collapse duplicate edges (constant region in the reference). Use
    # the unique edges as the actual binning - if everything collapses
    # to two edges we degenerate to a single bin, which yields PSI=0.
    edges = np.unique(edges)
    if edges.shape[0] < 3:
        edges = np.array([-np.inf, np.inf])
    return edges


def _bin_counts(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Counts per bin defined by ``edges`` (length nbins+1)."""
    counts, _ = np.histogram(x, bins=edges)
    return counts.astype(np.float64)


# ----- PSI -----


@numba.njit(**NUMBA_NJIT_PARAMS)
def _psi_kernel(p: np.ndarray, q: np.ndarray, eps: float) -> float:
    """sum_i (p_i - q_i) * log(p_i / q_i) with eps clamping for empty bins."""
    s = 0.0
    for i in range(p.shape[0]):
        pi = p[i]
        qi = q[i]
        if pi < eps:
            pi = eps
        if qi < eps:
            qi = eps
        s += (pi - qi) * log(pi / qi)
    return s


def population_stability_index(
    reference: np.ndarray, target: np.ndarray,
    *, nbins: int = 10, eps: float = 1e-4,
) -> float:
    """Population Stability Index between two 1-D samples.

    The banking standard for monitoring feature / prediction drift.
    Conventional thresholds:
        PSI < 0.10   => no significant change
        0.10 <= PSI < 0.25 => moderate shift, investigate
        PSI >= 0.25  => major shift, model likely degraded

    Uses quantile bins from the REFERENCE distribution (so reference is
    uniformly populated across bins by construction); target is binned
    against the SAME edges. Empty bins are clamped to ``eps`` to avoid
    log(0).
    """
    # iter608: dropped the unconditional ``dtype=np.float64`` cast.
    # ``_safe_quantile_bins`` / ``_bin_counts`` already produce float64
    # outputs internally; the input arrays only need to be ndarray (any
    # numeric dtype) for those helpers + the kernel's dispatch. Bench
    # nbins=10..50: PSI 1.29-3.09x across (f64, f32, mixed) dtype pairs.
    ref = np.asarray(reference)
    tgt = np.asarray(target)
    if ref.size == 0 or tgt.size == 0:
        return np.nan
    edges = _safe_quantile_bins(ref, int(nbins))
    p = _bin_counts(ref, edges); p /= p.sum() if p.sum() > 0 else 1.0
    q = _bin_counts(tgt, edges); q /= q.sum() if q.sum() > 0 else 1.0
    return float(_psi_kernel(p, q, float(eps)))


# ----- KL / JS -----


@numba.njit(**NUMBA_NJIT_PARAMS)
def _kl_kernel(p: np.ndarray, q: np.ndarray, eps: float) -> float:
    """KL(P || Q) = sum p_i log(p_i / q_i) with eps clamping."""
    s = 0.0
    for i in range(p.shape[0]):
        pi = p[i]
        qi = q[i]
        if pi <= 0.0:
            continue  # 0 * log 0 == 0 convention
        if qi < eps:
            qi = eps
        s += pi * log(pi / qi)
    return s


def kl_divergence(
    reference: np.ndarray, target: np.ndarray,
    *, nbins: int = 50, eps: float = 1e-12,
    pre_binned: bool = False,
) -> float:
    """Kullback-Leibler divergence KL(P || Q) where P=target, Q=reference.

    Non-symmetric: KL(P||Q) != KL(Q||P). The convention here is "how
    surprising is the target under the reference distribution".

    When ``pre_binned=True`` the inputs are taken as already-normalised
    probability vectors (e.g. multinomial parameters); when False they
    are binned into ``nbins`` quantile bins of the reference first.
    """
    # iter608: skip-cast (see psi_score). Bench nbins=10..50: KL
    # 1.16-3.42x across (f64, f32, mixed) dtype pairs. Bit-equiv.
    p = np.asarray(target)
    q = np.asarray(reference)
    if p.size == 0 or q.size == 0:
        return np.nan
    if pre_binned:
        if p.shape != q.shape:
            raise ValueError(f"shape mismatch p={p.shape}, q={q.shape}")
        return float(_kl_kernel(p, q, float(eps)))
    edges = _safe_quantile_bins(q, int(nbins))
    pn = _bin_counts(p, edges); pn /= pn.sum() if pn.sum() > 0 else 1.0
    qn = _bin_counts(q, edges); qn /= qn.sum() if qn.sum() > 0 else 1.0
    return float(_kl_kernel(pn, qn, float(eps)))


def js_divergence(
    reference: np.ndarray, target: np.ndarray,
    *, nbins: int = 50, eps: float = 1e-12,
    pre_binned: bool = False,
) -> float:
    """Jensen-Shannon divergence between two distributions.

    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), M = (P + Q) / 2.

    Symmetric, bounded in [0, log 2] (natural log) - the user-friendly
    cousin of KL. Use over KL when comparing two empirical distributions
    of comparable nature (e.g. train vs val predictions).

    Symmetry note: when ``pre_binned=False`` we bin both samples against
    quantile edges computed from the POOLED reference+target set. Binning
    against only the reference would break symmetry
    (js(a,b) != js(b,a)) because the bin edges depend on which sample is
    the reference.
    """
    # iter608: skip-cast (see psi_score / kl_divergence). Same
    # _kl_kernel under the hood, same dispatch behavior.
    p = np.asarray(target)
    q = np.asarray(reference)
    if p.size == 0 or q.size == 0:
        return np.nan
    if pre_binned:
        if p.shape != q.shape:
            raise ValueError(f"shape mismatch p={p.shape}, q={q.shape}")
        pn = p; qn = q
    else:
        # Pool both samples for the bin grid to keep js symmetric.
        pooled = np.concatenate([p, q])
        edges = _safe_quantile_bins(pooled, int(nbins))
        pn = _bin_counts(p, edges); pn /= pn.sum() if pn.sum() > 0 else 1.0
        qn = _bin_counts(q, edges); qn /= qn.sum() if qn.sum() > 0 else 1.0
    m = 0.5 * (pn + qn)
    return 0.5 * float(_kl_kernel(pn, m, float(eps))) + 0.5 * float(_kl_kernel(qn, m, float(eps)))


# ----- Wasserstein-1 -----


def wasserstein_1d(reference: np.ndarray, target: np.ndarray) -> float:
    """1-D Wasserstein-1 distance (earth mover's distance).

    For 1-D distributions, EMD equals the integral of the absolute
    difference between empirical CDFs:
        W1 = sum_i |F_P(x_i) - F_Q(x_i)| * (x_{i+1} - x_i)
    over the merged sorted support.

    Numpy implementation via merge + cumsum (O((n+m) log(n+m))).
    """
    # bench-attempt-rejected (iter614, 2026-05-31): tried dropping the
    # upfront ``dtype=np.float64`` cast for int reference / target.
    # Result REGRESSED 0.71-0.82x at n=1k..50k -- int dtype slows down
    # np.sort + np.searchsorted (bitwise compares are slower than
    # SIMD-friendly float64 ops), and the final ``/ size`` produces
    # float64 anyway. The upfront cast wins because it lets the entire
    # sort+searchsorted chain run on SIMD float64. Don't re-try.
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.nan
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    all_values = np.concatenate((a, b))
    all_values.sort(kind="quicksort")
    deltas = np.diff(all_values)
    # Empirical CDFs at all support points
    cdf_a = np.searchsorted(np.sort(a), all_values[:-1], side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b), all_values[:-1], side="right") / b.size
    return float(np.sum(np.abs(cdf_a - cdf_b) * deltas))


# ----- KS distribution distance -----


def ks_distribution_distance(
    reference: np.ndarray, target: np.ndarray,
) -> float:
    """Two-sample Kolmogorov-Smirnov distance.

    KS_dist = max_x |F_P(x) - F_Q(x)|

    Distinct from ``ks_statistic`` in ``_classification_extras.py``,
    which is the discrimination KS between class-conditional CDFs of a
    classifier's score. This one compares two arbitrary samples - the
    standard drift / population-distribution sanity check.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.nan
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    a_sorted = np.sort(a); b_sorted = np.sort(b)
    all_values = np.concatenate((a_sorted, b_sorted))
    all_values.sort()
    cdf_a = np.searchsorted(a_sorted, all_values, side="right") / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, all_values, side="right") / b_sorted.size
    return float(np.max(np.abs(cdf_a - cdf_b)))
