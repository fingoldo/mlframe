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
    pre_binned: bool = False, bias_correction: bool = True,
) -> float:
    """Kullback-Leibler divergence KL(P || Q) where P=target, Q=reference.

    Non-symmetric: KL(P||Q) != KL(Q||P). The convention here is "how
    surprising is the target under the reference distribution".

    When ``pre_binned=True`` the inputs are taken as already-normalised
    probability vectors (e.g. multinomial parameters); when False they
    are binned into ``nbins`` quantile bins of the reference first.

    The binned plug-in KL is positively biased in finite samples: the empirical histograms over-resolve the two
    distributions so KL_hat(P||Q) sits well above the true value (and stays clearly positive even when P and Q are
    drawn from the SAME distribution, where the truth is 0). The leading 1/n term is a sum of two Miller-Madow
    entropy-bias contributions -- ``(Kp-1)/(2 np)`` from the empirical -H(P) plus ``(Kq-1)/(2 nq)`` from the cross
    term -H(P, Q_hat), where Kp/Kq are the occupied-bin counts and np/nq the sample sizes. ``bias_correction=True``
    (default) subtracts that floor and clamps at 0; pass ``bias_correction=False`` for the legacy plug-in. The
    correction needs integer counts + sample sizes, so it only applies on the binned path (``pre_binned=False``).
    """
    p = np.asarray(target)
    q = np.asarray(reference)
    if p.size == 0 or q.size == 0:
        return np.nan
    if pre_binned:
        if p.shape != q.shape:
            raise ValueError(f"shape mismatch p={p.shape}, q={q.shape}")
        return float(_kl_kernel(p, q, float(eps)))
    edges = _safe_quantile_bins(q, int(nbins))
    pc = _bin_counts(p, edges); qc = _bin_counts(q, edges)
    np_tot = pc.sum(); nq_tot = qc.sum()
    pn = pc / (np_tot if np_tot > 0 else 1.0)
    qn = qc / (nq_tot if nq_tot > 0 else 1.0)
    kl = float(_kl_kernel(pn, qn, float(eps)))
    if bias_correction:
        kl = max(kl - _kl_mm_bias(pc, qc, np_tot, nq_tot), 0.0)
    return kl


def _kl_mm_bias(
    pc: np.ndarray, qc: np.ndarray, np_tot: float, nq_tot: float,
) -> float:
    """Miller-Madow leading-order bias of the binned plug-in KL: (Kp-1)/(2 np) + (Kq-1)/(2 nq)."""
    kp = float(np.count_nonzero(pc)); kq = float(np.count_nonzero(qc))
    bias = 0.0
    if np_tot > 0 and kp > 1.0:
        bias += (kp - 1.0) / (2.0 * np_tot)
    if nq_tot > 0 and kq > 1.0:
        bias += (kq - 1.0) / (2.0 * nq_tot)
    return bias


def js_divergence(
    reference: np.ndarray, target: np.ndarray,
    *, nbins: int = 50, eps: float = 1e-12,
    pre_binned: bool = False, bias_correction: bool = True,
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

    JS equals the mutual information I(L; B) between the sample-label L in {P, Q} and the bin B, so the binned
    plug-in is the same positively-biased MI plug-in: even when P and Q come from the SAME distribution (true JS=0)
    JS_hat sits clearly above 0, the bias growing as the bin count rises and the sample shrinks. ``bias_correction=
    True`` (default) subtracts the Miller-Madow MI floor ``(K-1)/(2 N)`` (K = occupied pooled bins, N = total
    samples np+nq) and clamps at 0; pass ``bias_correction=False`` for the legacy plug-in. The correction needs the
    raw counts + sample sizes, so it only applies on the binned path (``pre_binned=False``).
    """
    p = np.asarray(target)
    q = np.asarray(reference)
    if p.size == 0 or q.size == 0:
        return np.nan
    if pre_binned:
        if p.shape != q.shape:
            raise ValueError(f"shape mismatch p={p.shape}, q={q.shape}")
        pn = p; qn = q
        m = 0.5 * (pn + qn)
        return 0.5 * float(_kl_kernel(pn, m, float(eps))) + 0.5 * float(_kl_kernel(qn, m, float(eps)))
    # Pool both samples for the bin grid to keep js symmetric.
    pooled = np.concatenate([p, q])
    edges = _safe_quantile_bins(pooled, int(nbins))
    pc = _bin_counts(p, edges); qc = _bin_counts(q, edges)
    np_tot = pc.sum(); nq_tot = qc.sum()
    pn = pc / (np_tot if np_tot > 0 else 1.0)
    qn = qc / (nq_tot if nq_tot > 0 else 1.0)
    m = 0.5 * (pn + qn)
    js = 0.5 * float(_kl_kernel(pn, m, float(eps))) + 0.5 * float(_kl_kernel(qn, m, float(eps)))
    if bias_correction:
        n_tot = np_tot + nq_tot
        k_active = float(np.count_nonzero(pc + qc))
        if n_tot > 0 and k_active > 1.0:
            js = max(js - (k_active - 1.0) / (2.0 * n_tot), 0.0)
    return js


# ----- Wasserstein-1 -----


@numba.njit(**NUMBA_NJIT_PARAMS)
def _wasserstein_1d_fused(a_s: np.ndarray, b_s: np.ndarray) -> float:
    """Fused merge of two pre-sorted arrays -> W1 = sum |F_a - F_b| * delta in one O(na+nb) pass.

    Replaces the numpy ``concatenate+sort`` of the merged support plus two ``searchsorted`` scans.
    Equivalent by construction: at each consecutive merged support point the right-side empirical CDFs
    are the running counts ``i/na`` and ``j/nb`` after consuming all entries <= the current support value
    (ties handled by the two inner while-loops), exactly as ``searchsorted(side='right')`` would yield.
    """
    na = a_s.size
    nb = b_s.size
    i = 0
    j = 0
    total = 0.0
    have_prev = False
    prevx = 0.0
    ca = 0.0
    cb = 0.0
    while i < na or j < nb:
        if j >= nb or (i < na and a_s[i] <= b_s[j]):
            x = a_s[i]
        else:
            x = b_s[j]
        if have_prev:
            total += abs(ca - cb) * (x - prevx)
        while i < na and a_s[i] <= x:
            i += 1
        while j < nb and b_s[j] <= x:
            j += 1
        ca = i / na
        cb = j / nb
        prevx = x
        have_prev = True
    return total


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
    return float(_wasserstein_1d_fused(np.sort(a), np.sort(b)))


# ----- KS distribution distance -----


@numba.njit(**NUMBA_NJIT_PARAMS)
def _ks_distance_fused(a_s: np.ndarray, b_s: np.ndarray) -> float:
    """Fused merge of two pre-sorted arrays -> KS = max |F_a - F_b| in one O(na+nb) pass.

    Same right-side running-CDF construction as ``_wasserstein_1d_fused``; takes the running max of the
    CDF gap at every merged support point instead of the delta-weighted sum.
    """
    na = a_s.size
    nb = b_s.size
    i = 0
    j = 0
    m = 0.0
    while i < na or j < nb:
        if j >= nb or (i < na and a_s[i] <= b_s[j]):
            x = a_s[i]
        else:
            x = b_s[j]
        while i < na and a_s[i] <= x:
            i += 1
        while j < nb and b_s[j] <= x:
            j += 1
        d = abs(i / na - j / nb)
        if d > m:
            m = d
    return m


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
    return float(_ks_distance_fused(np.sort(a), np.sort(b)))
