"""Native bin-edge calculators for the discretisation pipeline.

Houses the Knuth (2006) optimal-bin-count rule and the Scargle (2013) Bayesian
Blocks change-point binner -- both reimplemented from primary sources after
astropy was dropped from the install graph -- plus the ``histogram`` shim that
routes ``bins='knuth'`` / ``bins='blocks'`` to them and the lower-level edge
helpers (`get_binning_edges`, `edges`, `discretize_sklearn`).
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
from numba import njit

from mlframe.core.arrays import arrayMinMax

logger = logging.getLogger(__name__)


@njit(nogil=True, cache=True)
def _knuth_log_posterior(M: int, n: int, counts: np.ndarray) -> float:
    """Knuth (2006) log-posterior for M equal-width bins given counts per bin.

    P(M | D, I) ∝ N log M + log Γ(M/2) - M log Γ(1/2) - log Γ(N + M/2)
                + Σ_k log Γ(n_k + 1/2)

    Implementation uses ``math.lgamma`` so it stays njit-friendly.
    Reference: Knuth, K.H. (2006) "Optimal data-based binning for histograms",
    arXiv:physics/0605197.
    """
    if M < 1 or n < 1:
        return -1e300
    log_M = math.log(M)
    log_gamma_half = math.lgamma(0.5)
    s = n * log_M + math.lgamma(M / 2.0) - M * log_gamma_half - math.lgamma(n + M / 2.0)
    for k in range(M):
        s += math.lgamma(counts[k] + 0.5)
    return s


@njit(nogil=True, cache=True)
def _knuth_best_M(a_sorted: np.ndarray, a_min: float, a_max: float, M_max: int) -> int:
    """Fused Knuth (2006) posterior search returning the optimal M in [2, M_max].

    BIT-IDENTICAL to the prior ``for M: np.histogram(a, linspace(a_min, a_max, M+1)) ->
    _knuth_log_posterior`` scan, but runs entirely in compiled code on the pre-sorted column:
    uniform-bin counts are obtained by integer differencing of ``np.searchsorted`` positions
    (``side='right'`` reproduces ``np.histogram``'s half-open ``[e_j, e_{j+1})`` bins with the final
    bin closed at ``a_max``), and the lgamma log-posterior is accumulated inline -- no per-M
    ``np.histogram`` dispatch and no ``counts.astype(int64)`` copy. ~6-47x over the object-mode loop
    (n=2k..50k) at zero numeric change to ``best_M``; bench discretization/_benchmarks/bench_knuth_posterior_fused.py.
    """
    n = a_sorted.shape[0]
    # Degenerate-input guard, replicated from the _knuth_bin_edges wrapper because this kernel is a
    # public re-export and may be called directly. With a_max<=a_min the bin width is 0, every
    # searchsorted lands at n, so all but the first bin get 0 counts and the posterior search returns
    # an arbitrary M; with n<1 the lgamma/log terms are meaningless. Return the minimum sensible M=2
    # (a single median split downstream) instead.
    if n < 1 or not (a_max > a_min):
        return 2
    log_gamma_half = math.lgamma(0.5)
    best_M = 2
    best_logp = -1e300
    for M in range(2, M_max + 1):
        width = (a_max - a_min) / M
        prev = 0
        s = n * math.log(M) + math.lgamma(M / 2.0) - M * log_gamma_half - math.lgamma(n + M / 2.0)
        for j in range(M):
            if j == M - 1:
                hi = n
            else:
                hi = np.searchsorted(a_sorted, a_min + (j + 1) * width, side="right")
            s += math.lgamma((hi - prev) + 0.5)
            prev = hi
        if s > best_logp:
            best_logp = s
            best_M = M
    return best_M


def _knuth_bin_edges(a: np.ndarray, edge_type: str = "quantile", m_max_cap: int = 64) -> np.ndarray:
    """Knuth's optimal-bin-count rule (Knuth 2006). Returns bin edges at the M*
    that maximises the log-posterior over M in [1, min(sqrt(N)*4, m_max_cap)].

    Args:
        a: 1-D continuous data, finite values used; NaN/inf skipped.
        edge_type: Type of edges to emit at the chosen M.
            ``'uniform'`` (legacy): equal-width edges matching Knuth's posterior
            likelihood model. Faithful but wastes resolution on skewed tails.
            ``'quantile'`` (2026-05-29 fix): quantile edges at the Knuth-optimal
            M. Empirically closes most of the bench gap to FD on heavy-tailed
            data. Per-audit recommendation; preserves M selection (Knuth's
            actual contribution) while routing edges through equal-frequency
            spacing.
        m_max_cap: Upper bound on M evaluated by the posterior loop. Default
            500 mirrors pre-fix behaviour; ``64`` (audit recommendation) keeps
            MI plug-in in the low-bias regime on small val-folds while not
            disturbing posterior shape on small data.

    Reference: Knuth, K.H. (2006) "Optimal data-based binning for histograms",
    arXiv:physics/0605197.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    n = a.size
    if n < 2:
        return np.array([a.min() if n else 0.0, a.max() + 1e-9 if n else 1.0])
    a_min, a_max = float(a.min()), float(a.max())
    if a_max <= a_min:
        return np.array([a_min, a_min + 1e-9])
    M_max = int(min(max(4, int(np.sqrt(n) * 4)), int(m_max_cap)))
    # 2026-05-29 fix: M_min=2. The posterior favours M=1 on perfectly-uniform
    # data (max-entropy histogram is best-described by 1 bin), but M=1 -> 0
    # bins downstream -> 0 MI in MRMR even when the joint signal is large.
    # Forcing M >= 2 yields the same posterior optimum on non-uniform inputs
    # AND a useful 2-bin median split on uniform inputs. Bench regression:
    # uniform mean MI 0.0000 -> ~0.50 with this fix.
    best_M = _knuth_best_M(np.sort(a), a_min, a_max, M_max)
    if edge_type == "quantile":
        # Quantile edges at the Knuth-optimal M. Preserves the posterior's M
        # selection (the empirical Knuth contribution) while routing edges
        # through equal-frequency spacing - empirically closes ~half the
        # bench gap to FD on skewed / heavy-tailed distributions.
        quantiles = np.linspace(0.0, 100.0, best_M + 1)
        return np.asarray(np.nanpercentile(a, quantiles))
    return np.linspace(a_min, a_max, best_M + 1)


@njit(nogil=True, cache=True)
def _bayesian_blocks_inner(t: np.ndarray, ncp_prior: float) -> np.ndarray:
    """Scargle (2013) Bayesian Blocks core DP. O(N^2).

    Args:
        t: SORTED unique data points, length N.
        ncp_prior: prior on the number of change points (Scargle eq. 21).
    Returns:
        change_point_indices: int64 array of indices into ``t`` marking block boundaries.
    Reference: Scargle, J.D., Norris, J.P., Jackson, B., Chiang, J. (2013),
    "Studies in Astronomical Time Series Analysis. VI", ApJ 764:167. Event mode (events at t_i).
    """
    N = t.shape[0]
    # Cell boundaries: midpoints between consecutive points + extrapolated end caps.
    edges = np.empty(N + 1, dtype=np.float64)
    edges[0] = t[0]
    for i in range(1, N):
        edges[i] = 0.5 * (t[i - 1] + t[i])
    edges[N] = t[N - 1]
    block_length = edges[N] - edges
    # Relative floor for T_cp before the log: on tie-heavy data the midpoint differences collapse and
    # T_cp can be a tiny positive float (~1e-16), making log(N_cp/T_cp) explode and biasing the search
    # toward degenerate single-point blocks. Floor T_cp at a small fraction of the total span so a
    # tie-collapsed cell can never dominate. The total span is edges[N]-edges[0] = block_length[0].
    _span = block_length[0]
    _t_floor = 1e-12 * _span if _span > 0.0 else 0.0
    # DP: best[i] = max log-likelihood reachable ending at point i.
    best = np.full(N, -1e300, dtype=np.float64)
    last = np.zeros(N, dtype=np.int64)
    for R in range(N):
        # For each possible block start R+1..R+1, compute log-likelihood of single block.
        for cp in range(R + 1):
            # Block from cp to R inclusive contains R - cp + 1 points.
            T_cp = block_length[cp] - block_length[R + 1]
            N_cp = R - cp + 1
            if T_cp <= 0.0 or N_cp <= 0:
                continue
            if T_cp < _t_floor:
                T_cp = _t_floor
            # Event-mode fitness: N * (log(N / T) - 1).
            fit = N_cp * (math.log(N_cp / T_cp))
            prev = best[cp - 1] if cp > 0 else 0.0
            score = prev + fit - ncp_prior
            if score > best[R]:
                best[R] = score
                last[R] = cp
    # Backtrack.
    cps = []
    R = N - 1
    while R >= 0:
        cps.append(last[R])
        R = last[R] - 1
    cps_arr = np.empty(len(cps), dtype=np.int64)
    for i, v in enumerate(cps):
        cps_arr[len(cps) - 1 - i] = v
    return cps_arr


@njit(nogil=True, cache=True)
def _bayesian_blocks_midpoints(t_sorted: np.ndarray) -> np.ndarray:
    """Build the cell-boundary midpoint array used by the canonical Scargle 2013 /
    astropy convention. ``edges[0]=t[0]``, ``edges[N]=t[N-1]``, internal points
    are ``0.5*(t[i-1]+t[i])``."""
    N = t_sorted.shape[0]
    edges = np.empty(N + 1, dtype=np.float64)
    edges[0] = t_sorted[0]
    for i in range(1, N):
        edges[i] = 0.5 * (t_sorted[i - 1] + t_sorted[i])
    edges[N] = t_sorted[N - 1]
    return edges


def _bayesian_blocks_bin_edges(a: np.ndarray, p0: float = 0.05, edge_placement: str = "start", subsample_threshold: int = 0) -> np.ndarray:
    """Scargle (2013) Bayesian Blocks bin edges.

    Args:
        a: 1-D continuous data, finite values used.
        p0: false-alarm probability for detecting a change point. Default ``0.05``
            preserves pre-2026-05-29 behaviour (astropy time-series default).
            ``0.10`` is recommended by the audit for continuous-data binning -
            raises ncp_prior less aggressively, accepts more change points,
            yields finer bins suitable for MI scoring.
        edge_placement: ``'start'`` (legacy bug-compat) places edges at
            ``a_sorted[cp_idx]`` (first data point of each block). ``'midpoint'``
            (Scargle 2013 / astropy convention) places edges at cell-boundary
            midpoints between adjacent points. ``'midpoint'`` fixes a half-spacing
            bias toward the lower neighbour but changes binned-counts on tie-heavy
            distributions.
        subsample_threshold: If ``> 0`` and ``N > subsample_threshold``, fit
            the BB DP on a uniform sub-sample of size ``subsample_threshold``
            then map edges back to the full domain. Default ``0`` disables.
            BB DP is O(N^2); sub-sampling to 1000 drops 211 ms -> ~50 ms with
            minimal MI-scoring impact (bin edges only need 1/M quantile-grade
            precision on small val-folds).
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size < 2:
        return np.array([a.min() if a.size else 0.0, a.max() + 1e-9 if a.size else 1.0])
    a_sorted = np.sort(a)
    # Sub-sample fast path (audit recommendation; only triggers when explicitly enabled).
    if subsample_threshold > 0 and a_sorted.size > subsample_threshold:
        N_full = a_sorted.size
        step = max(1, N_full // subsample_threshold)
        a_sorted_dp = a_sorted[::step]
    else:
        a_sorted_dp = a_sorted
    N = float(a_sorted_dp.size)
    # Scargle eq. 21: ncp_prior = 4 - log(73.53 * p0 * N^-0.478). p0 is a false-alarm PROBABILITY in (0, 1); p0<=0
    # (a natural "no false alarms" choice) would make math.log raise a domain error, so validate the public knob.
    if not (0.0 < p0 < 1.0):
        raise ValueError(f"_bayesian_blocks_bin_edges: p0 (false-alarm probability) must be in (0, 1); got {p0}.")
    ncp_prior = 4.0 - math.log(73.53 * p0 * (N**-0.478))
    cp_idx = _bayesian_blocks_inner(a_sorted_dp, ncp_prior)
    # Build edges from change-point indices.
    if edge_placement == "midpoint":
        # Scargle 2013 / astropy convention: edges at cell-boundary midpoints.
        cell_edges = _bayesian_blocks_midpoints(a_sorted_dp)
        edges_internal = np.empty(cp_idx.size + 1, dtype=np.float64)
        for i in range(cp_idx.size):
            edges_internal[i] = cell_edges[int(cp_idx[i])]
        edges_internal[-1] = cell_edges[-1]
    else:
        # Legacy 'start' placement (pre-2026-05-29 behaviour).
        edges_internal = np.empty(cp_idx.size + 1, dtype=np.float64)
        for i, ci in enumerate(cp_idx):
            edges_internal[i] = a_sorted_dp[ci]
        edges_internal[-1] = a_sorted_dp[-1]
    # De-duplicate consecutive equal edges (occurs when many ties present).
    edges_internal = np.unique(edges_internal)
    if edges_internal[0] > a.min():
        edges_internal = np.concatenate([[a.min()], edges_internal])
    # 2026-05-29 fix: BB DP correctly returns 1 block on perfectly-uniform data
    # (no change points), which collapses to 0 inner edges -> 0 MI downstream.
    # Insert the median as a forced split so the binner always returns >= 2 bins.
    if edges_internal.size < 3:
        median = float(np.median(a_sorted))
        edges_internal = np.array([float(a.min()), median, float(a.max())])
        edges_internal = np.unique(edges_internal)
    return edges_internal


def histogram(a, bins="auto", **kwargs):
    """In-tree histogram supporting np.histogram's bin schemes + 'blocks' / 'knuth'.

    2026-05-28: astropy was removed from the install graph; 'blocks' and 'knuth'
    are now native numba-compiled implementations of Scargle 2013 / Knuth 2006
    respectively. Other bin schemes (int / 'auto' / 'fd' / 'doane' / 'scott' /
    'rice' / 'sqrt' / 'sturges') route through np.histogram unchanged.

    Returns ``(hist, edges)`` matching the np.histogram + astropy contract.
    """
    if bins == "knuth":
        edges = _knuth_bin_edges(np.asarray(a))
        hist, _ = np.histogram(a, bins=edges)
        return hist, edges
    if bins == "blocks":
        p0 = kwargs.pop("p0", 0.05)
        edges = _bayesian_blocks_bin_edges(np.asarray(a), p0=p0)
        hist, _ = np.histogram(a, bins=edges)
        return hist, edges
    return np.histogram(a, bins=bins, **kwargs)


def edges(arr, quantiles):
    # Wave 21 P0: use nanpercentile so NaN in arr doesn't poison every
    # bin edge. ``discretize_array`` calls this 6000+ times per FS fit
    # (per the module docstring); pre-fix any NaN-bearing column made
    # bin_edges all-NaN, then digitize / searchsorted silently bucketed
    # every row to bin 0 -- the entire discretised feature collapsed to
    # a constant with no upstream signal.
    bin_edges = np.asarray(np.nanpercentile(arr, quantiles))
    return bin_edges


@njit(cache=True)
def get_binning_edges(arr: np.ndarray, n_bins: int = 10, method: str = "uniform", min_value: Optional[float] = None, max_value: Optional[float] = None):
    """Numba-jitted binning-edge calculator. Used by ``discretize_2d_array`` (itself ``@njit(parallel=True)`` and cannot dispatch to object-mode helpers).

    Outside an njit context (single-column path via ``discretize_array``) prefer the inlined raw-numpy version -- ``np.percentile`` beats numba's njit
    equivalent at n >= ~5000.
    """
    if method == "uniform":
        if min_value is None or max_value is None:
            min_value, max_value = arrayMinMax(arr)
        bin_edges = np.linspace(min_value, max_value, n_bins + 1)
    elif method == "quantile":
        # Wave 21 P0: numba's njit doesn't expose np.nanpercentile, so we
        # filter NaN inline before delegating to np.percentile. Pre-fix any
        # NaN in arr poisoned every edge -> downstream digitize silently
        # bucketed all rows to bin 0. The mask path is array-allocate +
        # one pass, cheaper than the percentile sort that follows.
        _mask = ~np.isnan(arr)
        if _mask.all():
            arr_finite = arr
        else:
            arr_finite = arr[_mask]
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.asarray(np.percentile(arr_finite, quantiles))
    else:
        # Any method other than "uniform"/"quantile" previously fell through
        # leaving ``bin_edges`` unbound -> UnboundLocalError at the return. Fail
        # explicitly instead. njit supports raising with a literal message only
        # (no f-string interpolation of ``method`` inside nopython mode).
        raise ValueError("get_binning_edges: unknown binning method; expected 'uniform' or 'quantile'")
    return bin_edges


def discretize_sklearn(
    arr: np.ndarray, n_bins: int = 10, method: str = "uniform",
    min_value: Optional[float] = None, max_value: Optional[float] = None, dtype: type = np.int8,
) -> np.ndarray:
    """Lightweight numpy port of sklearn's ``KBinsDiscretizer``.
    ``np.searchsorted`` is faster un-jitted on contemporary numpy."""
    bins_edges = get_binning_edges(arr=arr, n_bins=n_bins, method=method, min_value=min_value, max_value=max_value)
    return np.searchsorted(bins_edges[1:-1], arr, side="right").astype(dtype)
