"""Analytic large-n null for the MI permutation test (2026-06-16).

The mRMR confidence / debiasing step gates features by a PERMUTATION null: shuffle ``y`` many
times, measure how often the shuffled MI ties/beats the observed MI (the p-value), and average the
shuffled MIs (the null-mean bias floor). Profiling at scale (D:/Temp/bench_scaling: 400k fit) showed
this permutation null is the dominant large-n cost -- thousands of O(n) shuffles per FE scan, routed
to a cupy ``argsort`` permutation generator that was 72% of the 400k wall.

At large n the permutation null has an exact asymptotic form, so the shuffles are unnecessary:
  * plug-in MI of two INDEPENDENT discrete variables has the Miller-Madow bias
        E[MI_hat] approx (Bx - 1) * (By - 1) / (2 * N)        [nats]
    where Bx, By are the numbers of OCCUPIED bins -- this IS the permutation null mean.
  * the G-test / likelihood-ratio statistic ``2 * N * MI`` (MI in nats) is asymptotically
        chi-square with df = (Bx - 1) * (By - 1)
    under independence, so the permutation p-value approx ``chi2.sf(2 * N * MI, df)``.

Empirically validated against the actual permutation kernel (mi_direct, npermutations=64) across
n in {5k, 20k, 50k, 200k}: the analytic null mean matches the permutation null mean to 3+ digits
even at n=5000, and the analytic p reproduces the significance decision (signal -> ~0, noise -> high).
See D:/Temp/validate_analytic_null.py for the comparison table.

IMPORTANT validity conditions (the caller MUST gate on these):
  * MI must be RAW (nats), NOT symmetric-uncertainty-normalised -- the 2*N*MI ~ chi2 identity only
    holds for raw MI. Gate on ``not use_su_normalization()``.
  * Large n -- the asymptotic tightens with N. Default floor ``_ANALYTIC_NULL_MIN_N`` (env-tunable),
    above which the analytic path replaces permutations; below it the cheap permutation path runs
    unchanged (small-n behaviour byte-for-byte preserved).
"""
from __future__ import annotations

import os

import numpy as np
import numba
from numba import njit

try:  # scipy is a hard mlframe dep, but keep the import defensive so an env without it degrades.
    from scipy.special import gammaincc as _gammaincc  # chi2.sf(x, df) == gammaincc(df/2, x/2), ~20x cheaper per call
    # ``_chi2`` is consumed ONLY for the inverse-CDF FLOOR (``_chi2.ppf(quantile, df)``) in the
    # conditional/marginal permutation-null fallback and the CMI redundancy gate -- a rare per-fallback scalar,
    # NOT the hot vectorised ``sf`` path (that uses the gammaincc ufunc above). It MUST be exported here: both
    # consumers do ``from ._analytic_mi_null import _HAVE_CHI2, _chi2, ...`` under a bare ``except: _HAVE_CHI2 =
    # False``, so a missing ``_chi2`` silently disabled the ENTIRE analytic null (every call fell to the slow
    # permutation path). Regression-pinned in test_analytic_mi_null_chi2_export.
    from scipy.stats import chi2 as _chi2
    _HAVE_CHI2 = True
except Exception:  # pragma: no cover
    _HAVE_CHI2 = False
    _chi2 = None


def _chi2_sf(x, df):
    """``chi2.sf(x, df)`` via the regularized upper incomplete gamma ``gammaincc(df/2, x/2)`` -- BIT-IDENTICAL to
    ``scipy.stats.chi2.sf`` (verified maxdiff 0.0) but a direct C ufunc, skipping the ``_distn_infrastructure`` dispatch
    that made ``chi2.sf`` the #1 MRMR-screen hotspot (~20x scalar, ~170x vectorized). Accepts scalars or arrays."""
    return _gammaincc(df / 2.0, x / 2.0)


# Minimum n at which the analytic null replaces the permutation null. Validated tight from ~20k. Lowered
# 50k -> 25k (2026-07-03): the prospective-pair prevalence gate scores its candidates on a ~30k active
# subsample, so the old 50k floor gated the analytic OFF for that whole phase and every candidate ran the
# CPU permutation shuffle (permutation.py:parallel_mi_prange -- the #1 host caller on the F2 STRICT
# cProfile, ~11.7k shuffle invocations). At 30k the cells are dense (N/(Bx*By) ~ 75, far above the
# expected-cell-5 sparsity floor that STILL guards high-cardinality tables) so the chi-square/G-test
# asymptotic is accurate: a direct A/B at the pair shape (n=30k, Bx=By=20) matched the 200-shuffle
# permutation null_mean to ~4 digits with 0/12 significance-decision flips. 25k keeps a margin above the
# ~20k validated floor. The sparsity safe-condition is unchanged, so sparse/high-card tables still fall to
# the permutation path. Env-tunable (MLFRAME_MI_ANALYTIC_NULL_MIN_N); a KTC sweep can refine per host.
_ANALYTIC_NULL_MIN_N_DEFAULT = 25_000


def analytic_null_enabled() -> bool:
    """Off-switch: ``MLFRAME_MI_ANALYTIC_NULL=0`` forces the legacy permutation path everywhere."""
    return _HAVE_CHI2 and os.environ.get("MLFRAME_MI_ANALYTIC_NULL", "1").strip() not in ("0", "false", "False")


def analytic_null_min_n() -> int:
    """Minimum row count at/above which the analytic MI null replaces the permutation null.

    Reads ``MLFRAME_MI_ANALYTIC_NULL_MIN_N`` (positive int); falls back to the calibrated default
    when unset/invalid. Below this n the chi-square/G-test asymptotics are unreliable, so the
    permutation null is used instead."""
    raw = os.environ.get("MLFRAME_MI_ANALYTIC_NULL_MIN_N", "").strip()
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return _ANALYTIC_NULL_MIN_N_DEFAULT


# Minimum AVERAGE expected count per contingency cell for the chi-square approximation to be
# trustworthy. The G-test tail (and the Miller-Madow bias) degrade when cells are sparse -- the
# classic "expected count >= 5" rule. With Bx*By cells over N rows the average expected count is
# N/(Bx*By); below this floor the analytic null is NOT applicable and the caller must fall back to
# the (sparsity-correct) permutation test. Env-tunable. This is the safe-condition the n-only gate
# was missing: a fixed N threshold does not bound cardinality, so a high-cardinality raw feature can
# have sparse cells even at large N.
_ANALYTIC_NULL_MIN_EXPECTED_CELL_DEFAULT = 5.0


def _min_expected_cell() -> float:
    """Minimum per-cell expected count required for the analytic chi-square null to be valid (contingency-table sparsity gate); env-overridable via ``MLFRAME_MI_ANALYTIC_NULL_MIN_CELL``, falling back to the default on an unparsable or non-positive value."""
    raw = os.environ.get("MLFRAME_MI_ANALYTIC_NULL_MIN_CELL", "").strip()
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return _ANALYTIC_NULL_MIN_EXPECTED_CELL_DEFAULT


def analytic_null_applicable(n_rows: int, n_bins_x: int, n_bins_y: int) -> bool:
    """True when BOTH safe-conditions hold: n >= threshold AND the contingency cells are not sparse
    (average expected count N/(Bx*By) >= the min-cell floor). When False the caller must use the
    permutation test -- the chi-square approximation is unreliable on sparse / high-cardinality tables.
    """
    if int(n_rows) < analytic_null_min_n():
        return False
    cells = max(1, int(n_bins_x) * int(n_bins_y))
    return (float(n_rows) / cells) >= _min_expected_cell()


def analytic_mi_null(original_mi: float, n_rows: int, n_bins_x: int, n_bins_y: int) -> tuple[float, float]:
    """Return ``(null_mean, p_value)`` for the MI permutation test, computed analytically.

    ``original_mi`` MUST be raw MI in NATS (not SU-normalised). ``n_bins_x`` / ``n_bins_y`` are the
    numbers of OCCUPIED bins of x / y (i.e. ``len(freqs_x)`` / ``len(freqs_y)`` from ``merge_vars``).

    ``null_mean`` is the Miller-Madow plug-in bias ``(Bx-1)(By-1)/(2N)``; ``p_value`` is the G-test
    tail ``chi2.sf(2N*MI, df)``. Degenerate cases (df <= 0, N <= 0, no chi2) return ``(0.0, 1.0)`` --
    an uninformative feature is maximally non-significant, matching the permutation path's no-perm
    default.
    """
    df = (int(n_bins_x) - 1) * (int(n_bins_y) - 1)
    if df <= 0 or n_rows <= 0:
        return 0.0, 1.0
    null_mean = df / (2.0 * float(n_rows))
    if not _HAVE_CHI2 or original_mi <= 0.0:
        # no observed signal -> sits at/below its null -> non-significant.
        return null_mean, 1.0
    g_stat = 2.0 * float(n_rows) * float(original_mi)
    p_value = float(_chi2_sf(g_stat, df))
    # clamp into [0, 1] against any FP underflow/overflow at the extreme tail.
    if p_value < 0.0:
        p_value = 0.0
    elif p_value > 1.0:
        p_value = 1.0
    return null_mean, p_value


def analytic_mi_null_batch(raw_mi_row: np.ndarray, n_rows: int, bx_per_col: np.ndarray, by: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized ``analytic_mi_null`` over a whole candidate row -> ``(null_mean_row, p_values)``, each ``(p,)``.

    Bit-identical to looping the scalar ``analytic_mi_null`` per column (verified maxdiff 0.0) but issues ONE
    ``gammaincc`` over the significant candidates instead of ``p`` scalar ``chi2.sf`` calls -- ~170x on the row (this was
    the #1 MRMR-screen hotspot: 138k scalar ``chi2.sf`` calls). ``bx_per_col`` is per-column occupied bins; ``by`` the
    target's occupied bins; non-finite ``raw_mi`` is treated as 0 (non-significant), matching the scalar path.
    """
    bx = np.asarray(bx_per_col, dtype=np.float64)
    mi = np.where(np.isfinite(raw_mi_row), raw_mi_row, 0.0).astype(np.float64)
    df = (bx - 1.0) * (float(by) - 1.0)
    ok_df = (df > 0.0) & (n_rows > 0)
    null_mean_row = np.where(ok_df, df / (2.0 * float(n_rows)), 0.0)
    p_values = np.ones(bx.shape[0], dtype=np.float64)  # default 1.0 == non-significant
    sig = ok_df & (mi > 0.0) & _HAVE_CHI2
    if np.any(sig):
        g = 2.0 * float(n_rows) * mi[sig]
        p_values[sig] = np.clip(_chi2_sf(g, df[sig]), 0.0, 1.0)
    return null_mean_row, p_values


@njit(nogil=True, cache=True, parallel=True)
def _occupied_bins_per_col(disc_2d: np.ndarray, nt: int) -> np.ndarray:
    """Count OCCUPIED (distinct, non-negative) bin codes per column in one O(n*K) pass.

    Exact replacement for the per-column ``np.unique(disc_2d[:, k]).size`` loop, which sorted each
    length-n column (O(K * n log n)) only to count distinct codes. Candidate codes are low-cardinality
    non-negative integer bin labels, so a per-column presence array sized to ``max_code+1`` counts the
    occupied bins directly. Returns int64[K] of per-column occupied-bin counts (identical to np.unique).

    Parallelised by ROW CHUNK (not by column): each thread walks a CONTIGUOUS row block in cache-friendly
    row-major order into its own ``(K, M)`` presence matrix, then the (tiny) per-column merge ORs across
    threads. Column-parallel would stride ``disc_2d[i, k]`` by K and thrash cache (measured 0.48x at
    K=200); row-chunk keeps the sequential access AND parallelises -> 1.34x @300k/K=200, 1.39x @K=800,
    bit-identical at every K. ``M = max_code+1`` is tiny (quantile bin codes: nbins + sentinel ~<=21).

    ``nt`` (thread count) is a CALLER-supplied argument, not queried internally via
    ``numba.get_num_threads()``: that call reads numba's runtime-mutable threading-layer state, which numba
    treats as a "dynamic global" and refuses to disk-cache the function for (warned every fresh process:
    "Cannot cache compiled function ... uses dynamic globals") -- paying a full LLVM recompile on every
    process launch despite ``cache=True``. Passing the thread count as a plain int argument removes the
    dynamic global and restores real disk-cache hits (verified: no cache warning with this form)."""
    n, K = disc_2d.shape
    out = np.zeros(K, dtype=np.int64)
    if n == 0:
        return out
    gmax = -1
    for i in range(n):
        for k in range(K):
            v = int(disc_2d[i, k])
            if v > gmax:
                gmax = v
    if gmax < 0:
        return out
    M = gmax + 1
    local = np.zeros((nt, K, M), dtype=np.bool_)
    chunk = (n + nt - 1) // nt
    for t in numba.prange(nt):
        lo = t * chunk
        hi = min(lo + chunk, n)
        for i in range(lo, hi):
            for k in range(K):
                v = int(disc_2d[i, k])
                if v >= 0:
                    local[t, k, v] = True
    for k in numba.prange(K):
        c = 0
        for v in range(M):
            for t in range(nt):
                if local[t, k, v]:
                    c += 1
                    break
        out[k] = c
    return out


def analytic_batch_noise_gate(
    disc_2d: "np.ndarray | None",
    observed_mi: np.ndarray,
    classes_y: np.ndarray,
    n_rows: int,
    min_nonzero_confidence: float,
    bx_per_col: "np.ndarray | None" = None,
    by: "int | None" = None,
) -> np.ndarray:
    """Analytic large-n form of the batched FE-candidate permutation noise gate.

    The permutation gate rejects candidate ``k`` (sets ``fe_mi[k]=0``) when its permutation p-value
    ``nfailed/npermutations >= 1 - min_nonzero_confidence``, else keeps the observed MI. At large n the
    p-value is the G-test tail, so this reproduces the keep/reject decision WITHOUT any shuffles.

    ``observed_mi`` is the per-column ungated observed MI in NATS (compute it once via the CPU kernel
    with ``npermutations=0``). ``disc_2d`` is the (n, K) discretised candidate matrix (integer codes);
    occupied marginal bin counts drive each column's G-test df. Returns ``fe_mi[K]``.
    """
    observed = np.asarray(observed_mi, dtype=np.float64)
    # K from whichever the caller supplied: the device-resident FE pair-MI path passes disc_2d=None with a
    # precomputed bx_per_col (the (n,K) codes never leave the device), so deriving K from disc_2d.shape would
    # AttributeError and silently drop the whole GPU branch to the CPU fallback (the "GPU never helped" mode).
    if disc_2d is not None:
        K = int(disc_2d.shape[1])
    elif bx_per_col is not None:
        K = int(np.asarray(bx_per_col).shape[0])
    else:
        raise ValueError("analytic_batch_noise_gate requires disc_2d or bx_per_col")
    fe_mi = observed.copy()
    alpha_reject = 1.0 - float(min_nonzero_confidence)  # reject when analytic p >= this
    # Occupied y-category count. When the caller already has it (the dispatcher's ``np.count_nonzero(freqs_y)``,
    # O(nbins)), take it -- avoids an O(n log n) ``np.unique`` over all n target rows recomputed on EVERY
    # per-pair-batch dispatch (thousands of calls, classes_y fit-invariant).
    by = int(by) if by is not None else int(np.unique(np.asarray(classes_y)).size)
    # Per-column occupied-bin counts in one O(n*K) njit pass (was O(K * n log n): a np.unique sort per
    # column just to count distinct codes). Bit-identical for non-negative bin codes -- see _occupied_bins_per_col.
    # ``bx_per_col`` may be PRECOMPUTED by the caller (the device-resident FE pair-MI path counts occupied bins ON
    # device from the resident codes, so the (n,K) code matrix never crosses the bus just to run this host njit).
    if bx_per_col is None:
        bx_per_col = _occupied_bins_per_col(np.ascontiguousarray(disc_2d), numba.get_num_threads())
    else:
        bx_per_col = np.asarray(bx_per_col, dtype=np.int64)
    # VECTORISED (2026-07-03): the per-column loop below previously called scipy ``chi2.sf`` ONCE per
    # candidate -- K up to thousands, ~12k scalar calls per fit -- purely for Python-call overhead.
    # ``scipy.stats.chi2.sf`` is elementwise, so one array call reproduces every per-column p-value
    # bit-for-bit while dropping the loop. The keep/reject decision is IDENTICAL to the scalar
    # analytic_null_applicable + analytic_mi_null path: reject candidate k iff observed MI_k > 0 AND the
    # analytic null is applicable (n >= min_n AND avg expected cell N/(Bx*By) >= floor) AND the G-test tail
    # p_k >= alpha_reject; df<=0 or missing scipy -> analytic_mi_null returns p=1.0 (always rejected when
    # applicable), which the p=1.0 default below reproduces.
    _by = int(by)
    _bx = bx_per_col.astype(np.int64, copy=False)
    _df = (_bx - 1) * (_by - 1)  # (K,) G-test degrees of freedom
    _cells = np.maximum(1, _bx * _by)
    # OCCUPANCY CAVEAT (mrmr_critique N-F7, DOC): the G-test df uses the DECLARED bin counts (Bx, By), but
    # equi-frequency binning makes the actually-OCCUPIED Bx data-dependent on a tied/low-cardinality column (many rows
    # share a value -> fewer distinct bins than requested), so df is slightly overstated and the chi-square tail p is
    # mildly conservative there. The min-expected-cell floor below is the partial safeguard, not an exact occupancy
    # correction; not selection-altering in practice (the gate is conservative), documented for rigor.
    # PASS-THROUGH CAVEAT (mrmr_critique N-F8, DOC): a column where the analytic null is NOT applicable (n below min_n,
    # or a sparse joint below the expected-cell floor) is NOT permutation-tested here -- it keeps its raw MI ungated by
    # THIS gate and relies on the caller's conservative applicability gate (_pairs_dispatch) to have excluded it. A
    # per-column permutation fallback for the inapplicable case is a FUTURE hardening.
    _applicable = (int(n_rows) >= analytic_null_min_n()) & ((float(n_rows) / _cells) >= _min_expected_cell())  # (K,) == analytic_null_applicable per col
    _p = np.ones(K, dtype=np.float64)  # df<=0 / mi<=0 / no-chi2 -> p=1.0
    _use_chi2 = (_df > 0) & (fe_mi > 0.0)
    if _HAVE_CHI2 and bool(np.any(_use_chi2)):
        _g = 2.0 * float(n_rows) * fe_mi[_use_chi2]
        _p[_use_chi2] = np.clip(_chi2_sf(_g, _df[_use_chi2]), 0.0, 1.0)
    _reject = (fe_mi > 0.0) & _applicable & (_p >= alpha_reject)
    fe_mi[fe_mi <= 0.0] = 0.0
    fe_mi[_reject] = 0.0
    return np.asarray(fe_mi)


__all__ = [
    "analytic_mi_null",
    "analytic_batch_noise_gate",
    "analytic_null_enabled",
    "analytic_null_min_n",
    "analytic_null_applicable",
    "_occupied_bins_per_col",
]
