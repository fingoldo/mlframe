"""Native univariate hypothesis-test prescreen for RFECV (Wave 5, 2026-05-28).

Implements per-feature univariate significance tests against the target plus
Benjamini-Yekutieli FDR correction. Numpy / pandas only at the API surface;
heavy inner loops are numba-compiled (cache=True). No external dependencies.

Backend selection per (feature dtype, target dtype):
  - target binary / multiclass  + feature numeric    -> Mann-Whitney U / Kruskal-Wallis (per pair)
  - target binary / multiclass  + feature categorical -> chi-squared independence test
  - target continuous           + feature numeric    -> Kendall tau (rank correlation)
  - target continuous           + feature categorical -> one-way ANOVA (Kruskal-Wallis fallback)

Performance: heavy inner loops (rank computation, U-statistic accumulation) are
numba-compiled when ``numba`` is available; otherwise a numpy fallback is used.

API:
  ``calculate_relevance_table(X, y, ml_task='auto', fdr_level=0.05, n_jobs=1)``
  returns a pandas DataFrame indexed by feature name with columns
  ``['feature', 'p_value', 'relevant']``. Compatible with the
  call-site in ``_rfecv_fit._apply_prescreen``.
"""
from __future__ import annotations

import logging
import math
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    logger.warning(
        "_univariate_ht: numba is not available; falling back to pure-Python kernels "
        "(~10-30x slower on rank/U/H/tau). Install numba for the fast path."
    )
    def njit(*args, **kwargs):  # type: ignore
        if args and callable(args[0]):
            return args[0]
        def _dec(f):
            return f
        return _dec


def is_numba_active() -> bool:
    """Public probe for callers / tests that want to assert the fast path is live."""
    return _NUMBA_AVAILABLE


# ---------------------------------------------------------------------------
# Helpers


# L7 (Wave 5) fix: a true classification target almost never has more than a few
# dozen labels, so cap the multiclass branch absolutely (independent of n). The
# old max(10, sqrt(n)) threshold grew with n (1000 at n=1e6), mis-classifying a
# high-cardinality integer regression target (counts/ages/integer-coded ordinals
# with hundreds of distinct values) as 'multiclass' and routing it to Kruskal-
# Wallis with hundreds of near-singleton groups (meaningless H/p-values). We also
# keep the cardinality-ratio guard (<= 0.05 * n, mirroring _is_discrete_v2) so a
# moderate distinct count that is still a large fraction of the rows is treated as
# continuous. Callers with a genuine >50-class target should pass ml_task explicitly.
_MULTICLASS_MAX_LABELS = 50


def _is_multiclass_cardinality(n_unique: int, n_rows: int) -> bool:
    """Multiclass iff few distinct labels (absolute cap) AND cardinality << n_rows."""
    return (
        n_unique <= _MULTICLASS_MAX_LABELS
        and n_unique <= max(10, int(np.sqrt(max(n_rows, 1))))
        and n_unique <= 0.05 * n_rows
    )


def _classify_target(y: np.ndarray) -> str:
    """Return one of 'binary', 'multiclass', 'continuous'."""
    if y.dtype.kind in "iub":
        # Integer / boolean target.
        uniq = np.unique(y[~_isnan_mask(y)])
        if uniq.size == 2:
            return "binary"
        if _is_multiclass_cardinality(uniq.size, len(y)):
            return "multiclass"
        return "continuous"
    # Float target: check if it's all integer values in disguise.
    arr = y[~_isnan_mask(y)]
    if arr.size == 0:
        return "continuous"
    if np.all(arr == np.floor(arr)):
        uniq = np.unique(arr)
        if uniq.size == 2:
            return "binary"
        if _is_multiclass_cardinality(uniq.size, len(y)):
            return "multiclass"
    return "continuous"


def _isnan_mask(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind in "fc":
        return np.isnan(arr)
    return np.zeros(arr.shape, dtype=bool)


def _is_numeric_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)


def _benjamini_yekutieli(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Yekutieli (2001) FDR-controlling threshold.

    Returns a boolean mask of features rejected (relevant). The BY procedure
    is more conservative than Benjamini-Hochberg but valid under arbitrary
    p-value dependency structure - the right choice for feature-selection
    contexts where features can be heavily correlated.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    # BY harmonic-sum correction.
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, m + 1) / (m * c_m)) * alpha
    passing = ranked <= thresholds
    # Largest k that passes -> reject all rank <= k.
    if not passing.any():
        cutoff_idx = -1
    else:
        cutoff_idx = int(np.where(passing)[0].max())
    rejected = np.zeros(m, dtype=bool)
    if cutoff_idx >= 0:
        rejected[order[: cutoff_idx + 1]] = True
    return rejected


# ---------------------------------------------------------------------------
# Numeric feature vs binary / multiclass target: Mann-Whitney U (binary) /
# Kruskal-Wallis (multiclass).


@njit(cache=True)
def _rank_with_ties(x: np.ndarray) -> np.ndarray:
    """Mean-rank for tied values (matches scipy.stats.rankdata method='average').

    Stable: ties get the mean of the rank range they span. Returns float64 ranks 1-indexed.
    """
    n = x.shape[0]
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) * 0.5 + 1.0  # 1-indexed mean rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


@njit(cache=True)
def _rank_and_tiesum(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Mean-ranks AND the tie-correction sum (sum t^3 - t over tie groups) in a SINGLE argsort pass.

    The U / H kernels both need the average-ranks and the tie sum; computing them together avoids a second
    O(n log n) sort of the same array (the dominant per-feature cost on n>=1000 prescreens).
    """
    n = x.shape[0]
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    tie_sum = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) * 0.5 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        t = j - i + 1
        if t > 1:
            tie_sum += t * t * t - t
        i = j + 1
    return ranks, tie_sum


@njit(cache=True)
def _mann_whitney_u_z_v2(x: np.ndarray, group: np.ndarray) -> Tuple[float, float, float]:
    """Single-sort Mann-Whitney U (ranks + tie sum from one pass). Bit-identical to _mann_whitney_u_z."""
    n = x.shape[0]
    ranks, tie_sum = _rank_and_tiesum(x)
    n1 = 0
    rank_sum_1 = 0.0
    for i in range(n):
        if group[i] == 1:
            n1 += 1
            rank_sum_1 += ranks[i]
    n2 = n - n1
    if n1 == 0 or n2 == 0:
        return 0.0, 0.0, 0.0
    U1 = rank_sum_1 - n1 * (n1 + 1) / 2.0
    mu_U = n1 * n2 / 2.0
    if n > 1:
        var_U = (n1 * n2 / 12.0) * ((n + 1) - tie_sum / (n * (n - 1)))
    else:
        var_U = n1 * n2 * (n + 1) / 12.0
    if var_U <= 0:
        return U1, mu_U, 0.0
    return U1, mu_U, math.sqrt(var_U)


@njit(cache=True)
def _mann_whitney_u_z(x: np.ndarray, group: np.ndarray) -> Tuple[float, float, float]:
    """Mann-Whitney U two-sided z-statistic with tie correction.

    Returns (U_stat, mu_U, sigma_U). p-value is 2 * (1 - Phi(|z|)).
    Caller computes Phi via math.erf or scipy because numba-compatible erf
    is available via math.erf.
    """
    n = x.shape[0]
    ranks = _rank_with_ties(x)
    n1 = 0
    rank_sum_1 = 0.0
    for i in range(n):
        if group[i] == 1:
            n1 += 1
            rank_sum_1 += ranks[i]
    n2 = n - n1
    if n1 == 0 or n2 == 0:
        return 0.0, 0.0, 0.0
    U1 = rank_sum_1 - n1 * (n1 + 1) / 2.0
    mu_U = n1 * n2 / 2.0
    # Tie-corrected variance:
    # var_U = n1*n2/12 * ((n+1) - sum(t_k^3 - t_k) / (n*(n-1)))
    # We need to enumerate tie groups; do a small accumulator.
    order = np.argsort(x, kind="mergesort")
    tie_sum = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        t = j - i + 1
        if t > 1:
            tie_sum += t * t * t - t
        i = j + 1
    if n > 1:
        var_U = (n1 * n2 / 12.0) * ((n + 1) - tie_sum / (n * (n - 1)))
    else:
        var_U = n1 * n2 * (n + 1) / 12.0
    if var_U <= 0:
        return U1, mu_U, 0.0
    return U1, mu_U, math.sqrt(var_U)


def _normal_two_sided_p(z: float) -> float:
    if not np.isfinite(z):
        return 1.0
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def _mann_whitney_p_numeric_binary(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(x) & ~_isnan_mask(y)
    if mask.sum() < 5:
        return 1.0
    xm = x[mask].astype(np.float64, copy=False)
    # Binarise y into {0, 1} by comparison against the unique-sorted values.
    uniq = np.unique(y[mask])
    if uniq.size != 2:
        return 1.0
    grp = (y[mask] == uniq[1]).astype(np.int64)
    U, mu, sigma = _mann_whitney_u_z_v2(xm, grp)
    if sigma <= 0:
        return 1.0
    z = (U - mu) / sigma
    return _normal_two_sided_p(z)


@njit(cache=True)
def _kruskal_wallis_h(x: np.ndarray, group: np.ndarray, n_groups: int) -> Tuple[float, int]:
    """Kruskal-Wallis H statistic with tie correction. Returns (H, df=k-1)."""
    n = x.shape[0]
    ranks = _rank_with_ties(x)
    group_sizes = np.zeros(n_groups, dtype=np.int64)
    group_rank_sums = np.zeros(n_groups, dtype=np.float64)
    for i in range(n):
        g = group[i]
        if g >= 0:
            group_sizes[g] += 1
            group_rank_sums[g] += ranks[i]
    if n < 2:
        return 0.0, 0
    H_raw = 0.0
    for k in range(n_groups):
        if group_sizes[k] > 0:
            H_raw += (group_rank_sums[k] * group_rank_sums[k]) / group_sizes[k]
    H = 12.0 / (n * (n + 1)) * H_raw - 3.0 * (n + 1)
    # Tie correction
    order = np.argsort(x, kind="mergesort")
    tie_sum = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        t = j - i + 1
        if t > 1:
            tie_sum += t * t * t - t
        i = j + 1
    if n > 1:
        C = 1.0 - tie_sum / (n * n * n - n)
        if C > 0:
            H = H / C
    return H, n_groups - 1


@njit(cache=True)
def _kruskal_wallis_h_v2(x: np.ndarray, group: np.ndarray, n_groups: int) -> Tuple[float, int]:
    """Single-sort Kruskal-Wallis H (ranks + tie sum from one pass). Bit-identical to _kruskal_wallis_h."""
    n = x.shape[0]
    ranks, tie_sum = _rank_and_tiesum(x)
    group_sizes = np.zeros(n_groups, dtype=np.int64)
    group_rank_sums = np.zeros(n_groups, dtype=np.float64)
    for i in range(n):
        g = group[i]
        if g >= 0:
            group_sizes[g] += 1
            group_rank_sums[g] += ranks[i]
    if n < 2:
        return 0.0, 0
    H_raw = 0.0
    for k in range(n_groups):
        if group_sizes[k] > 0:
            H_raw += (group_rank_sums[k] * group_rank_sums[k]) / group_sizes[k]
    H = 12.0 / (n * (n + 1)) * H_raw - 3.0 * (n + 1)
    if n > 1:
        C = 1.0 - tie_sum / (n * n * n - n)
        if C > 0:
            H = H / C
    return H, n_groups - 1


def _regularized_upper_gamma_q(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x).

    Numerical-Recipes gammq: series expansion for x < a+1, continued fraction
    otherwise. Used as the df-aware scipy-free fallback for the chi-squared SF.
    """
    if x <= 0.0:
        return 1.0
    if a <= 0.0:
        return 0.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        # Series representation of the lower incomplete gamma P(a, x).
        ap = a
        term = 1.0 / a
        total = term
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        p = total * math.exp(-x + a * math.log(x) - gln)
        return 1.0 - p
    # Continued-fraction (Lentz) for the upper incomplete gamma Q(a, x).
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h * math.exp(-x + a * math.log(x) - gln)


def _chi2_sf(x: float, df: int) -> float:
    """Survival function of chi-squared. Uses scipy if available, else manual gamma."""
    if not np.isfinite(x) or x <= 0 or df <= 0:
        return 1.0
    try:
        from scipy.stats import chi2 as _c2
        return float(_c2.sf(x, df))
    except ImportError:
        # df-aware fallback: chi-squared SF == Q(df/2, x/2). The previous
        # erfc(sqrt(x/2)) collapsed every df to the df=1 case, producing wrong
        # p-values (Kruskal-Wallis/chi-squared independence have df = k-1 or
        # (r-1)(c-1) >> 1), corrupting the BY-FDR prescreen.
        return float(_regularized_upper_gamma_q(df / 2.0, x / 2.0))


def _kw_p_numeric_multiclass(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(x) & ~_isnan_mask(y)
    if mask.sum() < 5:
        return 1.0
    xm = x[mask].astype(np.float64, copy=False)
    uniq, inv = np.unique(y[mask], return_inverse=True)
    if uniq.size < 2:
        return 1.0
    H, df = _kruskal_wallis_h_v2(xm, inv.astype(np.int64), int(uniq.size))
    return _chi2_sf(H, df)


# ---------------------------------------------------------------------------
# Numeric feature vs continuous target: Kendall tau test.


@njit(cache=True)
def _tie_sums(v: np.ndarray) -> Tuple[float, float, float]:
    """Tie-group sums for Kendall's tie-corrected variance: returns (sum t(t-1), sum t(t-1)(t-2), sum t(t-1)(2t+5)) over the
    distinct-value group sizes t of ``v``. Numba-friendly sorted pass (avoids np.unique(return_counts=), which numba can't type)."""
    n = v.shape[0]
    if n == 0:
        return 0.0, 0.0, 0.0
    vs = np.sort(v)
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    run = 1.0
    for i in range(1, n):
        if vs[i] == vs[i - 1]:
            run += 1.0
        else:
            s0 += run * (run - 1.0)
            s1 += run * (run - 1.0) * (run - 2.0)
            s2 += run * (run - 1.0) * (2.0 * run + 5.0)
            run = 1.0
    s0 += run * (run - 1.0)
    s1 += run * (run - 1.0) * (run - 2.0)
    s2 += run * (run - 1.0) * (2.0 * run + 5.0)
    return s0, s1, s2


@njit(cache=True)
def _kendall_tau_z(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Kendall tau-b with tie correction. Returns (tau, z-statistic).

    O(n^2) reference impl. For p<=200 this is fine; for huge n the user can
    fall back to scipy's mergesort-O(n log n) via the scipy backend.
    """
    n = x.shape[0]
    concordant = 0
    discordant = 0
    tx = 0
    ty = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
                continue
            if dy == 0:
                ty += 1
                continue
            if (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                concordant += 1
            else:
                discordant += 1
    n_pairs = n * (n - 1) // 2
    denom_x = math.sqrt(n_pairs - tx)
    denom_y = math.sqrt(n_pairs - ty)
    if denom_x <= 0 or denom_y <= 0:
        return 0.0, 0.0
    tau = (concordant - discordant) / (denom_x * denom_y)
    if n < 3:
        return tau, 0.0
    # z under H0 of independence using the TIE-CORRECTED variance of S = C - D (Kendall 1945; same form scipy.stats.kendalltau
    # uses for its normal approximation). The previous (4n+10)/(9 n (n-1)) form is the NO-TIES variance of tau and gives the
    # wrong z/p on tied integer / low-cardinality features -- exactly the columns that feed the BY-FDR family here. We need the
    # per-value tie-group sizes of x and y to assemble v0/vt/vu/v1/v2. np.unique(return_counts=) is not numba-typed, so the tie
    # sums are accumulated from a sorted pass. _tie_sums returns (sum t(t-1), sum t(t-1)(t-2), sum t(t-1)(2t+5)).
    sx0, sx1, sx2 = _tie_sums(x)
    sy0, sy1, sy2 = _tie_sums(y)
    s = float(concordant - discordant)
    v0 = n * (n - 1.0) * (2.0 * n + 5.0)
    v1 = sx0 * sy0 / (2.0 * n * (n - 1.0))
    v2 = sx1 * sy1 / (9.0 * n * (n - 1.0) * (n - 2.0))
    var_s = (v0 - sx2 - sy2) / 18.0 + v1 + v2
    if var_s <= 0:
        return tau, 0.0
    z = s / math.sqrt(var_s)
    return tau, z


def _kendall_p_numeric_continuous(x: np.ndarray, y: np.ndarray, random_state: int = 0) -> float:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 5:
        return 1.0
    xm = x[mask].astype(np.float64, copy=False)
    ym = y[mask].astype(np.float64, copy=False)
    # Test at FULL n. Subsampling large features to 2000 rows before testing mixed heterogeneous effective-n p-values into a
    # single BY family -- a 2000-row p-value and a full-n p-value are NOT exchangeable, which loses power and breaks the
    # BY-FDR validity (the procedure assumes a homogeneous family). scipy.stats.kendalltau runs the tie-corrected tau-b in
    # O(n log n), so full-n is affordable; fall back to the O(n^2) reference (tie-corrected variance) only when scipy is
    # unavailable. random_state is retained for signature compatibility but no longer drives a subsample draw.
    try:
        from scipy.stats import kendalltau as _kendalltau

        _tau, _p = _kendalltau(xm, ym, variant="b", nan_policy="omit")
        if _p != _p:  # NaN (e.g. a constant column)
            return 1.0
        return float(_p)
    except ImportError:
        _, z = _kendall_tau_z(xm, ym)
        return _normal_two_sided_p(z)


# ---------------------------------------------------------------------------
# Categorical feature vs target: chi-squared independence test.


def _chi2_independence_p(x_cat: np.ndarray, y_cat: np.ndarray) -> float:
    """Pearson chi-squared p-value from a contingency table built on (x_cat, y_cat)."""
    if x_cat.size == 0 or y_cat.size == 0:
        return 1.0
    uniq_x = pd.unique(x_cat)
    uniq_y = pd.unique(y_cat)
    if len(uniq_x) < 2 or len(uniq_y) < 2:
        return 1.0
    table = pd.crosstab(pd.Series(x_cat), pd.Series(y_cat)).to_numpy(dtype=np.float64)
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    total = float(table.sum())
    if total <= 0:
        return 1.0
    expected = row_sums @ col_sums / total
    # Avoid division-by-zero on impossible expected cells.
    expected_safe = np.where(expected > 0, expected, 1e-12)
    # Cochran's rule: the asymptotic chi-squared is unreliable when expected cell counts fall below 5 (small
    # contingency tables then give inflated significance). Warn rather than silently trust the p-value; BY-FDR only
    # partially compensates. A Fisher-exact r x c fallback is the rigorous fix but needs scipy + is exponential in
    # table size, so it stays out of this hot prescreen path.
    n_low = int(np.sum(expected < 5.0))
    if n_low > 0 and n_low > 0.2 * expected.size:
        logger.warning(
            "chi2_independence: %d/%d expected cell(s) < 5 (Cochran's rule); the asymptotic chi-squared p-value "
            "is unreliable on this small/sparse contingency table.", n_low, expected.size)
    chi2 = float(np.sum((table - expected) ** 2 / expected_safe))
    df = (table.shape[0] - 1) * (table.shape[1] - 1)
    return _chi2_sf(chi2, df)


# ---------------------------------------------------------------------------
# Public API


def calculate_relevance_table(
    X: pd.DataFrame,
    y,
    *,
    ml_task: str = "auto",
    fdr_level: float = 0.05,
    n_jobs: int = 1,
    random_state: int = 0,
) -> pd.DataFrame:
    """Per-feature univariate relevance test with BY-FDR correction.

    Args:
        X: pandas DataFrame of features (numeric / categorical mix supported).
        y: 1-D target (array-like).
        ml_task: 'auto' / 'classification' / 'regression'. 'auto' inspects ``y``.
        fdr_level: Benjamini-Yekutieli FDR alpha (default 0.05).
        n_jobs: reserved for future parallelisation. Currently runs serial.
        random_state: retained for signature/back-compat only. The Kendall-tau path now tests at FULL n via
            ``scipy.stats.kendalltau`` (O(n log n)); it no longer draws a subsample, so this seed has no effect.

    Returns:
        pd.DataFrame indexed by feature name with columns ['feature', 'p_value', 'relevant'].
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("calculate_relevance_table: X must be a pandas DataFrame.")
    y_arr = np.asarray(y)
    if ml_task == "auto":
        target_type = _classify_target(y_arr)
    elif ml_task == "classification":
        # Drop NaN before counting labels: a single NaN in a float-coded target otherwise inflates the unique
        # count and mis-routes a genuine binary target to the multiclass (Kruskal-Wallis) branch.
        target_type = "binary" if np.unique(y_arr[~_isnan_mask(y_arr)]).size == 2 else "multiclass"
    elif ml_task == "regression":
        target_type = "continuous"
    else:
        raise ValueError(f"ml_task must be 'auto'/'classification'/'regression'; got {ml_task!r}")

    p_values: list = []
    names: list = []
    for col in X.columns:
        ser = X[col]
        names.append(col)
        try:
            if _is_numeric_dtype(ser):
                xv = ser.to_numpy(dtype=np.float64, na_value=np.nan)
                if target_type == "binary":
                    pv = _mann_whitney_p_numeric_binary(xv, y_arr)
                elif target_type == "multiclass":
                    pv = _kw_p_numeric_multiclass(xv, y_arr)
                else:
                    yv = y_arr.astype(np.float64, copy=False)
                    pv = _kendall_p_numeric_continuous(xv, yv, random_state=random_state)
            else:
                # Categorical / object feature: chi-squared independence.
                # For continuous target, bin into deciles to make a contingency table.
                if target_type == "continuous":
                    yb = pd.qcut(pd.Series(y_arr), q=min(10, max(2, len(y_arr) // 50)),
                                 labels=False, duplicates="drop")
                    # ``duplicates="drop"`` (heavily-tied y) and NaN in y both leave NaN bins in ``yb``; pairing them
                    # against ``x_cat`` and dropping the NaN-bin rows from BOTH keeps the contingency table aligned
                    # (else crosstab silently drops only the NaN-y rows, mismatching the two series' lengths).
                    yb_arr = yb.to_numpy()
                    bin_ok = ~pd.isna(yb_arr)
                    pv = _chi2_independence_p(ser.to_numpy()[bin_ok], yb_arr[bin_ok])
                else:
                    pv = _chi2_independence_p(ser.to_numpy(), y_arr)
        except Exception as exc:
            logger.warning("univariate_ht: feature %r raised %s; assigning p=1.0", col, exc)
            pv = 1.0
        p_values.append(float(pv))

    p_arr = np.asarray(p_values, dtype=float)
    # Clamp into [0, 1] in case of small fp noise.
    p_arr = np.clip(p_arr, 0.0, 1.0)
    rejected = _benjamini_yekutieli(p_arr, alpha=float(fdr_level))
    out = pd.DataFrame({"feature": names, "p_value": p_arr, "relevant": rejected})
    out.index = pd.Index(names, name=None)
    return out
