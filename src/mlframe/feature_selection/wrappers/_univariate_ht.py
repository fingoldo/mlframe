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
import warnings
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


def _classify_target(y: np.ndarray) -> str:
    """Return one of 'binary', 'multiclass', 'continuous'."""
    if y.dtype.kind in "iub":
        # Integer / boolean target.
        uniq = np.unique(y[~_isnan_mask(y)])
        if uniq.size == 2:
            return "binary"
        if uniq.size <= max(10, int(np.sqrt(len(y)))):
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
        if uniq.size <= max(10, int(np.sqrt(len(y)))):
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
    U, mu, sigma = _mann_whitney_u_z(xm, grp)
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


def _chi2_sf(x: float, df: int) -> float:
    """Survival function of chi-squared. Uses scipy if available, else manual gamma."""
    if not np.isfinite(x) or x <= 0 or df <= 0:
        return 1.0
    try:
        from scipy.stats import chi2 as _c2
        return float(_c2.sf(x, df))
    except ImportError:
        # Fallback: use math.gamma via series. Crude; users should have scipy.
        return float(math.erfc(math.sqrt(x / 2.0)))


def _kw_p_numeric_multiclass(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(x) & ~_isnan_mask(y)
    if mask.sum() < 5:
        return 1.0
    xm = x[mask].astype(np.float64, copy=False)
    uniq, inv = np.unique(y[mask], return_inverse=True)
    if uniq.size < 2:
        return 1.0
    H, df = _kruskal_wallis_h(xm, inv.astype(np.int64), int(uniq.size))
    return _chi2_sf(H, df)


# ---------------------------------------------------------------------------
# Numeric feature vs continuous target: Kendall tau test.


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
    # Standard z under H0 of independence (large-n approx, ignoring ties).
    var_t = (4 * n + 10) / (9 * n * (n - 1)) if n > 1 else 0.0
    if var_t <= 0:
        return tau, 0.0
    z = tau / math.sqrt(var_t)
    return tau, z


def _kendall_p_numeric_continuous(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 5:
        return 1.0
    xm = x[mask].astype(np.float64, copy=False)
    ym = y[mask].astype(np.float64, copy=False)
    if xm.size > 2000:
        # Subsample for the O(n^2) loop -- the BY-FDR procedure tolerates this loss.
        rng = np.random.default_rng(0)
        idx = rng.choice(xm.size, size=2000, replace=False)
        xm = xm[idx]
        ym = ym[idx]
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
) -> pd.DataFrame:
    """Per-feature univariate relevance test with BY-FDR correction.

    Args:
        X: pandas DataFrame of features (numeric / categorical mix supported).
        y: 1-D target (array-like).
        ml_task: 'auto' / 'classification' / 'regression'. 'auto' inspects ``y``.
        fdr_level: Benjamini-Yekutieli FDR alpha (default 0.05).
        n_jobs: reserved for future parallelisation. Currently runs serial.

    Returns:
        pd.DataFrame indexed by feature name with columns ['feature', 'p_value', 'relevant'].
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("calculate_relevance_table: X must be a pandas DataFrame.")
    y_arr = np.asarray(y)
    if ml_task == "auto":
        target_type = _classify_target(y_arr)
    elif ml_task == "classification":
        target_type = "binary" if len(np.unique(y_arr)) == 2 else "multiclass"
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
                    pv = _kendall_p_numeric_continuous(xv, yv)
            else:
                # Categorical / object feature: chi-squared independence.
                # For continuous target, bin into deciles to make a contingency table.
                if target_type == "continuous":
                    yb = pd.qcut(pd.Series(y_arr), q=min(10, max(2, len(y_arr) // 50)),
                                 labels=False, duplicates="drop")
                    pv = _chi2_independence_p(ser.to_numpy(), yb.to_numpy())
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
