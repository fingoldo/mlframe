"""``outlier_cap_or_missing``: winsorize outliers vs. treat-as-missing-then-impute, one togglable transform.

Source: bestpractice_feature-engineering-general.md section 4 -- IQR-based capping vs. converting outliers to
NaN then imputing, two established outlier-treatment recipes. mlframe's existing outlier handling
(``preprocessing.outliers.reject_outliers`` drops flagged rows; ``preprocessing.outlier_policy.apply_outlier_policy``
does a single fixed quantile-clip for non-tree models or an added score column for tree models) has neither a
capping-vs-missingness MODE TOGGLE nor a skewness-driven threshold rule -- this module is the genuinely missing
piece: per column, pick the outlier bound rule (mean +/- 3*std for near-symmetric data, IQR*1.5 for skewed data,
selected by a skewness test) then either clip values to that bound (``mode="cap"``) or replace them with NaN and
median-impute (``mode="missing_impute"``, composing with the sibling ``missing_indicator_pairing`` module for a
paired flag when the caller wants one).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew

# |skewness| below this is treated as "near-symmetric" -> mean +/- 3*std bound; above it the distribution is
# skewed enough that a symmetric bound would clip one tail far more aggressively than the other, so the more
# robust IQR*1.5 rule is used instead. 1.0 is the common textbook skewness-classification cutoff (|skew| > 1 =
# "highly skewed").
_SKEW_THRESHOLD = 1.0

# Symmetric heavy-tailed contamination (e.g. outliers injected on BOTH sides) can leave skewness near zero
# while still inflating the raw std enough that mean+/-3*std stops being a useful bound -- the std is computed
# FROM the very values it is meant to bound, so a few extreme points on each side can blow it out to where the
# resulting bound no longer catches anything (empirically verified: biz_value synthetic with symmetric +/-
# corruption originally produced a bound wide enough that capping did nothing and RMSE got WORSE than
# untreated). Guard by comparing the raw std against a robust sigma estimate from the IQR (IQR / 1.349 ~=
# std under normality); if the raw std is inflated well beyond that, fall back to the outlier-robust IQR rule
# regardless of skewness.
_STD_INFLATION_RATIO = 1.5


def _column_bounds(values: np.ndarray) -> Tuple[float, float]:
    """Return ``(lower, upper)`` outlier bounds for one column, rule chosen by skewness + std-inflation check."""
    finite = values[np.isfinite(values)]
    if finite.size < 4:
        return -np.inf, np.inf
    q1, q3 = np.percentile(finite, [25, 75])
    iqr = q3 - q1

    def _iqr_bounds() -> Tuple[float, float]:
        if iqr == 0.0:
            return -np.inf, np.inf
        return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)

    s = float(skew(finite))
    mean, std = float(finite.mean()), float(finite.std())
    if abs(s) > _SKEW_THRESHOLD or std == 0.0:
        return _iqr_bounds()
    robust_sigma = iqr / 1.349 if iqr > 0.0 else std
    if robust_sigma > 0.0 and std > robust_sigma * _STD_INFLATION_RATIO:
        return _iqr_bounds()
    return mean - 3.0 * std, mean + 3.0 * std


def outlier_cap_or_missing(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    mode: str = "cap",
) -> pd.DataFrame:
    """Winsorize (``mode="cap"``) or NaN-then-median-impute (``mode="missing_impute"``) outliers per column.

    Per column, the outlier bound rule is auto-selected by a skewness test: near-symmetric columns
    (|skewness| <= 1.0) use ``mean +/- 3*std``; skewed columns use ``IQR * 1.5`` (Tukey's fences), which is
    more robust to a heavy tail than a symmetric bound would be. Bounds are always computed from the values
    passed in ``df`` (fit-on-train discipline is the caller's responsibility -- pass only the train frame here
    and apply the returned bounds' equivalent transform to test data via a fitted pipeline if leakage-free
    train/test bounds are required).

    Parameters
    ----------
    df
        Source frame.
    columns
        Numeric columns to treat; defaults to all numeric columns.
    mode
        ``"cap"`` clips values to the computed bound (winsorization); ``"missing_impute"`` replaces
        out-of-bound values with NaN, then median-imputes (median computed AFTER the outlier-to-NaN swap, so
        it is not itself distorted by the outliers being replaced).

    Returns
    -------
    pd.DataFrame
        ``df`` (shallow copy) with each treated column's outliers capped or replaced+imputed in place.
    """
    if mode not in ("cap", "missing_impute"):
        raise ValueError(f"outlier_cap_or_missing: mode must be 'cap' or 'missing_impute'; got {mode!r}.")
    cols = list(columns) if columns is not None else [c for c in df.select_dtypes(include=[np.number]).columns]

    out = df.copy(deep=False)
    for col in cols:
        values = out[col].to_numpy(dtype=np.float64)
        lower, upper = _column_bounds(values)
        if not np.isfinite(lower) and not np.isfinite(upper):
            continue
        is_outlier = (values < lower) | (values > upper)
        if not is_outlier.any():
            continue
        if mode == "cap":
            out[col] = np.clip(values, lower, upper)
        else:
            treated = values.copy()
            treated[is_outlier] = np.nan
            median = np.nanmedian(treated)
            treated[is_outlier] = median
            out[col] = treated

    return out


__all__ = ["outlier_cap_or_missing"]
