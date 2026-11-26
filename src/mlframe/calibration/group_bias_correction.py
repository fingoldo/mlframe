"""``fit_group_bias_correction``/``apply_group_bias_correction``: per-group multiplicative bias correction.

Source: 5th_m5-forecasting-accuracy.md - "I decided to create an average correction factor for each
store/department, based on the last week (validation)... instead of using a magic multiplier, I used a magic
multiplier for each store/department." A model can be well-calibrated on average but systematically
over/under-predict for specific segments (a store, a department, a regime) - a per-group ratio correction
fixes that residual bias cheaply, without retraining, using only a held-out validation slice.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _canonical_group_key_series(group: np.ndarray) -> pd.Series:
    """Stringify group labels the SAME way at fit and apply time, so an int64<->float64 dtype drift between
    the two calls (e.g. a stray NaN elsewhere in the same store_id column upcasting it between the validation
    run and a later scoring run) can't silently split one logical group into two different string keys
    (``3`` vs ``3.0``). A genuine NaN group label is preserved as NaN (not stringified) so it still falls
    through pandas' own dropna-groupby / dict-lookup-miss behavior unchanged.
    """
    arr = np.asarray(group)
    if np.issubdtype(arr.dtype, np.floating):
        is_nan = np.isnan(arr)
        # inf == floor(inf) is True, so infinities must be excluded here or their int64 cast is undefined (22-F40).
        is_whole = np.isfinite(arr) & (arr == np.floor(arr))
        safe_int_arr = np.where(is_whole, arr, 0).astype(np.int64)
        keys = np.where(is_whole, safe_int_arr.astype(str), arr.astype(str)).astype(object)
        keys[is_nan] = np.nan
        return pd.Series(keys)
    ser = pd.Series(arr)
    is_missing = ser.isna()
    keys_obj = ser.astype(str).astype(object)
    keys_obj[is_missing] = np.nan
    return keys_obj


def _finite_rows(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(y_true, y_pred, group)`` restricted to the rows where both targets are finite, warning when rows are dropped.

    One NaN prediction made the global mean NaN, so with shrinkage every group's ratio was NaN and fell back to 1.0 silently; a NaN inside a group's
    own column could do the same to that group (audit 22-F15).
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    gr = _canonical_group_key_series(group).to_numpy()
    ok = np.isfinite(yt) & np.isfinite(yp)
    n_bad = int((~ok).sum())
    if n_bad:
        logger.warning("fit_group_bias_correction: %d row(s) have a non-finite y_true or y_pred and are excluded from the fit.", n_bad)
        return yt[ok], yp[ok], gr[ok]
    return yt, yp, gr


def fit_group_bias_correction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
    min_group_size: int = 5,
    clip_range: Optional[Tuple[float, float]] = (0.5, 2.0),
    shrinkage_k: Optional[float] = None,
) -> Dict[str, float]:
    """Compute per-group ``mean(y_true) / mean(y_pred)`` correction ratios on a held-out validation slice.

    Parameters
    ----------
    y_true, y_pred
        ``(n,)`` validation-slice ground truth and model predictions.
    group
        ``(n,)`` group/segment label per row (e.g. store/department id).
    min_group_size
        Groups with fewer than this many validation rows get NO correction (ratio ``1.0``) - too few
        observations to trust a group-specific ratio; avoids overfitting to validation noise for rare
        segments. Ignored when ``shrinkage_k`` is set (shrinkage handles small groups continuously instead
        of this binary cutoff).
    clip_range
        ``(low, high)`` bounds on the correction ratio, or ``None`` for no clipping - guards against a
        near-zero ``mean(y_pred)`` producing an extreme multiplier.
    shrinkage_k
        Opt-in empirical-Bayes-style shrinkage strength. ``None`` (default) preserves the exact prior
        behavior (hard ``min_group_size`` cutoff, ratio ``1.0`` below it). When set, every group's raw ratio
        is blended toward the GLOBAL ``mean(y_true)/mean(y_pred)`` ratio with weight
        ``count / (count + shrinkage_k)`` - large groups (``count >> shrinkage_k``) keep ~their full raw
        ratio (real bias gets corrected), tiny groups (``count << shrinkage_k``) shrink toward the global
        ratio instead of overfitting a noisy few-row estimate. ``min_group_size`` is not applied in this
        mode; shrinkage itself is the small-group safeguard. Larger ``shrinkage_k`` shrinks harder; a
        reasonable starting point is the group size below which per-group ratios feel unstable (e.g. 10-50).

    Returns
    -------
    dict
        ``{group_value: correction_ratio}`` - store this and reapply via ``apply_group_bias_correction`` at
        inference; never recompute on rows without ground truth (that's what this validation-slice-only fit
        exists to avoid). Keys are canonicalized (see :func:`_canonical_group_key_series`) so a numeric
        group column that drifts int64<->float64 between the fit and apply calls still matches. Rows whose
        ``group`` is NaN are excluded from the table (logged as a warning) and fall back to ``default_ratio``
        at apply time.
    """
    n_nan_groups = int(pd.isna(pd.Series(group)).sum())
    if n_nan_groups:
        logger.warning(
            "fit_group_bias_correction: %d row(s) have a NaN group label and are excluded from the fitted "
            "correction table (pandas groupby dropna=True); those rows will silently receive default_ratio "
            "at apply time unless handled upstream.",
            n_nan_groups,
        )

    y_true_arr, y_pred_arr, group_arr = _finite_rows(y_true, y_pred, group)
    df = pd.DataFrame({"y_true": y_true_arr, "y_pred": y_pred_arr, "group": group_arr})
    # A per-group loop calling sub["y_true"].mean()/sub["y_pred"].mean() separately pays pandas per-group
    # column-access + reduction overhead TWICE per group (measured as the dominant cProfile cost at
    # n_groups=2000). A single groupby(...).agg(["mean", "count"]) computes both means and the group size for
    # EVERY group in one vectorized pass, then the ratio/clip/min-size logic is cheap elementwise arithmetic
    # over just the (small) per-group aggregate table.
    agg = df.groupby("group", sort=False).agg(y_true_mean=("y_true", "mean"), y_pred_mean=("y_pred", "mean"), count=("y_true", "size"))

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_ratio = np.where(agg["y_pred_mean"].to_numpy() != 0, agg["y_true_mean"].to_numpy() / agg["y_pred_mean"].to_numpy(), 1.0)

    if shrinkage_k is not None:
        global_pred_mean = y_pred_arr.mean()
        global_ratio = float(y_true_arr.mean() / global_pred_mean) if global_pred_mean != 0 else 1.0
        counts = agg["count"].to_numpy(dtype=np.float64)
        weight = counts / (counts + shrinkage_k)
        final_ratio = weight * raw_ratio + (1.0 - weight) * global_ratio
        if clip_range is not None:
            final_ratio = np.clip(final_ratio, clip_range[0], clip_range[1])
    else:
        if clip_range is not None:
            raw_ratio = np.clip(raw_ratio, clip_range[0], clip_range[1])
        final_ratio = np.where(agg["count"].to_numpy() >= min_group_size, raw_ratio, 1.0)

    return {str(group_value): float(ratio) for group_value, ratio in zip(agg.index, final_ratio)}


def apply_group_bias_correction(y_pred: np.ndarray, group: np.ndarray, ratios: Dict[str, float], default_ratio: float = 1.0) -> np.ndarray:
    """Apply previously-fitted per-group ratios: ``corrected = y_pred * ratios.get(group, default_ratio)``.

    ``group`` is canonicalized the same way ``fit_group_bias_correction`` keyed ``ratios`` (see
    :func:`_canonical_group_key_series`), so an int64<->float64 dtype drift in the group column between the
    fit and apply calls does not silently miss every lookup.
    """
    # A per-row Python dict.get() list comprehension is a real bottleneck at large n (measured as the
    # dominant cost at n=500k). pandas.Series.map with a dict does the same lookup via a vectorized C-level
    # hashtable pass instead of a Python-level loop.
    group_arr = _canonical_group_key_series(group)
    n_nan_groups = int(pd.isna(group_arr).sum())
    if n_nan_groups:
        logger.warning(
            "apply_group_bias_correction: %d row(s) have a NaN group label and will receive default_ratio=%s.",
            n_nan_groups,
            default_ratio,
        )
    ratio_lookup = group_arr.map(ratios).fillna(default_ratio).to_numpy(dtype=np.float64)
    return np.asarray(np.asarray(y_pred, dtype=np.float64) * ratio_lookup)


__all__ = ["fit_group_bias_correction", "apply_group_bias_correction"]
