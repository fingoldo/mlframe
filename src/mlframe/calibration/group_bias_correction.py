"""``fit_group_bias_correction``/``apply_group_bias_correction``: per-group multiplicative bias correction.

Source: 5th_m5-forecasting-accuracy.md -- "I decided to create an average correction factor for each
store/department, based on the last week (validation)... instead of using a magic multiplier, I used a magic
multiplier for each store/department." A model can be well-calibrated on average but systematically
over/under-predict for specific segments (a store, a department, a regime) -- a per-group ratio correction
fixes that residual bias cheaply, without retraining, using only a held-out validation slice.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def fit_group_bias_correction(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray, min_group_size: int = 5, clip_range: Optional[Tuple[float, float]] = (0.5, 2.0)) -> Dict[str, float]:
    """Compute per-group ``mean(y_true) / mean(y_pred)`` correction ratios on a held-out validation slice.

    Parameters
    ----------
    y_true, y_pred
        ``(n,)`` validation-slice ground truth and model predictions.
    group
        ``(n,)`` group/segment label per row (e.g. store/department id).
    min_group_size
        Groups with fewer than this many validation rows get NO correction (ratio ``1.0``) -- too few
        observations to trust a group-specific ratio; avoids overfitting to validation noise for rare
        segments.
    clip_range
        ``(low, high)`` bounds on the correction ratio, or ``None`` for no clipping -- guards against a
        near-zero ``mean(y_pred)`` producing an extreme multiplier.

    Returns
    -------
    dict
        ``{group_value: correction_ratio}`` -- store this and reapply via ``apply_group_bias_correction`` at
        inference; never recompute on rows without ground truth (that's what this validation-slice-only fit
        exists to avoid).
    """
    df = pd.DataFrame({"y_true": np.asarray(y_true, dtype=np.float64), "y_pred": np.asarray(y_pred, dtype=np.float64), "group": group})
    # A per-group loop calling sub["y_true"].mean()/sub["y_pred"].mean() separately pays pandas per-group
    # column-access + reduction overhead TWICE per group (measured as the dominant cProfile cost at
    # n_groups=2000). A single groupby(...).agg(["mean", "count"]) computes both means and the group size for
    # EVERY group in one vectorized pass, then the ratio/clip/min-size logic is cheap elementwise arithmetic
    # over just the (small) per-group aggregate table.
    agg = df.groupby("group", sort=False).agg(y_true_mean=("y_true", "mean"), y_pred_mean=("y_pred", "mean"), count=("y_true", "size"))

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_ratio = np.where(agg["y_pred_mean"].to_numpy() != 0, agg["y_true_mean"].to_numpy() / agg["y_pred_mean"].to_numpy(), 1.0)
    if clip_range is not None:
        raw_ratio = np.clip(raw_ratio, clip_range[0], clip_range[1])
    final_ratio = np.where(agg["count"].to_numpy() >= min_group_size, raw_ratio, 1.0)

    return {str(group_value): float(ratio) for group_value, ratio in zip(agg.index, final_ratio)}


def apply_group_bias_correction(y_pred: np.ndarray, group: np.ndarray, ratios: Dict[str, float], default_ratio: float = 1.0) -> np.ndarray:
    """Apply previously-fitted per-group ratios: ``corrected = y_pred * ratios.get(group, default_ratio)``."""
    # A per-row Python dict.get() list comprehension is a real bottleneck at large n (measured as the
    # dominant cost at n=500k). pandas.Series.map with a dict does the same lookup via a vectorized C-level
    # hashtable pass instead of a Python-level loop.
    group_arr = pd.Series(group).astype(str)
    ratio_lookup = group_arr.map(ratios).fillna(default_ratio).to_numpy(dtype=np.float64)
    return np.asarray(np.asarray(y_pred, dtype=np.float64) * ratio_lookup)


__all__ = ["fit_group_bias_correction", "apply_group_bias_correction"]
