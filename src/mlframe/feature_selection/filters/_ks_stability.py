"""Per-feature train/test KS-test stability filter.

A feature can pass the model-based adversarial-AUC diagnostic (the CLASSIFIER can't separate train from test
using it, or its contribution to the combined signal is small) yet still have a distribution that shifted
meaningfully between splits when examined ALONE. A 1st-place LANL-earthquake-prediction team's rule: only
keep a feature if its train-vs-test Kolmogorov-Smirnov test p-value clears a threshold (``<= 0.05`` rejects
the feature) -- a per-feature, distribution-based check that complements (not duplicates) adversarial
validation's model-based, joint-feature-set check.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def ks_stability_filter(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    p_value_threshold: float = 0.05,
) -> pd.DataFrame:
    """Per-feature two-sample KS test between ``train_df`` and ``test_df``; flag unstable features.

    Parameters
    ----------
    train_df, test_df
        Frames sharing the columns to screen; only finite values are compared per column.
    feature_cols
        Numeric columns to screen; defaults to every numeric column present in both frames.
    p_value_threshold
        A feature is flagged unstable when its KS p-value is AT OR BELOW this threshold (the source
        convention: ``p <= 0.05`` means the train/test distributions are significantly different, i.e. the
        feature is unstable and a candidate to drop).

    Returns
    -------
    pd.DataFrame
        One row per screened column: ``{"column", "ks_statistic", "p_value", "stable"}``, sorted by
        ``p_value`` ascending (most unstable first). ``stable=False`` marks features to consider dropping.
    """
    from scipy.stats import ks_2samp

    if feature_cols is None:
        feature_cols = [c for c in train_df.columns if c in test_df.columns and pd.api.types.is_numeric_dtype(train_df[c])]
    feature_cols = list(feature_cols)

    rows = []
    for col in feature_cols:
        train_vals = train_df[col].to_numpy(dtype=np.float64)
        test_vals = test_df[col].to_numpy(dtype=np.float64)
        train_vals = train_vals[np.isfinite(train_vals)]
        test_vals = test_vals[np.isfinite(test_vals)]
        if train_vals.size == 0 or test_vals.size == 0:
            rows.append({"column": col, "ks_statistic": np.nan, "p_value": np.nan, "stable": True})
            continue
        result = ks_2samp(train_vals, test_vals)
        p_value = float(result.pvalue)
        rows.append({"column": col, "ks_statistic": float(result.statistic), "p_value": p_value, "stable": p_value > p_value_threshold})

    report = pd.DataFrame(rows)
    return report.sort_values("p_value", ascending=True, na_position="last").reset_index(drop=True)


__all__ = ["ks_stability_filter"]
