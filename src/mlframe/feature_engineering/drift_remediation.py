"""Auto-remediate adversarially-flagged drifting features via within-group rank transforms.

Composes two existing mlframe pieces rather than reimplementing either:

- ``reporting.charts.drift.adversarial_auc`` — trains a train-vs-test classifier and reports per-feature gain
  importance as a drift score (the Kaggle "will my CV transfer" diagnostic).
- ``feature_engineering.grouped.per_group_rank`` — within-group rank transform.

A feature that drifts in absolute LEVEL between train and test (e.g. a volume/count aggregate that grows
over time) but is still informative in its RELATIVE ordering within a natural grouping (e.g. rank among all
entities at the same ``time_id``) does not need to be dropped — converting it to a within-group rank strips
the level drift while keeping the relative signal, exactly the remediation Optiver's 1st-place solution
applied to `order_count`/`total_volume`. Dropping flagged features outright throws away real signal;
blanket-keeping them risks a model that overfits to the train-only regime.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def remediate_drifting_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str,
    feature_names: Sequence[str] | None = None,
    n_std: float = 1.0,
    rank_pct: bool = True,
    **adversarial_kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flag adversarially-drifting features and replace them with within-``group_col`` rank transforms.

    Parameters
    ----------
    train_df, test_df
        Feature frames sharing the same columns.
    group_col
        Column defining the natural grouping for the rank transform (e.g. ``time_id``, ``date``). Must be
        present in both frames; never itself flagged/remediated.
    feature_names
        Columns to scan for drift. Defaults to every shared column except ``group_col``.
    n_std
        A feature is flagged when its adversarial-validation gain importance exceeds
        ``mean(importances) + n_std * std(importances)`` — an anomaly threshold relative to the OTHER
        features' importance distribution, not an absolute magic number, so it adapts to how many features
        are scanned and how separable the dataset is overall.
    rank_pct
        Passed to ``per_group_rank`` — ``True`` returns normalised ``[0, 1]`` ranks (the default; comparable
        across groups of different sizes), ``False`` returns raw integer ranks.
    **adversarial_kwargs
        Forwarded to ``reporting.charts.drift.adversarial_auc`` (e.g. ``n_splits``, ``seed``, ``lgbm_params``).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_out, test_out, report)`` — copies of the input frames with flagged feature columns replaced
        by their within-``group_col`` rank, and a report frame with columns
        ``{"feature", "drift_importance", "flagged"}`` sorted by importance descending.
    """
    from mlframe.feature_engineering.grouped import per_group_rank
    from mlframe.reporting.charts.drift import adversarial_auc

    if group_col not in train_df.columns or group_col not in test_df.columns:
        raise ValueError(f"remediate_drifting_features: group_col={group_col!r} must be present in both frames")

    scan_cols = [c for c in feature_names] if feature_names is not None else [c for c in train_df.columns if c != group_col and c in test_df.columns]

    _, _, _, importances, names = adversarial_auc(train_df[scan_cols], test_df[scan_cols], feature_names=scan_cols, **adversarial_kwargs)
    names_list = list(names)
    importances_arr = np.asarray(importances, dtype=np.float64)

    threshold = float(importances_arr.mean() + n_std * importances_arr.std())
    flagged_mask = importances_arr > threshold

    train_out = train_df.copy()
    test_out = test_df.copy()
    train_groups = train_df[group_col].to_numpy()
    test_groups = test_df[group_col].to_numpy()
    for name, flagged in zip(names_list, flagged_mask):
        if not flagged:
            continue
        train_out[name] = per_group_rank(train_df[name].to_numpy(dtype=np.float64), train_groups, pct=rank_pct)
        test_out[name] = per_group_rank(test_df[name].to_numpy(dtype=np.float64), test_groups, pct=rank_pct)

    report = pd.DataFrame({"feature": names_list, "drift_importance": importances_arr, "flagged": flagged_mask})
    report = report.sort_values("drift_importance", ascending=False).reset_index(drop=True)
    return train_out, test_out, report


__all__ = ["remediate_drifting_features"]
