"""Flag subgroups whose train/test prevalence ratio has shifted, as an overfit-risk signal.

A feature engineered from only ONE subgroup of the data (e.g. "average balance among revolving-loan
customers only") is exactly as reliable as that subgroup's prevalence is STABLE between train and test. If
the subgroup's share of the population shifts (train is 90% subgroup-A / 10% subgroup-B, test is 99%/1%),
a subgroup-only feature's apparent CV improvement is at real risk of being an artifact of the shifted mix
rather than genuine signal — the home-credit 5th place team diagnosed exactly this (a revolving-loan-only
feature that boosted CV 0.805->0.811 turned out to be tied to the cash/revolving prevalence mismatch and
was excluded). ``subpopulation_ratio_drift_check`` automates that diagnosis for any categorical column.
"""
from __future__ import annotations

import pandas as pd


def subpopulation_ratio_drift_check(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    subgroup_col: str,
    ratio_threshold: float = 2.0,
) -> pd.DataFrame:
    """Compare train vs test prevalence of every value of ``subgroup_col``; flag strong shifts.

    Parameters
    ----------
    train_df, test_df
        Frames both containing ``subgroup_col``.
    subgroup_col
        Categorical column defining the subgroups a subset-only feature might be built from (e.g. loan type,
        product tier, region).
    ratio_threshold
        A subgroup is flagged when ``max(train_prevalence, test_prevalence) / min(train_prevalence,
        test_prevalence)`` exceeds this. Default ``2.0`` (one side's share is more than double the other's)
        matches the magnitude of the diagnosed home-credit case (~90/10 train vs ~99/1 test, a ~9x-11x ratio
        on the minority subgroup).

    Returns
    -------
    pd.DataFrame
        One row per subgroup value: ``{"subgroup_value", "train_prevalence", "test_prevalence",
        "prevalence_ratio", "flagged"}``, sorted by ``prevalence_ratio`` descending. A value present in only
        one split gets ``prevalence_ratio = inf`` (maximally flagged, not a division error).
    """
    if subgroup_col not in train_df.columns or subgroup_col not in test_df.columns:
        raise ValueError(f"subpopulation_ratio_drift_check: subgroup_col={subgroup_col!r} must be present in both frames")

    train_prev = train_df[subgroup_col].value_counts(normalize=True)
    test_prev = test_df[subgroup_col].value_counts(normalize=True)
    all_values = sorted(set(train_prev.index.tolist()) | set(test_prev.index.tolist()), key=lambda v: (str(type(v)), v))

    rows = []
    for value in all_values:
        t_prev = float(train_prev.get(value, 0.0))
        e_prev = float(test_prev.get(value, 0.0))
        ratio = _prevalence_ratio(t_prev, e_prev)
        rows.append(
            {
                "subgroup_value": value,
                "train_prevalence": t_prev,
                "test_prevalence": e_prev,
                "prevalence_ratio": ratio,
                "flagged": ratio > ratio_threshold,
            }
        )

    report = pd.DataFrame(rows)
    return report.sort_values("prevalence_ratio", ascending=False).reset_index(drop=True)


def _prevalence_ratio(train_prevalence: float, test_prevalence: float) -> float:
    if train_prevalence <= 0.0 or test_prevalence <= 0.0:
        return float("inf")
    hi, lo = max(train_prevalence, test_prevalence), min(train_prevalence, test_prevalence)
    return hi / lo


__all__ = ["subpopulation_ratio_drift_check"]
