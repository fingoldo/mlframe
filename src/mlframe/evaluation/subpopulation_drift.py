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

from typing import Sequence, Union

import pandas as pd


def subpopulation_ratio_drift_check(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    subgroup_col: str,
    ratio_threshold: float = 2.0,
    include_severity_score: bool = False,
) -> pd.DataFrame:
    """Compare train vs test prevalence of every value of ``subgroup_col``; flag strong shifts.

    Any cardinality of ``subgroup_col`` is supported natively (binary or multi-way categorical) -- one row is
    produced per distinct value seen in either split, regardless of how many there are.

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
    include_severity_score
        Opt-in (default ``False``, output unchanged when omitted). When ``True``, adds a
        ``"drift_severity_score"`` column: a continuous overfit-risk score in ``[0, 1)`` (see
        :func:`_drift_severity_score`), letting callers rank subgroup values -- and, via
        :func:`rank_subpopulation_drift_severity`, whole candidate columns -- by degree of drift rather than
        only a threshold pass/fail flag.

    Returns
    -------
    pd.DataFrame
        One row per subgroup value: ``{"subgroup_value", "train_prevalence", "test_prevalence",
        "prevalence_ratio", "flagged"}`` (plus ``"drift_severity_score"`` if ``include_severity_score``),
        sorted by ``prevalence_ratio`` descending. A value present in only one split gets
        ``prevalence_ratio = inf`` (maximally flagged, not a division error).
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
        row: dict[str, Union[str, float, bool]] = {
            "subgroup_value": value,
            "train_prevalence": t_prev,
            "test_prevalence": e_prev,
            "prevalence_ratio": ratio,
            "flagged": ratio > ratio_threshold,
        }
        if include_severity_score:
            row["drift_severity_score"] = _drift_severity_score(ratio)
        rows.append(row)

    report = pd.DataFrame(rows)
    return report.sort_values("prevalence_ratio", ascending=False).reset_index(drop=True)


def _prevalence_ratio(train_prevalence: float, test_prevalence: float) -> float:
    if train_prevalence <= 0.0 or test_prevalence <= 0.0:
        return float("inf")
    hi, lo = max(train_prevalence, test_prevalence), min(train_prevalence, test_prevalence)
    return hi / lo


def _drift_severity_score(ratio: float) -> float:
    """Map a prevalence ratio in ``[1, inf)`` to a graded overfit-risk severity in ``[0, 1)``.

    ``1 - 2 / (ratio + 1)`` is ``0.0`` at ``ratio=1`` (no shift), rises smoothly and concavely (so the
    difference between ratio 2 and 4 registers more than between ratio 20 and 22), and saturates to ``1.0``
    as ``ratio -> inf`` -- a continuous score for ranking, instead of only a threshold pass/fail flag.
    """
    if ratio == float("inf"):
        return 1.0
    return 1.0 - 2.0 / (ratio + 1.0)


def rank_subpopulation_drift_severity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    subgroup_cols: Sequence[str],
    ratio_threshold: float = 2.0,
) -> pd.DataFrame:
    """Rank multiple candidate subgroup columns by overall train/test drift severity.

    For each column in ``subgroup_cols`` (categorical, any cardinality -- binary or multi-way), runs
    :func:`subpopulation_ratio_drift_check` with ``include_severity_score=True`` and aggregates the per-value
    severity scores into a single column-level score: the prevalence-weighted average severity across all of
    the column's subgroup values (weight = average of train/test prevalence), so a shift concentrated in a
    populous subgroup dominates the score over noise in a near-empty one. Lets callers rank candidate
    subset-only features by overfit risk rather than only checking a single column pass/fail.

    Returns
    -------
    pd.DataFrame
        One row per column in ``subgroup_cols``: ``{"subgroup_col", "drift_severity_score",
        "max_prevalence_ratio", "any_flagged"}``, sorted by ``drift_severity_score`` descending (highest
        overfit risk first).
    """
    rows = []
    for col in subgroup_cols:
        report = subpopulation_ratio_drift_check(
            train_df, test_df, subgroup_col=col, ratio_threshold=ratio_threshold, include_severity_score=True
        )
        weights = (report["train_prevalence"] + report["test_prevalence"]) / 2.0
        total_weight = float(weights.sum())
        agg_score = float((report["drift_severity_score"] * weights).sum() / total_weight) if total_weight > 0.0 else 0.0
        rows.append(
            {
                "subgroup_col": col,
                "drift_severity_score": agg_score,
                "max_prevalence_ratio": float(report["prevalence_ratio"].max()),
                "any_flagged": bool(report["flagged"].any()),
            }
        )

    ranking = pd.DataFrame(rows)
    return ranking.sort_values("drift_severity_score", ascending=False).reset_index(drop=True)


__all__ = ["subpopulation_ratio_drift_check", "rank_subpopulation_drift_severity"]
