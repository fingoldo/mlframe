"""Cross-reference a subgroup-only feature's CV gain against that subgroup's train/test prevalence shift.

``subpopulation_ratio_drift_check`` (this package) already flags WHICH subgroups have shifted train/test
prevalence. This module adds the second half of the diagnosis: given a feature that was engineered from only
ONE subgroup's rows, and the CV delta that feature produced, flag it as "possible overfit to train/test mix
shift" rather than genuine signal whenever its source subgroup is itself flagged as shifted -- the home-credit
5th place team's exact reasoning for excluding a revolving-loan-only feature that boosted CV 0.805->0.811.
"""
from __future__ import annotations

import pandas as pd

from mlframe.evaluation.subpopulation_drift import subpopulation_ratio_drift_check


def flag_subgroup_only_feature_overfit_risk(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    subgroup_col: str,
    feature_subgroup_value: object,
    cv_delta: float,
    ratio_threshold: float = 2.0,
) -> dict:
    """Flag a subgroup-only feature's CV gain as overfit-risk when its subgroup's prevalence has shifted.

    Parameters
    ----------
    train_df, test_df
        Frames both containing ``subgroup_col`` (passed straight through to
        :func:`subpopulation_ratio_drift_check`).
    subgroup_col
        The categorical column defining the subgroup the feature was built from.
    feature_subgroup_value
        Which value of ``subgroup_col`` the feature is scoped to (e.g. ``"revolving_loans"``).
    cv_delta
        The CV score improvement attributed to this feature (any sign convention; only its presence/
        magnitude is reported, not judged against a fixed threshold -- the caller knows their own metric's
        scale).
    ratio_threshold
        Passed through to :func:`subpopulation_ratio_drift_check`.

    Returns
    -------
    dict
        ``subgroup_report_row`` (the matching row of the full drift report, or ``None`` if the subgroup
        value doesn't appear in either split), ``prevalence_ratio``, ``subgroup_shifted`` (bool),
        ``cv_delta``, ``overfit_risk_flag`` (bool -- ``True`` when the subgroup is shifted AND ``cv_delta``
        is nonzero, i.e. there IS a CV gain riding on a shifted-mix subgroup).
    """
    report = subpopulation_ratio_drift_check(train_df, test_df, subgroup_col, ratio_threshold=ratio_threshold)
    matching = report[report["subgroup_value"] == feature_subgroup_value]
    if matching.empty:
        return {
            "subgroup_report_row": None,
            "prevalence_ratio": float("nan"),
            "subgroup_shifted": False,
            "cv_delta": cv_delta,
            "overfit_risk_flag": False,
        }

    row = matching.iloc[0]
    subgroup_shifted = bool(row["flagged"])
    overfit_risk_flag = subgroup_shifted and cv_delta != 0.0

    return {
        "subgroup_report_row": row.to_dict(),
        "prevalence_ratio": float(row["prevalence_ratio"]),
        "subgroup_shifted": subgroup_shifted,
        "cv_delta": cv_delta,
        "overfit_risk_flag": overfit_risk_flag,
    }


__all__ = ["flag_subgroup_only_feature_overfit_risk"]
