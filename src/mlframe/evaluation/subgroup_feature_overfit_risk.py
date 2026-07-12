"""Cross-reference a subgroup-only feature's CV gain against that subgroup's train/test prevalence shift.

``subpopulation_ratio_drift_check`` (this package) already flags WHICH subgroups have shifted train/test
prevalence. This module adds the second half of the diagnosis: given a feature that was engineered from only
ONE subgroup's rows, and the CV delta that feature produced, flag it as "possible overfit to train/test mix
shift" rather than genuine signal whenever its source subgroup is itself flagged as shifted -- the home-credit
5th place team's exact reasoning for excluding a revolving-loan-only feature that boosted CV 0.805->0.811.
"""
from __future__ import annotations

from typing import Mapping, Sequence

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


def rank_subgroup_feature_overfit_risk(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidates: Sequence[Mapping[str, object]],
    ratio_threshold: float = 2.0,
) -> pd.DataFrame:
    """Rank multiple candidate subgroup-only features by overfit-risk severity in one call.

    Companion to :func:`flag_subgroup_only_feature_overfit_risk`, mirroring the boolean-flag ->
    severity-ranking upgrade that :func:`~mlframe.evaluation.subpopulation_drift.rank_subpopulation_drift_severity`
    already does for whole subgroup columns. Where the single-feature function only answers "is THIS feature
    at risk", this answers "which of MY candidate features should I drop/investigate FIRST" -- useful once a
    model has several subgroup-only features and only some CV budget to re-validate them.

    Parameters
    ----------
    train_df, test_df
        Frames both containing every ``subgroup_col`` referenced in ``candidates``.
    candidates
        One mapping per candidate feature, each with keys ``"feature_name"`` (any hashable label),
        ``"subgroup_col"``, ``"feature_subgroup_value"``, and ``"cv_delta"`` -- the same inputs
        :func:`flag_subgroup_only_feature_overfit_risk` takes per-call, batched here. Candidates may reuse
        the same ``subgroup_col`` (its drift report is computed once and reused across them).
    ratio_threshold
        Passed through to :func:`~mlframe.evaluation.subpopulation_drift.subpopulation_ratio_drift_check`
        for the ``subgroup_shifted``/``overfit_risk_flag`` columns; the ranking itself is driven by the
        continuous ``risk_score``, not this threshold.

    Returns
    -------
    pd.DataFrame
        One row per candidate: ``{"feature_name", "subgroup_col", "feature_subgroup_value", "cv_delta",
        "prevalence_ratio", "subgroup_shifted", "overfit_risk_flag", "drift_severity_score", "risk_score"}``,
        sorted by ``risk_score`` descending (highest overfit risk first). ``risk_score`` is
        ``drift_severity_score * abs(cv_delta)`` -- a candidate riding a badly-shifted subgroup with a large
        CV gain ranks above one with either a mild shift or a negligible gain; unlike ``drift_severity_score``
        alone it is NOT bounded to ``[0, 1)`` since ``cv_delta`` is caller-scaled. A candidate whose subgroup
        value doesn't appear in either split gets ``risk_score = 0.0`` (nothing to rank -- same convention as
        ``overfit_risk_flag=False`` in the single-feature function).
    """
    # cache per subgroup_col so candidates sharing a column don't re-run the drift check redundantly.
    reports: dict[str, pd.DataFrame] = {}
    rows = []
    for candidate in candidates:
        subgroup_col = str(candidate["subgroup_col"])
        feature_subgroup_value = candidate["feature_subgroup_value"]
        cv_delta = float(candidate["cv_delta"])  # type: ignore[arg-type]
        feature_name = candidate.get("feature_name", feature_subgroup_value)

        if subgroup_col not in reports:
            reports[subgroup_col] = subpopulation_ratio_drift_check(
                train_df, test_df, subgroup_col, ratio_threshold=ratio_threshold, include_severity_score=True
            )
        report = reports[subgroup_col]
        matching = report[report["subgroup_value"] == feature_subgroup_value]

        if matching.empty:
            rows.append(
                {
                    "feature_name": feature_name,
                    "subgroup_col": subgroup_col,
                    "feature_subgroup_value": feature_subgroup_value,
                    "cv_delta": cv_delta,
                    "prevalence_ratio": float("nan"),
                    "subgroup_shifted": False,
                    "overfit_risk_flag": False,
                    "drift_severity_score": 0.0,
                    "risk_score": 0.0,
                }
            )
            continue

        row = matching.iloc[0]
        subgroup_shifted = bool(row["flagged"])
        drift_severity_score = float(row["drift_severity_score"])
        rows.append(
            {
                "feature_name": feature_name,
                "subgroup_col": subgroup_col,
                "feature_subgroup_value": feature_subgroup_value,
                "cv_delta": cv_delta,
                "prevalence_ratio": float(row["prevalence_ratio"]),
                "subgroup_shifted": subgroup_shifted,
                "overfit_risk_flag": subgroup_shifted and cv_delta != 0.0,
                "drift_severity_score": drift_severity_score,
                "risk_score": drift_severity_score * abs(cv_delta),
            }
        )

    ranking = pd.DataFrame(rows)
    return ranking.sort_values("risk_score", ascending=False).reset_index(drop=True)


__all__ = ["flag_subgroup_only_feature_overfit_risk", "rank_subgroup_feature_overfit_risk"]
