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

For SEVERELY drifting features the rank transform is not always enough — a feature whose within-group rank
is itself still separable (e.g. its rank distribution shifts because the group composition itself changed
between train and test) keeps leaking. ``drop_n_std``/``auto_tune_drop_threshold`` add an opt-in second,
higher severity tier: features that clear that higher bar get dropped outright instead of rank-transformed,
while everything between the two thresholds still gets the (cheaper, signal-preserving) rank remedy.
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
    drop_n_std: float | None = None,
    auto_tune_drop_threshold: bool = False,
    auto_tune_candidates: Sequence[float] | None = None,
    **adversarial_kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flag adversarially-drifting features and remediate them by severity tier.

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
    drop_n_std
        Opt-in second, higher severity threshold (in the same ``mean + k * std`` units as ``n_std``; must be
        ``> n_std``). Features whose importance clears ``mean + drop_n_std * std`` are DROPPED outright from
        both frames instead of rank-transformed; features between the two thresholds still get the rank
        remedy. ``None`` (the default) disables the tier entirely — every flagged feature is rank-transformed,
        identical to the original single-remedy behaviour.
    auto_tune_drop_threshold
        When ``True``, ignore any explicit ``drop_n_std`` and instead search ``auto_tune_candidates`` for the
        drop threshold that minimises the POST-remediation adversarial AUC on the scanned columns (a held-out
        re-check via ``adversarial_auc``, reusing the same helper this function already depends on), breaking
        ties toward the higher threshold (drops fewer features for the same de-drifting effect). Adds one
        extra ``adversarial_auc`` call per candidate — opt-in because it is materially more expensive than the
        default single-pass remediation.
    auto_tune_candidates
        Candidate ``k`` values (std multiples, each must be ``> n_std``) to search when
        ``auto_tune_drop_threshold=True``. Defaults to ``n_std + [0.5, 1.0, 1.5, 2.0, 3.0]``.
    **adversarial_kwargs
        Forwarded to ``reporting.charts.drift.adversarial_auc`` (e.g. ``n_splits``, ``seed``, ``lgbm_params``).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_out, test_out, report)`` — copies of the input frames with flagged feature columns either
        replaced by their within-``group_col`` rank or dropped entirely, and a report frame with columns
        ``{"feature", "drift_importance", "flagged", "action"}`` sorted by importance descending. ``action``
        is one of ``{"none", "rank_transform", "drop"}``.
    """
    from mlframe.feature_engineering.grouped import per_group_rank
    from mlframe.reporting.charts.drift import adversarial_auc

    if group_col not in train_df.columns or group_col not in test_df.columns:
        raise ValueError(f"remediate_drifting_features: group_col={group_col!r} must be present in both frames")

    if drop_n_std is not None and drop_n_std <= n_std:
        raise ValueError(f"remediate_drifting_features: drop_n_std={drop_n_std!r} must be strictly greater than n_std={n_std!r}")

    scan_cols = [c for c in feature_names] if feature_names is not None else [c for c in train_df.columns if c != group_col and c in test_df.columns]

    _, _, _, importances, names = adversarial_auc(train_df[scan_cols], test_df[scan_cols], feature_names=scan_cols, **adversarial_kwargs)
    names_list = list(names)
    importances_arr = np.asarray(importances, dtype=np.float64)

    mean_imp = float(importances_arr.mean())
    std_imp = float(importances_arr.std())
    threshold = mean_imp + n_std * std_imp
    flagged_mask = importances_arr > threshold

    def _build(effective_drop_n_std: float | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_out = train_df.copy()
        test_out = test_df.copy()
        train_groups = train_df[group_col].to_numpy()
        test_groups = test_df[group_col].to_numpy()

        drop_threshold = mean_imp + effective_drop_n_std * std_imp if effective_drop_n_std is not None else None
        actions = []
        drop_names = []
        for name, importance, flagged in zip(names_list, importances_arr, flagged_mask):
            if not flagged:
                actions.append("none")
                continue
            if drop_threshold is not None and importance > drop_threshold:
                actions.append("drop")
                drop_names.append(name)
                continue
            actions.append("rank_transform")
            train_out[name] = per_group_rank(train_df[name].to_numpy(dtype=np.float64), train_groups, pct=rank_pct)
            test_out[name] = per_group_rank(test_df[name].to_numpy(dtype=np.float64), test_groups, pct=rank_pct)

        if drop_names:
            train_out = train_out.drop(columns=drop_names)
            test_out = test_out.drop(columns=drop_names)

        report_out = pd.DataFrame({"feature": names_list, "drift_importance": importances_arr, "flagged": flagged_mask, "action": actions})
        report_out = report_out.sort_values("drift_importance", ascending=False).reset_index(drop=True)
        return train_out, test_out, report_out

    if not auto_tune_drop_threshold:
        return _build(drop_n_std)

    candidates = list(auto_tune_candidates) if auto_tune_candidates is not None else [n_std + 0.5, n_std + 1.0, n_std + 1.5, n_std + 2.0, n_std + 3.0]
    for c in candidates:
        if c <= n_std:
            raise ValueError(f"remediate_drifting_features: auto_tune_candidates entries must be > n_std={n_std!r}, got {c!r}")

    best_candidate: float | None = None
    best_auc = float("inf")
    for c in candidates:
        cand_train, cand_test, _ = _build(c)
        cand_cols = [col for col in scan_cols if col in cand_train.columns]
        recheck_auc, *_ = adversarial_auc(cand_train[cand_cols], cand_test[cand_cols], feature_names=cand_cols, **adversarial_kwargs)
        # tie-break toward the higher threshold: fewer dropped columns for the same de-drifting effect.
        if recheck_auc < best_auc or (recheck_auc == best_auc and (best_candidate is None or c > best_candidate)):
            best_auc = recheck_auc
            best_candidate = c

    return _build(best_candidate)


__all__ = ["remediate_drifting_features"]
