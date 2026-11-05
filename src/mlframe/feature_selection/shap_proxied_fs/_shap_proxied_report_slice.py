"""An unbiased figure for the chosen subset: a report-only slice carved off the revalidation holdout.

Every candidate subset is scored on the same holdout and the winner is the minimum over them, so the winner's
``honest_loss`` is the most favourable draw of a noisy estimate - fine for ranking, optimistic as a number. With
``report_holdout_fraction > 0`` that share of the holdout rows is set aside before any candidate sees them; after
selection the final subset is refitted on the search rows and scored once on the slice, reported as
``shap_proxy_report_["report_holdout"]``. Selection then runs on fewer holdout rows, which is the trade the default
records (see ``ShapProxiedFS.report_holdout_fraction``).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def split_report_slice(
    idx_hold: np.ndarray, X: Any, y: Any, fraction: float, classification: bool, random_state: int, groups: Any = None
) -> Tuple[np.ndarray, Optional[tuple]]:
    """``(selection_holdout_idx, (X_report, y_report))``; the slice is None when ``fraction`` is 0 or the holdout is too small.

    The report rows are materialised here (full width, a small share of the rows) because the fit releases the full frame
    right after the split. ``groups`` (a label per row of the FULL frame) makes the slice group-disjoint from the rest of the holdout, as the holdout
    itself is group-disjoint from the search rows: a row-wise split put one group on both sides and the reported figure was still optimistic.
    """
    if not fraction or fraction <= 0.0 or fraction >= 1.0 or len(idx_hold) < 20:
        return idx_hold, None
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    y_arr = np.asarray(y)
    if groups is not None:
        hold_groups = np.asarray(groups)[idx_hold]
        if np.unique(hold_groups).size >= 2:
            keep_pos, rep_pos = next(GroupShuffleSplit(n_splits=1, test_size=float(fraction), random_state=int(random_state)).split(idx_hold, groups=hold_groups))
            keep, rep = idx_hold[keep_pos], idx_hold[rep_pos]
            X_report = X.iloc[rep].reset_index(drop=True) if hasattr(X, "iloc") else np.asarray(X)[rep]
            return np.asarray(keep), (X_report, y_arr[rep])
    y_hold = y_arr[idx_hold]
    _, counts = np.unique(y_hold, return_counts=True)
    n_rep = int(np.ceil(float(fraction) * len(idx_hold)))
    # a stratified split needs at least one row per class on both sides: a tiny fraction of a small holdout cannot give that and used to raise
    stratify = y_hold if classification and counts.size > 1 and counts.min() >= 2 and n_rep >= counts.size and len(idx_hold) - n_rep >= counts.size else None
    keep, rep = train_test_split(idx_hold, test_size=float(fraction), random_state=int(random_state), shuffle=True, stratify=stratify)  # group-blind-ok: reached only without groups or with a single holdout group
    X_report = X.iloc[rep].reset_index(drop=True) if hasattr(X, "iloc") else np.asarray(X)[rep]
    return np.asarray(keep), (X_report, y_arr[rep])


def score_on_report_slice(est: Any, report: dict, X_search: Any, y_search: Any, report_slice: Optional[tuple], working_cols: Any,
                          member_cols: Any, model_template: Any) -> None:
    """Refit the chosen subset on the search rows and record its loss, once, on the untouched report slice."""
    if report_slice is None or len(member_cols) == 0:
        return
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_objective import resolve_metric
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate import _honest_loss

    X_report, y_report = report_slice
    wc = [int(c) for c in working_cols]
    X_rep = X_report.iloc[:, wc] if hasattr(X_report, "iloc") else np.asarray(X_report)[:, wc]  # same column space as X_search
    loss = _honest_loss(model_template, X_search, y_search, X_rep, y_report, [int(c) for c in member_cols], est.classification,
                        resolve_metric(est.classification, est.metric))
    report["report_holdout"] = dict(loss=float(loss), n_rows=len(y_report), fraction=float(est.report_holdout_fraction), selection_optimistic=False)
