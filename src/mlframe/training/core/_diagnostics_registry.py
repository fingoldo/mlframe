"""Opt-in registry adapting standalone ``mlframe.evaluation`` diagnostics to the standard suite-local
variables (train/val/test frames, target column, cat features, group ids, y) available right after
``_phase_global_outlier_detection``.

Every adapter shares the signature ``(train_df, val_df, test_df, target_col, cat_features, group_ids,
y, **kwargs) -> dict`` and must never raise -- the call site in ``_main_train_suite.py`` also wraps each
call in try/except, but adapters catch internally too so a partial failure (e.g. missing optional kwarg)
reports as ``{"error": ...}`` instead of masking a real KeyError from a bug in the adapter itself.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DiagnosticAdapter = Callable[..., dict]


def _numeric_frame(df: Optional[pd.DataFrame], cat_features: Optional[List[str]]) -> pd.DataFrame:
    """Best-effort numeric-only view of ``df`` for diagnostics that need a plain feature matrix."""
    if df is None:
        return pd.DataFrame()
    cat_set = set(cat_features or [])
    num_df = df.drop(columns=[c for c in cat_set if c in df.columns], errors="ignore")
    num_df = num_df.select_dtypes(include=[np.number])
    return num_df.fillna(0.0)


def _looks_classification(y: Optional[np.ndarray]) -> bool:
    if y is None:
        return False
    uniq = np.unique(y[~pd.isnull(y)]) if len(y) else np.array([])
    return uniq.size <= max(20, int(0.05 * max(len(y), 1)))


def _default_model_and_metric(y: Optional[np.ndarray]):
    """Lightweight sklearn model_factory + metric_fn pair picked from the target's apparent task type.

    ``metric_fn`` must accept BOTH ``model.predict()`` output (hard 0/1 labels for a classifier) and the
    caller-side "leaked dummy" prediction (a continuous per-fold mean/median of y_test) -- an accuracy-style
    metric chokes on the latter (``ValueError: mix of binary and continuous targets``), so ``r2_score`` is
    used for both task types: it's defined for any pair of numeric arrays regardless of label cardinality.
    """
    from sklearn.metrics import r2_score

    if _looks_classification(y):
        from sklearn.linear_model import LogisticRegression

        return (lambda: LogisticRegression(max_iter=200)), r2_score
    from sklearn.linear_model import Ridge

    return (lambda: Ridge()), r2_score


def adapt_cv_informativeness(train_df, val_df, test_df, target_col, cat_features, group_ids, y, **kwargs) -> dict:
    try:
        from mlframe.evaluation.cv_informativeness import cv_informativeness_check
        from sklearn.model_selection import KFold

        X = _numeric_frame(train_df, cat_features)
        if X.empty or y is None or len(y) != len(X):
            return {"error": "cv_informativeness: no usable numeric train_df / aligned y"}
        model_factory, metric_fn = kwargs.pop("model_factory", None), kwargs.pop("metric_fn", None)
        if model_factory is None or metric_fn is None:
            model_factory, metric_fn = _default_model_and_metric(y)
        n_splits = min(3, len(X))
        if n_splits < 2:
            return {"error": "cv_informativeness: not enough rows for a CV split"}
        cv_splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=0).split(X))
        return cv_informativeness_check(
            X=X, y=np.asarray(y), cv_splits=cv_splits, model_factory=model_factory, metric_fn=metric_fn, **kwargs,
        )
    except Exception as e:
        logger.debug("diagnostics.cv_informativeness failed: %s", e)
        return {"error": str(e)}


def adapt_compare_cv_schemes(train_df, val_df, test_df, target_col, cat_features, group_ids, y, **kwargs) -> dict:
    try:
        from mlframe.evaluation.compare_cv_schemes import compare_cv_schemes
        from sklearn.model_selection import KFold, ShuffleSplit

        X = _numeric_frame(train_df, cat_features)
        if X.empty or y is None or len(y) != len(X):
            return {"error": "compare_cv_schemes: no usable numeric train_df / aligned y"}
        n = len(X)
        if n < 10:
            return {"error": "compare_cv_schemes: not enough rows for an out-of-time holdout"}
        split_point = int(n * 0.8)
        train_part_idx = np.arange(split_point)
        ooo_idx = np.arange(split_point, n)
        model_factory, metric_fn = kwargs.pop("model_factory", None), kwargs.pop("metric_fn", None)
        if model_factory is None or metric_fn is None:
            model_factory, metric_fn = _default_model_and_metric(y)
        n_splits = min(3, len(train_part_idx))
        if n_splits < 2:
            return {"error": "compare_cv_schemes: not enough in-sample rows for CV schemes"}
        from typing import Iterable, Tuple

        schemes: Dict[str, Iterable[Tuple[np.ndarray, np.ndarray]]] = {
            "kfold": list(KFold(n_splits=n_splits, shuffle=True, random_state=0).split(train_part_idx)),
            "shuffled": list(ShuffleSplit(n_splits=n_splits, random_state=1).split(train_part_idx)),
        }
        return compare_cv_schemes(
            X=X, y=np.asarray(y), schemes=schemes, ooo_time_idx=(train_part_idx, ooo_idx),
            model_factory=model_factory, metric_fn=metric_fn, **kwargs,
        )
    except Exception as e:
        logger.debug("diagnostics.compare_cv_schemes failed: %s", e)
        return {"error": str(e)}


def adapt_group_leakage(train_df, val_df, test_df, target_col, cat_features, group_ids, y, **kwargs) -> dict:
    try:
        from mlframe.evaluation.group_leakage_guard import assert_no_group_leakage

        if group_ids is None or train_df is None or val_df is None:
            return {"skipped": "no group_ids / train_df / val_df available"}
        groups = np.asarray(group_ids)
        n_train, n_val = len(train_df), len(val_df)
        if len(groups) < n_train + n_val:
            return {"skipped": "group_ids shorter than train+val; cannot align folds"}
        fold_groups = groups[: n_train + n_val]
        cv_splits = [(np.arange(n_train), np.arange(n_train, n_train + n_val))]
        try:
            assert_no_group_leakage(cv_splits=cv_splits, groups=fold_groups, **kwargs)
        except ValueError as leak_err:
            return {"leakage_detected": True, "detail": str(leak_err)}
        return {"leakage_detected": False}
    except Exception as e:
        logger.debug("diagnostics.group_leakage failed: %s", e)
        return {"error": str(e)}


def adapt_constant_group_leak(train_df, val_df, test_df, target_col, cat_features, group_ids, y, **kwargs) -> dict:
    try:
        from mlframe.evaluation.constant_group_leak_scan import constant_group_target_scan

        if train_df is None or y is None or not cat_features:
            return {"skipped": "no train_df / y / cat_features available"}
        candidate_cols = [c for c in cat_features if c in train_df.columns]
        if not candidate_cols:
            return {"skipped": "no candidate categorical columns present in train_df"}
        report = constant_group_target_scan(df=train_df, y=np.asarray(y), candidate_cols=candidate_cols, **kwargs)
        return {"report": report.to_dict(orient="records")}
    except Exception as e:
        logger.debug("diagnostics.constant_group_leak failed: %s", e)
        return {"error": str(e)}


def adapt_adversarial_fold_selection(train_df, val_df, test_df, target_col, cat_features, group_ids, y, **kwargs) -> dict:
    try:
        from mlframe.evaluation.adversarial_fold_selection import build_test_like_validation_fold

        X_train = _numeric_frame(train_df, cat_features)
        X_test = _numeric_frame(test_df, cat_features)
        common_cols = [c for c in X_train.columns if c in X_test.columns]
        if not common_cols:
            return {"error": "adversarial_fold_selection: no shared numeric columns between train_df/test_df"}
        X_train, X_test = X_train[common_cols], X_test[common_cols]
        kwargs.pop("return_history", None)  # keep the 2-tuple contract this adapter returns
        _fold_result = build_test_like_validation_fold(X_train=X_train, X_test=X_test, return_history=False, **kwargs)
        val_idx, remaining_idx = _fold_result[0], _fold_result[1]
        return {"val_idx": val_idx.tolist(), "n_selected": int(len(val_idx)), "n_remaining": int(len(remaining_idx))}
    except Exception as e:
        logger.debug("diagnostics.adversarial_fold_selection failed: %s", e)
        return {"error": str(e)}


def adapt_subpopulation_drift(train_df, val_df, test_df, target_col, cat_features, group_ids, y, **kwargs) -> dict:
    try:
        from mlframe.evaluation.subpopulation_drift import subpopulation_ratio_drift_check

        subgroup_col = kwargs.pop("subgroup_col", None)
        if subgroup_col is None and cat_features:
            subgroup_col = next((c for c in cat_features if c in (train_df.columns if train_df is not None else [])), None)
        if subgroup_col is None or train_df is None or test_df is None:
            return {"skipped": "no subgroup_col resolvable; pass one via diagnostics_kwargs['subpopulation_drift']"}
        report = subpopulation_ratio_drift_check(train_df=train_df, test_df=test_df, subgroup_col=subgroup_col, **kwargs)
        return {"report": report.to_dict(orient="records")}
    except Exception as e:
        logger.debug("diagnostics.subpopulation_drift failed: %s", e)
        return {"error": str(e)}


# "cv_delta_triage" is deliberately NOT registered: ``triage_cv_delta`` needs paired per-fold score
# histories (baseline vs candidate) from repeated CV runs, which the suite does not accumulate at the
# diagnostics call site (right after outlier detection, before any model is trained). Wiring it would
# require fabricating fold scores, i.e. forcing something broken -- callers who track fold-score
# history across suite runs should call ``triage_cv_delta`` directly.
DIAGNOSTICS_REGISTRY: Dict[str, DiagnosticAdapter] = {
    "cv_informativeness": adapt_cv_informativeness,
    "compare_cv_schemes": adapt_compare_cv_schemes,
    "group_leakage": adapt_group_leakage,
    "constant_group_leak": adapt_constant_group_leak,
    "adversarial_fold_selection": adapt_adversarial_fold_selection,
    "subpopulation_drift": adapt_subpopulation_drift,
}
