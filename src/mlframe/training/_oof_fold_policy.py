"""Fold-model policy for OOF predictions: early-stopping cap on the clone and the splitter for i.i.d. rows."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from mlframe.training._data_helpers import get_function_param_names
from mlframe.training._oof_round_budget import apply_round_budget, deployed_round_budget


def cap_early_stopping_clone(estimator: Any, deployed_model: Any, diag: dict) -> None:
    """Switch off native early stopping on the OOF clone and cap its rounds at the point the deployed model stopped at.

    A model configured with native early stopping (LGB_GENERAL_PARAMS bakes in early_stopping_rounds for every "lgb"/"xgb" registry entry)
    requires an eval_set at fit time; cross_val_predict's internal per-fold .fit(X_train, y_train) never supplies one, so every fold would
    raise and the whole OOF computation would no-op. Only the CLONE is changed (the already-fit model is untouched) and its round count is
    set from the deployed model's early-stopping point, so OOF predictions come from models as fitted as the shipped one. The applied budget
    is recorded in ``diag["round_budget"]``.
    """
    try:
        if getattr(estimator, "early_stopping_rounds", None):
            estimator.set_params(early_stopping_rounds=None)
            budget = deployed_round_budget(deployed_model)
            if apply_round_budget(estimator, budget):
                diag["round_budget"] = budget
    except (ValueError, TypeError):
        pass


def apply_feature_roles_to_clone(estimator: Any, fit_params: Optional[dict], columns: Any) -> None:
    """Copy the deployed fit's ``cat_features`` / ``text_features`` / ``embedding_features`` onto the OOF clone's constructor params.

    Those roles reach the deployed model through ``.fit(**fit_params)``, which ``cross_val_predict``'s per-fold ``.fit(X, y)`` never sees; a
    CatBoost clone without them reads every categorical column as numeric and raises. Only estimators exposing the param (CatBoost) are touched,
    and only columns still present in the frame are kept.
    """
    if not fit_params or not hasattr(estimator, "get_params"):
        return
    available = set(columns) if columns is not None else None
    ctor_params = get_function_param_names(type(estimator).__init__)
    roles = {}
    for key in ("cat_features", "text_features", "embedding_features"):
        names = fit_params.get(key)
        if names and key in ctor_params:
            roles[key] = [c for c in names if available is None or c in available]
    if roles:
        estimator.set_params(**roles)


def iid_oof_splitter(train_target: Any, n_splits: int, random_seed: int, is_classifier_model: bool, diag: dict) -> Any:
    """Shuffled ``StratifiedKFold`` for a 1-D classifier target whose rarest class has at least ``n_splits`` members, else shuffled ``KFold``.

    Plain ``KFold`` on a rare class can leave a training fold without it, which makes the whole OOF computation skip. The scheme chosen is
    recorded in ``diag["fold_scheme"]``.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    y_flat = np.asarray(train_target)
    if is_classifier_model and y_flat.ndim == 1:
        _, class_counts = np.unique(y_flat, return_counts=True)
        if class_counts.size >= 2 and int(class_counts.min()) >= n_splits:
            diag["fold_scheme"] = "stratified_kfold"
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    diag["fold_scheme"] = "kfold"
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
