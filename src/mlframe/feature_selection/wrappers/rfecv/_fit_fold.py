"""Per-fold CV evaluator carved out of ``RFECV.fit``.

Holds ``_eval_fold_body`` -- the function previously defined as a nested closure inside ``fit``. Closure-captured outer locals are now explicit kwargs; the four mutable containers (``scores``, ``feature_importances``, ``fitted_estimators``, ``dummy_scores``) are passed by reference so the parent's outer-loop state observes the worker-thread appends exactly as it did with the closure (GIL-atomic ``list.append`` / dict assignment).

Behavioural equivalence: the body is the verbatim pre-carve body. The signature is the union of every name the original closure resolved: 4 mutable containers + 25 readonly locals.
"""
from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

import numpy as np
import pandas as pd

from pyutilz.pythonlib import suppress_stdout_stderr

from sklearn.base import clone

from mlframe.config import CATBOOST_MODEL_TYPES
from mlframe.core.helpers import has_early_stopping_support
from mlframe.training.helpers import compute_cb_text_processing
from mlframe.preprocessing.transforms import pack_val_set_into_fit_params
from mlframe.estimators.baselines import get_best_dummy_score

from .._helpers import (
    get_feature_importances,
    split_into_train_test,
)


logger = logging.getLogger("mlframe.feature_selection.wrappers.rfecv")


def _eval_fold_body(
    nfold,
    train_index,
    test_index,
    fold_seed,
    *,
    # readonly captures
    self,
    current_features,
    frac,
    must_include_resolved,
    X,
    y,
    X_estimator,
    col_pos,
    val_cv,
    estimator_type,
    groups,
    verbose,
    cat_features,
    early_stopping_rounds,
    fit_params,
    estimators_list,
    estimator,
    scoring,
    importance_getter,
    keep_estimators,
    evaluated_scores_mean,
    # mutable containers (passed by reference; mutations observable in caller)
    scores: list,
    feature_importances: dict,
    fitted_estimators: dict,
    dummy_scores: list,
):
    """Per-fold evaluation. Returns a dict or None on skip. Fold-local state (RNG, estimator clone, fit_params) is built fresh inside.

    Under ``n_jobs_effective > 1`` the per-fold ``scores.append`` calls happen from worker threads concurrently. The GIL makes ``list.append`` atomic so the result is order-INDEPENDENT for ``np.min(...)``-style aggregation, but per-fold log line ordering is best-effort. Don't rely on log order for fold attribution; emit ``nfold`` explicitly when needed.
    """
    if self.min_train_size and len(train_index) < self.min_train_size:
        return None
    if frac:
        size = int(len(train_index) * frac)
        if size > 10:
            # Per-fold local RNG seeded deterministically; avoids races on self._rng when joblib runs folds in parallel.
            local_rng = np.random.default_rng(fold_seed)
            train_index = local_rng.choice(train_index, size=size, replace=False)

    # Actual fit/score uses must_include + optimiser's pick. current_features already lives in the search-universe complement (must_include filtered out at fit entry), so concatenation never duplicates.
    if must_include_resolved:
        fit_features = list(must_include_resolved) + list(current_features)
    else:
        fit_features = current_features
    X_train, y_train, X_test, y_test = split_into_train_test(
        X=X, y=y, train_index=train_index, test_index=test_index, features_indices=fit_features,
        X_estimator=X_estimator, col_pos=col_pos,
    )
    if verbose > 2:
        print(f"Train set size={len(y_train):_}, train idx sum={train_index.sum():_}")

    _filtered_fit_params: dict = {}
    # Records the early-stopping val re-split (assigned inside the block below).
    # Stays None when no re-split runs, so the per-fold sample_weight slicing
    # knows whether X_train was narrowed to true_train_index.
    true_train_index = None
    if val_cv and has_early_stopping_support(estimator_type):

        # Additional early-stopping split from X_train.
        if groups is not None:
            if isinstance(groups, pd.Series):
                train_groups = groups.iloc[train_index]
            else:
                train_groups = groups[train_index]
        else:
            train_groups = None

        splits = list(val_cv.split(X=X_train, y=y_train, groups=train_groups))
        true_train_index, val_index = splits[-1]

        X_train, y_train, X_val, y_val = split_into_train_test(X=X_train, y=y_train, train_index=true_train_index, test_index=val_index)
        if verbose > 2:
            print(f"Val set size={len(y_val):_}, val idx sum={val_index.sum():_}")

        # If the estimator type is known, craft early-stopping fit params tailored to it.
        # Index cat features against fit_features (the ACTUAL X_train/X_val column
        # order = must_include_resolved + current_features), NOT current_features:
        # otherwise must_include's prepended columns shift every index by
        # len(must_include) and any categorical must_include column is dropped,
        # so CatBoost treats the wrong columns as categorical.
        temp_cat_features = [fit_features.index(var) for var in cat_features if var in fit_features] if cat_features else None

        temp_fit_params = pack_val_set_into_fit_params(
            model=estimator,
            X_val=X_val,
            y_val=y_val,
            early_stopping_rounds=early_stopping_rounds,
            cat_features=temp_cat_features,
        )
        # Filter feature-list keys (cat_features / text_features / embedding_features) coming in via fit_params to only columns present in the current selector iteration. Otherwise names from the outer call reference columns dropped by the current iteration and CB raises ``Error while processing column for feature 'cat_0'``.
        # cat_features: pack_val_set_into_fit_params above already injected index-based temp_cat_features IFF that list was non-empty. When empty (current_features doesn't intersect self.cat_features) we MUST still pass a name-list filtered to current_features so CB doesn't fall back to auto-detect on numerically-encoded category columns (target-encoded cats look like floats and trip CB's "Invalid type for cat_feature").
        # When current_features holds integer indices (ndarray X path), set-membership against string column names always misses; in that case the user-supplied lists pass through unfiltered (the inner estimator can deal with missing column references on its own).
        _features_are_integer = (
            len(current_features) > 0
            and isinstance(current_features[0], (int, np.integer))
        )
        _current_set = set(current_features) if not _features_are_integer else None
        for _k, _v in fit_params.items():
            if _k == "cat_features":
                if "cat_features" in temp_fit_params:
                    # temp's index-based cat_features wins; drop outer name list.
                    continue
                if _v and _current_set is not None:
                    _filtered = [c for c in _v if c in _current_set]
                    _filtered_fit_params[_k] = _filtered or None
                elif _v:
                    _filtered_fit_params[_k] = _v
                continue
            if _k in ("text_features", "embedding_features") and _v:
                if _current_set is not None:
                    _filtered = [c for c in _v if c in _current_set]
                    _filtered_fit_params[_k] = _filtered or None
                else:
                    _filtered_fit_params[_k] = _v
                continue
            _filtered_fit_params[_k] = _v
        temp_fit_params.update(_filtered_fit_params)

    else:

        temp_fit_params = {}
        X_val = None

    # Per-fold sample_weight slicing. Estimator-side: pass via temp_fit_params only when the estimator's fit signature accepts sample_weight (sklearn convention). Scorer-side: detected at score-call time below (sklearn's _BaseScorer.__call__ accepts sample_weight=).
    _fold_train_sw = None
    _fold_test_sw = None
    if self._fit_sample_weight_ is not None:
        _full_train_sw = self._fit_sample_weight_[train_index]
        # When val_cv re-split X_train down to true_train_index above, the train
        # weights MUST follow the same re-slice -- otherwise sample_weight is
        # longer than the (narrowed) X_train and estimator.fit() raises a
        # length-mismatch. true_train_index is None when no early-stopping val
        # re-split ran (both indices are positional; _fit_sample_weight_ is an
        # ndarray per _rfecv_fit_init).
        _fold_train_sw = _full_train_sw if true_train_index is None else _full_train_sw[true_train_index]
        _fold_test_sw = self._fit_sample_weight_[test_index]

    # Always clone per fold via sklearn.base.clone. copy.copy is a SHALLOW copy that shares mutable state (cat_features list, set_params side effects, warm_start buffers) across folds. clone() returns an unfitted estimator with the same constructor params and NO fitted state.
    fitted_estimator = clone(estimator)

    # Dynamic CB ``text_processing`` calibration for THIS fold's clone (not the outer estimator). RFECV folds are typically much smaller than the outer training set; with CB's default ``occurrence_lower_bound=50`` words that occur < 50 times in the fold are pruned, leaving an empty dictionary and HANGING CB's C++ ``_train`` loop. ``compute_cb_text_processing`` returns a config that scales the floor proportionally to fold rows, or None when the fold is large enough.
    if val_cv and has_early_stopping_support(estimator_type):
        _temp_text_feats = _filtered_fit_params.get("text_features") or []
        if _temp_text_feats and "CatBoost" in type(fitted_estimator).__name__:
            _fold_rows = X_train.shape[0] if hasattr(X_train, "shape") else None
            _tp = compute_cb_text_processing(_fold_rows) if _fold_rows is not None else None
            if _tp is not None and hasattr(fitted_estimator, "set_params"):
                try:
                    fitted_estimator.set_params(text_processing=_tp)
                except Exception as _tp_exc:
                    logger.warning(
                        "RFECV inner fold: failed to set CB text_processing "
                        "(fold_rows=%s, exc=%s).", _fold_rows, _tp_exc,
                    )

    # Empty-train guard: heavy upstream filtering (small n + outlier_detection + trainset_aging_limit) can collapse X_train / y_train to 0 rows on a CV fold; CatBoost then raises "Labels variable is empty" deep in C++ Pool init. Skip the fold cleanly with a NaN score; sklearn's RFECV does the same on degenerate inner folds.
    _x_n = X_train.shape[0] if hasattr(X_train, "shape") else None
    _y_n = len(y_train) if y_train is not None and hasattr(y_train, "__len__") else None
    if (_x_n is not None and _x_n == 0) or (_y_n is not None and _y_n == 0):
        # Always-on ERROR (not verbose-gated) so the operator sees the empty-fold collapse. Root cause is upstream filter aggression (OD + trainset_aging_limit together can shrink the inner-CV training fraction below cv splits) - fix that, not this guard.
        logger.error(
            "wrappers.fit: skipping fold %s - empty train slice "
            "(rows=%s, target_len=%s). Upstream filters reduced "
            "the train batch to zero. Investigate "
            "outlier_detection contamination + "
            "trainset_aging_limit interactions.",
            nfold, _x_n, _y_n,
        )
        scores.append(np.nan)
        feature_importances[f"{len(current_features)}_{nfold}"] = {}
        return None
    # Multi-estimator: fit ALL estimators (singular case is a len-1 list). Score per fold = mean across estimators; FI runs stored under separate keys ("{N}_{fold}" for singular, "{N}_{fold}_e{idx}" for multi) so the voting layer treats each estimator's importances as an independent run.
    _est_scores = []
    _est_fi_runs = []  # list of (key, fi_dict)
    for _est_idx, _est_proto in enumerate(estimators_list):
        if _est_idx == 0:
            # First estimator was already cloned + text_processing tuned above as ``fitted_estimator``.
            _fitted = fitted_estimator
        else:
            _fitted = clone(_est_proto)
            # Apply CB text_processing if applicable for THIS clone.
            if val_cv and has_early_stopping_support(estimator_type):
                _temp_text_feats = _filtered_fit_params.get("text_features") or []
                if _temp_text_feats and "CatBoost" in type(_fitted).__name__:
                    _fold_rows = X_train.shape[0] if hasattr(X_train, "shape") else None
                    _tp = compute_cb_text_processing(_fold_rows) if _fold_rows is not None else None
                    if _tp is not None and hasattr(_fitted, "set_params"):
                        try:
                            _fitted.set_params(text_processing=_tp)
                        except Exception:
                            pass

        _model_type_name = type(_fitted).__name__
        _ctx = suppress_stdout_stderr() if _model_type_name in CATBOOST_MODEL_TYPES else nullcontext()
        # Build per-call fit kwargs: only attach sample_weight when the estimator's fit signature accepts it. CatBoost / LGBM / sklearn estimators all accept it; some custom shims may not, so introspect.
        _per_est_fit_params = dict(temp_fit_params)
        if _fold_train_sw is not None and "sample_weight" not in _per_est_fit_params:
            try:
                import inspect as _inspect
                _sig = _inspect.signature(_fitted.fit)
                if "sample_weight" in _sig.parameters or any(p.kind == _inspect.Parameter.VAR_KEYWORD for p in _sig.parameters.values()):
                    _per_est_fit_params["sample_weight"] = _fold_train_sw
            except (TypeError, ValueError):
                pass
        with _ctx:
            _fitted.fit(X=X_train, y=y_train, **_per_est_fit_params)
        # Scorer-side: forward fold-test sample_weight when sklearn scorer accepts it. make_scorer-wrapped callables expose sample_weight via _BaseScorer.__call__; bare callables may not.
        if _fold_test_sw is not None:
            try:
                _score = scoring(_fitted, X_test, y_test, sample_weight=_fold_test_sw)
            except TypeError:
                _score = scoring(_fitted, X_test, y_test)
        else:
            _score = scoring(_fitted, X_test, y_test)
        _est_scores.append(_score)

        # FI is computed on the actual fit_features.
        # F4/F5/F10/F11 (Wave 3, 2026-05-28): pass through the rescale-source
        # and aggregation knobs so coef_ uses train-side stds (not test) and
        # multiclass collapses via max (not sum); CPI uses min_samples_leaf
        # instead of a fixed depth cap.
        _fi_full = get_feature_importances(
            model=_fitted, current_features=fit_features,
            data=X_test, reference_data=X_val, target=y_test,
            train_data=X_train,
            importance_getter=importance_getter,
            multiclass_coef_aggregation=getattr(self, "multiclass_coef_aggregation", "max"),
            coef_scale_source=getattr(self, "coef_scale_source", "train"),
            cpi_max_depth=getattr(self, "cpi_max_depth", None),
            cpi_min_samples_leaf=int(getattr(self, "cpi_min_samples_leaf", 10)),
            # Prefer the wide-data-guard's effective n_repeats (set in RFECV.fit) over the user-facing self.n_repeats.
            n_repeats=int(getattr(self, "_effective_n_repeats", None) or getattr(self, "n_repeats", 5)),
        )
        if must_include_resolved:
            must_set = set(must_include_resolved)
            _fi = {k: v for k, v in _fi_full.items() if k not in must_set}
        else:
            _fi = _fi_full

        _est_suffix = f"_e{_est_idx}" if len(estimators_list) > 1 else ""
        _key = f"{len(current_features)}_{nfold}{_est_suffix}"
        # F14 (Wave 3, 2026-05-28): if THIS estimator's score on THIS fold was NaN, drop its FI from the voting pool. A failed/degenerate
        # estimator-fold pair should not contribute to the next-subset rank vote. Keep the score in _est_scores so the across-estimator
        # aggregation (min) still sees it (and propagates NaN to the score curve, which is the diagnostic signal). Opt-out via
        # drop_nan_score_fi=False.
        _drop_nan_fi = getattr(self, "drop_nan_score_fi", True)
        if not (_drop_nan_fi and isinstance(_score, float) and np.isnan(_score)):
            _est_fi_runs.append((_key, _fi))
        if keep_estimators:
            fitted_estimators[_key] = _fitted

    # Aggregate fold score via worst-case (min) across estimators given sklearn's "higher is better". Mean would let one strong estimator (e.g. RF on 2 informative features) mask the fact that another (e.g. LR) needs more features to converge; worst-case forces N to be where ALL estimators agree it's sufficient. For the singular path (len-1 list), min == mean.
    if _est_scores:
        valid_scores = [s for s in _est_scores if not np.isnan(s)]
        score = float(np.min(valid_scores)) if valid_scores else float("nan")
    else:
        score = float("nan")
    scores.append(score)
    # Persist every estimator's FI run.
    for _k, _fi in _est_fi_runs:
        feature_importances[_k] = _fi

    if 0 not in evaluated_scores_mean:

        # Dummy baselines serve as fitness @ 0 features.
        if not self.nofeatures_dummy_scoring:
            # Sign-direction-agnostic "worse than model" placeholder. ``score/10`` silently put the dummy ABOVE the model when _sign==+1 and score was negative (e.g. R^2 < 0). Subtracting a positive number always makes the score worse under both sklearn conventions; use a magnitude-relative fudge so it scales with the metric in play (log-loss ~0.5, MSE ~1e6, R^2 ~1).
            fudge = max(abs(score), 1e-3) * 9.0
            dummy_scores.append(score - fudge)
        else:
            dummy_scores.append(
                get_best_dummy_score(estimator=estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, scoring=scoring)
            )
