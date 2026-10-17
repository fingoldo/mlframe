"""K-fold and single-split branches of ``_oof_holdout_predictions_with_rows``; parent-module helpers are reached through ``_ens`` at call time."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

import mlframe.training.composite.ensemble as _ens

logger = logging.getLogger("mlframe.training.composite.ensemble")


def _oof_kfold_branch(
    train_X: Any,
    y_train_full: np.ndarray,
    component_models: list[Any],
    component_names: list[str],
    component_specs: list[dict[str, Any] | None],
    base_train_full_per_spec: dict[str, np.ndarray],
    group_ids: np.ndarray | None,
    sample_weight: np.ndarray | None,
    n_train: int,
    random_state: int,
    kfold: int,
    external_holdout_X: Any | None,
    _time_monotone: bool,
    _full_key: Any,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    """K-fold (or forward-walking TimeSeriesSplit) OOF branch of ``_oof_holdout_predictions_with_rows``; caches its result under ``_full_key``."""
    if external_holdout_X is not None:
        logger.warning(
            "compute_oof_holdout_predictions: both kfold>1 and external_holdout_X supplied; kfold "
            "takes precedence and the external holdout frame is IGNORED. Pass kfold=1 to use the "
            "external holdout."
        )
    # Outer OOF split must be group-aware when group_ids is supplied: plain shuffled K-fold lets same-group rows span refit-train and holdout, inflating the OOF surface the NNLS weights + dummy-floor gate consume (the inner eval-carve is group-aware but the OUTER split was not). GroupKFold keeps whole groups in one fold.
    _kf_split = _ens._outer_oof_split(n_train, int(kfold), int(random_state), component_specs, group_ids, _time_monotone)
    oof_preds_by_name: dict[str, np.ndarray] = {}
    survived_set: set[str] | None = None
    survived_set = _ens._oof_holdout_predic_fold_train_idx_fold(_kf_split, train_X, y_train_full, component_models, component_names, component_specs, base_train_full_per_spec, group_ids, sample_weight, n_train, random_state, kfold, survived_set, oof_preds_by_name)
    if not oof_preds_by_name or not survived_set:
        # Shape consistency -- match the (0, 0) tiny-data short-circuit. No components survived.
        _empty: tuple = (np.zeros((0, 0)), np.zeros(0), [], np.zeros(0, dtype=np.int64))
        if _full_key is not None:
            _ens._oof_cache_put(_full_key, _empty)
        return _empty
    surviving_names = [n for n in component_names if n in survived_set]
    cols = [oof_preds_by_name[n] for n in surviving_names]
    oof_matrix = np.column_stack(cols)
    # Drop rows with any NaN (folds that lost every component) - rare.
    finite_rows = np.all(np.isfinite(oof_matrix), axis=1)
    _result = (
        oof_matrix[finite_rows],
        y_train_full.astype(np.float64)[finite_rows],
        surviving_names,
        np.flatnonzero(finite_rows),
    )
    if _full_key is not None:
        _ens._oof_cache_put(_full_key, _result)
    return _result


def _oof_single_split_branch(
    train_X: Any,
    y_train_full: np.ndarray,
    component_models: list[Any],
    component_names: list[str],
    component_specs: list[dict[str, Any] | None],
    base_train_full_per_spec: dict[str, np.ndarray],
    group_ids: np.ndarray | None,
    sample_weight: np.ndarray | None,
    n_train: int,
    random_state: int,
    holdout_frac: float,
    time_ordering: np.ndarray | None,
    _full_key: Any,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    """Single trailing/random/group-aware holdout branch of ``_oof_holdout_predictions_with_rows``; caches its result under ``_full_key``."""
    use_time_split = False
    _time_order = None
    _time_order, use_time_split = _ens._oof_holdout_predic_decide_whether_do_time(time_ordering, _time_order, use_time_split)

    n_holdout = max(round(n_train * holdout_frac), 1)
    # Group-aware outer holdout: carve WHOLE groups so no group spans the refit-train and holdout slices (a random row permutation lets same-group rows leak across the split, inflating the OOF surface). Only when group_ids covers all rows with enough distinct groups and we are not on the explicit time-split path.
    _group_holdout = None
    _group_holdout = _ens._oof_holdout_predic_group_aware_outer_holdout(use_time_split, group_ids, n_train, _group_holdout)
    holdout_idx, train_idx = _ens._oof_holdout_predic_use_time_split(use_time_split, n_train, n_holdout, _time_order, _group_holdout, random_state)

    # Subset X. Branch on type so we don't pull pandas APIs on polars frames.
    X_holdout, X_stack = _ens._oof_holdout_predic_subset_branch_type_don(train_X, train_idx, holdout_idx)

    y_stack = y_train_full[train_idx].astype(np.float64)
    y_holdout = y_train_full[holdout_idx].astype(np.float64)

    # Group-aware inner eval-carve parity with the raw branch + kfold path: subset the caller-supplied group_ids to the stack rows so neither the composite nor the raw inner-eval carve splits a group across fit/eval (within-group leakage under-stops the booster and degrades OOF RMSE on group-aware splits).
    _group_stack = None
    _group_stack = _ens._oof_holdout_predic_group_aware_inner_eval(group_ids, n_train, train_idx, _group_stack)

    holdout_cols: list[np.ndarray] = []
    surviving_names: list[str] = []
    _pair_memo: dict[Any, Any] = {}
    _ens._oof_holdout_predic_model_name_spec_zip(component_models, component_names, component_specs, _pair_memo, X_stack, X_holdout, y_stack, base_train_full_per_spec, train_idx, group_ids, sample_weight, _group_stack, random_state, holdout_cols, surviving_names)

    # Summary log so operators can see "ensemble built with N of K components" at INFO without grepping per-component WARN lines. ``component_names`` is the full caller-supplied list; ``surviving_names`` is the subset whose refit succeeded.
    _surviving_n = len(surviving_names)
    _total_n = len(component_names)
    if _surviving_n < _total_n:
        _dropped = [n for n in component_names if n not in set(surviving_names)]
        logger.info(
            "compute_oof_holdout_predictions: built OOF matrix with %d of %d components "
            "(dropped %d: %s). Per-component drop reasons logged at WARN above.",
            _surviving_n, _total_n, _total_n - _surviving_n, _dropped,
        )
    if not holdout_cols:
        # Shape consistency -- match the tiny-data + kfold short-circuits.
        _empty: tuple[Any, ...] = (np.zeros((0, 0)), np.zeros(0), [], np.zeros(0, dtype=np.int64))
        if _full_key is not None:
            _ens._oof_cache_put(_full_key, _empty)
        return _empty
    _final = (np.column_stack(holdout_cols), y_holdout, surviving_names, np.asarray(holdout_idx))
    if _full_key is not None:
        _ens._oof_cache_put(_full_key, _final)
    return _final
