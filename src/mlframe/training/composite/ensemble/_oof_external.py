"""The external-holdout OOF path of ``compute_oof_holdout_predictions``: refit each component on the full train, predict the val frame.

Carved out of ``ensemble/__init__.py`` to keep that module under the 1000-line limit; re-exported there, so existing
imports of ``mlframe.training.composite.ensemble._compute_oof_with_external_holdout`` are unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]
from sklearn.base import clone

from .._composite_utils import is_polars_df as _is_polars_df
from ..estimator import CompositeTargetEstimator
from mlframe.training.composite.estimator.shared import extract_groups as _extract_groups
from ..transforms import get_transform
from ._oof_split import _align_fit_sw, _carve_inner_eval_split
from mlframe.utils.log_throttle import log_throttle
from mlframe.training.composite.transforms.shared import call_transform

logger = logging.getLogger("mlframe.training.composite.ensemble")


def _compute_oof_with_external_holdout(
    *,
    # Slice-stable ES (mlframe.training.SliceStableESConfig) is NOT propagated into the inner OOF refit loop: this function builds its own per-fold ``eval_set`` via ``_carve_eval_set_from_train_with_groups`` and a single (X_holdout, y_holdout) pair, incompatible with the multi-eval-set / per-shard registration path slice-ES needs. Callers wanting robust ES inside OOF refit should use full-K-fold CV with an outer selector (see ``_cv_aggregation.aggregate_fold_scores``).
    component_models: list[Any],
    component_names: list[str],
    component_specs: list[dict[str, Any] | None],
    train_X: Any,
    y_train_full: np.ndarray,
    base_train_full_per_spec: dict[str, np.ndarray],
    external_holdout_X: Any,
    external_holdout_y: np.ndarray,
    sample_weight: np.ndarray | None,
    full_key: tuple | None,
    group_ids: np.ndarray | None = None,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fit each component clone on full train, predict on caller-supplied external holdout (typically the suite's val split).

    Mirrors the per-component branch in :func:`compute_oof_holdout_predictions` but skips the internal train/holdout slicing.

    Holdout-side base columns are NOT taken as an argument: the ``CompositeTargetEstimator`` wrapper re-extracts its base column from ``external_holdout_X`` itself during ``predict``, so a parallel per-spec holdout-base dict would be dead weight (the train-side ``base_train_full_per_spec`` is still needed because it drives the transform.forward that produces the T values the inner is re-fit on).
    """
    from mlframe.training.composite import ensemble as _ens  # helpers stay in the package module; resolved at call time (import order)

    y_train_full = y_train_full.astype(np.float64)
    holdout_cols: list[np.ndarray] = []
    surviving_names: list[str] = []
    _pair_memo: dict = {}
    for model, name, spec in zip(
        component_models, component_names, component_specs,
    ):
        try:
            inner, pp = _ens._unwrap_shim(model)
            X_stack_t, X_holdout_t = _ens._transform_pair_cached(
                _pair_memo, pp, train_X, external_holdout_X, y_train=y_train_full,
            )
            if isinstance(inner, CompositeTargetEstimator):
                if spec is None:
                    raise ValueError("composite component with no spec")
                base_full = base_train_full_per_spec.get(
                    str(spec.get("name") or spec.get("base_column")),
                )
                if base_full is None:
                    raise ValueError(f"missing base column '{spec['base_column']}' " "for external-holdout OOF (train side)")
                transform = get_transform(spec["transform_name"])
                valid = transform.domain_check(y_train_full, base_full)
                if valid.sum() < 10:
                    raise ValueError("too few valid rows after domain filter")
                # A grouped transform reads the component's own group column from the raw train frame.
                _t_group_col = getattr(inner, "group_column", None)
                _t_groups = _extract_groups(train_X, _t_group_col)[valid] if _t_group_col else None
                t_train = call_transform(transform, "forward", y_train_full[valid], base_full[valid], spec["fitted_params"], groups=_t_groups)
                inner_clone = clone(inner.estimator_)
                if isinstance(X_stack_t, pd.DataFrame):
                    X_train_valid = X_stack_t.iloc[valid].reset_index(
                        drop=True,
                    )
                elif _is_polars_df(X_stack_t):
                    X_train_valid = X_stack_t.filter(pl.Series(valid))
                else:
                    X_train_valid = X_stack_t[valid]
                _sw_train_valid = None if sample_weight is None else sample_weight[valid]
                _group_for_valid = None
                if group_ids is not None:
                    try:
                        _g_arr = np.asarray(group_ids)
                        if _g_arr.shape[0] == valid.shape[0]:
                            _group_for_valid = _g_arr[valid]
                    except (TypeError, IndexError):
                        _group_for_valid = None
                _X_fit_c, _t_fit_c, _X_ev_c, _t_ev_c, _fm_c = (
                    _carve_inner_eval_split(
                        X_train_valid, t_train, random_state=int(random_state),
                        group_ids=_group_for_valid, return_fit_mask=True,
                    )
                )
                _eval_set_c = (_X_ev_c, _t_ev_c) if _X_ev_c is not None else None
                _sw_fit_c = _align_fit_sw(_sw_train_valid, _fm_c, len(_t_fit_c))
                _ens._maybe_pass_sample_weight(
                    inner_clone, _X_fit_c, _t_fit_c, _sw_fit_c,
                    eval_set=_eval_set_c, fitted_source=inner.estimator_,
                )
                wrapped = _ens._wrap_fitted_inner(spec, inner_clone, spec["fitted_params"], y_train_full[valid], base_full[valid], getattr(inner, "group_column", None),
                                                  train_X, valid)
                preds = wrapped.predict(external_holdout_X, inner_X=X_holdout_t)
            else:
                inner_clone = clone(inner)
                _X_fit_r, _y_fit_r, _X_ev_r, _y_ev_r, _fm_r = (
                    _carve_inner_eval_split(
                        X_stack_t, y_train_full, random_state=int(random_state),
                        group_ids=group_ids, return_fit_mask=True,
                    )
                )
                _eval_set_r = (_X_ev_r, _y_ev_r) if _X_ev_r is not None else None
                _sw_fit_r = _align_fit_sw(sample_weight, _fm_r, len(_y_fit_r))
                _ens._maybe_pass_sample_weight(
                    inner_clone, _X_fit_r, _y_fit_r, _sw_fit_r,
                    eval_set=_eval_set_r, fitted_source=inner,
                )
                preds = inner_clone.predict(X_holdout_t)
            preds = np.asarray(preds).reshape(-1).astype(np.float64)
            if preds.shape[0] != external_holdout_y.shape[0]:
                raise ValueError(f"component '{name}' predicted " f"{preds.shape[0]} rows but external holdout has " f"{external_holdout_y.shape[0]}")
            if not np.all(np.isfinite(preds)):
                raise ValueError("non-finite holdout predictions")
            holdout_cols.append(preds)
            surviving_names.append(name)
        except Exception as exc:  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
            log_throttle(
                logger, "ensemble_external_holdout_oof_refit_failed", logging.WARNING,
                "[CompositeCrossTargetEnsemble] external-holdout OOF " "refit failed for component '%s': %s. Excluded from " "ensemble weights.",
                name,
                exc,
            )
            continue
    _surviving_n = len(surviving_names)
    _total_n = len(component_names)
    if _surviving_n < _total_n:
        _dropped = [n for n in component_names if n not in set(surviving_names)]
        logger.info(
            "compute_oof_holdout_predictions (external-holdout): built "
            "OOF matrix with %d of %d components (dropped %d: %s).",
            _surviving_n, _total_n, _total_n - _surviving_n, _dropped,
        )
    if not holdout_cols:
        _empty: tuple = (np.zeros((0, 0)), np.zeros(0), [])
        if full_key is not None:
            _ens._oof_cache_put(full_key, _empty)
        return _empty
    _final = (
        np.column_stack(holdout_cols),
        external_holdout_y,
        surviving_names,
    )
    if full_key is not None:
        _ens._oof_cache_put(full_key, _final)
    return _final
