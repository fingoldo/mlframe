"""CV resolution + early-stopping val_cv build for ``RFECV.fit``.

Carved out of ``_rfecv_fit``'s pre-while setup. Resolves the ``cv``
argument into a concrete splitter:

* int cv -> auto-detect time-series via a 4-source priority chain
  (suite-level ``timestamps`` hint -> polars schema hint ->
  pandas DatetimeIndex monotonicity -> single polars datetime column
  monotonicity). If detected, swap to ``TimeSeriesSplit``.
* int cv + classifier + groups -> ``StratifiedGroupKFold``.
* int cv + classifier + no groups -> ``StratifiedKFold``.
* int cv + regressor + groups -> ``GroupKFold``.
* int cv + regressor + no groups -> ``KFold``.

If ``early_stopping_val_nsplits`` is set, builds ``val_cv`` by cloning
``cv`` with the override; falls back to ``copy.copy + setattr(n_splits)``
for splitters whose ``get_params()`` doesn't expose the slot (LOO etc.)
and WARNS loudly because the override may be silently ignored.

Re-imported at the parent's module bottom so historical
``from ._rfecv_fit import _resolve_cv_and_val_cv`` keeps resolving
transparently.
"""
from __future__ import annotations

import copy
import logging

import numpy as np
import pandas as pd

from sklearn.base import is_classifier
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

logger = logging.getLogger("mlframe.feature_selection.wrappers._rfecv")


def _resolve_cv_and_val_cv(
    *,
    cv,
    X,
    y,
    groups,
    estimator,
    cv_shuffle: bool,
    random_state,
    fit_params,
    early_stopping_val_nsplits,
    early_stopping_rounds,
    _polars_time_series_hint: bool,
    verbose,
):
    """Resolve ``cv`` -> concrete splitter and build optional ``val_cv``.

    Returns ``(cv, val_cv, early_stopping_rounds)``. ``val_cv`` is None
    when ``early_stopping_val_nsplits`` was falsy. ``cv`` is the original
    object when it was already a splitter (no int / numeric-string).
    """
    if cv is None or str(cv).isnumeric():
        if cv is None:
            cv = 3
        # Time-series auto-detect: a monotonic datetime axis means KFold-style shuffles would leak future into past; TimeSeriesSplit is
        # the correct choice. Triggers ONLY when groups is None (group-aware splits already handle temporal grouping) and the
        # detected datetime axis is strictly increasing (avoid TSS on randomly-shuffled datetime data, which has no ordering meaning).
        # Pandas path: DatetimeIndex.is_monotonic_increasing. Polars path: scan for a single datetime / date column that is
        # monotonically increasing (polars has no row index; the time axis is necessarily a column).
        _is_time_series = False
        # Suite-level timestamps hint: callers pass ``timestamps=`` via fit_params (1-D monotonic
        # array-like). This catches the case where X has no DatetimeIndex / no polars datetime col
        # but the suite knows the row order is temporal (e.g. integer epoch seconds in a separate
        # array). Honour the hint regardless of X's schema.
        _ts_hint = fit_params.pop("timestamps", None) if isinstance(fit_params, dict) else None
        if groups is None and _ts_hint is not None:
            try:
                _ts_arr = np.asarray(_ts_hint)
                if _ts_arr.ndim == 1 and _ts_arr.size == (X.shape[0] if hasattr(X, "shape") else len(X)):
                    # Monotonic-non-decreasing -> time axis.
                    if bool(np.all(_ts_arr[1:] >= _ts_arr[:-1])):
                        _is_time_series = True
            except (TypeError, ValueError):
                pass
        # Polars-input path: the schema-level monotonic-datetime check happens BEFORE the to_pandas() at fit entry; the hint is set
        # there so the conversion doesn't erase the per-column polars dtype information needed for unambiguous detection.
        if not _is_time_series and groups is None and _polars_time_series_hint:
            _is_time_series = True
        elif groups is None and isinstance(X, pd.DataFrame):
            _idx = X.index
            if isinstance(_idx, pd.DatetimeIndex) and _idx.is_monotonic_increasing:
                _is_time_series = True
        elif groups is None:
            try:
                import polars as _pl
                if isinstance(X, _pl.DataFrame):
                    _dt_cols = [
                        n for n, d in X.schema.items()
                        if d in (_pl.Datetime, _pl.Date) or str(d).startswith(("Datetime", "Date"))
                    ]
                    # Exactly one datetime column = unambiguous time axis; multiple datetimes would require the
                    # caller to disambiguate via an explicit cv= (we won't guess which column orders the rows).
                    if len(_dt_cols) == 1:
                        _col = X.get_column(_dt_cols[0])
                        if _col.is_sorted(descending=False) and _col.null_count() == 0:
                            _is_time_series = True
            except ImportError:
                pass
        if _is_time_series:
            cv = TimeSeriesSplit(n_splits=cv)
            if verbose:
                logger.info(
                    "Using cv=%s (auto-detected from monotonic DatetimeIndex; "
                    "pass cv=KFold(...) explicitly to override).", cv,
                )
        elif is_classifier(estimator):
            if groups is not None:
                cv = StratifiedGroupKFold(n_splits=cv, shuffle=cv_shuffle, random_state=random_state if cv_shuffle else None)
            else:
                if cv_shuffle:
                    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
                else:
                    cv = StratifiedKFold(n_splits=cv, shuffle=False)
        else:
            if groups is not None:
                cv = GroupKFold(n_splits=cv)  # GroupKFold doesn't support shuffle/random_state
            else:
                if cv_shuffle:
                    cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)
                else:
                    cv = KFold(n_splits=cv, shuffle=False)
        if verbose and not _is_time_series:
            logger.info("Using cv=%s", cv)

    if early_stopping_val_nsplits:
        try:
            # sklearn KFold-family raises ValueError if random_state is set while shuffle=False. Drop random_state in that case.
            _cv_params = dict(cv.get_params())
            if _cv_params.get("shuffle") is False and _cv_params.get("random_state") is not None:
                _cv_params["random_state"] = None
            _cv_params["n_splits"] = early_stopping_val_nsplits
            val_cv = type(cv)(**_cv_params)
        except (AttributeError, TypeError, ValueError):
            # Fallback for LeaveOneOut / iterator-based custom CVs whose get_params() may not exist or whose n_splits is computed
            # from the data. The setattr-style fallback below may silently write to a meaningless attribute and the CV runs with its
            # original split count, ignoring the user's early_stopping_val_nsplits request - warn loudly.
            logger.warning(
                "RFECV: cv=%r does not accept n_splits via get_params(); "
                "falling back to copy.copy + attribute assignment. The user's "
                "early_stopping_val_nsplits=%s may be IGNORED if this CV "
                "computes its split count from the data (e.g. LeaveOneOut). "
                "Pass an explicit val_cv-compatible splitter to silence this.",
                type(cv).__name__, early_stopping_val_nsplits,
            )
            val_cv = copy.copy(cv)
            val_cv.n_splits = early_stopping_val_nsplits
        if not early_stopping_rounds:
            early_stopping_rounds = 20
    else:
        val_cv = None

    return cv, val_cv, early_stopping_rounds
