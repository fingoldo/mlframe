"""CV resolution + early-stopping val_cv build for ``RFECV.fit``.

Carved out of ``_rfecv_fit``'s pre-while setup. Resolves the ``cv``
argument into a concrete splitter:

* a splitter name (``"KFold"``) or class -> instantiated (``n_splits=3`` when it takes one); a numeric string -> int.
* int cv -> auto-detect time-series via a 4-source priority chain
  (suite-level ``timestamps`` hint -> polars schema hint ->
  pandas DatetimeIndex monotonicity -> single polars datetime column
  monotonicity). If detected, swap to ``TimeSeriesSplit``.
* int cv + a ``timestamps`` hint whose values are NOT sorted -> ``TimestampOrderedSplit`` (forward-chains in timestamp
  order, at the group level when groups are given), since every positional splitter would chain in the wrong order.
* int cv + groups + a temporal signal -> ``GroupTimeSeriesSplit`` (entity isolation AND forward-chained time
  order; falls back to the group KFold below when there are too few groups, or when ``cv_shuffle=True`` opts out).
* int cv + classifier + groups -> ``StratifiedGroupKFold``.
* int cv + classifier + no groups -> ``StratifiedKFold``.
* int cv + regressor + groups -> ``GroupKFold``.
* int cv + regressor + no groups -> ``KFold``.

If ``early_stopping_val_nsplits`` is set, ``val_cv`` is ``cv`` rebuilt with that many folds (see ``_derive_val_cv``).

Re-imported at the parent's module bottom so historical
``from ._fit import _resolve_cv_and_val_cv`` keeps resolving
transparently.
"""
from __future__ import annotations

import copy
import inspect
import logging
from typing import Any

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

from ._timestamp_ordered_split import TimestampOrderedSplit

logger = logging.getLogger("mlframe.feature_selection.wrappers.rfecv")


_DEFAULT_N_SPLITS = 3


def _splitter_class_by_name(name: str) -> type:
    """Resolve a splitter class from its name: sklearn.model_selection first, then this package's own splitters."""
    import sklearn.model_selection as _skms

    from ._group_time_series_split import GroupTimeSeriesSplit

    own = {"GroupTimeSeriesSplit": GroupTimeSeriesSplit, "TimestampOrderedSplit": TimestampOrderedSplit}
    cls = own.get(name) or getattr(_skms, name, None)
    if not isinstance(cls, type) or not hasattr(cls, "split"):
        raise ValueError(f"RFECV: cv={name!r} is not a known CV splitter name (sklearn.model_selection or {sorted(own)}).")
    return cls


def _init_param_names(cls: type) -> set:
    """Keyword parameters of ``cls.__init__``, the same introspection sklearn's own splitter ``__repr__`` relies on."""
    try:
        return {p.name for p in inspect.signature(cls).parameters.values() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    except (TypeError, ValueError):
        return set()


def _coerce_cv_spec(cv: Any, cv_shuffle: bool, random_state: Any) -> Any:
    """Turn a numeric string into an int and a splitter name or class into an instance; anything else passes through.

    A class gets ``n_splits=3`` (RFECV's ``cv=None`` default) when it takes one, plus ``shuffle`` / ``random_state`` when
    ``cv_shuffle`` asks for a shuffle and the class accepts them.
    """
    if isinstance(cv, str):
        if cv.strip().isnumeric():
            return int(cv)
        cv = _splitter_class_by_name(cv.strip())
    if isinstance(cv, type):
        params = _init_param_names(cv)
        kwargs: dict = {}
        if "n_splits" in params:
            kwargs["n_splits"] = _DEFAULT_N_SPLITS
        if cv_shuffle and "shuffle" in params:
            kwargs["shuffle"] = True
            if "random_state" in params:
                kwargs["random_state"] = random_state
        return cv(**kwargs)
    return cv


def _derive_val_cv(cv: Any, n_splits: int) -> Any:
    """The early-stopping splitter: ``cv`` rebuilt with ``n_splits`` folds.

    sklearn splitters have no ``get_params``; like their own ``__repr__``, the constructor signature is read and every
    parameter taken from the same-named attribute. Only a splitter with no ``n_splits`` slot at all (LeaveOneOut and
    other data-sized splitters) cannot honour the override, and only that case warns.
    """
    custom = getattr(cv, "early_stopping_val_cv", None)
    if callable(custom):
        return custom(n_splits)
    params = _init_param_names(type(cv))
    if "n_splits" in params and all(hasattr(cv, p) for p in params):
        kwargs = {p: getattr(cv, p) for p in params}
        kwargs["n_splits"] = n_splits
        if kwargs.get("shuffle") is False and kwargs.get("random_state") is not None:
            kwargs["random_state"] = None
        try:
            return type(cv)(**kwargs)
        except (TypeError, ValueError) as exc:
            logger.debug("RFECV: rebuilding %s with n_splits=%d failed (%s); falling back to attribute assignment.", type(cv).__name__, n_splits, exc)
    if not hasattr(cv, "n_splits"):
        logger.warning(
            "RFECV: cv=%s has no n_splits (its fold count comes from the data), so early_stopping_val_nsplits=%d cannot "
            "apply; its early-stopping hold-out uses the same splitter. Pass an n_splits-based cv to control it.",
            type(cv).__name__, n_splits,
        )
        return copy.deepcopy(cv)
    # deepcopy, not copy: a shallow copy shares mutable internals, so the n_splits write would leak into the caller's cv.
    val_cv = copy.deepcopy(cv)
    val_cv.n_splits = n_splits
    return val_cv


def _make_group_time_series(n_splits, groups, fallback, verbose):
    """Build a GroupTimeSeriesSplit (entity isolation + temporal fold order), or fall back to ``fallback`` when
    there are too few distinct groups for the requested n_splits (need >= n_splits + 1)."""
    from ._group_time_series_split import GroupTimeSeriesSplit

    try:
        import pandas as _pd
        n_groups = _pd.unique(np.asarray(groups)).shape[0]
    except Exception as e:
        logger.debug("pandas unique() group-count failed, falling back to np.unique: %s", e)
        n_groups = np.unique(np.asarray(groups)).shape[0]
    if n_groups < n_splits + 1:
        logger.warning(
            "RFECV: groups+temporal detected but only %d distinct groups for n_splits=%d (need >= %d); falling "
            "back to %s. Fold time-ordering is NOT guaranteed on this fallback.",
            n_groups, n_splits, n_splits + 1, type(fallback).__name__,
        )
        return fallback
    if verbose:
        logger.info(
            "RFECV: groups + temporal signal detected; using GroupTimeSeriesSplit (forward-chaining over %d "
            "time-ordered groups) so folds isolate entities AND respect time order.", n_groups,
        )
    return GroupTimeSeriesSplit(n_splits=n_splits)


def _fixed_shuffle_seed(cv_shuffle: bool, random_state):
    """A concrete seed for a shuffled splitter, so every ``.split()`` inside one fit returns the same folds.

    With ``random_state=None`` a shuffled splitter repartitions on EVERY call, and RFECV splits once per outer
    iteration: the per-fold prescreen universes, keyed by each fold's train-index bytes, then never matched a later split
    and every fold silently fell back to the leaky full-data prescreen. One seed per fit keeps the shuffle random across
    fits and identical across the splits inside this one.
    """
    if cv_shuffle and random_state is None:
        return int(np.random.default_rng().integers(0, 2**31 - 1))
    return random_state


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
    random_state = _fixed_shuffle_seed(cv_shuffle, random_state)
    cv = _coerce_cv_spec(cv, cv_shuffle, random_state)
    if cv is None or isinstance(cv, (int, np.integer)):
        if cv is None:
            cv = 3
        # Time-series auto-detect: a monotonic datetime axis means KFold-style shuffles would leak future into past; TimeSeriesSplit is
        # the correct choice. Triggers ONLY when groups is None (group-aware splits already handle temporal grouping) and the
        # detected datetime axis is strictly increasing (avoid TSS on randomly-shuffled datetime data, which has no ordering meaning).
        # Pandas path: DatetimeIndex.is_monotonic_increasing. Polars path: scan for a single datetime / date column that is
        # monotonically increasing (polars has no row index; the time axis is necessarily a column).
        _is_time_series = False
        # Suite-level timestamps hint: callers pass ``timestamps=`` via fit_params (1-D monotonic array-like). This
        # catches the case where X has no DatetimeIndex / no polars datetime col but the suite knows the row order is
        # temporal (e.g. integer epoch seconds in a separate array). Honour the hint regardless of X's schema.
        _ts_hint = fit_params.pop("timestamps", None) if isinstance(fit_params, dict) else None
        # A hint whose values are not sorted still says the rows are temporal, just not in row order: every positional
        # splitter (TimeSeriesSplit, GroupTimeSeriesSplit) would then chain in the wrong order, so it gets TimestampOrderedSplit.
        _ts_hint_unsorted = False
        if _ts_hint is not None:
            try:
                _ts_arr = np.asarray(_ts_hint)
                if _ts_arr.ndim == 1 and _ts_arr.size == (X.shape[0] if hasattr(X, "shape") else len(X)):
                    if bool(np.all(_ts_arr[1:] >= _ts_arr[:-1])):
                        _is_time_series = groups is None
                    else:
                        _ts_hint_unsorted = True
            except (TypeError, ValueError):
                pass
        # Polars-input path: the schema-level monotonic-datetime check happens BEFORE the to_pandas() at fit entry; the hint is set
        # there so the conversion doesn't erase the per-column polars dtype information needed for unambiguous detection.
        if not _is_time_series and groups is None and _polars_time_series_hint:
            _is_time_series = True
        elif groups is None and isinstance(X, pd.DataFrame):
            _idx = X.index
            if isinstance(_idx, pd.DatetimeIndex):
                # E13: NaT in DatetimeIndex makes
                # is_monotonic_increasing False; pre-fix silently falls back
                # to KFold and loses the temporal guarantee. Warn loudly.
                if _idx.hasnans:
                    if verbose:
                        logger.warning(
                            "RFECV: X.index is a DatetimeIndex with NaT; "
                            "temporal auto-detect disabled. Drop NaT rows "
                            "or pass cv=TimeSeriesSplit() explicitly to "
                            "preserve the time-ordering guarantee.",
                        )
                elif _idx.is_monotonic_increasing:
                    _is_time_series = True
        elif groups is None:
            try:
                import polars as _pl
                if isinstance(X, _pl.DataFrame):
                    _dt_cols = [n for n, d in X.schema.items() if d in (_pl.Datetime, _pl.Date) or str(d).startswith(("Datetime", "Date"))]
                    # Exactly one datetime column = unambiguous time axis; multiple datetimes would require the
                    # caller to disambiguate via an explicit cv= (we won't guess which column orders the rows).
                    if len(_dt_cols) == 1:
                        _col = X.get_column(_dt_cols[0])
                        if _col.is_sorted(descending=False) and _col.null_count() == 0:
                            _is_time_series = True
            except ImportError:
                pass
        # groups + a temporal signal: every branch above is gated on ``groups is None``, so a caller with BOTH a group
        # key and time-ordered rows would otherwise get GroupKFold / StratifiedGroupKFold, which isolate entities but
        # do NOT order folds in time (a future-dated group can land in train while a past-dated group is in test),
        # inflating the CV score on any non-stationary signal. Detect the temporal signal here and route to
        # GroupTimeSeriesSplit, which forward-chains at the group level (entity isolation AND temporal order). Honour
        # an explicit cv_shuffle=True opt-out, mirroring the non-group temporal path.
        _use_group_time_series = False
        if _ts_hint_unsorted:
            pass
        elif groups is not None and not cv_shuffle:
            _use_group_time_series = (
                _ts_hint is not None
                or _polars_time_series_hint
                or (isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex) and not X.index.hasnans and X.index.is_monotonic_increasing)
            )
        elif groups is not None and cv_shuffle and (
            _ts_hint is not None or _polars_time_series_hint
            or (isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex) and not X.index.hasnans and X.index.is_monotonic_increasing)
        ):
            logger.warning(
                "RFECV: explicit cv_shuffle=True overrides the groups+temporal auto-detect; GroupTimeSeriesSplit "
                "will NOT be substituted and fold time-ordering is not guaranteed. Drop cv_shuffle or pass a "
                "group-time-aware splitter via cv= if temporal ordering matters.",
            )
        # Respect explicit cv_shuffle=True: the auto-swap to TimeSeriesSplit voids the user's explicit shuffle request silently. Treat ``cv_shuffle=True`` as an opt-out from temporal auto-detect and
        # WARN so the caller knows their explicit choice took precedence (and that the temporal-leakage guarantee is consequently their responsibility, not the auto-detector's).
        if _is_time_series and cv_shuffle:
            logger.warning(
                "RFECV: explicit cv_shuffle=True overrides temporal auto-detect; "
                "TimeSeriesSplit will NOT be substituted. This voids the time-ordering "
                "guarantee on polars-datetime / DatetimeIndex / timestamps-hint input; "
                "pass cv=TimeSeriesSplit(...) explicitly if you want temporal folds.",
            )
            _is_time_series = False
        if _ts_hint_unsorted and cv_shuffle:
            logger.warning(
                "RFECV: explicit cv_shuffle=True overrides the fit_params['timestamps'] hint; folds are shuffled, not "
                "ordered in time. Drop cv_shuffle if temporal ordering matters.",
            )
        if _ts_hint_unsorted and not cv_shuffle:
            cv = TimestampOrderedSplit(n_splits=cv, timestamps=_ts_hint)
            logger.info(
                "RFECV: rows are not sorted by the fit_params['timestamps'] hint; using %s, which forward-chains in timestamp order%s.",
                cv, " at the group level" if groups is not None else "",
            )
        elif _is_time_series:
            cv = TimeSeriesSplit(n_splits=cv)
            if verbose:
                logger.info(
                    "Using cv=%s (auto-detected from monotonic DatetimeIndex; " "pass cv=KFold(...) explicitly to override).",
                    cv,
                )
            # Distinguish the auto-upgrade case so callers running the outer suite in temporal mode see an explicit confirmation that the inner FS CV is temporal too. Without this line, a caller who set ctx.split_config.timestamps had to inspect cv_ attribute manually to confirm the upgrade reached RFECV.
            logger.info(
                "RFECV: temporal CV upgrade applied because outer split_config is temporal " "(detected via %s); inner FS folds will respect time ordering.",
                "fit_params['timestamps'] hint" if _ts_hint is not None else ("polars datetime schema" if _polars_time_series_hint else "pandas DatetimeIndex"),
            )
        elif is_classifier(estimator):
            if groups is not None and _use_group_time_series:
                cv = _make_group_time_series(cv, groups, StratifiedGroupKFold(n_splits=cv, shuffle=False), verbose)
            elif groups is not None:
                cv = StratifiedGroupKFold(n_splits=cv, shuffle=cv_shuffle, random_state=random_state if cv_shuffle else None)
            else:
                if cv_shuffle:
                    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
                else:
                    cv = StratifiedKFold(n_splits=cv, shuffle=False)
        else:
            if groups is not None and _use_group_time_series:
                cv = _make_group_time_series(cv, groups, GroupKFold(n_splits=cv), verbose)
            elif groups is not None:
                cv = GroupKFold(n_splits=cv)  # GroupKFold doesn't support shuffle/random_state
            else:
                if cv_shuffle:
                    cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)
                else:
                    cv = KFold(n_splits=cv, shuffle=False)
        if verbose and not _is_time_series and not _ts_hint_unsorted:
            logger.info("Using cv=%s", cv)

    if early_stopping_val_nsplits:
        val_cv = _derive_val_cv(cv, int(early_stopping_val_nsplits))
        if not early_stopping_rounds:
            early_stopping_rounds = 20
    else:
        val_cv = None

    return cv, val_cv, early_stopping_rounds
