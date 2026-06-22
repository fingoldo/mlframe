"""``MRMR._fit_impl`` Layer 92 temporal-aggregate FE stage.

Carved verbatim out of the giant ``_fit_impl`` orchestration body in
``_fit_impl_core.py`` (Tier E partial split) to shrink the parent below the
monolith budget. ``_fe_stage_temporal_agg`` is the self-contained
``if fe_temporal_agg_enable:`` block: temporal leak-safe grouped aggregations
keyed on a time column, only ever seeing the strict past (expanding / rolling /
lag), each survivor MI-gated against y, recipes stored for transform-time
replay against TRAIN history only. Routing piggybacks on
``hybrid_orth_features_``.

The block reads the ``MRMR`` instance + a couple of pure fit-body locals
(``_y_np`` / ``verbose``) and mutates ``self`` (``temporal_agg_features_`` /
``hybrid_orth_features_``) plus the passed-in ``_temporal_agg_pre_recipes``
dict in place. It reassigns the working ``X`` frame, so the (possibly
replaced) ``X`` is RETURNED and the single call site rebinds it. Behaviour is
byte-for-byte identical to the inlined block -- same appended columns, same
order, same recipes, same attributes set, same RNG (none used here). The lazy
in-body ``from .._temporal_agg_fe import ...`` import stays inside the function
to preserve the original import timing.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def _fe_stage_temporal_agg(self, X, _y_np, verbose, _temporal_agg_pre_recipes):
    """Layer 92 temporal-aggregate FE stage carved out of ``_fit_impl``.

    Threads the ``MRMR`` instance + fit-body locals explicitly, mutates
    ``self`` (``temporal_agg_features_`` / ``hybrid_orth_features_``) and the
    passed-in ``_temporal_agg_pre_recipes`` dict in place, and RETURNS the
    (possibly replaced) working ``X`` frame so the caller can rebind it.
    """
    # Layer 92 (2026-06-01): temporal leak-safe grouped aggregations. Keyed on
    # a time column, only ever seeing the strict past (expanding / rolling /
    # lag). Each survivor MI-gated against y; recipes store the fit-time
    # per-entity sorted history so transform() replays test rows against TRAIN
    # history only. Routing piggybacks on hybrid_orth_features_.
    if bool(getattr(self, "fe_temporal_agg_enable", False)):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Layer 92 temporal_agg FE enabled but X is not a pandas "
                "DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._temporal_agg_fe import hybrid_temporal_agg_fe

                _ta_time = getattr(self, "fe_temporal_agg_time_col", None)
                _ta_entities = [
                    c for c in (getattr(self, "fe_temporal_agg_entity_cols", ()) or ())
                    if c in X.columns
                ]
                _ta_values = [
                    c for c in (getattr(self, "fe_temporal_agg_value_cols", ()) or ())
                    if c in X.columns
                ]
                if _ta_time is None or _ta_time not in X.columns or not _ta_entities or not _ta_values:
                    if verbose:
                        logger.info(
                            "MRMR.fit temporal_agg: skipped (need time_col + "
                            "entity_cols + value_cols all present in X)."
                        )
                else:
                    _y_for_ta = (
                        _y_np
                    )
                    if _y_for_ta.dtype.kind in "fc":
                        if int(np.unique(_y_for_ta).size) <= 32:
                            _y_for_ta = _y_for_ta.astype(np.int64)
                        else:
                            try:
                                _y_for_ta = pd.qcut(
                                    _y_for_ta, q=10, labels=False, duplicates="drop",
                                ).astype(np.int64)
                            except Exception:
                                _y_for_ta = _y_for_ta.astype(np.int64)
                    _ta_stats = tuple(
                        getattr(self, "fe_temporal_agg_stats", ())
                        or ("mean", "std", "count")
                    )
                    _ta_windows = tuple(getattr(self, "fe_temporal_agg_windows", ()) or ())
                    _ta_lags = tuple(getattr(self, "fe_temporal_agg_lags", (1,)) or ())
                    _ta_top_k = int(getattr(self, "fe_temporal_agg_top_k", 10))
                    _X_before_ta_cols = list(X.columns)
                    X_ta, _ta_appended, _ta_recipes, _ta_scores = hybrid_temporal_agg_fe(
                        X, _y_for_ta,
                        entity_cols=_ta_entities, value_cols=_ta_values,
                        time_col=_ta_time, stats=_ta_stats,
                        windows=_ta_windows, lags=_ta_lags, top_k=_ta_top_k,
                    )
                    _ta_appended = [
                        c for c in _ta_appended if c not in _X_before_ta_cols
                    ]
                    if _ta_appended:
                        X = X_ta
                        self.temporal_agg_features_ = list(_ta_appended)
                        self.hybrid_orth_features_ = (
                            list(self.hybrid_orth_features_ or []) + list(_ta_appended)
                        )
                        for _r in _ta_recipes:
                            if _r.name in _ta_appended:
                                _temporal_agg_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit temporal_agg: appended %d engineered "
                                "column(s): %s",
                                len(_ta_appended), _ta_appended[:8],
                            )
            except Exception as _ta_exc:
                logger.warning(
                    "MRMR.fit temporal_agg FE raised %s: %s; continuing without "
                    "temporal-aggregate columns.",
                    type(_ta_exc).__name__, _ta_exc,
                )
    return X
