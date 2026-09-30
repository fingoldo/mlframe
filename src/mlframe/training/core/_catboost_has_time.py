"""Let CatBoost's ``has_time`` follow the suite's unified CV policy, but only when the train rows really are in time order.

``has_time=True`` makes CatBoost treat ROW ORDER as time order (no shuffling when it draws the permutations behind ordered target statistics and
ordered boosting). The suite's train frame keeps the caller's row order, so the flag is only correct when the train rows are already non-decreasing in
time. Reordering would mean copying the frame and re-aligning every row-aligned artifact (weights, group ids, eval sets, OOF / predictions), so an unsorted
train set simply leaves ``has_time`` off and says why.

Enabled only when ALL hold: the policy is temporal, neither ``hyperparams_config.has_time`` nor ``cb_kwargs['has_time']`` was set by the caller,
the train rows' timestamps are known, and they are non-decreasing. An explicit ``has_time`` (True or False) is never overridden.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["timestamps_are_chronological", "decide_catboost_has_time", "apply_catboost_has_time"]


def timestamps_are_chronological(timestamps: Any) -> bool:
    """True when ``timestamps`` is non-decreasing with no missing value (O(n), no sort, no copy of the caller's array)."""
    if timestamps is None or len(timestamps) < 2:
        return False
    series = timestamps if isinstance(timestamps, pd.Series) else pd.Series(np.asarray(timestamps))
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        series = series.dt.tz_convert("UTC").dt.tz_localize(None)
    elif series.dtype == object:
        series = pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)
    values = series.to_numpy()
    if values.dtype.kind in "mM":
        if np.isnat(values).any():
            return False
        keys = values.view(np.int64)
    elif values.dtype.kind in "iuf":
        if values.dtype.kind == "f" and np.isnan(values).any():
            return False
        keys = values
    else:
        return False
    return bool(np.all(keys[1:] >= keys[:-1]))


def _caller_set_has_time(hyperparams_config: Any) -> Optional[str]:
    """Where the caller set CatBoost's ``has_time`` explicitly, or None."""
    if "has_time" in set(getattr(hyperparams_config, "model_fields_set", None) or ()):
        return "hyperparams_config.has_time"
    cb_kwargs = getattr(hyperparams_config, "cb_kwargs", None) or {}
    if "has_time" in cb_kwargs:
        return "hyperparams_config.cb_kwargs['has_time']"
    return None


def decide_catboost_has_time(policy: Any, hyperparams_config: Any) -> Tuple[bool, str]:
    """``(enable, reason)`` for CatBoost's ``has_time`` given the suite's ``CVPolicy`` (whose ``timestamps`` are the train rows' own)."""
    explicit = _caller_set_has_time(hyperparams_config)
    if explicit:
        return False, f"{explicit} was set explicitly (left as given)"
    if policy is None or not getattr(policy, "temporal", False):
        return False, "the unified CV policy is not temporal"
    train_ts = getattr(policy, "timestamps", None)
    if train_ts is None:
        return False, "no train-row timestamps are available"
    if not timestamps_are_chronological(train_ts):
        return False, "the train rows are not in chronological order (or have missing timestamps) and reordering would copy the frame"
    return True, "the policy is temporal and the train rows are already in chronological order"


def apply_catboost_has_time(models_params: Optional[Dict[str, Any]], policy: Any, hyperparams_config: Any, verbose: Any = 0) -> bool:
    """Switch ``has_time`` on for the suite's CatBoost estimator(s) when ``decide_catboost_has_time`` says so. Returns whether anything changed."""
    if not models_params:
        return False
    targets = []
    for entry in models_params.values():
        model = entry.get("model") if isinstance(entry, dict) else None
        if model is None or not hasattr(model, "get_params"):
            continue
        try:
            keys = [k for k in model.get_params(deep=True) if k == "has_time" or k.endswith("__has_time")]
        except Exception as exc:
            logger.debug("get_params failed while looking for has_time: %s", exc)
            continue
        if not keys and type(model).__module__.startswith("catboost"):
            keys = ["has_time"]  # CatBoost reports only explicitly-set params
        if keys:
            targets.append((model, keys))
    if not targets:
        return False
    enable, reason = decide_catboost_has_time(policy, hyperparams_config)
    if not enable:
        if getattr(policy, "temporal", False) and _caller_set_has_time(hyperparams_config) is None:
            logger.info("CatBoost has_time stays off: %s.", reason)
        return False
    for model, keys in targets:
        model.set_params(**{k: True for k in keys})
    if verbose:
        logger.info("CatBoost has_time=True: %s.", reason)
    return True
