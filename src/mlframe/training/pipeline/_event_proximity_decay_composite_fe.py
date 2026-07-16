"""Event-proximity decay FE (``feature_engineering.event_proximity_decay.event_proximity_decay_features``),
applied at the same pre-encoding point as the other composite-FE steps.

The "reference" this trick needs isn't derivable from the training frame OR an auxiliary table at
all -- it's caller-supplied domain knowledge (e.g. a known holiday calendar), so the event dates are
a literal config value, not a column/table. Dates themselves come from ``timestamps`` (the same
full-length, pre-split array threaded from the FeaturesAndTargetsExtractor's own ts_field
resolution used by the entity/time composite-FE steps) rather than a column lookup on
train/val/test: the suite decomposes datetime COLUMNS into numeric day/month/weekday parts BEFORE
this pre-encoding insertion point runs (a separate, earlier phase), so a raw datetime column is
already gone from the frame by the time any composite-FE step here would see it -- ``timestamps``
is captured earlier, before that decomposition, and survives.
No fit-time state; predict-time replay re-runs the same computation using the SAME persisted
event-date list against the predict frame's own ``timestamps``.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import polars as pl

from mlframe.feature_engineering.event_proximity_decay import event_proximity_decay_features

logger = logging.getLogger(__name__)


def _attach_new_columns(df: Any, new_cols: "pd.DataFrame") -> Any:
    """Attach new_cols (a pandas frame) onto df, matching df's own polars/pandas type."""
    if new_cols.shape[1] == 0:
        return df
    if isinstance(df, pl.DataFrame):
        return df.with_columns([pl.Series(c, new_cols[c].to_numpy()) for c in new_cols.columns])
    return df.join(new_cols) if hasattr(df, "join") else pd.concat([df, new_cols], axis=1)


def _row_count(df: Any) -> int:
    """Row count of df, or 0 if df is None."""
    return df.shape[0] if df is not None else 0


def apply_event_proximity_decay_composite_fe(
    train_df: Any,
    val_df: Any,
    test_df: Any,
    config: Any,
    timestamps: Any,
    train_idx: Optional[np.ndarray],
    val_idx: Optional[np.ndarray],
    test_idx: Optional[np.ndarray],
    metadata: Optional[dict] = None,
    verbose: int = 0,
) -> tuple:
    """No-op unless ``config.event_proximity_decay_event_dates`` is set AND ``timestamps`` is available."""
    event_dates: List = list(getattr(config, "event_proximity_decay_event_dates", None) or [])
    if not event_dates or timestamps is None or train_df is None:
        return train_df, val_df, test_df

    cap = getattr(config, "event_proximity_decay_cap", 30)
    cap_before = getattr(config, "event_proximity_decay_cap_before", None)
    cap_after = getattr(config, "event_proximity_decay_cap_after", None)
    column_prefix = getattr(config, "event_proximity_decay_column_prefix", "event_proximity")

    if metadata is not None:
        metadata["event_proximity_decay_event_dates"] = event_dates
        metadata["event_proximity_decay_cap"] = cap
        metadata["event_proximity_decay_cap_before"] = cap_before
        metadata["event_proximity_decay_cap_after"] = cap_after
        metadata["event_proximity_decay_column_prefix"] = column_prefix

    ts_arr = np.asarray(timestamps)

    def _slice(idx: Optional[np.ndarray], n_rows: int) -> Optional[np.ndarray]:
        """Slice ts_arr down to idx, or return None if idx doesn't match n_rows."""
        if idx is None:
            return None
        idx_arr = np.asarray(idx)
        if len(idx_arr) != n_rows or len(ts_arr) <= int(idx_arr.max()):
            return None
        return np.asarray(ts_arr[idx_arr])

    _splits = {"train": (train_df, train_idx), "val": (val_df, val_idx), "test": (test_df, test_idx)}
    out: dict = {}
    for split_name, (df, idx) in _splits.items():
        if df is None:
            out[split_name] = None
            continue
        ts_split = _slice(idx, _row_count(df))
        if ts_split is None:
            out[split_name] = df
            continue
        try:
            result = event_proximity_decay_features(
                pd.Series(ts_split), event_dates, cap=cap, column_prefix=column_prefix,
                cap_before=cap_before, cap_after=cap_after,
            )
            out[split_name] = _attach_new_columns(df, result.reset_index(drop=True))
            if verbose:
                logger.info("apply_event_proximity_decay_composite_fe[%s]: added %d column(s)", split_name, result.shape[1])
        except Exception:
            logger.warning("apply_event_proximity_decay_composite_fe: step failed for split %r; skipping.", split_name, exc_info=True)
            out[split_name] = df

    return out["train"], out["val"], out["test"]


class _ReplayConfig:
    """Replays this FE step's config from fitted-pipeline metadata (for inference-time reapplication)."""

    def __init__(self, metadata: dict):
        self.event_proximity_decay_event_dates = metadata.get("event_proximity_decay_event_dates") or []
        self.event_proximity_decay_cap = metadata.get("event_proximity_decay_cap", 30)
        self.event_proximity_decay_cap_before = metadata.get("event_proximity_decay_cap_before")
        self.event_proximity_decay_cap_after = metadata.get("event_proximity_decay_cap_after")
        self.event_proximity_decay_column_prefix = metadata.get("event_proximity_decay_column_prefix", "event_proximity")


def replay_event_proximity_decay_composite_fe(df: Any, metadata: dict, timestamps: Any, verbose: int = 0) -> Any:
    """Predict-time replay: identical computation using the SAME persisted event-date list against
    the predict frame's own ``timestamps``."""
    if df is None or timestamps is None:
        return df
    config = _ReplayConfig(metadata)
    if not config.event_proximity_decay_event_dates:
        return df
    train, _, _ = apply_event_proximity_decay_composite_fe(
        df, None, None, config, timestamps, train_idx=np.arange(_row_count(df)), val_idx=None, test_idx=None, verbose=verbose,
    )
    return train


__all__ = ["apply_event_proximity_decay_composite_fe", "replay_event_proximity_decay_composite_fe"]
