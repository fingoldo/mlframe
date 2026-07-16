"""Two-step recency-weighted target encoding
(``feature_engineering.two_step_target_encode.two_step_recency_weighted_target_encode``), applied
at the same pre-encoding point as the other composite-FE steps.

Unlike ``_entity_time_composite_fe.py``'s tricks, this one genuinely needs FIT-TIME STATE: the
underlying function needs ``y`` directly, which doesn't exist at predict time. The wiring here
builds a standard fit/lookup target-encoder around it:

  * TRAIN rows get the function's own ``causal=True`` expanding-window encoding (leak-free: row i's
    value only sees its own entity's events up to and including its own time).
  * A separate per-ENTITY lookup table is built from ``causal=False`` on TRAIN ONLY (each entity's
    terminal/full-train-history recency-weighted encoding) and persisted onto ``metadata``.
  * VAL/TEST/predict rows look up their entity in that table; entities unseen in train fall back to
    a smoothed global prior (persisted alongside).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import polars as pl

from mlframe.feature_engineering.two_step_target_encode import two_step_recency_weighted_target_encode

logger = logging.getLogger(__name__)

_ENTITY_COL = "__mlframe_tste_entity__"
_TIME_COL = "__mlframe_tste_time__"


def _to_pandas(df: Any) -> Optional[pd.DataFrame]:
    """Convert a polars DataFrame to pandas; pass through pandas/None unchanged."""
    if df is None:
        return None
    return df.to_pandas() if isinstance(df, pl.DataFrame) else df


def _attach_new_columns(df: Any, new_cols: "pd.DataFrame") -> Any:
    """Attach new_cols (a pandas frame) onto df, matching df's own polars/pandas type."""
    if new_cols.shape[1] == 0:
        return df
    if isinstance(df, pl.DataFrame):
        return df.with_columns([pl.Series(c, new_cols[c].to_numpy()) for c in new_cols.columns])
    return df.join(new_cols) if hasattr(df, "join") else pd.concat([df, new_cols], axis=1)


def _out_col_name(columns: list) -> str:
    """Generated output column name for a two-step target encoding of the given grouping columns."""
    return "__".join(columns) + "__two_step_target_encode"


def apply_target_encoding_composite_fe(
    train_df: Any,
    val_df: Any,
    test_df: Any,
    config: Any,
    group_ids: Optional[np.ndarray],
    timestamps: Any,
    y_train: Optional[np.ndarray],
    train_idx: Optional[np.ndarray],
    val_idx: Optional[np.ndarray],
    test_idx: Optional[np.ndarray],
    metadata: Optional[dict] = None,
    verbose: int = 0,
) -> tuple:
    """No-op unless ``config.two_step_target_encode_columns`` is set AND ``group_ids``/``y_train``
    are both available (needs an entity key and a train-only target -- no leakage-safe way to fit
    without either)."""
    columns = list(getattr(config, "two_step_target_encode_columns", None) or [])
    if not columns or group_ids is None or y_train is None or train_df is None or train_idx is None:
        return train_df, val_df, test_df

    group_ids = np.asarray(group_ids)
    ts_arr = np.asarray(timestamps) if timestamps is not None else np.arange(len(group_ids), dtype=np.float64)
    train_idx_arr = np.asarray(train_idx)
    if len(train_idx_arr) != _row_count(train_df) or len(group_ids) <= int(train_idx_arr.max()):
        return train_df, val_df, test_df

    train_pd = _to_pandas(train_df)
    if train_pd is None or not all(c in train_pd.columns for c in columns):
        return train_df, val_df, test_df

    g_train = group_ids[train_idx_arr]
    t_train = ts_arr[train_idx_arr]
    y_train = np.asarray(y_train, dtype=np.float64)
    if len(y_train) != len(train_pd):
        return train_df, val_df, test_df

    decay_half_life = float(getattr(config, "two_step_target_encode_decay_half_life", 30.0) or 30.0)
    smoothing = float(getattr(config, "two_step_target_encode_smoothing", 1.0) or 1.0)

    work = train_pd[columns].copy()
    work[_ENTITY_COL] = g_train
    work[_TIME_COL] = t_train

    try:
        causal_encoding = two_step_recency_weighted_target_encode(
            work, entity_col=_ENTITY_COL, feature_cols=columns, y=y_train, time_col=_TIME_COL,
            decay_half_life=decay_half_life, smoothing=smoothing, causal=True,
        )
        terminal_encoding = two_step_recency_weighted_target_encode(
            work, entity_col=_ENTITY_COL, feature_cols=columns, y=y_train, time_col=_TIME_COL,
            decay_half_life=decay_half_life, smoothing=smoothing, causal=False,
        )
    except Exception:
        logger.warning("apply_target_encoding_composite_fe: two_step_recency_weighted_target_encode failed; skipping.", exc_info=True)
        return train_df, val_df, test_df

    out_col = _out_col_name(columns)
    entity_lookup = pd.Series(terminal_encoding, index=g_train).groupby(level=0).first().to_dict()
    global_prior = float(np.average(y_train))

    if metadata is not None:
        metadata["two_step_target_encode_columns"] = columns
        metadata["two_step_target_encode_entity_lookup"] = {str(k): float(v) for k, v in entity_lookup.items()}
        metadata["two_step_target_encode_global_prior"] = global_prior
        metadata["two_step_target_encode_out_col"] = out_col

    train_new = pd.DataFrame({out_col: causal_encoding}, index=range(len(train_pd)))
    out_train = _attach_new_columns(train_df, train_new)

    def _lookup_split(df: Any, idx: Optional[np.ndarray]) -> Any:
        """Attach the train-fitted per-entity target encoding onto a val/test split by group id."""
        if df is None:
            return None
        if idx is None or len(np.asarray(idx)) != _row_count(df) or len(group_ids) <= int(np.asarray(idx).max()):
            return df
        g = group_ids[np.asarray(idx)]
        vals = np.array([entity_lookup.get(gid, global_prior) for gid in g], dtype=np.float64)
        return _attach_new_columns(df, pd.DataFrame({out_col: vals}, index=range(len(vals))))

    out_val = _lookup_split(val_df, val_idx)
    out_test = _lookup_split(test_df, test_idx)

    if verbose:
        logger.info("apply_target_encoding_composite_fe: added %r (%d train entities in lookup)", out_col, len(entity_lookup))

    return out_train, out_val, out_test


def _row_count(df: Any) -> int:
    """Row count of df, or 0 if df is None."""
    return df.shape[0] if df is not None else 0


def replay_target_encoding_composite_fe(df: Any, metadata: dict, group_ids: Optional[np.ndarray], verbose: int = 0) -> Any:
    """Predict-time replay: entity lookup only (no y needed) -- unseen entities fall back to the
    persisted global prior."""
    if df is None or group_ids is None:
        return df
    out_col = metadata.get("two_step_target_encode_out_col")
    entity_lookup = metadata.get("two_step_target_encode_entity_lookup")
    global_prior = metadata.get("two_step_target_encode_global_prior")
    if not out_col or entity_lookup is None or global_prior is None:
        return df
    g = np.asarray(group_ids)
    if len(g) != _row_count(df):
        return df
    vals = np.array([entity_lookup.get(str(gid), global_prior) for gid in g], dtype=np.float64)
    if verbose:
        logger.info("replay_target_encoding_composite_fe: replayed %r via entity lookup", out_col)
    return _attach_new_columns(df, pd.DataFrame({out_col: vals}, index=range(len(vals))))


__all__ = ["apply_target_encoding_composite_fe", "replay_target_encoding_composite_fe"]
