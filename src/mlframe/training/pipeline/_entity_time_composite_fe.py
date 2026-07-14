"""Entity/time-keyed FE (state_duration / recency_aggregation), applied at the same pre-encoding
point as ``_categorical_composite_fe.py`` -- both need ``group_ids``, which the suite already
resolves via the FeaturesAndTargetsExtractor but never threaded this far before.

Unlike the categorical composite steps, these are pure functions of (values, group_ids, order) with
NO fit-time state to persist: predict-time replay just re-runs the same computation against the
predict frame's own group_ids/timestamps (also resolved via the FTE). No metadata bookkeeping needed
beyond the config itself.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import polars as pl

from mlframe.feature_engineering.recency_aggregation import per_group_recency_weighted_agg
from mlframe.feature_engineering.state_duration import time_since_state_change

logger = logging.getLogger(__name__)


def _attach_new_columns(df: Any, new_cols: "pd.DataFrame") -> Any:
    """Attach ``new_cols`` (a pandas frame) onto ``df``, matching df's own polars/pandas type."""
    if new_cols.shape[1] == 0:
        return df
    if isinstance(df, pl.DataFrame):
        return df.with_columns([pl.Series(c, new_cols[c].to_numpy()) for c in new_cols.columns])
    return df.join(new_cols) if hasattr(df, "join") else pd.concat([df, new_cols], axis=1)


def _to_numpy_column(df: Any, col: str) -> Optional[np.ndarray]:
    """Return column ``col`` of ``df`` as a numpy array, or None if df/col is missing."""
    if df is None or col not in (df.columns if hasattr(df, "columns") else []):
        return None
    if isinstance(df, pl.DataFrame):
        return np.asarray(df[col].to_numpy())
    return np.asarray(df[col].to_numpy())


def _row_count(df: Any) -> int:
    """Row count of ``df``, or 0 if ``df`` is None."""
    return df.shape[0] if df is not None else 0


def apply_entity_time_composite_fe(
    train_df: Any,
    val_df: Any,
    test_df: Any,
    config: Any,
    group_ids: Optional[np.ndarray],
    timestamps: Any,
    train_idx: Optional[np.ndarray],
    val_idx: Optional[np.ndarray],
    test_idx: Optional[np.ndarray],
    metadata: Optional[dict] = None,
    verbose: int = 0,
) -> tuple:
    """Generate state_duration / recency_aggregation columns on train/val/test, opt-in per ``config``.

    When ``metadata`` is given, persists the exact column lists/params used so ``predict.py`` can
    replay identically via ``replay_entity_time_composite_fe`` without needing the original config
    object (which isn't otherwise available at predict time).

    ``group_ids``/``timestamps`` are FULL-LENGTH (pre-split) arrays aligned to the original input
    frame; sliced here by ``train_idx``/``val_idx``/``test_idx`` to align with each split's own rows.
    No-op entirely when ``group_ids`` is ``None`` (no FeaturesAndTargetsExtractor resolved a group key).
    """
    state_cols = list(getattr(config, "state_duration_columns", None) or [])
    recency_cols = list(getattr(config, "recency_aggregation_columns", None) or [])
    if group_ids is None or (not state_cols and not recency_cols) or train_df is None:
        return train_df, val_df, test_df

    if metadata is not None:
        metadata["state_duration_columns"] = state_cols
        metadata["state_duration_include_activation_count"] = bool(getattr(config, "state_duration_include_activation_count", False))
        metadata["recency_aggregation_columns"] = recency_cols
        metadata["recency_aggregation_scheme"] = str(getattr(config, "recency_aggregation_scheme", "poly") or "poly")
        metadata["recency_aggregation_param"] = float(getattr(config, "recency_aggregation_param", 1.0) or 1.0)
        metadata["recency_aggregation_agg"] = str(getattr(config, "recency_aggregation_agg", "mean") or "mean")

    group_ids = np.asarray(group_ids)
    ts_arr = np.asarray(timestamps) if timestamps is not None else None

    def _slice(idx: Optional[np.ndarray], n_rows: int):
        """Slice group_ids/timestamps down to ``idx``, or return None if idx doesn't match n_rows."""
        if idx is None:
            return None
        idx_arr = np.asarray(idx)
        if len(idx_arr) != n_rows:
            return None
        _g = group_ids[idx_arr] if len(group_ids) > int(idx_arr.max()) else None
        _t = ts_arr[idx_arr] if ts_arr is not None and len(ts_arr) > int(idx_arr.max()) else None
        return _g, _t

    _splits = {"train": (train_df, train_idx), "val": (val_df, val_idx), "test": (test_df, test_idx)}
    out: dict = {}
    for split_name, (df, idx) in _splits.items():
        if df is None:
            out[split_name] = None
            continue
        sliced = _slice(idx, _row_count(df))
        if sliced is None:
            # idx not usable for this split (length mismatch) -- skip the step for this split rather
            # than risk misaligning group_ids/timestamps against df's actual rows.
            out[split_name] = df
            continue
        g, t = sliced
        new_cols = pd.DataFrame(index=range(_row_count(df)))

        for col in state_cols:
            state_vals = _to_numpy_column(df, col)
            if state_vals is None:
                continue
            try:
                include_ac = bool(getattr(config, "state_duration_include_activation_count", False))
                result = time_since_state_change(state_vals, g, include_activation_count=include_ac)
                for out_name, out_vals in result.items():
                    new_cols[f"{col}__{out_name}"] = out_vals
            except Exception:
                logger.warning("apply_entity_time_composite_fe: state_duration step failed for column %r; skipping.", col, exc_info=True)

        for col in recency_cols:
            vals = _to_numpy_column(df, col)
            if vals is None:
                continue
            try:
                scheme = str(getattr(config, "recency_aggregation_scheme", "poly") or "poly")
                param = float(getattr(config, "recency_aggregation_param", 1.0) or 1.0)
                agg = str(getattr(config, "recency_aggregation_agg", "mean") or "mean")
                recency_vals = per_group_recency_weighted_agg(
                    np.asarray(vals, dtype=np.float64), g, agg=agg, order=t, scheme=scheme, param=param,
                )
                new_cols[f"{col}__recency_{agg}"] = recency_vals
            except Exception:
                logger.warning("apply_entity_time_composite_fe: recency_aggregation step failed for column %r; skipping.", col, exc_info=True)

        out[split_name] = _attach_new_columns(df, new_cols)
        if verbose and new_cols.shape[1] > 0:
            logger.info("apply_entity_time_composite_fe[%s]: added %d column(s)", split_name, new_cols.shape[1])

    return out["train"], out["val"], out["test"]


class _ReplayConfig:
    """Duck-typed config stand-in built from persisted ``metadata`` -- the real
    ``PreprocessingExtensionsConfig`` instance used at fit time isn't otherwise available at predict
    time (only its fitted artifacts are, via ``extensions_pipeline``/``datetime_methods``-style keys)."""

    def __init__(self, metadata: dict):
        self.state_duration_columns = metadata.get("state_duration_columns") or []
        self.state_duration_include_activation_count = metadata.get("state_duration_include_activation_count", False)
        self.recency_aggregation_columns = metadata.get("recency_aggregation_columns") or []
        self.recency_aggregation_scheme = metadata.get("recency_aggregation_scheme", "poly")
        self.recency_aggregation_param = metadata.get("recency_aggregation_param", 1.0)
        self.recency_aggregation_agg = metadata.get("recency_aggregation_agg", "mean")


def replay_entity_time_composite_fe(
    df: Any,
    metadata: dict,
    group_ids: Optional[np.ndarray],
    timestamps: Any,
    verbose: int = 0,
) -> Any:
    """Predict-time replay: identical computation to ``apply_entity_time_composite_fe``, no fit-time
    DISCOVERY state needed (pure function of the predict frame's own group_ids/timestamps) -- just the
    column lists/params persisted onto ``metadata`` at fit time."""
    if df is None or group_ids is None:
        return df
    config = _ReplayConfig(metadata)
    if not config.state_duration_columns and not config.recency_aggregation_columns:
        return df
    train, _, _ = apply_entity_time_composite_fe(
        df, None, None, config, group_ids, timestamps,
        train_idx=np.arange(_row_count(df)), val_idx=None, test_idx=None, verbose=verbose,
    )
    return train


__all__ = ["apply_entity_time_composite_fe", "replay_entity_time_composite_fe"]
