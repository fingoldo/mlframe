"""MA-crossover FE (``feature_engineering.ma_crossover.ma_crossover_features``), applied at the
same pre-encoding point as the other composite-FE steps.

``ma_crossover_features`` takes PRECOMPUTED moving averages (``{window: series}``) -- the suite has
no rolling-MA step of its own, so this module computes them first (per-entity if ``group_ids`` is
available, else a single global series), ordered by ``timestamps`` (row order if unavailable), then
feeds the result to the underlying function. No fit-time state: pure function of the declared
columns/windows, predict-time replay re-runs the same computation.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import polars as pl

from mlframe.feature_engineering.ma_crossover import ma_crossover_features

logger = logging.getLogger(__name__)


def _to_pandas(df: Any) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    return df.to_pandas() if isinstance(df, pl.DataFrame) else df


def _attach_new_columns(df: Any, new_cols: "pd.DataFrame") -> Any:
    if new_cols.shape[1] == 0:
        return df
    if isinstance(df, pl.DataFrame):
        return df.with_columns([pl.Series(c, new_cols[c].to_numpy()) for c in new_cols.columns])
    return df.join(new_cols) if hasattr(df, "join") else pd.concat([df, new_cols], axis=1)


def _rolling_means(values: np.ndarray, group_ids: Optional[np.ndarray], order: np.ndarray, windows: List[int]) -> dict:
    """Per-entity (or global, if ``group_ids`` is None) rolling means at each window, computed in
    time order and returned aligned to the ORIGINAL row order of ``values``."""
    n = len(values)
    sort_idx = np.lexsort((order, group_ids)) if group_ids is not None else np.argsort(order, kind="stable")
    inv_idx = np.argsort(sort_idx, kind="stable")
    s = pd.Series(values[sort_idx])
    out = {}
    if group_ids is not None:
        # ``sort_idx`` already makes ``s`` group-contiguous (lexsort's primary key), so
        # ``groupby(sort=False).rolling()`` -- pandas' own vectorized C-level per-group rolling --
        # processes each contiguous block in place and returns rows in the SAME order as ``s``,
        # matching ``groupby().transform(lambda x: x.rolling(...))`` exactly but ~3x faster at
        # n=300k (measured): the transform+lambda form pays a per-group Python callback, the same
        # anti-pattern flagged elsewhere in this codebase (e.g. two_step_target_encode's docstring).
        g_sorted = pd.Series(group_ids[sort_idx])
        for w in windows:
            rolled = s.groupby(g_sorted, sort=False).rolling(window=w, min_periods=1).mean()
            out[w] = pd.Series(rolled.to_numpy()[inv_idx], index=range(n))
    else:
        for w in windows:
            rolled = s.rolling(window=w, min_periods=1).mean()
            out[w] = pd.Series(rolled.to_numpy()[inv_idx], index=range(n))
    return out


def apply_ma_crossover_composite_fe(
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
    """No-op unless ``config.ma_crossover_columns`` is set and has >=2 windows declared."""
    columns = list(getattr(config, "ma_crossover_columns", None) or [])
    windows = sorted(set(int(w) for w in (getattr(config, "ma_crossover_windows", None) or [])))
    if not columns or len(windows) < 2 or train_df is None:
        return train_df, val_df, test_df

    weight_power = float(getattr(config, "ma_crossover_long_window_weight_power", 0.0) or 0.0)
    if metadata is not None:
        metadata["ma_crossover_columns"] = columns
        metadata["ma_crossover_windows"] = windows
        metadata["ma_crossover_long_window_weight_power"] = weight_power

    ts_arr = np.asarray(timestamps) if timestamps is not None else None

    def _slice(idx: Optional[np.ndarray], n_rows: int):
        if idx is None or group_ids is None:
            return None, None
        idx_arr = np.asarray(idx)
        if len(idx_arr) != n_rows:
            return None, None
        _g = group_ids[idx_arr] if len(group_ids) > int(idx_arr.max()) else None
        _t = ts_arr[idx_arr] if ts_arr is not None and len(ts_arr) > int(idx_arr.max()) else None
        return _g, _t

    _splits = {"train": (train_df, train_idx), "val": (val_df, val_idx), "test": (test_df, test_idx)}
    out: dict = {}
    for split_name, (df, idx) in _splits.items():
        if df is None:
            out[split_name] = None
            continue
        pd_df = _to_pandas(df)
        if pd_df is None or not all(c in pd_df.columns for c in columns):
            out[split_name] = df
            continue
        n_rows = pd_df.shape[0]
        g, t = _slice(idx, n_rows)
        order = t if t is not None else np.arange(n_rows, dtype=np.float64)
        try:
            new_cols = pd.DataFrame(index=range(n_rows))
            for col in columns:
                mas = _rolling_means(pd_df[col].to_numpy(dtype=np.float64), g, order, windows)
                mas_series = {w: pd.Series(v.to_numpy()) for w, v in mas.items()}
                result = ma_crossover_features(mas_series, column_prefix=f"{col}_ma_crossover", group_ids=g, long_window_weight_power=weight_power)
                for c in result.columns:
                    new_cols[c] = result[c].to_numpy()
            out[split_name] = _attach_new_columns(df, new_cols)
            if verbose:
                logger.info("apply_ma_crossover_composite_fe[%s]: added %d column(s)", split_name, new_cols.shape[1])
        except Exception:
            logger.warning("apply_ma_crossover_composite_fe: step failed for split %r; skipping.", split_name, exc_info=True)
            out[split_name] = df

    return out["train"], out["val"], out["test"]


class _ReplayConfig:
    def __init__(self, metadata: dict):
        self.ma_crossover_columns = metadata.get("ma_crossover_columns") or []
        self.ma_crossover_windows = metadata.get("ma_crossover_windows") or []
        self.ma_crossover_long_window_weight_power = metadata.get("ma_crossover_long_window_weight_power", 0.0)


def replay_ma_crossover_composite_fe(df: Any, metadata: dict, group_ids: Optional[np.ndarray], timestamps: Any, verbose: int = 0) -> Any:
    """Predict-time replay: identical computation, config persisted onto ``metadata`` at fit time."""
    if df is None:
        return df
    config = _ReplayConfig(metadata)
    if not config.ma_crossover_columns:
        return df
    train, _, _ = apply_ma_crossover_composite_fe(
        df, None, None, config, group_ids, timestamps,
        train_idx=np.arange(df.shape[0]), val_idx=None, test_idx=None, verbose=verbose,
    )
    return train


__all__ = ["apply_ma_crossover_composite_fe", "replay_ma_crossover_composite_fe"]
