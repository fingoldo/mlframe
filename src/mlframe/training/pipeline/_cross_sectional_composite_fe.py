"""Cross-sectional-neighbor FE (``feature_engineering.cross_sectional_neighbors``), applied at the
same pre-encoding point as ``_categorical_composite_fe.py``/``_entity_time_composite_fe.py``.

Unlike those two, this trick's key ("snapshot") is a COLUMN already present in the frame (e.g.
``time_id``), not a side-array threaded from the FeaturesAndTargetsExtractor -- so it's purely
column-declaration-driven (mirrors ``tfidf_columns``) and needs no group_ids/timestamps plumbing.
No fit-time state either: it's a pure function of (snapshot_col, feature_cols) present in the
frame, so predict-time replay just re-runs the same computation, config persisted onto metadata.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
import polars as pl

from mlframe.feature_engineering.cross_sectional_neighbors import compute_cross_sectional_neighbor_features

logger = logging.getLogger(__name__)


def _to_pandas(df: Any) -> Optional[pd.DataFrame]:
    """Convert a polars DataFrame to pandas; pass through pandas/None unchanged."""
    if df is None:
        return None
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    return df


def _attach_new_columns(df: Any, new_cols: "pd.DataFrame") -> Any:
    """Attach ``new_cols`` (a pandas frame) onto ``df``, matching df's own polars/pandas type."""
    if new_cols.shape[1] == 0:
        return df
    if isinstance(df, pl.DataFrame):
        return df.with_columns([pl.Series(c, new_cols[c].to_numpy()) for c in new_cols.columns])
    return df.join(new_cols) if hasattr(df, "join") else pd.concat([df, new_cols], axis=1)


def apply_cross_sectional_composite_fe(
    train_df: Any,
    val_df: Any,
    test_df: Any,
    config: Any,
    metadata: Optional[dict] = None,
    verbose: int = 0,
) -> tuple:
    """Generate cross-sectional-neighbor columns on train/val/test, opt-in per ``config``.

    No-op when ``config.cross_sectional_neighbors_snapshot_col`` is unset, when it or any declared
    ``cross_sectional_neighbors_feature_cols`` is missing from a split's schema, or fewer than
    ``k+1`` distinct snapshots exist in a split (kNN needs neighbors to find).
    """
    snapshot_col = getattr(config, "cross_sectional_neighbors_snapshot_col", None)
    feature_cols = list(getattr(config, "cross_sectional_neighbors_feature_cols", None) or [])
    if not snapshot_col or not feature_cols or train_df is None:
        return train_df, val_df, test_df

    k = int(getattr(config, "cross_sectional_neighbors_k", 10) or 10)
    agg_stats = tuple(getattr(config, "cross_sectional_neighbors_agg_stats", None) or ("mean", "std"))

    if metadata is not None:
        metadata["cross_sectional_neighbors_snapshot_col"] = snapshot_col
        metadata["cross_sectional_neighbors_feature_cols"] = feature_cols
        metadata["cross_sectional_neighbors_k"] = k
        metadata["cross_sectional_neighbors_agg_stats"] = list(agg_stats)

    out: dict = {}
    for split_name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        if df is None:
            out[split_name] = None
            continue
        pd_df = _to_pandas(df)
        if pd_df is None or snapshot_col not in pd_df.columns or not all(c in pd_df.columns for c in feature_cols):
            out[split_name] = df
            continue
        n_snapshots = pd_df[snapshot_col].nunique()
        if n_snapshots < 2:
            out[split_name] = df
            continue
        try:
            result = compute_cross_sectional_neighbor_features(
                pd_df, snapshot_col=snapshot_col, feature_cols=feature_cols, k=min(k, n_snapshots - 1), agg_stats=agg_stats,
            )
            new_cols = result.to_pandas() if isinstance(result, pl.DataFrame) else result
            new_cols = new_cols.reset_index(drop=True)
            out[split_name] = _attach_new_columns(df, new_cols)
            if verbose:
                logger.info("apply_cross_sectional_composite_fe[%s]: added %d column(s)", split_name, new_cols.shape[1])
        except Exception:
            logger.warning("apply_cross_sectional_composite_fe: step failed for split %r; skipping.", split_name, exc_info=True)
            out[split_name] = df

    return out["train"], out["val"], out["test"]


class _ReplayConfig:
    """Replays this FE step's config from fitted-pipeline metadata (for inference-time reapplication)."""

    def __init__(self, metadata: dict):
        self.cross_sectional_neighbors_snapshot_col = metadata.get("cross_sectional_neighbors_snapshot_col")
        self.cross_sectional_neighbors_feature_cols = metadata.get("cross_sectional_neighbors_feature_cols") or []
        self.cross_sectional_neighbors_k = metadata.get("cross_sectional_neighbors_k", 10)
        self.cross_sectional_neighbors_agg_stats = metadata.get("cross_sectional_neighbors_agg_stats") or ["mean", "std"]


def replay_cross_sectional_composite_fe(df: Any, metadata: dict, verbose: int = 0) -> Any:
    """Predict-time replay: identical computation, config persisted onto ``metadata`` at fit time."""
    if df is None:
        return df
    config = _ReplayConfig(metadata)
    if not config.cross_sectional_neighbors_snapshot_col:
        return df
    train, _, _ = apply_cross_sectional_composite_fe(df, None, None, config, metadata=None, verbose=verbose)
    return train


__all__ = ["apply_cross_sectional_composite_fe", "replay_cross_sectional_composite_fe"]
