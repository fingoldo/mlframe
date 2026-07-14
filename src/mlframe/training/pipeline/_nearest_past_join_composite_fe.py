"""Nearest-past as-of join (``feature_engineering.nearest_past_join.nearest_past_join``), applied
at the same pre-encoding point as the other composite-FE steps.

Like ``latent_interaction_svd``, this consumes the SEPARATE ``auxiliary_events_df`` reference table
rather than columns generated from train/val/test alone -- but unlike that module, there's no
fit-time state to persist: the join itself is inherently leak-safe by construction (backward as-of
match, ``right_df[on] <= left_df[on]``), so predict-time replay just re-runs the same join against a
FRESH ``auxiliary_events_df`` (the predict-time entities' own up-to-date historical reference rows).
``on``/``by`` must be real columns present in BOTH ``train_df``/``val_df``/``test_df`` AND
``auxiliary_events_df`` -- explicit declaration, mirrors ``tfidf_columns``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
import polars as pl

from mlframe.feature_engineering.nearest_past_join import nearest_past_join

logger = logging.getLogger(__name__)


def _to_pandas(df: Any) -> Optional[pd.DataFrame]:
    """Convert a polars DataFrame to pandas; pass through pandas/None unchanged."""
    if df is None:
        return None
    return df.to_pandas() if isinstance(df, pl.DataFrame) else df


def _restore_frame_type(original: Any, result_pd: pd.DataFrame) -> Any:
    """Convert ``result_pd`` back to polars if ``original`` was a polars DataFrame."""
    return pl.from_pandas(result_pd) if isinstance(original, pl.DataFrame) else result_pd


def apply_nearest_past_join_composite_fe(
    train_df: Any,
    val_df: Any,
    test_df: Any,
    config: Any,
    auxiliary_events_df: Any,
    metadata: Optional[dict] = None,
    verbose: int = 0,
) -> tuple:
    """No-op unless ``config.nearest_past_join_on``/``by`` are set AND ``auxiliary_events_df`` is available."""
    on = getattr(config, "nearest_past_join_on", None)
    by = list(getattr(config, "nearest_past_join_by", None) or [])
    if not on or not by or auxiliary_events_df is None or train_df is None:
        return train_df, val_df, test_df

    right_pd = _to_pandas(auxiliary_events_df)
    if right_pd is None or on not in right_pd.columns or not all(c in right_pd.columns for c in by):
        logger.warning(
            "apply_nearest_past_join_composite_fe: on=%r/by=%r not both present in auxiliary_events_df; skipping.", on, by,
        )
        return train_df, val_df, test_df

    value_cols = getattr(config, "nearest_past_join_value_cols", None)
    fallback_by_chain = getattr(config, "nearest_past_join_fallback_by_chain", None)
    min_group_size = int(getattr(config, "nearest_past_join_min_group_size", 1) or 1)

    if metadata is not None:
        metadata["nearest_past_join_on"] = on
        metadata["nearest_past_join_by"] = by
        metadata["nearest_past_join_value_cols"] = value_cols
        metadata["nearest_past_join_fallback_by_chain"] = fallback_by_chain
        metadata["nearest_past_join_min_group_size"] = min_group_size

    out: dict = {}
    for split_name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        if df is None:
            out[split_name] = None
            continue
        left_pd = _to_pandas(df)
        if left_pd is None or on not in left_pd.columns or not all(c in left_pd.columns for c in by):
            out[split_name] = df
            continue
        try:
            # The suite's own preprocessing (e.g. ensure_float32_dtypes) can coerce train/val/test's
            # by/on columns to a dtype that no longer matches auxiliary_events_df's (e.g. an entity
            # id widened to float32 while the auxiliary table keeps int64) -- the underlying
            # polars as-of join hard-rejects a by-key dtype mismatch. Align right_pd's by/on columns
            # to left_pd's dtypes (not the reverse: left_df's dtype reflects what the REST of the
            # pipeline downstream already committed to) before joining.
            right_aligned = right_pd
            for _key_col in [on, *by]:
                if right_aligned[_key_col].dtype != left_pd[_key_col].dtype:
                    right_aligned = right_aligned.copy() if right_aligned is right_pd else right_aligned
                    right_aligned[_key_col] = right_aligned[_key_col].astype(left_pd[_key_col].dtype)
            joined = nearest_past_join(
                left_pd, right_aligned, on=on, by=by, right_value_cols=value_cols,
                fallback_by_chain=fallback_by_chain, min_group_size=min_group_size,
            )
            out[split_name] = _restore_frame_type(df, joined)
            if verbose:
                logger.info("apply_nearest_past_join_composite_fe[%s]: joined, %d cols now", split_name, joined.shape[1])
        except Exception:
            logger.warning("apply_nearest_past_join_composite_fe: join failed for split %r; skipping.", split_name, exc_info=True)
            out[split_name] = df

    return out["train"], out["val"], out["test"]


class _ReplayConfig:
    """Replays this FE step's join config from fitted-pipeline metadata (for inference-time reapplication)."""

    def __init__(self, metadata: dict):
        self.nearest_past_join_on = metadata.get("nearest_past_join_on")
        self.nearest_past_join_by = metadata.get("nearest_past_join_by") or []
        self.nearest_past_join_value_cols = metadata.get("nearest_past_join_value_cols")
        self.nearest_past_join_fallback_by_chain = metadata.get("nearest_past_join_fallback_by_chain")
        self.nearest_past_join_min_group_size = metadata.get("nearest_past_join_min_group_size", 1)


def replay_nearest_past_join_composite_fe(df: Any, metadata: dict, auxiliary_events_df: Any, verbose: int = 0) -> Any:
    """Predict-time replay: identical join against a FRESH ``auxiliary_events_df``, config persisted
    onto ``metadata`` at fit time (no fit-time STATE -- the join itself needs no basis to freeze)."""
    if df is None or auxiliary_events_df is None:
        return df
    config = _ReplayConfig(metadata)
    if not config.nearest_past_join_on or not config.nearest_past_join_by:
        return df
    train, _, _ = apply_nearest_past_join_composite_fe(df, None, None, config, auxiliary_events_df, metadata=None, verbose=verbose)
    return train


__all__ = ["apply_nearest_past_join_composite_fe", "replay_nearest_past_join_composite_fe"]
