"""Bring the disjoint calibration slice to the same feature schema the models' test frames carry.

The calib slice is carved from the raw suite frame before ``_phase_fit_pipeline`` runs, so none of the suite-level stages
that train/val/test go through (datetime decomposition, the composite FE families, the main pipeline, row-wise extension
columns, the extensions pipeline) ever touched it. A model fit on those columns then failed on the calib predict
(LightGBM: "train and valid dataset categorical_feature do not match"; positional ``iloc`` out of bounds elsewhere), and
finalize silently fell back to no conformal residuals. The calib rows are unseen at fit time exactly like predict-time
rows, so they are replayed through the same fitted stages ``predict_from_models`` uses, then aligned per model to the
column set of that model's own test frame (feature tiers and per-model column drops differ between models).
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


def _slice_rows(arr: Any, idx: Optional[np.ndarray]) -> Any:
    """Slice a full-length side array (group ids / timestamps) to the calib rows; ``None`` passes through."""
    if arr is None or idx is None:
        return None
    if isinstance(arr, (pd.Series, pd.Index)):
        return np.asarray(arr)[idx]
    if isinstance(arr, pl.Series):
        return arr.gather(idx)
    return np.asarray(arr)[idx]


def replay_suite_fe_on_calib(
    calib_df: Any,
    metadata: dict,
    calib_idx: Optional[np.ndarray] = None,
    group_ids: Any = None,
    timestamps: Any = None,
    auxiliary_events_df: Any = None,
    verbose: bool = False,
) -> Tuple[Any, Any]:
    """Replay the fitted suite-level stages on the raw calib slice, in ``predict_from_models`` order.

    Returns ``(calib_post, calib_pre)``: ``calib_post`` has gone through the main pipeline, row-wise extensions and the
    extensions pipeline (the frame non-native models are fit on); ``calib_pre`` stops before the main pipeline and gets
    every column the later stages added hstacked on, mirroring the polars-pre back-merge models on the native fastpath
    are fit on.
    """
    from .._feature_name_sanitize import sanitize_frame_columns
    from .._fixed_splits import split_id_columns_from_metadata
    from .predict import _apply_extensions_pipeline, _apply_row_wise_extensions, _replay_suite_datetime_decomposition
    from .utils import _drop_cols_df, _validate_input_columns_against_metadata
    from mlframe.training.pipeline.shared import (
        replay_categorical_composite_fe,
        replay_cross_sectional_composite_fe,
        replay_entity_time_composite_fe,
        replay_event_proximity_decay_composite_fe,
        replay_latent_interaction_svd_composite_fe,
        replay_ma_crossover_composite_fe,
        replay_nearest_past_join_composite_fe,
        replay_per_target_supervised_fe,
        replay_target_encoding_composite_fe,
    )

    gids = _slice_rows(group_ids, calib_idx)
    ts = _slice_rows(timestamps, calib_idx)

    df = _drop_cols_df(calib_df, split_id_columns_from_metadata(metadata))
    df = _replay_suite_datetime_decomposition(df, metadata, verbose=verbose)
    df = replay_categorical_composite_fe(df, metadata, verbose=verbose)
    df = replay_entity_time_composite_fe(df, metadata, gids, ts, verbose=verbose)
    df = replay_cross_sectional_composite_fe(df, metadata, verbose=verbose)
    df = replay_target_encoding_composite_fe(df, metadata, gids, verbose=verbose)
    df = replay_per_target_supervised_fe(df, metadata, gids, verbose=verbose)
    df = replay_ma_crossover_composite_fe(df, metadata, gids, ts, verbose=verbose)
    df = replay_latent_interaction_svd_composite_fe(df, metadata, auxiliary_events_df, gids, verbose=verbose)
    df = replay_nearest_past_join_composite_fe(df, metadata, auxiliary_events_df, verbose=verbose)
    df = replay_event_proximity_decay_composite_fe(df, metadata, ts, verbose=verbose)
    df = _validate_input_columns_against_metadata(df, metadata, verbose=bool(verbose))
    calib_pre = df

    pipeline = metadata.get("pipeline")
    if pipeline is not None:
        df = sanitize_frame_columns(pipeline.transform(df))
    row_wise_cfg = metadata.get("row_wise_extensions_config")
    if row_wise_cfg is not None:
        df = _apply_row_wise_extensions(df, row_wise_cfg, verbose=verbose)
    extensions_pipeline = metadata.get("extensions_pipeline")
    if extensions_pipeline is not None:
        df = _apply_extensions_pipeline(df, extensions_pipeline, verbose=verbose)
    calib_post = df

    return calib_post, _back_merge_new_columns(calib_pre, calib_post)


def _back_merge_new_columns(pre: Any, post: Any) -> Any:
    """Append to ``pre`` the columns ``post`` gained over it, keeping ``pre``'s own (raw, unencoded) columns as they are."""
    pre_cols = set(pre.columns)
    new_cols = [c for c in post.columns if c not in pre_cols]
    if not new_cols or len(pre) != len(post):
        return pre
    if isinstance(pre, pl.DataFrame):
        if isinstance(post, pl.DataFrame):
            return pre.hstack(post.select(new_cols))
        return pre.hstack(pl.DataFrame({c: post[c].to_numpy() for c in new_cols}))
    added = post.select(new_cols).to_pandas() if isinstance(post, pl.DataFrame) else post[new_cols]
    added = added.set_axis(pre.index) if not added.index.equals(pre.index) else added
    return pd.concat([pre, added], axis=1)


def align_calib_to_test_schema(calib_post: Any, calib_pre: Optional[Any], test_df: Any) -> Any:
    """Return the calib variant holding every column of ``test_df``, subset to ``test_df``'s columns and frame type.

    The variant matching ``test_df``'s frame type is tried first: the polars fastpath frame is the pre-pipeline one with
    raw categoricals, the pandas tier frame is the post-pipeline one. Without a test frame to align to, ``calib_post``
    is returned as is.
    """
    if test_df is None or not hasattr(test_df, "columns"):
        return calib_post
    test_cols = list(test_df.columns)
    candidates = [c for c in (calib_post, calib_pre) if c is not None]
    candidates.sort(key=lambda c: isinstance(c, pl.DataFrame) != isinstance(test_df, pl.DataFrame))
    for cand in candidates:
        cand_cols = set(cand.columns)
        if all(c in cand_cols for c in test_cols):
            out = cand.select(test_cols) if isinstance(cand, pl.DataFrame) else cand[test_cols]
            if isinstance(test_df, pl.DataFrame) and not isinstance(out, pl.DataFrame):
                return pl.from_pandas(out)
            if isinstance(test_df, pd.DataFrame) and isinstance(out, pl.DataFrame):
                return out.to_pandas()
            return out
    missing = [c for c in test_cols if c not in set(calib_post.columns)]
    logger.warning(
        "calib slice: no replayed variant carries every test-frame column (missing from the post-pipeline variant: %s); "
        "the calib predict will see a different schema than test.",
        missing[:10],
    )
    return calib_post


__all__ = ["align_calib_to_test_schema", "replay_suite_fe_on_calib"]
