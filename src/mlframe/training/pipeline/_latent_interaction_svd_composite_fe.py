"""SVD latent-interaction embeddings (``feature_engineering.latent_interaction_svd``), applied at
the same pre-encoding point as the other composite-FE steps.

Unlike every other composite-FE module, this one consumes a SEPARATE auxiliary reference table
(``auxiliary_events_df`` -- an entity x item interaction log, e.g. customer x product purchases)
rather than columns already on train/val/test. The SVD basis is fit ONCE on the auxiliary table;
row (entity) embeddings are then joined onto train/val/test by matching each row's ``group_ids``
against the fitted basis's row-entity index. Entities absent from the auxiliary table (never
interacted) get an all-zero embedding (the SVD-origin cold-start fallback, matching
``FittedLatentInteractionSvd.transform_new_entities``'s own convention) plus
``oov_weight_fraction=1.0``.

Real fit-time state (the fitted TF-IDF/SVD basis itself): persisted directly onto ``metadata``
(pickled with the rest of the model bundle, not JSON -- same precedent as ``extensions_pipeline``),
so predict-time replay calls ``FittedLatentInteractionSvd.transform_new_entities`` on the SAME
frozen basis without refitting. Predict-time callers must supply a FRESH ``auxiliary_events_df``
(the entities being predicted' own recent interaction history) via the ``auxiliary_events_df`` kwarg
threaded through ``predict_from_models``/``predict_from_suite``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, cast

import numpy as np
import pandas as pd
import polars as pl

from mlframe.feature_engineering.latent_interaction_svd import FittedLatentInteractionSvd, latent_interaction_features

logger = logging.getLogger(__name__)


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


def _join_embeddings(df: Any, group_ids_split: Optional[np.ndarray], row_emb: "pd.DataFrame", column_prefix: str) -> Any:
    """Attach each row's SVD embedding vector (looked up by group id) onto df as new prefixed columns."""
    if df is None or group_ids_split is None:
        return df
    n = df.shape[0] if hasattr(df, "shape") else 0
    lut = {idx: row for idx, row in zip(row_emb.index, row_emb.to_numpy())}
    zero_vec = np.zeros(row_emb.shape[1])
    vecs = np.array([lut.get(gid, zero_vec) for gid in group_ids_split])
    cols = [f"{column_prefix}_{c}" for c in row_emb.columns]
    new_cols = pd.DataFrame(vecs, columns=cols, index=range(n))
    return _attach_new_columns(df, new_cols)


def apply_latent_interaction_svd_composite_fe(
    train_df: Any,
    val_df: Any,
    test_df: Any,
    config: Any,
    auxiliary_events_df: Any,
    group_ids: Optional[np.ndarray],
    train_idx: Optional[np.ndarray],
    val_idx: Optional[np.ndarray],
    test_idx: Optional[np.ndarray],
    metadata: Optional[dict] = None,
    verbose: int = 0,
) -> tuple:
    """No-op unless ``config.latent_interaction_svd_row_entity``/``col_entity`` are both set AND
    ``auxiliary_events_df``/``group_ids`` are available."""
    row_entity = getattr(config, "latent_interaction_svd_row_entity", None)
    col_entity = getattr(config, "latent_interaction_svd_col_entity", None)
    if not row_entity or not col_entity or auxiliary_events_df is None or group_ids is None or train_df is None:
        return train_df, val_df, test_df

    events_pd = _to_pandas(auxiliary_events_df)
    if events_pd is None or row_entity not in events_pd.columns or col_entity not in events_pd.columns:
        logger.warning(
            "apply_latent_interaction_svd_composite_fe: row_entity=%r/col_entity=%r not both present in "
            "auxiliary_events_df; skipping.", row_entity, col_entity,
        )
        return train_df, val_df, test_df

    weight_col = getattr(config, "latent_interaction_svd_weight_col", None)
    time_col = getattr(config, "latent_interaction_svd_time_col", None)
    decay_half_life = getattr(config, "latent_interaction_svd_decay_half_life", None)
    use_tfidf = bool(getattr(config, "latent_interaction_svd_use_tfidf", True))
    n_components = int(getattr(config, "latent_interaction_svd_n_components", 10) or 10)

    try:
        _fit_result = latent_interaction_features(
            events_pd, row_entity=row_entity, col_entity=col_entity, weight_col=weight_col,
            time_col=time_col, time_decay_half_life=decay_half_life, use_tfidf=use_tfidf,
            n_components=n_components, return_fitted=True,
        )
        row_emb, _col_emb, fitted = cast(Tuple[pd.DataFrame, pd.DataFrame, FittedLatentInteractionSvd], _fit_result)
    except Exception:
        logger.warning("apply_latent_interaction_svd_composite_fe: latent_interaction_features fit failed; skipping.", exc_info=True)
        return train_df, val_df, test_df

    if metadata is not None:
        metadata["latent_interaction_svd_fitted"] = fitted
        metadata["latent_interaction_svd_column_prefix"] = f"{row_entity}_{col_entity}_svd"

    column_prefix = f"{row_entity}_{col_entity}_svd"
    group_ids = np.asarray(group_ids)

    def _slice(idx: Optional[np.ndarray], n_rows: int) -> Optional[np.ndarray]:
        """Slice group_ids down to idx, or return None if idx doesn't match n_rows."""
        if idx is None:
            return None
        idx_arr = np.asarray(idx)
        if len(idx_arr) != n_rows or len(group_ids) <= int(idx_arr.max()):
            return None
        return np.asarray(group_ids[idx_arr])

    out_train = _join_embeddings(train_df, _slice(train_idx, train_df.shape[0]), row_emb, column_prefix)
    out_val = _join_embeddings(val_df, _slice(val_idx, val_df.shape[0]) if val_df is not None else None, row_emb, column_prefix)
    out_test = _join_embeddings(test_df, _slice(test_idx, test_df.shape[0]) if test_df is not None else None, row_emb, column_prefix)

    if verbose:
        logger.info("apply_latent_interaction_svd_composite_fe: added %d embedding column(s), %d fitted entities", row_emb.shape[1], row_emb.shape[0])

    return out_train, out_val, out_test


def replay_latent_interaction_svd_composite_fe(
    df: Any,
    metadata: dict,
    auxiliary_events_df: Any,
    group_ids: Optional[np.ndarray],
    verbose: int = 0,
) -> Any:
    """Predict-time replay: embeds group_ids' entities via the FROZEN fitted basis using a FRESH
    ``auxiliary_events_df`` (the predict-time entities' own recent interaction history) --
    ``transform_new_entities`` handles cold-start (never-seen) entities with an all-zero embedding."""
    if df is None or group_ids is None or auxiliary_events_df is None:
        return df
    fitted: Optional[FittedLatentInteractionSvd] = metadata.get("latent_interaction_svd_fitted")
    column_prefix = metadata.get("latent_interaction_svd_column_prefix")
    if fitted is None or not column_prefix:
        return df
    events_pd = _to_pandas(auxiliary_events_df)
    if events_pd is None:
        return df
    try:
        row_emb = fitted.transform_new_entities(events_pd)
    except Exception:
        logger.warning("replay_latent_interaction_svd_composite_fe: transform_new_entities failed; skipping.", exc_info=True)
        return df
    svd_cols = [c for c in row_emb.columns if c != "oov_weight_fraction"]
    if verbose:
        logger.info("replay_latent_interaction_svd_composite_fe: replayed %d embedding column(s)", len(svd_cols))
    return _join_embeddings(df, np.asarray(group_ids), row_emb[svd_cols], column_prefix)


__all__ = ["apply_latent_interaction_svd_composite_fe", "replay_latent_interaction_svd_composite_fe"]
