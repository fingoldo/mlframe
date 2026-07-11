"""``build_panel_sequence_tensor``: multi-channel (entities, channels, time) tensor for sequence models.

Source: 5th_home-credit-default-risk.md -- a per-user "image" (n_characteristics x 96 months) combining
signals from multiple source tables, normalized by per-row (per-entity) max, fed through a sequence encoder
(Conv1D -> BiLSTM -> Dense); the OOF encoder output used as a single stacked feature in the main GBM. The
sequence-model architecture itself is out of scope here (the source's own measured lift was tiny, +0.001, for
a large deep-learning subsystem; any framework's sequence model can plug into mlframe's existing stacking
pipeline by returning an OOF vector to `composite_oof_predictions`, no new adapter code needed) -- what IS
genuinely reusable is the TENSOR ASSEMBLY step: stacking multiple time-indexed value columns (channels) into
one aligned ``(n_entities, n_channels, n_time_steps)`` array with per-entity joint normalization, which
:func:`mlframe.feature_engineering.panel_pivot.pivot_time_indexed_panel` doesn't do on its own (it only
produces a flat 2D wide-column slab per call, with no channel axis or entity-level normalization).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from mlframe.feature_engineering.panel_pivot import pivot_time_indexed_panel


def build_panel_sequence_tensor(df: pd.DataFrame, id_col: str, time_index_col: str, channel_cols: Sequence[str], max_lags: int = 13, normalize: bool = True) -> np.ndarray:
    """Assemble multiple time-indexed value columns into one ``(n_entities, n_channels, max_lags)`` tensor.

    Reuses :func:`pivot_time_indexed_panel` (right-aligned: the most recent observation is always at
    ``lag_0``, regardless of an entity's history length) to build one aligned wide slab per channel, then
    stacks them along a new channel axis and (optionally) rescales each ENTITY's whole
    ``(n_channels, max_lags)`` block by that entity's own max absolute value across all channels and time
    steps jointly -- the source's own per-row normalization, needed because raw magnitudes can vary wildly
    across entities (e.g. high-vs-low-balance accounts) in a way that would otherwise dominate a sequence
    model's learned features.

    Parameters
    ----------
    df
        Long-format panel: one row per (entity, time_step).
    id_col
        Entity identifier column.
    time_index_col
        Column giving each row's chronological position within its entity.
    channel_cols
        Value columns to stack as channels.
    max_lags
        Number of most-recent time steps retained per entity (right-aligned).
    normalize
        If True, divide each entity's whole ``(n_channels, max_lags)`` block by that entity's own max
        absolute value (missing/NaN lags stay NaN; an all-zero/all-NaN entity is left unscaled to avoid
        division by zero).

    Returns
    -------
    np.ndarray
        ``(n_entities, n_channels, max_lags)`` float64 tensor, entity order matching
        ``df[id_col].drop_duplicates()`` sorted order (the same order `pivot_time_indexed_panel` returns).
    """
    wide = pivot_time_indexed_panel(df, id_col, time_index_col, channel_cols, max_lags=max_lags)
    n_entities = wide.shape[0]
    n_channels = len(channel_cols)

    tensor = np.empty((n_entities, n_channels, max_lags), dtype=np.float64)
    for c, col in enumerate(channel_cols):
        lag_cols = [f"{col}_lag_{k}" for k in range(max_lags)]
        tensor[:, c, :] = wide[lag_cols].to_numpy()

    if normalize:
        with np.errstate(invalid="ignore"):
            entity_max = np.nanmax(np.abs(tensor), axis=(1, 2))
        safe_scale = np.where((entity_max > 0) & np.isfinite(entity_max), entity_max, 1.0)
        tensor = tensor / safe_scale[:, np.newaxis, np.newaxis]

    return tensor


__all__ = ["build_panel_sequence_tensor"]
