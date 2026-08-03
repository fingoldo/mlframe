"""Shared output-column builder for the ``baseline_disagreement*`` transformer family: independently
duplicated across those modules, consolidated here so a fix can't silently drift out of sync across
copies.
"""
from __future__ import annotations

import numpy as np


def disagreement_feats_to_cols(feats: np.ndarray, column_prefix: str, dtype: "np.dtype | type") -> dict[str, np.ndarray]:
    """Map the fixed 8-column layout of ``feats`` (3 baseline predictions + 5 disagreement stats) to their
    ``{column_prefix}_*`` output names, cast to the requested output ``dtype``."""
    cols: dict[str, np.ndarray] = {}
    cols[f"{column_prefix}_p_lgbd3"] = feats[:, 0].astype(dtype, copy=False)
    cols[f"{column_prefix}_p_lgbd5"] = feats[:, 1].astype(dtype, copy=False)
    cols[f"{column_prefix}_p_linear"] = feats[:, 2].astype(dtype, copy=False)
    cols[f"{column_prefix}_mean"] = feats[:, 3].astype(dtype, copy=False)
    cols[f"{column_prefix}_std"] = feats[:, 4].astype(dtype, copy=False)
    cols[f"{column_prefix}_range"] = feats[:, 5].astype(dtype, copy=False)
    cols[f"{column_prefix}_depth_diff"] = feats[:, 6].astype(dtype, copy=False)
    cols[f"{column_prefix}_lgb_vs_linear"] = feats[:, 7].astype(dtype, copy=False)
    return cols
