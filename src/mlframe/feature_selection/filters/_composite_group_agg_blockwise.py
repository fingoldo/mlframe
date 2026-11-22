"""Block-wise scoring of the composite group aggregates: the broadcast of all engineered columns is never held at once.

Split out of ``_composite_group_agg_fe`` (re-exported there, so the old import path keeps working). The parent's names are imported inside the function: the parent imports this module.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _composite_full_matrix_bytes(n_rows: int, n_keys: int, n_num: int, n_stats: int) -> int:
    """Bytes of the full broadcast of composite aggregates: a column per (source, stat) plus the z and ratio residuals, 8 bytes each, n rows."""
    return int(n_rows) * 8 * int(n_num) * (int(n_stats) + 2) * max(1, min(int(n_keys), 1))


def _composite_block_budget_bytes() -> int:
    """Working-memory budget of one column block of the composite broadcast: ``MLFRAME_COMPOSITE_BLOCK_MB`` (default 256 MB)."""
    import os

    try:
        return int(float(os.environ.get("MLFRAME_COMPOSITE_BLOCK_MB", "256")) * 1024 * 1024)
    except ValueError:
        logger.warning("MLFRAME_COMPOSITE_BLOCK_MB is not a number; using the 256 MB default")
        return 256 * 1024 * 1024


def _block_columns(n_rows: int) -> int:
    """How many engineered columns one block may hold: a column is ``n_rows`` 8-byte values, and ``5`` columns per numeric source (one per stat, z, ratio) are built per block."""
    per_col = max(1, n_rows) * 8
    return max(1, _composite_block_budget_bytes() // max(1, per_col * 5))


def composite_group_agg_top_blockwise(
    X: pd.DataFrame,
    y: np.ndarray,
    group_col_sets: Sequence[Sequence[str]],
    num_cols: Sequence[str],
    *,
    base_cols: Sequence[str],
    top_k: int,
    n_bins: int,
    min_cmi: float,
    min_uplift: float,
    heldout_uplift: bool = False,
    eng_to_source: dict,
    stats: Optional[Sequence[str]] = None,
    max_card_frac: float = 0.5,
    _precomputed_keys: Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The composite aggregates scored and reduced block by block, so the broadcast of ALL engineered columns is never held at once.

    The engineered columns are generated per numeric column (one call of :func:`_emit_num_col_features` per source), each source's block is scored with the same CMI scorer as the
    full path, and only the columns that pass the gate are kept: the running best ``top_k`` by CMI are retained (the full path keeps the same set, ranked by the same score). Returns
    ``(winners_frame, scores, raw_recipes_of_winners)``. The scores of every scored column are returned in ``scores``, as the full path does. Peak memory is one block of
    ``_block_columns`` source columns times the stats, not the number of keys times the rows times the sources.
    """
    from ._composite_group_agg_fe import (
        _VALID_STATS,
        COMPOSITE_STAT_NAMES,
        _emit_num_col_features,
        _unique_inverse,
        build_composite_keys,
        composite_cardinality_ok,
    )
    from ._grouped_agg_fe import score_grouped_agg_by_cmi_uplift

    stats = [s for s in (COMPOSITE_STAT_NAMES if stats is None else stats) if s in _VALID_STATS]
    norm_sets: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for gset in group_col_sets:
        cols = tuple(c for c in gset if c in X.columns)
        if len(cols) < 2 or cols in seen:
            continue
        seen.add(cols)
        norm_sets.append(cols)
    n_rows = len(X)
    keep_frames: list[pd.DataFrame] = []
    keep_recipes: dict = {}
    scores_frames: list[pd.DataFrame] = []
    support_cache: dict = {}  # the conditioning joint and the source-column bins/MI are the same for every block of candidates: built once for this call
    for group_cols in norm_sets:
        _pre = _precomputed_keys.get(group_cols) if _precomputed_keys is not None else None
        if _pre is not None:
            keys, key_uniq, key_inverse, n_distinct = _pre
        else:
            keys = build_composite_keys(X, group_cols)
            key_uniq, key_inverse = _unique_inverse(keys)
            n_distinct = int(key_uniq.size) if key_uniq is not None else int(np.unique(keys).size)
        if not composite_cardinality_ok(n_distinct, n_rows, max_card_frac):
            continue
        cur_num = [c for c in num_cols if c in X.columns and c not in set(group_cols) and pd.api.types.is_numeric_dtype(X[c])]
        block = _block_columns(n_rows)
        for start in range(0, len(cur_num), block):
            encoded: dict[str, np.ndarray] = {}
            recipes: dict[str, dict] = {}
            for num_col in cur_num[start : start + block]:
                _emit_num_col_features(X, num_col, keys, key_uniq, key_inverse, group_cols, stats, encoded, recipes)
            if not encoded:
                continue
            enc_df = pd.DataFrame(encoded, index=X.index, copy=False)
            del encoded
            # the source column of every candidate comes from its recipe, as on the full path; an explicit map from the caller wins. Without it the redundancy test against the source was off on this path.
            block_sources = {name: rec["num_col"] for name, rec in recipes.items() if name in enc_df.columns}
            block_sources.update(eng_to_source)
            scored = score_grouped_agg_by_cmi_uplift(X, enc_df, y, list(base_cols), n_bins=n_bins, eng_to_source=block_sources, support_cache=support_cache, heldout_uplift=heldout_uplift)
            scores_frames.append(scored)
            keep = scored[(scored["cmi"] >= float(min_cmi)) & (scored["uplift"] >= float(min_uplift))]
            winners = list(keep["engineered_col"].head(int(top_k)))
            if winners:
                keep_frames.append(enc_df[winners])
                for name in winners:
                    keep_recipes[name] = recipes[name]
            del enc_df, recipes
            # the best ``top_k`` so far: earlier blocks' winners compete with this block's, by the same CMI ranking
            if keep_frames:
                merged = pd.concat(keep_frames, axis=1)
                merged_scores = pd.concat(scores_frames, ignore_index=True)
                ranked = merged_scores[merged_scores["engineered_col"].isin(merged.columns)].sort_values("cmi", ascending=False)
                best = list(ranked["engineered_col"].head(int(top_k)))
                keep_frames = [merged[best]] if best else []
                keep_recipes = {k: v for k, v in keep_recipes.items() if k in best}
    winners_frame = pd.concat(keep_frames, axis=1) if keep_frames else pd.DataFrame(index=X.index)
    all_scores = pd.concat(scores_frames, ignore_index=True) if scores_frames else pd.DataFrame()
    return winners_frame, all_scores, keep_recipes


def conditioning_leaves_no_candidate(X: pd.DataFrame, base_cols: Sequence[str], n_bins: int, min_cmi: float, support_cache: dict) -> bool:
    """Whether no engineered column can reach ``min_cmi``: the joint class of the binned conditioning columns puts every row in a cell of its own, and then
    ``CMI(candidate; y | support)`` is exactly 0 for every candidate (``_batched_cmi_in_chunks`` and ``_binned_cmi_in_column_blocks`` return zeros there without scoring).

    The joint is built as the scorer builds it and left in ``support_cache`` under the scorer's key, so a support that does leave room is not binned twice. With a threshold at or below
    zero a zero score still passes, so the answer is ``False`` and the candidates are generated."""
    from ._mi_greedy_cmi_fe import _conditioning_is_all_singletons, _support_joint

    cols = [c for c in base_cols if c in X.columns]
    if not cols or not float(min_cmi) > 0.0 or len(X) < 2:
        return False
    key = ("joint", tuple(map(str, cols)), int(n_bins))
    z_joint = support_cache.get(key)
    if z_joint is None:
        z_joint = support_cache[key] = _support_joint(X[cols].to_numpy(), int(n_bins))
    return bool(_conditioning_is_all_singletons(z_joint, len(X)))
