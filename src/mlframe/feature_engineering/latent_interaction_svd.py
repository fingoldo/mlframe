"""SVD latent-factor embeddings of an (entity x item) interaction matrix, optionally TF-IDF/time weighted.

Distinct from the pre-existing ``cat_cooccurrence_svd`` (a per-ROW categorical-pair encoder producing a
single-column embedding for a `src_col` category from its contingency with `other_col`): this builds a full
sparse ENTITY x ITEM interaction matrix from an events log (one row per interaction, e.g. customer x item
purchase), optionally TF-IDF-reweighted and/or time-decayed, and returns BOTH row (entity) and column (item)
latent embeddings via truncated SVD -- the pattern used for customer-item, customer-brand, customer-coupon
relational features in a 5th-place AmExpert-2019 solution.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


def latent_interaction_features(
    events_df: pd.DataFrame,
    row_entity: str,
    col_entity: str,
    weight_col: Optional[str] = None,
    time_col: Optional[str] = None,
    time_decay_half_life: Optional[float] = None,
    reference_time: Optional[float] = None,
    use_tfidf: bool = True,
    n_components: int = 10,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build a sparse (row_entity x col_entity) interaction matrix and return SVD row/column embeddings.

    Parameters
    ----------
    events_df
        One row per interaction event (e.g. a purchase, a click).
    row_entity, col_entity
        Columns defining the two axes of the interaction matrix (e.g. ``"customer_id"``, ``"item_id"``).
    weight_col
        Optional numeric column to sum per ``(row_entity, col_entity)`` cell instead of a plain interaction
        count (e.g. purchase amount).
    time_col, time_decay_half_life, reference_time
        When all three are given, each event's weight (count or ``weight_col``) is multiplied by
        ``0.5 ** ((reference_time - time_col) / time_decay_half_life)`` before aggregation -- more recent
        events contribute more. ``reference_time`` defaults to ``events_df[time_col].max()``.
    use_tfidf
        Apply TF-IDF reweighting (via ``sklearn.feature_extraction.text.TfidfTransformer``, treating each
        row-entity's interaction-count vector as a "document" over item "terms") to the (possibly time-
        decayed) count matrix before SVD -- downweights ubiquitous items, upweights entity-distinctive ones.
    n_components
        Number of SVD components (capped at the matrix's rank).
    random_state
        Passed to ``TruncatedSVD``.

    Returns
    -------
    tuple
        ``(row_embeddings, col_embeddings)`` -- DataFrames indexed by the entity id, columns
        ``svd_0..svd_{n-1}``.
    """
    if row_entity not in events_df.columns or col_entity not in events_df.columns:
        raise ValueError(f"latent_interaction_features: {row_entity!r}/{col_entity!r} must both be columns of events_df")

    weights = events_df[weight_col].to_numpy(dtype=np.float64) if weight_col is not None else np.ones(len(events_df), dtype=np.float64)

    if time_col is not None and time_decay_half_life is not None:
        t = events_df[time_col].to_numpy(dtype=np.float64)
        ref = float(reference_time) if reference_time is not None else float(t.max())
        decay = 0.5 ** ((ref - t) / time_decay_half_life)
        weights = weights * decay

    row_uniq, row_inv = np.unique(events_df[row_entity].to_numpy(), return_inverse=True)
    col_uniq, col_inv = np.unique(events_df[col_entity].to_numpy(), return_inverse=True)
    n_rows, n_cols = len(row_uniq), len(col_uniq)

    M = sparse.coo_matrix((weights, (row_inv, col_inv)), shape=(n_rows, n_cols)).tocsr()

    if use_tfidf:
        from sklearn.feature_extraction.text import TfidfTransformer

        M = TfidfTransformer().fit_transform(M)

    from sklearn.decomposition import TruncatedSVD

    n_eff = min(n_components, max(1, min(n_rows, n_cols) - 1))
    svd = TruncatedSVD(n_components=n_eff, random_state=random_state)
    row_emb = svd.fit_transform(M)
    col_emb = svd.components_.T  # (n_cols, n_eff), the item-side embedding sharing the same latent axes

    col_names = [f"svd_{i}" for i in range(n_eff)]
    row_df = pd.DataFrame(row_emb, index=row_uniq, columns=col_names)
    col_df = pd.DataFrame(col_emb, index=col_uniq, columns=col_names)
    row_df.index.name = row_entity
    col_df.index.name = col_entity
    return row_df, col_df


__all__ = ["latent_interaction_features"]
