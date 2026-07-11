"""``tfidf_svd_entity_embedding``: TF-IDF + TruncatedSVD embedding of an entity's bag-of-categories.

Source: 7th_elo-merchant-category-recommendation.md -- TF-IDF + TruncatedSVD on the per-entity "bag of
categorical values" string (e.g. every ``merchant_category_id`` a card has ever transacted with, treated as
a document). Distinct from the existing ``sequence2vec_categorical`` word2vec-style embedding (which is
ORDER-aware, capturing local co-occurrence within a sequence) and from ``latent_interaction_svd`` (a global
interaction-matrix SVD) -- TF-IDF+SVD is order-AGNOSTIC, capturing which categories an entity uses and how
distinctively (down-weighting categories common across every entity), a genuinely different signal.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def tfidf_svd_entity_embedding(
    df: pd.DataFrame,
    entity_col: str,
    token_col: str,
    n_components: int = 10,
    random_state: int = 42,
    column_prefix: str = "tfidf_svd",
) -> pd.DataFrame:
    """Per-entity TF-IDF-weighted bag-of-categories, reduced to ``n_components`` via TruncatedSVD.

    Parameters
    ----------
    df
        Event-level frame with one row per ``(entity_col, token_col)`` occurrence.
    entity_col
        Grouping key (the "document" is each entity's full set of ``token_col`` occurrences).
    token_col
        Categorical id column (e.g. merchant/category id).
    n_components
        SVD output dimensionality (capped at ``min(n_components, n_entities - 1, n_unique_tokens - 1)``).
    random_state
        Seed for TruncatedSVD's randomized solver.
    column_prefix
        Output column-name prefix.

    Returns
    -------
    pd.DataFrame
        One row per unique entity (first-seen order), columns ``entity_col`` plus
        ``{column_prefix}_{0..n_components-1}``.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    entities = pd.unique(df[entity_col])
    # Casting token_col to str INSIDE the per-group lambda (s.astype(str)) re-pays pandas' astype dispatch
    # overhead once per group (10000 groups measured as ~80% of total cProfile time) -- casting the WHOLE
    # column to str ONCE before grouping, then joining via the canned ``" ".join`` aggregation, does the
    # same work in a single pass instead of len(entities) small ones.
    token_str = df[token_col].astype(str)
    grouped = token_str.groupby(df[entity_col], sort=False).agg(" ".join)
    documents = grouped.reindex(entities).to_numpy()

    tfidf = TfidfVectorizer(token_pattern=r"(?u)\S+")
    tfidf_matrix = tfidf.fit_transform(documents)

    k = min(n_components, len(entities) - 1, tfidf_matrix.shape[1] - 1)
    k = max(1, k)
    svd = TruncatedSVD(n_components=k, random_state=random_state)
    embedding = svd.fit_transform(tfidf_matrix)

    out: Dict[str, np.ndarray] = {entity_col: entities}
    for i in range(embedding.shape[1]):
        out[f"{column_prefix}_{i}"] = embedding[:, i]
    return pd.DataFrame(out)


__all__ = ["tfidf_svd_entity_embedding"]
