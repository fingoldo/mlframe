"""``tfidf_svd_entity_embedding``: TF-IDF + TruncatedSVD embedding of an entity's bag-of-categories.

Source: 7th_elo-merchant-category-recommendation.md -- TF-IDF + TruncatedSVD on the per-entity "bag of
categorical values" string (e.g. every ``merchant_category_id`` a card has ever transacted with, treated as
a document). Distinct from the existing ``sequence2vec_categorical`` word2vec-style embedding (which is
ORDER-aware, capturing local co-occurrence within a sequence) and from ``latent_interaction_svd`` (a global
interaction-matrix SVD) -- TF-IDF+SVD is order-AGNOSTIC, capturing which categories an entity uses and how
distinctively (down-weighting categories common across every entity), a genuinely different signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer


def _build_documents(df: pd.DataFrame, entity_col: str, token_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Group ``df`` into one whitespace-joined document per entity (first-seen order)."""
    entities = pd.unique(df[entity_col])
    # Casting token_col to str INSIDE the per-group lambda (s.astype(str)) re-pays pandas' astype dispatch
    # overhead once per group (10000 groups measured as ~80% of total cProfile time) -- casting the WHOLE
    # column to str ONCE before grouping, then joining via the canned ``" ".join`` aggregation, does the
    # same work in a single pass instead of len(entities) small ones.
    token_str = df[token_col].astype(str)
    grouped = token_str.groupby(df[entity_col], sort=False).agg(" ".join)
    documents = grouped.reindex(entities).to_numpy()
    return entities, documents


@dataclass
class FittedTfidfSvdEntityEmbedding:
    """Fitted TF-IDF vocabulary + SVD components, reusable to embed NEW entities without refitting.

    Refitting per inference batch would let the vocabulary/basis drift from the training corpus (and
    leaks future-batch statistics into the embedding); ``transform_new_entities`` instead applies the
    already-fitted ``tfidf``/``svd`` transformers as-is, plus reports each new entity's out-of-vocabulary
    (OOV) token fraction -- categories never seen during fit, which the fitted TF-IDF vocabulary silently
    drops -- as a reliability diagnostic (high OOV fraction means the embedding rests on little/no real
    signal for that entity, e.g. a cold-start entity whose categories are all novel).
    """

    tfidf: "TfidfVectorizer"
    svd: "TruncatedSVD"
    column_prefix: str

    def transform_new_entities(self, df: pd.DataFrame, entity_col: str, token_col: str) -> pd.DataFrame:
        """Embed entities in ``df`` using the already-fitted vocabulary/SVD basis (no refitting).

        Returns one row per unique entity (first-seen order): ``entity_col``, the
        ``{column_prefix}_{0..k-1}`` embedding columns, and ``{column_prefix}_oov_fraction`` -- the
        fraction of that entity's tokens absent from the fitted TF-IDF vocabulary (0 = every category was
        seen during fit, 1 = every category is novel and the embedding is effectively meaningless).
        """
        entities, documents = _build_documents(df, entity_col, token_col)

        vocabulary = self.tfidf.vocabulary_
        oov_fraction = np.empty(len(documents), dtype=float)
        for i, doc in enumerate(documents):
            tokens = doc.split()
            if not tokens:
                oov_fraction[i] = 1.0
                continue
            n_oov = sum(1 for tok in tokens if tok not in vocabulary)
            oov_fraction[i] = n_oov / len(tokens)

        tfidf_matrix = self.tfidf.transform(documents)
        embedding = self.svd.transform(tfidf_matrix)

        out: Dict[str, np.ndarray] = {entity_col: entities}
        for i in range(embedding.shape[1]):
            out[f"{self.column_prefix}_{i}"] = embedding[:, i]
        out[f"{self.column_prefix}_oov_fraction"] = oov_fraction
        return pd.DataFrame(out)


def tfidf_svd_entity_embedding(
    df: pd.DataFrame,
    entity_col: str,
    token_col: str,
    n_components: int = 10,
    random_state: int = 42,
    column_prefix: str = "tfidf_svd",
    return_fitted: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, FittedTfidfSvdEntityEmbedding]]:
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
    return_fitted
        When ``True``, also return a :class:`FittedTfidfSvdEntityEmbedding` capturing the fitted TF-IDF
        vocabulary and SVD basis, so NEW entities (e.g. cold-start entities seen only at inference) can be
        embedded later via ``transform_new_entities`` on the SAME basis, with an OOV-fraction reliability
        diagnostic, instead of leaking/refitting on the new batch. Default ``False`` preserves the prior
        return type and is bit-identical to the pre-extension behavior.

    Returns
    -------
    pd.DataFrame
        One row per unique entity (first-seen order), columns ``entity_col`` plus
        ``{column_prefix}_{0..n_components-1}``. If ``return_fitted=True``, a
        ``(DataFrame, FittedTfidfSvdEntityEmbedding)`` tuple instead.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    entities, documents = _build_documents(df, entity_col, token_col)

    tfidf = TfidfVectorizer(token_pattern=r"(?u)\S+")
    tfidf_matrix = tfidf.fit_transform(documents)

    k = min(n_components, len(entities) - 1, tfidf_matrix.shape[1] - 1)
    k = max(1, k)
    svd = TruncatedSVD(n_components=k, random_state=random_state)
    embedding = svd.fit_transform(tfidf_matrix)

    out: Dict[str, np.ndarray] = {entity_col: entities}
    for i in range(embedding.shape[1]):
        out[f"{column_prefix}_{i}"] = embedding[:, i]
    out_df = pd.DataFrame(out)

    if not return_fitted:
        return out_df
    return out_df, FittedTfidfSvdEntityEmbedding(tfidf=tfidf, svd=svd, column_prefix=column_prefix)


__all__ = ["tfidf_svd_entity_embedding", "FittedTfidfSvdEntityEmbedding"]
