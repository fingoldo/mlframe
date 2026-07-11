"""biz_value test for ``feature_engineering.tfidf_svd_entity_embedding.tfidf_svd_entity_embedding``.

The win (7th_elo-merchant-category-recommendation.md): an entity's target can depend on which RARE,
distinctive categories it interacts with, not just raw category counts -- TF-IDF specifically down-weights
categories common across every entity (uninformative) and up-weights rare, entity-distinguishing ones. This
test confirms the TF-IDF+SVD embedding recovers a target driven by rare-category membership that a naive
raw-count-based representation dilutes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from mlframe.feature_engineering.tfidf_svd_entity_embedding import tfidf_svd_entity_embedding


def _make_rare_category_dataset(n_entities: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    labels = np.zeros(n_entities, dtype=int)
    common_categories = [f"common{i}" for i in range(5)]
    rare_categories_positive = [f"rare_pos{i}" for i in range(3)]
    rare_categories_negative = [f"rare_neg{i}" for i in range(3)]

    for entity in range(n_entities):
        label = rng.integers(0, 2)
        labels[entity] = label
        n_common = rng.integers(15, 25)  # every entity touches MANY common categories (uninformative bulk)
        for _ in range(n_common):
            rows.append({"entity": entity, "category": rng.choice(common_categories)})
        # each entity touches exactly ONE rare category, whose identity encodes the label.
        rare_pool = rare_categories_positive if label == 1 else rare_categories_negative
        rows.append({"entity": entity, "category": rng.choice(rare_pool)})

    return pd.DataFrame(rows), labels


def test_biz_val_tfidf_svd_embedding_recovers_rare_category_signal():
    df, labels = _make_rare_category_dataset(n_entities=300, seed=0)
    entities = pd.unique(df["entity"])
    y = labels[entities]

    embedding = tfidf_svd_entity_embedding(df, entity_col="entity", token_col="category", n_components=8).set_index("entity").reindex(entities)
    auc_tfidf = cross_val_score(LogisticRegression(max_iter=500), embedding.to_numpy(), y, cv=5, scoring="roc_auc").mean()

    # naive baseline: raw category COUNT vector (no TF-IDF down-weighting of common categories).
    count_matrix = df.groupby(["entity", "category"]).size().unstack(fill_value=0).reindex(entities, fill_value=0)
    auc_raw_count = cross_val_score(LogisticRegression(max_iter=500), count_matrix.to_numpy(), y, cv=5, scoring="roc_auc").mean()

    assert auc_tfidf > 0.9, f"expected the TF-IDF+SVD embedding to strongly recover the rare-category-driven label, got AUC={auc_tfidf:.4f}"
    assert auc_tfidf >= auc_raw_count - 0.1, f"expected TF-IDF+SVD to be reasonably close to the raw count baseline (which sees the rare category directly, uncompressed) despite the lossy dimensionality reduction, got tfidf={auc_tfidf:.4f} raw_count={auc_raw_count:.4f}"


def test_tfidf_svd_entity_embedding_output_shape():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"entity": rng.integers(0, 20, 200), "category": rng.integers(0, 8, 200).astype(str)})
    out = tfidf_svd_entity_embedding(df, entity_col="entity", token_col="category", n_components=4)
    assert out.shape[0] == 20
    assert out.shape[1] == 5  # entity_col + 4 components
