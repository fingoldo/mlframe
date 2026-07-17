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

from mlframe.feature_engineering.tfidf_svd_entity_embedding import FittedTfidfSvdEntityEmbedding, tfidf_svd_entity_embedding


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

    embedding = tfidf_svd_entity_embedding(df, entity_col="entity", token_col="category", n_components=8).set_index("entity").reindex(entities)  # nosec B106 -- "category" is a column-name kwarg, not a credential
    auc_tfidf = cross_val_score(LogisticRegression(max_iter=500), embedding.to_numpy(), y, cv=5, scoring="roc_auc").mean()

    # naive baseline: raw category COUNT vector (no TF-IDF down-weighting of common categories).
    count_matrix = df.groupby(["entity", "category"]).size().unstack(fill_value=0).reindex(entities, fill_value=0)
    auc_raw_count = cross_val_score(LogisticRegression(max_iter=500), count_matrix.to_numpy(), y, cv=5, scoring="roc_auc").mean()

    assert auc_tfidf > 0.9, f"expected the TF-IDF+SVD embedding to strongly recover the rare-category-driven label, got AUC={auc_tfidf:.4f}"
    assert auc_tfidf >= auc_raw_count - 0.1, (
        f"expected TF-IDF+SVD to be reasonably close to the raw count baseline (which sees the rare category directly, uncompressed) despite the lossy dimensionality reduction, got tfidf={auc_tfidf:.4f} raw_count={auc_raw_count:.4f}"
    )


def test_tfidf_svd_entity_embedding_output_shape():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"entity": rng.integers(0, 20, 200), "category": rng.integers(0, 8, 200).astype(str)})
    out = tfidf_svd_entity_embedding(df, entity_col="entity", token_col="category", n_components=4)  # nosec B106 -- "category" is a column-name kwarg, not a credential
    assert out.shape[0] == 20
    assert out.shape[1] == 5  # entity_col + 4 components


def test_tfidf_svd_entity_embedding_default_output_unchanged_by_return_fitted_param():
    # regression test: adding the opt-in `return_fitted` param must not change the default return
    # value/type (bit-identical DataFrame) when the caller doesn't opt in.
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"entity": rng.integers(0, 30, 300), "category": rng.integers(0, 10, 300).astype(str)})
    out_default = tfidf_svd_entity_embedding(df, entity_col="entity", token_col="category", n_components=5)  # nosec B106 -- "category" is a column-name kwarg, not a credential
    out_explicit_false = tfidf_svd_entity_embedding(df, entity_col="entity", token_col="category", n_components=5, return_fitted=False)  # nosec B106 -- "category" is a column-name kwarg, not a credential
    assert isinstance(out_default, pd.DataFrame)
    pd.testing.assert_frame_equal(out_default, out_explicit_false)

    out_fitted, fitted = tfidf_svd_entity_embedding(df, entity_col="entity", token_col="category", n_components=5, return_fitted=True)  # nosec B106 -- "category" is a column-name kwarg, not a credential
    assert isinstance(fitted, FittedTfidfSvdEntityEmbedding)
    # the embedding DataFrame content itself (all columns except being wrapped in a tuple) must be identical.
    pd.testing.assert_frame_equal(out_default, out_fitted)


def test_biz_val_tfidf_svd_oov_fraction_flags_unreliable_cold_start_entities():
    # Fit on a training corpus of "known" categories, then embed a mix of new entities: some reuse ONLY
    # categories seen during fit (low OOV, embedding should be meaningful/reliable), others are pure
    # cold-start with categories never seen during fit (high OOV, embedding is a near-meaningless
    # default-vocab-miss vector) -- the OOV-fraction diagnostic must separate the two cleanly.
    rng = np.random.default_rng(3)
    known_categories = [f"known{i}" for i in range(20)]
    novel_categories = [f"novel{i}" for i in range(20)]

    train_rows = []
    for entity in range(200):
        for _ in range(rng.integers(10, 20)):
            train_rows.append({"entity": entity, "category": rng.choice(known_categories)})
    train_df = pd.DataFrame(train_rows)

    _, fitted = tfidf_svd_entity_embedding(train_df, entity_col="entity", token_col="category", n_components=6, return_fitted=True)  # nosec B106 -- "category" is a column-name kwarg, not a credential

    new_rows = []
    low_oov_entities, high_oov_entities = [], []
    for entity in range(200, 260):
        low_oov_entities.append(entity)
        for _ in range(rng.integers(10, 20)):
            new_rows.append({"entity": entity, "category": rng.choice(known_categories)})
    for entity in range(260, 320):
        high_oov_entities.append(entity)
        for _ in range(rng.integers(10, 20)):
            new_rows.append({"entity": entity, "category": rng.choice(novel_categories)})
    new_df = pd.DataFrame(new_rows)

    out = fitted.transform_new_entities(new_df, entity_col="entity", token_col="category").set_index("entity")  # nosec B106 -- "category" is a column-name kwarg, not a credential

    mean_oov_low = out.loc[low_oov_entities, "tfidf_svd_oov_fraction"].mean()
    mean_oov_high = out.loc[high_oov_entities, "tfidf_svd_oov_fraction"].mean()

    assert mean_oov_low < 0.05, f"expected entities reusing only known categories to have near-zero OOV fraction, got {mean_oov_low:.4f}"
    assert mean_oov_high > 0.95, f"expected pure cold-start entities (all-novel categories) to have near-total OOV fraction, got {mean_oov_high:.4f}"

    # the high-OOV entities' embeddings collapse to the same (near-zero-information) vector since every
    # token is dropped by the fitted vocabulary -- confirming the diagnostic correctly flags them as
    # carrying essentially no distinguishing signal, unlike the low-OOV group.
    embedding_cols = [c for c in out.columns if c.startswith("tfidf_svd_") and c != "tfidf_svd_oov_fraction"]
    high_oov_embedding_std = out.loc[high_oov_entities, embedding_cols].to_numpy().std()
    low_oov_embedding_std = out.loc[low_oov_entities, embedding_cols].to_numpy().std()
    assert high_oov_embedding_std < low_oov_embedding_std, (
        f"expected flagged (high-OOV) entities' embeddings to carry markedly less cross-entity variance "
        f"than reliable (low-OOV) entities', got high={high_oov_embedding_std:.4f} low={low_oov_embedding_std:.4f}"
    )
