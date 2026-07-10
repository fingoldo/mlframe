"""biz_value test for ``feature_engineering.sequence2vec_entity_features`` / ``train_sequence2vec``.

The win: when an entity's true label is determined by which behavioral cluster of categorical tokens
dominates its sequence, skip-gram embeddings should place same-cluster tokens near each other and different-
cluster tokens far apart, so a downstream classifier on the entity-level mean embedding recovers the true
label far better than a majority-class baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from mlframe.feature_engineering.sequence2vec_categorical import sequence2vec_entity_features, train_sequence2vec


def test_biz_val_sequence2vec_recovers_true_behavioral_cluster():
    rng = np.random.default_rng(0)
    n_entities = 200
    rows = []
    labels = {}
    for e in range(n_entities):
        label = int(rng.random() < 0.5)
        labels[e] = label
        tokens = rng.choice(["A", "B"] if label == 1 else ["C", "D"], size=10)
        for t, tok in enumerate(tokens):
            rows.append({"entity": e, "t": t, "tok": tok})

    df = pd.DataFrame(rows)
    result = sequence2vec_entity_features(df, "entity", "tok", time_col="t", embedding_dim=16, window=2, n_epochs=10, random_state=0)
    mean_cols = [c for c in result.columns if c.startswith("emb_mean_")]
    y = np.array([labels[e] for e in result["entity"]])

    auc = cross_val_score(LogisticRegression(max_iter=500), result[mean_cols], y, cv=5, scoring="roc_auc").mean()
    majority_baseline = max(float(y.mean()), 1.0 - float(y.mean()))

    assert auc > majority_baseline + 0.35, (
        f"sequence2vec entity embeddings should recover the true behavioral cluster far above the majority baseline: "
        f"auc={auc:.4f} baseline={majority_baseline:.4f}"
    )


def test_train_sequence2vec_within_cluster_similarity_exceeds_cross_cluster():
    rng = np.random.default_rng(1)
    sequences = []
    for _ in range(200):
        seq = list(rng.choice(["A", "B"], size=10)) if rng.random() < 0.5 else list(rng.choice(["C", "D"], size=10))
        sequences.append(seq)

    embeddings = train_sequence2vec(sequences, embedding_dim=16, window=2, n_negative=5, n_epochs=10, random_state=0)

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    within_cluster = (_cos(embeddings["A"], embeddings["B"]) + _cos(embeddings["C"], embeddings["D"])) / 2
    cross_cluster = (_cos(embeddings["A"], embeddings["C"]) + _cos(embeddings["A"], embeddings["D"])) / 2

    assert within_cluster > cross_cluster + 0.6


def test_train_sequence2vec_min_count_filters_rare_tokens():
    sequences = [["a", "a", "a", "rare"], ["a", "a", "b", "b"]]
    embeddings = train_sequence2vec(sequences, embedding_dim=4, min_count=2, n_epochs=1)
    assert "rare" not in embeddings
    assert "a" in embeddings and "b" in embeddings


def test_train_sequence2vec_empty_vocab_returns_empty_dict():
    embeddings = train_sequence2vec([], embedding_dim=4)
    assert embeddings == {}
