"""biz_value test for ``feature_engineering.sequence2vec_entity_features`` / ``train_sequence2vec``.

The win: when an entity's true label is determined by which behavioral cluster of categorical tokens
dominates its sequence, skip-gram embeddings should place same-cluster tokens near each other and different-
cluster tokens far apart, so a downstream classifier on the entity-level mean embedding recovers the true
label far better than a majority-class baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from mlframe.feature_engineering.sequence2vec_categorical import (
    sequence2vec_entity_features,
    sequence2vec_transform_new_entities,
    train_sequence2vec,
)


def _make_cluster_df(rng: np.random.Generator, n_entities: int, entity_offset: int = 0) -> tuple[pd.DataFrame, dict]:
    """Helper: Make cluster df."""
    rows = []
    labels = {}
    for i in range(n_entities):
        e = entity_offset + i
        label = int(rng.random() < 0.5)
        labels[e] = label
        tokens = rng.choice(["A", "B"] if label == 1 else ["C", "D"], size=10)
        for t, tok in enumerate(tokens):
            rows.append({"entity": e, "t": t, "tok": tok})
    return pd.DataFrame(rows), labels


def test_biz_val_sequence2vec_transform_new_entities_recovers_true_behavioral_cluster():
    """The win: a fitted embedding basis (trained once on a "training" entity population) transfers to a
    held-out entity population with genuinely NEW entity ids and some genuinely unseen tokens, without
    retraining -- a downstream classifier on the transformed mean embedding still recovers the true label
    far above the majority baseline, and unseen tokens are reported via emb_oov_fraction rather than
    silently corrupting the output.
    """
    rng = np.random.default_rng(2)
    train_df, _ = _make_cluster_df(rng, n_entities=200, entity_offset=0)
    holdout_df, holdout_labels = _make_cluster_df(rng, n_entities=200, entity_offset=1000)

    # Inject genuinely unseen tokens into the held-out population, absent from the fitted vocabulary.
    unseen_rows = pd.DataFrame({"entity": [1000, 1000, 1001], "t": [10, 11, 10], "tok": ["Z", "Z", "Z"]})
    holdout_df = pd.concat([holdout_df, unseen_rows], ignore_index=True)

    _, embeddings = sequence2vec_entity_features(
        train_df, "entity", "tok", time_col="t", embedding_dim=16, window=2, n_epochs=10, random_state=0, return_embeddings=True
    )
    assert "Z" not in embeddings  # confirms "Z" is a genuine OOV token for the fitted basis

    transformed = sequence2vec_transform_new_entities(holdout_df, "entity", "tok", embeddings, time_col="t")

    # OOV diagnostic: entity 1000 has 2/12 unseen tokens, entity 1001 has 1/11, every other entity 0.
    oov_by_entity = transformed.set_index("entity")["emb_oov_fraction"]
    assert oov_by_entity.loc[1000] == pytest.approx(2 / 12)
    assert oov_by_entity.loc[1001] == pytest.approx(1 / 11)
    assert (oov_by_entity.drop([1000, 1001]) == 0).all()

    mean_cols = [c for c in transformed.columns if c.startswith("emb_mean_")]
    y = np.array([holdout_labels[e] for e in transformed["entity"]])

    auc = cross_val_score(LogisticRegression(max_iter=500), transformed[mean_cols], y, cv=5, scoring="roc_auc").mean()
    majority_baseline = max(float(y.mean()), 1.0 - float(y.mean()))

    assert auc > majority_baseline + 0.35, (
        f"transform_new_entities on the fitted basis should recover the true behavioral cluster on held-out entities "
        f"far above the majority baseline: auc={auc:.4f} baseline={majority_baseline:.4f}"
    )


def test_sequence2vec_transform_new_entities_all_oov_entity_gets_zero_vector():
    """An entity whose entire sequence is unseen tokens must not crash and must not silently emit a
    meaningless-but-plausible-looking vector -- it gets the documented all-zero fallback plus oov_fraction=1.
    """
    embeddings = {"A": np.array([1.0, 2.0]), "B": np.array([3.0, 4.0])}
    df = pd.DataFrame({"entity": [1, 1, 2], "t": [0, 1, 0], "tok": ["Q", "Q", "A"]})
    out = sequence2vec_transform_new_entities(df, "entity", "tok", embeddings, time_col="t")
    row1 = out[out["entity"] == 1].iloc[0]
    assert row1["emb_oov_fraction"] == 1.0
    assert row1["emb_mean_0"] == 0.0 and row1["emb_mean_1"] == 0.0
    row2 = out[out["entity"] == 2].iloc[0]
    assert row2["emb_oov_fraction"] == 0.0
    assert row2["emb_mean_0"] == 1.0 and row2["emb_mean_1"] == 2.0


def test_sequence2vec_entity_features_return_embeddings_false_is_bit_identical_to_prior_default():
    """Default behavior (``return_embeddings=False``) must remain exactly the prior fit-only contract."""
    rng = np.random.default_rng(3)
    df, _ = _make_cluster_df(rng, n_entities=30)
    plain = sequence2vec_entity_features(df, "entity", "tok", time_col="t", embedding_dim=8, window=2, n_epochs=3, random_state=0)
    assert isinstance(plain, pd.DataFrame)
    with_embeddings, embeddings = sequence2vec_entity_features(
        df, "entity", "tok", time_col="t", embedding_dim=8, window=2, n_epochs=3, random_state=0, return_embeddings=True
    )
    pd.testing.assert_frame_equal(plain, with_embeddings)
    assert isinstance(embeddings, dict)


def test_biz_val_sequence2vec_recovers_true_behavioral_cluster():
    """Biz val sequence2vec recovers true behavioral cluster."""
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
    """Train sequence2vec within cluster similarity exceeds cross cluster."""
    rng = np.random.default_rng(1)
    sequences = []
    for _ in range(200):
        seq = list(rng.choice(["A", "B"], size=10)) if rng.random() < 0.5 else list(rng.choice(["C", "D"], size=10))
        sequences.append(seq)

    embeddings = train_sequence2vec(sequences, embedding_dim=16, window=2, n_negative=5, n_epochs=10, random_state=0)

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        """Helper: Cos."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    within_cluster = (_cos(embeddings["A"], embeddings["B"]) + _cos(embeddings["C"], embeddings["D"])) / 2
    cross_cluster = (_cos(embeddings["A"], embeddings["C"]) + _cos(embeddings["A"], embeddings["D"])) / 2

    assert within_cluster > cross_cluster + 0.6


def test_train_sequence2vec_min_count_filters_rare_tokens():
    """Train sequence2vec min count filters rare tokens."""
    sequences = [["a", "a", "a", "rare"], ["a", "a", "b", "b"]]
    embeddings = train_sequence2vec(sequences, embedding_dim=4, min_count=2, n_epochs=1)
    assert "rare" not in embeddings
    assert "a" in embeddings and "b" in embeddings


def test_train_sequence2vec_empty_vocab_returns_empty_dict():
    """Train sequence2vec empty vocab returns empty dict."""
    embeddings = train_sequence2vec([], embedding_dim=4)
    assert embeddings == {}
