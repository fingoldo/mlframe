"""biz_value test for ``feature_engineering.latent_interaction_features``.

The win: when entities and items share a true block/cluster structure (entities interact mostly with items
in their own latent cluster), the SVD embedding of the (entity x item) interaction matrix should recover that
cluster structure well enough for a downstream classifier to predict an entity's true cluster far better than
a majority-class baseline -- the whole point of using latent factors as tabular features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from mlframe.feature_engineering.latent_interaction_svd import latent_interaction_features


def _make_clustered_interactions(seed: int):
    rng = np.random.default_rng(seed)
    n_entities, n_items, n_clusters = 300, 200, 5
    entity_cluster = rng.integers(0, n_clusters, n_entities)
    item_cluster = rng.integers(0, n_clusters, n_items)

    rows = []
    for e in range(n_entities):
        n_interactions = rng.integers(20, 40)
        same_cluster_items = np.flatnonzero(item_cluster == entity_cluster[e])
        other_items = np.flatnonzero(item_cluster != entity_cluster[e])
        n_same = int(0.8 * n_interactions)
        n_other = n_interactions - n_same
        chosen_same = rng.choice(same_cluster_items, size=min(n_same, len(same_cluster_items)), replace=True)
        chosen_other = rng.choice(other_items, size=n_other, replace=True)
        for item in np.concatenate([chosen_same, chosen_other]):
            rows.append({"entity": e, "item": int(item)})

    return pd.DataFrame(rows), entity_cluster, n_entities


def test_biz_val_latent_interaction_features_recovers_true_cluster_structure():
    events_df, entity_cluster, n_entities = _make_clustered_interactions(seed=0)

    row_emb, col_emb = latent_interaction_features(events_df, "entity", "item", n_components=8, use_tfidf=True)
    row_emb_aligned = row_emb.reindex(np.arange(n_entities))
    assert not row_emb_aligned.isna().any().any()

    accuracy = cross_val_score(LogisticRegression(max_iter=500), row_emb_aligned.to_numpy(), entity_cluster, cv=5).mean()
    majority_baseline = float(np.bincount(entity_cluster).max() / n_entities)

    assert accuracy > majority_baseline + 0.4, (
        f"SVD embeddings should recover the latent cluster structure well above the majority-class baseline: "
        f"accuracy={accuracy:.4f} baseline={majority_baseline:.4f}"
    )
    assert col_emb.shape[1] == row_emb.shape[1]


def test_latent_interaction_features_time_decay_downweights_stale_events():
    rng = np.random.default_rng(1)
    events_df = pd.DataFrame(
        {
            "entity": ["a", "a", "b"],
            "item": ["x", "y", "x"],
            "t": [0.0, 100.0, 100.0],
        }
    )
    row_emb_no_decay, _ = latent_interaction_features(
        events_df, "entity", "item", n_components=1, use_tfidf=False, time_col=None
    )
    row_emb_decay, _ = latent_interaction_features(
        events_df, "entity", "item", n_components=1, use_tfidf=False, time_col="t", time_decay_half_life=10.0, reference_time=100.0
    )
    # with a short half-life, entity "a"'s stale (t=0) interaction with item "x" should be nearly zeroed out,
    # changing its embedding relative to the no-decay version.
    assert not np.allclose(row_emb_no_decay.loc["a"].to_numpy(), row_emb_decay.loc["a"].to_numpy())


def test_latent_interaction_features_missing_column_raises():
    import pytest

    with pytest.raises(ValueError):
        latent_interaction_features(pd.DataFrame({"a": [1]}), "entity", "item")
