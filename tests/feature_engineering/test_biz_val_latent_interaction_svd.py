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
    np.random.default_rng(1)
    events_df = pd.DataFrame(
        {
            "entity": ["a", "a", "b"],
            "item": ["x", "y", "x"],
            "t": [0.0, 100.0, 100.0],
        }
    )
    row_emb_no_decay, _ = latent_interaction_features(events_df, "entity", "item", n_components=1, use_tfidf=False, time_col=None)
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


def test_latent_interaction_features_return_fitted_default_unchanged():
    """``return_fitted=False`` (the default) must be bit-identical to the pre-extension return value."""
    events_df, _, _ = _make_clustered_interactions(seed=2)
    row_emb, col_emb = latent_interaction_features(events_df, "entity", "item", n_components=6, use_tfidf=True)
    row_emb2, col_emb2 = latent_interaction_features(events_df, "entity", "item", n_components=6, use_tfidf=True, return_fitted=False)
    pd.testing.assert_frame_equal(row_emb, row_emb2)
    pd.testing.assert_frame_equal(col_emb, col_emb2)


def test_biz_val_latent_interaction_features_transfers_to_disjoint_held_out_entities():
    """A basis fit on a TRAIN entity population should embed a DISJOINT test entity population usefully.

    Entities never seen at fit time are transformed via ``transform_new_entities`` on the frozen item
    vocabulary/SVD basis (no refitting), and a classifier trained ONLY on the train embeddings must predict
    the true cluster of the held-out entities far better than a majority-class baseline -- the point of a
    frozen basis is that it generalizes to a disjoint population, not just to the fit sample.
    """
    events_df, entity_cluster, n_entities = _make_clustered_interactions(seed=0)
    rng = np.random.default_rng(7)
    train_mask = rng.random(n_entities) < 0.7
    train_entities = np.flatnonzero(train_mask)
    test_entities = np.flatnonzero(~train_mask)

    train_events = events_df[events_df["entity"].isin(train_entities)]
    test_events = events_df[events_df["entity"].isin(test_entities)]

    row_emb, _, fitted = latent_interaction_features(train_events, "entity", "item", n_components=8, use_tfidf=True, return_fitted=True)
    row_emb_aligned = row_emb.reindex(train_entities)
    assert not row_emb_aligned.isna().any().any()

    clf = LogisticRegression(max_iter=500).fit(row_emb_aligned.to_numpy(), entity_cluster[train_entities])

    test_emb = fitted.transform_new_entities(test_events)
    test_emb_aligned = test_emb.reindex(test_entities)
    assert not test_emb_aligned.isna().any().any()

    feature_cols = [c for c in test_emb_aligned.columns if c != "oov_weight_fraction"]
    accuracy = clf.score(test_emb_aligned[feature_cols].to_numpy(), entity_cluster[test_entities])
    majority_baseline = float(np.bincount(entity_cluster[test_entities]).max() / len(test_entities))

    assert accuracy > majority_baseline + 0.3, (
        f"frozen-basis embedding of held-out entities should beat the majority-class baseline by a wide margin: "
        f"accuracy={accuracy:.4f} baseline={majority_baseline:.4f}"
    )
    # every test entity interacted only with items present at fit time (same item population), so OOV mass is 0.
    assert (test_emb_aligned["oov_weight_fraction"] == 0.0).all()


def test_biz_val_latent_interaction_features_oov_items_fall_back_to_origin():
    """An entity whose interactions are ENTIRELY with items unseen at fit time gets an all-zero embedding.

    That's the documented OOV fallback (mirrors ``FittedTfidfSvdEntityEmbedding``'s OOV-fraction diagnostic):
    with zero known-vocabulary signal the frozen SVD basis has nothing to project, so the entity lands at the
    origin with ``oov_weight_fraction == 1.0`` flagging the embedding as unreliable, rather than crashing or
    silently returning a misleading nonzero vector.
    """
    events_df, _, _ = _make_clustered_interactions(seed=0)
    _, _, fitted = latent_interaction_features(events_df, "entity", "item", n_components=8, use_tfidf=True, return_fitted=True)

    novel_events = pd.DataFrame({"entity": [9001, 9001, 9002], "item": [-1, -2, -3]})  # items never seen at fit time
    out = fitted.transform_new_entities(novel_events)

    assert (out["oov_weight_fraction"] == 1.0).all()
    feature_cols = [c for c in out.columns if c != "oov_weight_fraction"]
    assert np.allclose(out.loc[9001, feature_cols].to_numpy(), 0.0)
    assert np.allclose(out.loc[9002, feature_cols].to_numpy(), 0.0)


def test_latent_interaction_features_transform_new_entities_partial_oov():
    """An entity with a MIX of known and novel items gets a nonzero ``oov_weight_fraction`` in (0, 1)."""
    events_df, _, _ = _make_clustered_interactions(seed=0)
    _, _, fitted = latent_interaction_features(events_df, "entity", "item", n_components=8, use_tfidf=True, return_fitted=True)

    known_item = int(fitted.col_uniq[0])
    mixed_events = pd.DataFrame({"entity": [8001, 8001], "item": [known_item, -999]})
    out = fitted.transform_new_entities(mixed_events)
    assert 0.0 < out.loc[8001, "oov_weight_fraction"] < 1.0


def test_latent_interaction_features_transform_new_entities_empty_events_returns_empty_frame():
    """A predict-time batch whose entities have ZERO interaction history yet (a legitimate cold-start
    scenario, not a caller error) must return an empty-but-correctly-shaped DataFrame rather than
    crashing inside TfidfTransformer.transform on a 0-sample matrix."""
    events_df, _, _ = _make_clustered_interactions(seed=0)
    _, _, fitted = latent_interaction_features(events_df, "entity", "item", n_components=8, use_tfidf=True, return_fitted=True)

    empty_events = pd.DataFrame({"entity": pd.Series(dtype="int64"), "item": pd.Series(dtype="int64")})
    out = fitted.transform_new_entities(empty_events)
    assert len(out) == 0
    assert out.index.name == "entity"
    assert "oov_weight_fraction" in out.columns
    assert len([c for c in out.columns if c != "oov_weight_fraction"]) == 8
