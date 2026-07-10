"""biz_value + unit tests for ``training.reconstruct_pseudo_group_ids``.

The win: on synthetic replicate-structured data (each entity's rows are near-duplicate feature vectors, no
explicit entity-id column exposed), the reconstructed pseudo-group ids (a) exactly recover the true entity
partition when replicates are exact duplicates, and (b) when used with GroupKFold, close a REAL leakage gap
— a memorizing 1-NN classifier's cross-validated accuracy collapses from artificially-inflated (row-level
KFold lets a replicate leak into both train and validation) to honest (GroupKFold with the reconstructed
ids never splits an entity's replicates across folds).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from mlframe.training._pseudo_group_reconstruction import reconstruct_pseudo_group_ids


def _make_replicate_data(n_entities: int, n_replicates: int, n_features: int, seed: int, noise_std: float = 0.0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_entities, n_features)) * 5.0
    true_entity_id = np.repeat(np.arange(n_entities), n_replicates)
    n = true_entity_id.shape[0]
    X = centers[true_entity_id] + noise_std * rng.standard_normal((n, n_features))
    y = (rng.random(n_entities) < 0.5).astype(np.int64)[true_entity_id]  # label is per-ENTITY, not per-row
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)]), y, true_entity_id


def test_reconstruct_pseudo_group_ids_exact_duplicates_recovers_true_partition():
    X, _, true_entity_id = _make_replicate_data(n_entities=30, n_replicates=4, n_features=6, seed=0, noise_std=0.0)
    reconstructed = reconstruct_pseudo_group_ids(X)
    assert adjusted_rand_score(true_entity_id, reconstructed) == 1.0


def test_reconstruct_pseudo_group_ids_unique_rows_get_singleton_groups():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    ids = reconstruct_pseudo_group_ids(X)
    assert len(set(ids.tolist())) == 3


def test_reconstruct_pseudo_group_ids_respects_feature_cols_subset():
    X = pd.DataFrame({"a": [1.0, 1.0, 2.0], "b": [10.0, 99.0, 20.0]})
    ids = reconstruct_pseudo_group_ids(X, feature_cols=["a"])
    assert ids[0] == ids[1]  # match on "a" only, "b" differs but is ignored
    assert ids[0] != ids[2]


def test_reconstruct_pseudo_group_ids_empty_feature_cols_raises():
    import pytest

    X = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError):
        reconstruct_pseudo_group_ids(X, feature_cols=[])


def test_biz_val_reconstructed_groups_close_a_real_leakage_gap():
    # noise_std << the decimals=1 rounding bucket width (0.1) so no replicate can cross a rounding boundary
    # in any of the 8 feature dims (that boundary-crossing fragility is a distinct concern from what this
    # test measures -- exact-duplicate recovery is already covered by a dedicated unit test above).
    X, y, true_entity_id = _make_replicate_data(n_entities=150, n_replicates=5, n_features=8, seed=42, noise_std=0.0005)
    reconstructed = reconstruct_pseudo_group_ids(X, decimals=1)
    assert adjusted_rand_score(true_entity_id, reconstructed) > 0.95  # near-perfect recovery despite noise

    clf = KNeighborsClassifier(n_neighbors=1)

    # row-level KFold: replicate rows of the SAME entity can (and with 5 replicates/entity, routinely do)
    # land in both train and validation -- a 1-NN classifier then "cheats" by finding a near-duplicate.
    leaky_scores = cross_val_score(clf, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=0))

    # GroupKFold on the reconstructed pseudo-ids: every replicate of an entity stays in the SAME fold, so
    # the 1-NN classifier can no longer memorize via a near-duplicate leak -- score reflects genuine
    # generalization to unseen entities only.
    honest_scores = cross_val_score(clf, X, y, cv=GroupKFold(n_splits=5), groups=reconstructed)

    leaky_acc = float(np.mean(leaky_scores))
    honest_acc = float(np.mean(honest_scores))

    assert leaky_acc > 0.90, f"sanity: row-level KFold should show inflated near-perfect accuracy from the leak, got {leaky_acc:.3f}"
    assert honest_acc < leaky_acc - 0.15, (
        f"GroupKFold on reconstructed ids should show a materially lower, more honest accuracy: "
        f"leaky={leaky_acc:.3f} honest={honest_acc:.3f}"
    )
