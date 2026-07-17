"""biz_value test for ``evaluation.assert_no_group_leakage``.

The win: the guard correctly detects the exact failure mode from the source writeup -- a plain (non-grouped)
KFold over a child table with repeated parent-entity values lets a model memorize per-entity artifacts across
the fold boundary, giving an artificially inflated CV score that doesn't reflect real generalization. This
test both confirms the guard fires on the leaky split and quantifies the CV inflation it protects against
(leaky-split AUC vs honest GroupKFold AUC on the identical data).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

from mlframe.evaluation.group_leakage_guard import assert_no_group_leakage


def _make_nested_table(seed: int):
    rng = np.random.default_rng(seed)
    n_entities = 150
    rows_per_entity = 6
    entity_ids = np.repeat(np.arange(n_entities), rows_per_entity)
    n = len(entity_ids)

    # the target is a per-entity coin flip with NO relationship to any feature -- a real model should only
    # ever achieve chance-level performance on an unseen entity.
    entity_label = rng.integers(0, 2, n_entities)
    y = entity_label[entity_ids]

    # a near-constant-per-entity "artifact" column (like the writeup's repeated AMT_INSTALMENT): unique
    # enough per entity that a tree can effectively memorize "this artifact value -> this entity's label"
    # whenever it has seen ANY row from that entity during training -- exactly the leak the guard defends
    # against. Genuinely unrelated to y at the entity level (entity_label is independent of the artifact draw).
    entity_artifact = rng.normal(0, 1, n_entities)[entity_ids] + rng.normal(0, 0.01, n)
    real_feature = rng.normal(0, 1, n)  # pure noise, no signal either

    X = np.column_stack([entity_artifact, real_feature])
    return X, y, entity_ids


def test_biz_val_assert_no_group_leakage_catches_leaky_kfold():
    X, _y, entity_ids = _make_nested_table(seed=0)
    leaky_splits = list(KFold(n_splits=5, shuffle=True, random_state=0).split(X))

    import pytest

    with pytest.raises(ValueError, match="assert_no_group_leakage"):
        assert_no_group_leakage(leaky_splits, entity_ids)


def test_assert_no_group_leakage_passes_for_group_kfold():
    X, y, entity_ids = _make_nested_table(seed=1)
    safe_splits = list(GroupKFold(n_splits=5).split(X, y, groups=entity_ids))
    assert_no_group_leakage(safe_splits, entity_ids)  # must not raise


def test_biz_val_leaky_split_actually_inflates_cv_score_vs_group_kfold():
    X, y, entity_ids = _make_nested_table(seed=2)

    def _cv_auc(splits):
        scores = []
        for train_idx, test_idx in splits:
            model = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=0, n_jobs=1)
            model.fit(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[test_idx])[:, 1]
            scores.append(roc_auc_score(y[test_idx], proba))
        return float(np.mean(scores))

    leaky_auc = _cv_auc(KFold(n_splits=5, shuffle=True, random_state=2).split(X))
    honest_auc = _cv_auc(GroupKFold(n_splits=5).split(X, y, groups=entity_ids))

    assert leaky_auc > honest_auc + 0.03, (
        f"the leaky (non-grouped) split should show an inflated CV score relative to the honest group-aware split: "
        f"leaky={leaky_auc:.4f} honest={honest_auc:.4f}"
    )


def _make_implicit_duplicate_table(seed: int):
    """No explicit group id column, but every entity's rows are near-duplicate feature vectors (e.g. the same
    loan application re-submitted with tiny noise) -- exactly the leak the group-key-only check cannot see.
    """
    rng = np.random.default_rng(seed)
    n_entities = 120
    copies_per_entity = 3
    n_features = 8

    entity_center = rng.normal(0, 1, (n_entities, n_features))
    entity_ids = np.repeat(np.arange(n_entities), copies_per_entity)
    n = len(entity_ids)
    # near-duplicate: tiny noise around the entity's center vector, feature scale ~1 -> noise scale ~1e-3.
    X = entity_center[entity_ids] + rng.normal(0, 1e-3, (n, n_features))
    return X, entity_ids


def test_biz_val_assert_no_group_leakage_near_duplicate_mode_catches_implicit_group_leak():
    """The default (group-key-only) check misses this leak entirely because there IS no group id column --
    only the opt-in near-duplicate mode, given the raw features, can detect the implicit entity-sharing.
    """
    X, _entity_ids = _make_implicit_duplicate_table(seed=3)
    # a plain shuffled split has no notion of the implicit entity clusters, so near-duplicate copies of the
    # same entity land on both sides of most folds.
    leaky_splits = list(KFold(n_splits=5, shuffle=True, random_state=3).split(X))

    # anonymous_ids: no real group semantics available to the caller (simulates "no explicit group id column").
    anonymous_ids = np.arange(len(X))
    assert_no_group_leakage(leaky_splits, anonymous_ids)  # group-key-only mode: must NOT raise (misses the leak)

    import pytest

    with pytest.raises(ValueError, match="near-duplicate"):
        assert_no_group_leakage(
            leaky_splits,
            anonymous_ids,
            near_duplicate_features=X,
            near_duplicate_metric="euclidean",
            near_duplicate_max_neighbor_distance=0.05,
        )


def test_biz_val_assert_no_group_leakage_near_duplicate_mode_passes_on_deduped_split():
    """Sanity/threshold check: with entity clusters kept intact per fold (one fold gets ALL copies of each
    entity, i.e. a group-aware split simulated by construction), the near-duplicate mode must not fire, proving
    the flag is driven by actual cross-fold duplication rather than by the feature values alone.
    """
    X, entity_ids = _make_implicit_duplicate_table(seed=4)
    n_entities = int(entity_ids.max()) + 1
    fold_of_entity = np.arange(n_entities) % 5
    fold_of_row = fold_of_entity[entity_ids]

    safe_splits = []
    for fold in range(5):
        test_idx = np.where(fold_of_row == fold)[0]
        train_idx = np.where(fold_of_row != fold)[0]
        safe_splits.append((train_idx, test_idx))

    anonymous_ids = np.arange(len(X))
    assert_no_group_leakage(
        safe_splits,
        anonymous_ids,
        near_duplicate_features=X,
        near_duplicate_metric="euclidean",
        near_duplicate_max_neighbor_distance=0.05,
    )  # must not raise: no entity's near-duplicate copies are split across train/test


def test_assert_no_group_leakage_near_duplicate_mode_default_omitted_is_unchanged():
    """Omitting the new params must reproduce the exact pre-extension behavior (no new ValueError path taken)."""
    X, y, entity_ids = _make_nested_table(seed=5)
    safe_splits = list(GroupKFold(n_splits=5).split(X, y, groups=entity_ids))
    assert_no_group_leakage(safe_splits, entity_ids)  # bit-identical call signature to the original function
