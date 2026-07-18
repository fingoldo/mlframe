"""Unit + biz_value coverage for ``mlframe.training.pipeline._latent_interaction_svd_composite_fe``.

The underlying trick (``latent_interaction_features``/``FittedLatentInteractionSvd``) already has
its own biz_value test at the function level. This file covers the suite-wiring layer: the auxiliary
events-table contract (fit once, join by group_ids), schema alignment, and predict-time replay via
``transform_new_entities`` on a FRESH events table (no refitting) -- plus one biz_value test proving
the wired module recovers a latent customer-cluster signal a raw entity-id can't.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline._latent_interaction_svd_composite_fe import (
    apply_latent_interaction_svd_composite_fe,
    replay_latent_interaction_svd_composite_fe,
)


def _events_and_frame(n_customers=50, n_items=20, n_events=500, n=100, seed=0):
    """Events and frame."""
    rng = np.random.default_rng(seed)
    events = pd.DataFrame(
        {
            "customer_id": rng.integers(0, n_customers, n_events),
            "item_id": rng.integers(0, n_items, n_events),
            "qty": rng.integers(1, 5, n_events),
        }
    )
    group_ids = rng.integers(0, n_customers, n)
    df = pd.DataFrame({"dummy": np.arange(n)})
    return events, df, group_ids


def test_apply_latent_interaction_svd_composite_fe_noop_when_entities_unset():
    """Apply latent interaction svd composite fe noop when entities unset."""
    events, df, group_ids = _events_and_frame()
    cfg = PreprocessingExtensionsConfig()
    train, _val, _test = apply_latent_interaction_svd_composite_fe(
        df.iloc[:70],
        df.iloc[70:],
        None,
        cfg,
        events,
        group_ids,
        np.arange(70),
        np.arange(70, 100),
        None,
        verbose=0,
    )
    assert list(train.columns) == list(df.columns)


def test_apply_latent_interaction_svd_composite_fe_noop_without_auxiliary_events_or_group_ids():
    """Apply latent interaction svd composite fe noop without auxiliary events or group ids."""
    _events, df, group_ids = _events_and_frame()
    cfg = PreprocessingExtensionsConfig(latent_interaction_svd_row_entity="customer_id", latent_interaction_svd_col_entity="item_id")
    train, _, _ = apply_latent_interaction_svd_composite_fe(df, None, None, cfg, None, group_ids, np.arange(len(df)), None, None, verbose=0)
    assert list(train.columns) == list(df.columns)
    train2, _, _ = apply_latent_interaction_svd_composite_fe(df, None, None, cfg, _events, None, np.arange(len(df)), None, None, verbose=0)
    assert list(train2.columns) == list(df.columns)


def test_apply_latent_interaction_svd_composite_fe_schema_aligned_across_splits():
    """Apply latent interaction svd composite fe schema aligned across splits."""
    events, df, group_ids = _events_and_frame()
    train_idx, val_idx, test_idx = np.arange(0, 70), np.arange(70, 85), np.arange(85, 100)
    cfg = PreprocessingExtensionsConfig(
        latent_interaction_svd_row_entity="customer_id",
        latent_interaction_svd_col_entity="item_id",
        latent_interaction_svd_weight_col="qty",
        latent_interaction_svd_n_components=5,
    )
    metadata: dict = {}
    train, val, test = apply_latent_interaction_svd_composite_fe(
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
        cfg,
        events,
        group_ids,
        train_idx,
        val_idx,
        test_idx,
        metadata=metadata,
        verbose=0,
    )
    assert set(train.columns) == set(val.columns) == set(test.columns)
    svd_cols = [c for c in train.columns if "svd" in c]
    assert len(svd_cols) == 5
    assert metadata["latent_interaction_svd_fitted"] is not None


def test_replay_latent_interaction_svd_composite_fe_uses_fresh_events_no_refit():
    """Replay latent interaction svd composite fe uses fresh events no refit."""
    events, df, group_ids = _events_and_frame()
    train_idx = np.arange(0, 100)
    cfg = PreprocessingExtensionsConfig(
        latent_interaction_svd_row_entity="customer_id",
        latent_interaction_svd_col_entity="item_id",
        latent_interaction_svd_n_components=4,
    )
    metadata: dict = {}
    train, _, _ = apply_latent_interaction_svd_composite_fe(df, None, None, cfg, events, group_ids, train_idx, None, None, metadata=metadata, verbose=0)

    fresh_idx = np.arange(0, 10)
    fresh = df.iloc[fresh_idx][["dummy"]].reset_index(drop=True)
    fresh_group_ids = group_ids[fresh_idx]
    # FRESH events table for predict replay (distinct object from the fit-time one -- still a valid
    # source of the same customer_id/item_id vocabulary the fitted basis knows about).
    fresh_events = events[events["customer_id"].isin(fresh_group_ids)]
    replayed = replay_latent_interaction_svd_composite_fe(fresh, metadata, fresh_events, fresh_group_ids, verbose=0)
    assert set(replayed.columns) - {"dummy"} == set(train.columns) - {"dummy"}


def test_replay_latent_interaction_svd_composite_fe_cold_start_entity_gets_zero_vector():
    """Replay latent interaction svd composite fe cold start entity gets zero vector."""
    events, df, group_ids = _events_and_frame(n_customers=10)
    train_idx = np.arange(0, 100)
    cfg = PreprocessingExtensionsConfig(
        latent_interaction_svd_row_entity="customer_id", latent_interaction_svd_col_entity="item_id", latent_interaction_svd_n_components=3
    )
    metadata: dict = {}
    apply_latent_interaction_svd_composite_fe(df, None, None, cfg, events, group_ids, train_idx, None, None, metadata=metadata, verbose=0)

    cold_start_id = 999999
    fresh = pd.DataFrame({"dummy": [0]})
    # cold-start entity has ZERO events in the fresh table too -- transform_new_entities must still
    # produce a row (all-zero embedding), not crash or drop the row.
    empty_events = events.iloc[0:0]
    replayed = replay_latent_interaction_svd_composite_fe(fresh, metadata, empty_events, np.array([cold_start_id]), verbose=0)
    svd_cols = [c for c in replayed.columns if "svd" in c]
    assert len(svd_cols) == 3
    assert np.allclose(replayed[svd_cols].to_numpy(), 0.0)


def test_biz_val_latent_interaction_svd_composite_wiring_recovers_customer_cluster():
    """Customers belong to one of 3 hidden purchase-affinity clusters; the target depends on the
    cluster, not the raw customer_id (a fresh integer id has no numeric relationship to the
    cluster). Raw customer_id as a single numeric feature carries no usable signal (arbitrary id
    ordering); the wired SVD embedding (built from the customer x item interaction log) recovers
    the cluster structure directly."""
    rng = np.random.default_rng(4)
    n_customers, n_items, n_clusters = 150, 30, 3
    customer_cluster = rng.integers(0, n_clusters, n_customers)
    cluster_item_affinity = rng.integers(0, n_items, n_clusters)  # each cluster favors one item strongly

    events_rows = []
    for cust in range(n_customers):
        cl = customer_cluster[cust]
        for _ in range(20):
            if rng.random() < 0.7:
                item = cluster_item_affinity[cl]
            else:
                item = rng.integers(0, n_items)
            events_rows.append((cust, item))
    events = pd.DataFrame(events_rows, columns=["customer_id", "item_id"])

    y_by_cluster = {0: 0, 1: 1, 2: 0}
    group_ids = np.arange(n_customers)
    y = np.array([y_by_cluster[c] for c in customer_cluster])
    df = pd.DataFrame({"customer_id_raw": group_ids.astype(float)})

    train_idx, test_idx = train_test_split(np.arange(n_customers), test_size=0.3, random_state=0, stratify=y)
    cfg = PreprocessingExtensionsConfig(
        latent_interaction_svd_row_entity="customer_id",
        latent_interaction_svd_col_entity="item_id",
        latent_interaction_svd_n_components=5,
    )
    out_df, _, _ = apply_latent_interaction_svd_composite_fe(df, None, None, cfg, events, group_ids, np.arange(n_customers), None, None, verbose=0)

    svd_cols = [c for c in out_df.columns if "svd" in c]

    def _auc(cols):
        """Auc."""
        clf = LogisticRegression(max_iter=1000)
        clf.fit(out_df.iloc[train_idx][cols], y[train_idx])
        return roc_auc_score(y[test_idx], clf.predict_proba(out_df.iloc[test_idx][cols])[:, 1])

    auc_raw = _auc(["customer_id_raw"])
    auc_wired = _auc(svd_cols)

    assert (
        auc_wired > auc_raw + 0.2
    ), f"wired SVD embedding should recover the customer cluster far better than the raw id, got auc_wired={auc_wired:.3f} vs auc_raw={auc_raw:.3f}"
