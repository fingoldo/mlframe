"""biz_value test for ``feature_engineering.leakage_safe_aggregate``.

The win: a naive whole-history aggregate that accidentally includes rows AT OR AFTER the prediction cutoff
can leak the label through post-event data (a real, repeatedly-reported bug per the source writeups) --
producing a suspiciously high train AUC that will NOT generalize. ``leakage_safe_aggregate`` must exclude
those rows and produce the honest (much lower, chance-level) feature instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from mlframe.feature_engineering.as_of_aggregate import leakage_safe_aggregate


def _make_leaky_scenario(seed: int):
    rng = np.random.default_rng(seed)
    n_entities = 300
    rows = []
    labels = {}
    for entity_id in range(n_entities):
        label = int(rng.random() < 0.5)
        labels[entity_id] = label
        cutoff_t = 50

        n_pre = rng.integers(3, 8)
        for t in sorted(rng.choice(np.arange(0, cutoff_t), size=n_pre, replace=False)):
            rows.append({"entity": entity_id, "t": int(t), "amount": float(rng.normal(0, 1))})

        # POST-cutoff rows whose value directly encodes the label -- the leak. A naive aggregate that
        # doesn't respect the cutoff will pick this up; the leakage-safe aggregate must not.
        n_post = rng.integers(2, 5)
        for t in sorted(rng.choice(np.arange(cutoff_t, cutoff_t + 20), size=n_post, replace=False)):
            rows.append({"entity": entity_id, "t": int(t), "amount": float(label * 100 + rng.normal(0, 0.1))})

    history_df = pd.DataFrame(rows)
    query_df = pd.DataFrame({"entity": list(labels.keys()), "as_of": [50] * n_entities})
    y = np.array([labels[e] for e in query_df["entity"]])
    return history_df, query_df, y


def test_biz_val_leakage_safe_aggregate_avoids_label_leak_naive_aggregate_falls_for():
    history_df, query_df, y = _make_leaky_scenario(seed=0)

    safe_result = leakage_safe_aggregate(history_df, entity_col="entity", time_col="t", as_of=query_df, agg_funcs={"amount": ["mean"]})
    safe_auc = roc_auc_score(y, safe_result["amount_mean"].fillna(0.0))

    naive_result = history_df.groupby("entity")["amount"].mean().reindex(query_df["entity"]).to_numpy()
    naive_auc = roc_auc_score(y, naive_result)

    assert naive_auc > 0.95, f"the naive (leaky) aggregate should look suspiciously predictive: {naive_auc:.4f}"
    assert safe_auc < 0.65, f"the leakage-safe aggregate should NOT show the same inflated signal: {safe_auc:.4f}"


def test_leakage_safe_aggregate_matches_manual_computation():
    history_df = pd.DataFrame(
        {"entity": ["a", "a", "a", "b", "b"], "t": [1, 5, 9, 2, 8], "amount": [10.0, 20.0, 30.0, 100.0, 200.0]}
    )
    query_df = pd.DataFrame({"entity": ["a", "b"], "as_of": [6, 9]})

    result = leakage_safe_aggregate(history_df, entity_col="entity", time_col="t", as_of=query_df, agg_funcs={"amount": ["mean", "count"]})

    a_row = result[result["entity"] == "a"].iloc[0]
    assert a_row["amount_mean"] == 15.0  # rows at t=1,5 (t=9 excluded, as_of=6)
    assert a_row["amount_count"] == 2

    b_row = result[result["entity"] == "b"].iloc[0]
    assert b_row["amount_mean"] == 150.0  # rows at t=2,8 (both < as_of=9)
    assert b_row["amount_count"] == 2


def test_leakage_safe_aggregate_no_eligible_history_is_nan():
    history_df = pd.DataFrame({"entity": ["a"], "t": [10], "amount": [5.0]})
    query_df = pd.DataFrame({"entity": ["a"], "as_of": [1]})
    result = leakage_safe_aggregate(history_df, entity_col="entity", time_col="t", as_of=query_df, agg_funcs={"amount": ["mean"]})
    assert pd.isna(result["amount_mean"].iloc[0])


def test_leakage_safe_aggregate_rejects_non_dataframe_as_of():
    import pytest

    history_df = pd.DataFrame({"entity": ["a"], "t": [1], "amount": [5.0]})
    with pytest.raises(TypeError):
        leakage_safe_aggregate(history_df, entity_col="entity", time_col="t", as_of="not_a_dataframe", agg_funcs={"amount": ["mean"]})
