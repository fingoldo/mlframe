"""biz_value test for ``feature_engineering.relational_dfs`` (cutoff-time-aware relational aggregation).

The win: a naive multi-table join that aggregates a child table WITHOUT respecting each parent row's own
cutoff time silently includes future child events. If the (synthetic, ground-truth) label is partly driven
by events that happen at/after the cutoff -- a realistic leakage scenario, e.g. "events causally simultaneous
with the outcome being predicted" -- a model trained on the naive aggregate looks great in-sample but is
badly miscalibrated in production, where those future events genuinely haven't happened yet at serving time
and the feature can only legitimately be computed from pre-cutoff data. ``compute_relational_features``
enforces the cutoff at aggregation time, so the SAME feature-computation code path is used consistently for
both training and serving, eliminating the train/serve mismatch by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from mlframe.feature_engineering.relational_dfs import ChildTableSpec, compute_relational_features, stack_relational_features


def _make_relational_dataset(seed: int, n_parents: int = 300):
    rng = np.random.default_rng(seed)
    parent = pd.DataFrame({"cust_id": np.arange(n_parents), "cutoff": np.full(n_parents, 10.0)})

    rows = []
    y = np.zeros(n_parents)
    for cust_id in range(n_parents):
        n_pre = rng.integers(2, 6)
        pre_times = rng.uniform(0, 10, n_pre)
        pre_amounts = rng.normal(loc=5, scale=1, size=n_pre)
        for t, a in zip(pre_times, pre_amounts):
            rows.append({"cust_id": cust_id, "ts": t, "amount": a})

        n_post = rng.integers(1, 4)
        post_times = rng.uniform(10, 20, n_post)
        post_amounts = rng.normal(loc=5, scale=1, size=n_post)
        for t, a in zip(post_times, post_amounts):
            rows.append({"cust_id": cust_id, "ts": t, "amount": a})

        # Ground-truth target is dominated by the POST-cutoff amounts (a causal-leakage scenario: the label
        # and the "future" events are simultaneous/downstream of the same underlying process).
        y[cust_id] = 1.0 * pre_amounts.sum() + 6.0 * post_amounts.sum() + rng.normal(scale=1.0)

    child_df = pd.DataFrame(rows)
    return parent, child_df, y


def test_biz_val_relational_dfs_cutoff_safe_beats_leaky_join_at_serving_time():
    parent, child_df, y = _make_relational_dataset(seed=0)
    spec = ChildTableSpec(child_df=child_df, foreign_key_col="cust_id", time_col="ts", value_cols={"amount": ["sum"]}, prefix="txn")

    # Cutoff-safe: the SAME code path (aggregate only rows before cutoff) is used to build both the
    # training feature and the serving-time feature -- no train/serve mismatch possible by construction.
    safe_train = compute_relational_features(parent, "cust_id", "cutoff", [spec])[["txn_amount_sum"]].to_numpy()
    safe_test = safe_train  # identical computation would be used again at serving time

    # Leaky: naive full-history groupby-merge with NO cutoff filter (aggregates every child row regardless
    # of time) used to build the TRAINING feature -- the bug this idea exists to prevent.
    leaky_full = child_df.groupby("cust_id")["amount"].sum().reindex(parent["cust_id"]).to_numpy().reshape(-1, 1)

    # At actual serving time, the leaky pipeline can only legitimately see pre-cutoff data (post-cutoff
    # events haven't happened yet) -- so its SERVING feature is necessarily the pre-cutoff-only aggregate,
    # mismatched from what it was trained on.
    pre_only = child_df[child_df["ts"] < 10.0]
    leaky_serving = pre_only.groupby("cust_id")["amount"].sum().reindex(parent["cust_id"]).fillna(0.0).to_numpy().reshape(-1, 1)

    reg_leaky = LinearRegression().fit(leaky_full, y)
    reg_safe = LinearRegression().fit(safe_train, y)

    mse_leaky_serving = mean_squared_error(y, reg_leaky.predict(leaky_serving))
    mse_safe_serving = mean_squared_error(y, reg_safe.predict(safe_test))

    improvement = (mse_leaky_serving - mse_safe_serving) / mse_leaky_serving
    assert improvement > 0.5, f"expected >50% MSE reduction at serving time vs the leaky-trained model, got {improvement:.4f} (leaky={mse_leaky_serving:.4f}, safe={mse_safe_serving:.4f})"


def test_compute_relational_features_excludes_post_cutoff_rows():
    parent = pd.DataFrame({"cust_id": [1, 2], "cutoff": [10, 10]})
    child = pd.DataFrame({"cust_id": [1, 1, 1, 2], "ts": [1, 5, 12, 3], "amount": [100, 200, 999, 50]})
    spec = ChildTableSpec(child_df=child, foreign_key_col="cust_id", time_col="ts", value_cols={"amount": ["sum", "count"]}, prefix="txn")
    out = compute_relational_features(parent, "cust_id", "cutoff", [spec])
    assert out.loc[out["cust_id"] == 1, "txn_amount_sum"].item() == 300
    assert out.loc[out["cust_id"] == 1, "txn_amount_count"].item() == 2
    assert out.loc[out["cust_id"] == 2, "txn_amount_sum"].item() == 50


def test_stack_relational_features_depth_2_respects_both_hop_cutoffs():
    parent = pd.DataFrame({"store_id": [1], "cutoff": [100]})
    outlets = pd.DataFrame({"outlet_id": [10, 11], "store_id": [1, 1], "opened_at": [5, 90]})
    sales = pd.DataFrame({"outlet_id": [10, 10, 10, 11, 11], "sale_ts": [1, 3, 95, 91, 200], "revenue": [10, 20, 999, 30, 999]})
    grandchild_spec = ChildTableSpec(child_df=sales, foreign_key_col="outlet_id", time_col="sale_ts", value_cols={"revenue": ["sum"]}, prefix="sale")

    out = stack_relational_features(
        parent_df=parent,
        parent_id_col="store_id",
        cutoff_col="cutoff",
        child_df=outlets,
        child_id_col="outlet_id",
        child_time_col="opened_at",
        child_foreign_key_col="store_id",
        grandchild_specs=[grandchild_spec],
        child_value_cols={},
        prefix="l2",
    )
    # Outlet 10 (opened_at=5) sees sales before ts=5 -> revenue=10+20=30 (sale_ts=1,3); sale_ts=95 excluded.
    # Outlet 11 (opened_at=90) sees sales before ts=90 -> none (sale_ts=91 and 200 are both at/after its own
    # opened_at) -> 0 after the depth-2 zero-fill. Rolled up onto the store (cutoff=100, sees both outlets):
    # 30 + 0 = 30.
    result_col = [c for c in out.columns if "sale_revenue_sum" in c and "mean" not in c][0]
    assert out[result_col].item() == 30
