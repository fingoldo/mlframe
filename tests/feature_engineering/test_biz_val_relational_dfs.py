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

from mlframe.feature_engineering.relational_dfs import (
    ChildTableSpec,
    RelationalHop,
    compute_relational_features,
    stack_relational_chain,
    stack_relational_features,
)


def _make_relational_dataset(seed: int, n_parents: int = 300):
    """Helper: Make relational dataset."""
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
    """Biz val relational dfs cutoff safe beats leaky join at serving time."""
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
    assert improvement > 0.5, (
        f"expected >50% MSE reduction at serving time vs the leaky-trained model, got {improvement:.4f} (leaky={mse_leaky_serving:.4f}, safe={mse_safe_serving:.4f})"
    )


def test_compute_relational_features_excludes_post_cutoff_rows():
    """Compute relational features excludes post cutoff rows."""
    parent = pd.DataFrame({"cust_id": [1, 2], "cutoff": [10, 10]})
    child = pd.DataFrame({"cust_id": [1, 1, 1, 2], "ts": [1, 5, 12, 3], "amount": [100, 200, 999, 50]})
    spec = ChildTableSpec(child_df=child, foreign_key_col="cust_id", time_col="ts", value_cols={"amount": ["sum", "count"]}, prefix="txn")
    out = compute_relational_features(parent, "cust_id", "cutoff", [spec])
    assert out.loc[out["cust_id"] == 1, "txn_amount_sum"].item() == 300
    assert out.loc[out["cust_id"] == 1, "txn_amount_count"].item() == 2
    assert out.loc[out["cust_id"] == 2, "txn_amount_sum"].item() == 50


def test_stack_relational_features_depth_2_respects_both_hop_cutoffs():
    """Stack relational features depth 2 respects both hop cutoffs."""
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
    result_col = next(c for c in out.columns if "sale_revenue_sum" in c and "mean" not in c)
    assert out[result_col].item() == 30


def test_stack_relational_chain_depth_2_matches_stack_relational_features():
    # RelationalHop/stack_relational_chain generalizes stack_relational_features from a fixed depth-2 chain
    # to arbitrary depth; a single-hop chain must reproduce the depth-2-specific path BIT-IDENTICALLY --
    # this is the proof the generalization is correct, not just additive.
    """Stack relational chain depth 2 matches stack relational features."""
    parent = pd.DataFrame({"store_id": [1, 2], "cutoff": [100, 100]})
    outlets = pd.DataFrame({"outlet_id": [10, 11, 12], "store_id": [1, 1, 2], "opened_at": [5, 90, 20]})
    sales = pd.DataFrame({"outlet_id": [10, 10, 10, 11, 11, 12], "sale_ts": [1, 3, 95, 91, 200, 4], "revenue": [10, 20, 999, 30, 999, 7]})
    grandchild_spec = ChildTableSpec(child_df=sales, foreign_key_col="outlet_id", time_col="sale_ts", value_cols={"revenue": ["sum"]}, prefix="sale")

    via_depth2_api = stack_relational_features(
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
    via_generic_chain = stack_relational_chain(
        parent_df=parent,
        parent_id_col="store_id",
        cutoff_col="cutoff",
        hops=[RelationalHop(df=outlets, id_col="outlet_id", time_col="opened_at", foreign_key_col="store_id", value_cols={})],
        leaf_specs=[grandchild_spec],
        prefix="l2",
    )
    pd.testing.assert_frame_equal(via_depth2_api, via_generic_chain)


def _make_depth3_dataset(seed: int, n_accounts: int = 250):
    # account (grandparent) -> device (parent) -> session (child) -> event (grandchild). The real signal
    # driving y lives in event-level values, reachable from the account only via a depth-3 rollup (event ->
    # session sum -> device mean -> account mean); session/device-level raw columns are pure noise, so a
    # depth-2-limited rollup (which can only see device+session, never events) cannot recover it.
    """Helper: Make depth3 dataset."""
    rng = np.random.default_rng(seed)
    accounts = pd.DataFrame({"account_id": np.arange(n_accounts), "cutoff": np.full(n_accounts, 2000.0)})

    device_rows, session_rows, event_rows = [], [], []
    device_id = 0
    session_id = 0
    y = np.zeros(n_accounts)
    for account_id in range(n_accounts):
        n_devices = rng.integers(2, 4)
        account_signal = 0.0
        for _ in range(n_devices):
            device_rows.append({"device_id": device_id, "account_id": account_id, "registered_at": 1000.0})
            n_sessions = rng.integers(2, 4)
            device_signal = 0.0
            for _ in range(n_sessions):
                session_rows.append({"session_id": session_id, "device_id": device_id, "session_ts": 500.0, "session_duration": rng.normal(loc=30, scale=5)})
                n_events = rng.integers(2, 5)
                event_values = rng.normal(loc=2, scale=1, size=n_events)
                for v in event_values:
                    event_rows.append({"session_id": session_id, "event_ts": rng.uniform(0, 500), "event_value": v})
                device_signal += event_values.sum()
                session_id += 1
            account_signal += device_signal
            device_id += 1
        y[account_id] = 5.0 * account_signal + rng.normal(scale=1.0)

    return accounts, pd.DataFrame(device_rows), pd.DataFrame(session_rows), pd.DataFrame(event_rows), y


def test_biz_val_stack_relational_chain_depth_3_recovers_signal_invisible_to_depth_2():
    """Biz val stack relational chain depth 3 recovers signal invisible to depth 2."""
    accounts, devices, sessions, events, y = _make_depth3_dataset(seed=0)

    leaf_spec = ChildTableSpec(child_df=events, foreign_key_col="session_id", time_col="event_ts", value_cols={"event_value": ["sum"]}, prefix="ev")

    # Depth-3-aware: chains device -> session -> event, so account-level features carry the event-derived signal.
    depth3 = stack_relational_chain(
        parent_df=accounts,
        parent_id_col="account_id",
        cutoff_col="cutoff",
        hops=[
            RelationalHop(df=devices, id_col="device_id", time_col="registered_at", foreign_key_col="account_id", value_cols={}),
            RelationalHop(df=sessions, id_col="session_id", time_col="session_ts", foreign_key_col="device_id", value_cols={"session_duration": ["mean"]}),
        ],
        leaf_specs=[leaf_spec],
        prefix="l3",
    )
    depth3_cols = [c for c in depth3.columns if c not in accounts.columns]
    X_depth3 = depth3[depth3_cols].fillna(0.0).to_numpy()

    # Depth-2-limited: the existing stack_relational_features can only reach device -> session (its
    # grandchild_specs aggregate the SESSION table's own raw columns), it structurally cannot see events.
    depth2 = stack_relational_features(
        parent_df=accounts,
        parent_id_col="account_id",
        cutoff_col="cutoff",
        child_df=devices,
        child_id_col="device_id",
        child_time_col="registered_at",
        child_foreign_key_col="account_id",
        grandchild_specs=[
            ChildTableSpec(
                child_df=sessions, foreign_key_col="device_id", time_col="session_ts", value_cols={"session_duration": ["mean", "sum"]}, prefix="sess"
            )
        ],
        child_value_cols={},
        prefix="l2",
    )
    depth2_cols = [c for c in depth2.columns if c not in accounts.columns]
    X_depth2 = depth2[depth2_cols].fillna(0.0).to_numpy()

    reg_depth3 = LinearRegression().fit(X_depth3, y)
    reg_depth2 = LinearRegression().fit(X_depth2, y)

    mse_depth3 = mean_squared_error(y, reg_depth3.predict(X_depth3))
    mse_depth2 = mean_squared_error(y, reg_depth2.predict(X_depth2))

    improvement = (mse_depth2 - mse_depth3) / mse_depth2
    assert improvement > 0.85, (
        f"expected >85% MSE reduction from the depth-3 event-aware rollup vs the depth-2-limited one, got {improvement:.4f} (depth2={mse_depth2:.4f}, depth3={mse_depth3:.4f})"
    )
