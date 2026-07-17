"""biz_value test for ``feature_engineering.nearest_past_join``.

The win: attaching each row's nearest-PAST known value of a slowly-drifting per-entity signal (via the as-of
join) gives a materially better predictive feature than having no time-varying signal at all (a global-mean
baseline) -- and the join must respect the no-future-leakage backward-match contract (only past rows are
eligible), which is verified directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from mlframe.feature_engineering.nearest_past_join import nearest_past_join


def test_biz_val_nearest_past_join_feature_beats_no_signal_baseline():
    rng = np.random.default_rng(0)
    n_entities = 200

    history_rows = []
    query_rows = []
    for entity_id in range(n_entities):
        drift = rng.normal(0, 1)
        n_history = rng.integers(5, 10)
        history_times = np.sort(rng.choice(np.arange(1, 100), size=n_history, replace=False))
        history_values = drift + rng.normal(0, 0.1, n_history)
        for t, v in zip(history_times, history_values):
            history_rows.append({"entity": entity_id, "t": int(t), "val": float(v)})

        query_time = int(history_times[-1]) + int(rng.integers(1, 5))
        y = drift + rng.normal(0, 0.15)
        query_rows.append({"entity": entity_id, "t": query_time, "y": y})

    history_df = pd.DataFrame(history_rows)
    query_df = pd.DataFrame(query_rows)

    joined = nearest_past_join(query_df, history_df, on="t", by=["entity"], right_value_cols=["val"])
    assert joined["val"].notna().all()

    model_with_feature = LinearRegression().fit(joined[["val"]], joined["y"])
    pred_with_feature = model_with_feature.predict(joined[["val"]])
    rmse_with_feature = float(np.sqrt(mean_squared_error(joined["y"], pred_with_feature)))

    rmse_baseline = float(np.sqrt(mean_squared_error(joined["y"], np.full(len(joined), joined["y"].mean()))))

    assert rmse_with_feature < rmse_baseline * 0.5, (
        f"the as-of-joined feature should substantially beat a no-signal global-mean baseline: "
        f"with_feature={rmse_with_feature:.4f} baseline={rmse_baseline:.4f}"
    )


def test_nearest_past_join_respects_backward_only_no_future_leakage():
    left = pd.DataFrame({"entity": ["a", "a"], "t": [1, 3]})
    right = pd.DataFrame({"entity": ["a", "a", "a"], "t": [0, 2, 5], "val": [10, 20, 999]})

    result = nearest_past_join(left, right, on="t", by=["entity"], right_value_cols=["val"])
    assert result.loc[result["t"] == 1, "val"].iloc[0] == 10
    assert result.loc[result["t"] == 3, "val"].iloc[0] == 20  # NOT the t=5 future row (val=999)


def test_nearest_past_join_no_eligible_history_gives_nan():
    left = pd.DataFrame({"entity": ["a"], "t": [0]})
    right = pd.DataFrame({"entity": ["a"], "t": [5], "val": [42]})
    result = nearest_past_join(left, right, on="t", by=["entity"], right_value_cols=["val"])
    assert pd.isna(result["val"].iloc[0])


def test_nearest_past_join_preserves_row_count_and_order():
    left = pd.DataFrame({"entity": ["b", "a", "a"], "t": [10, 5, 1]})
    right = pd.DataFrame({"entity": ["a", "b"], "t": [0, 0], "val": [1.0, 2.0]})
    result = nearest_past_join(left, right, on="t", by=["entity"], right_value_cols=["val"])
    assert len(result) == 3
    assert list(result["entity"]) == ["b", "a", "a"]


def _make_sparse_key_dataset(rng: np.random.Generator, n_entities: int = 150):
    """Each entity's history covers only 6 of 28 possible (time-of-day, weekday) combos -- a query row's
    exact combo is usually unseen for that entity (finest key too sparse), but its time-of-day alone is
    almost always covered (coarser key resolves it), and every entity always has SOME history (global
    tier is the last resort, rarely needed here).
    """
    history_rows = []
    query_rows = []
    for entity_id in range(n_entities):
        level = rng.normal(0, 1)
        combos = rng.choice(28, size=6, replace=False)
        for combo in combos:
            tod, wd = int(combo % 4), int(combo // 4)
            t = int(rng.integers(1, 50))
            history_rows.append({"entity": entity_id, "tod": tod, "wd": wd, "t": t, "val": level + rng.normal(0, 0.1)})
        query_tod, query_wd = int(rng.integers(0, 4)), int(rng.integers(0, 7))
        y = level + rng.normal(0, 0.15)
        query_rows.append({"entity": entity_id, "tod": query_tod, "wd": query_wd, "t": 100, "y": y})
    return pd.DataFrame(history_rows), pd.DataFrame(query_rows)


def test_biz_val_nearest_past_join_fallback_chain_recovers_sparse_key_matches():
    rng = np.random.default_rng(0)
    history_df, query_df = _make_sparse_key_dataset(rng)

    fine_only = nearest_past_join(query_df, history_df, on="t", by=["entity", "tod", "wd"], right_value_cols=["val"])
    fine_nan_frac = float(fine_only["val"].isna().mean())
    assert fine_nan_frac > 0.5, "fixture must actually be sparse under the finest key for the test to be meaningful"

    chain = nearest_past_join(
        query_df,
        history_df,
        on="t",
        by=["entity", "tod", "wd"],
        right_value_cols=["val"],
        fallback_by_chain=[["entity", "tod"], ["entity"]],
        tier_col="tier",
    )
    chain_nan_frac = float(chain["val"].isna().mean())
    assert chain_nan_frac < fine_nan_frac * 0.1, (
        f"the fallback chain should recover the vast majority of matches the finest key misses: "
        f"fine_nan_frac={fine_nan_frac:.3f} chain_nan_frac={chain_nan_frac:.3f}"
    )
    # rows recovered via a coarser tier are actually tagged as such, not silently attributed to tier 0
    assert (chain["tier"] > 0).sum() > 0

    fine_mean_imputed = fine_only["val"].fillna(fine_only["val"].mean())
    rmse_fine = float(np.sqrt(mean_squared_error(query_df["y"], fine_mean_imputed)))
    rmse_chain = float(np.sqrt(mean_squared_error(query_df["y"], chain["val"])))
    assert rmse_chain < rmse_fine * 0.5, (
        f"the fallback-chain feature should substantially beat mean-imputing the finest-key-only feature's "
        f"gaps: rmse_fine_mean_imputed={rmse_fine:.4f} rmse_chain={rmse_chain:.4f}"
    )


def test_nearest_past_join_fallback_chain_omitted_matches_single_key_bit_identical():
    rng = np.random.default_rng(1)
    history_df, query_df = _make_sparse_key_dataset(rng)

    baseline = nearest_past_join(query_df, history_df, on="t", by=["entity", "tod", "wd"], right_value_cols=["val"])
    explicit_default = nearest_past_join(query_df, history_df, on="t", by=["entity", "tod", "wd"], right_value_cols=["val"], fallback_by_chain=None)
    pd.testing.assert_frame_equal(baseline, explicit_default)


def test_nearest_past_join_fallback_chain_min_group_size_treats_thin_groups_as_sparse():
    left = pd.DataFrame({"g": [1, 2], "t": [10, 10]})
    right = pd.DataFrame({"g": [1, 1, 2], "t": [1, 2, 1], "val": [10.0, 20.0, 5.0]})

    result = nearest_past_join(left, right, on="t", by=["g"], right_value_cols=["val"], fallback_by_chain=[[]], min_group_size=2, tier_col="tier")
    # g=2's group has only 1 historical row -- too sparse under min_group_size=2, falls to the global tier
    assert result.loc[result["g"] == 2, "tier"].iloc[0] == 1
    assert result.loc[result["g"] == 2, "val"].iloc[0] == 20.0
    # g=1's group has 2 historical rows -- meets min_group_size, resolved at tier 0
    assert result.loc[result["g"] == 1, "tier"].iloc[0] == 0
