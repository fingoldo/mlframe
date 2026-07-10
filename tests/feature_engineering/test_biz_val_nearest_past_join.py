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
