"""biz_value test for ``evaluation.compare_cv_schemes``.

The win: reproduces the source's counter-intuitive finding -- when the true out-of-time scenario has the
SAME entities continuing into the future (a realistic "existing customers keep transacting" production
setup), GroupKFold's held-out-entity simulation is an artificially pessimistic proxy (it tests a cold-start
scenario that never actually happens), while plain KFold (which lets the model learn each entity's fixed
effect from its other rows) tracks the true out-of-time score far more closely. ``compare_cv_schemes`` should
correctly pick KFold as the best scheme here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold

from mlframe.evaluation.compare_cv_schemes import compare_cv_schemes


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def test_biz_val_compare_cv_schemes_picks_kfold_when_entities_persist_into_the_future():
    rng = np.random.default_rng(0)
    n_entities = 60
    rows_hist, rows_future = 40, 10
    entity_offset = rng.normal(0, 3, n_entities)

    rows = []
    for e in range(n_entities):
        for t in range(rows_hist + rows_future):
            rows.append({"entity": e, "t": t, "y": 10 + 0.05 * t + entity_offset[e] + rng.normal(0, 0.5)})
    df = pd.DataFrame(rows)
    X = df[["entity", "t"]]
    y = df["y"].to_numpy()

    is_hist = df["t"].to_numpy() < rows_hist
    hist_idx = np.flatnonzero(is_hist)
    future_idx = np.flatnonzero(~is_hist)

    kfold_splits = [(hist_idx[tr], hist_idx[te]) for tr, te in KFold(5, shuffle=True, random_state=0).split(hist_idx)]
    groupkfold_splits = [
        (hist_idx[tr], hist_idx[te]) for tr, te in GroupKFold(5).split(hist_idx, groups=df["entity"].to_numpy()[hist_idx])
    ]

    result = compare_cv_schemes(
        X,
        y,
        schemes={"kfold": kfold_splits, "groupkfold": groupkfold_splits},
        ooo_time_idx=(hist_idx, future_idx),
        model_factory=lambda: RandomForestRegressor(n_estimators=100, max_depth=8, random_state=0),
        metric_fn=_rmse,
    )

    assert result["best_scheme"] == "kfold", result
    assert result["scheme_scores"]["kfold"]["gap_to_ooo_time"] < result["scheme_scores"]["groupkfold"]["gap_to_ooo_time"] * 0.5


def test_compare_cv_schemes_empty_schemes_returns_none_best():
    X = np.zeros((10, 1))
    y = np.zeros(10)
    result = compare_cv_schemes(
        X, y, schemes={}, ooo_time_idx=(np.arange(5), np.arange(5, 10)),
        model_factory=lambda: RandomForestRegressor(n_estimators=5), metric_fn=_rmse,
    )
    assert result["best_scheme"] is None
    assert result["scheme_scores"] == {}
