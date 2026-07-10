"""biz_value test for ``feature_engineering.randomize_as_of_lag``.

The win: per-entity history trends linearly with entity-specific slope, and a naive as-of feature pipeline
always computes TRAINING features at the freshest possible cutoff (lag=0). At TRUE serving time, a fixed
pipeline-refresh delay means features are always computed at a materially staler cutoff -- a train/serve
freshness mismatch. A model trained only on freshest-cutoff features learns a coefficient calibrated for
that regime, which is miscalibrated for the systematically-different staler feature distribution it will
actually see served. Training with a per-row RANDOMIZED lag (spanning the true serving-lag range) instead
teaches a coefficient that generalizes to the fixed staleness level actually encountered at serve time --
mirroring the Power Laws Forecasting 1st place's random-lag training / sequential-lag validation split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from mlframe.feature_engineering import leakage_safe_aggregate, randomize_as_of_lag


def _make_entity_history(n_entities: int, n_history: int, seed: int):
    rng = np.random.default_rng(seed)
    entity_level = rng.normal(scale=2.0, size=n_entities)
    slopes = rng.normal(scale=0.6, size=n_entities)
    rows = []
    values_at_label: dict[int, float] = {}
    label_t = float(n_history - 5)
    for e in range(n_entities):
        v = entity_level[e] + slopes[e] * np.arange(n_history) + rng.normal(scale=0.15, size=n_history)
        values_at_label[e] = float(v[int(label_t)])
        for t in range(n_history):
            rows.append({"entity": e, "t": float(t), "value": v[t]})
    history_df = pd.DataFrame(rows)
    y = np.array([values_at_label[e] for e in range(n_entities)])
    return history_df, y, label_t


def test_biz_val_randomize_as_of_lag_beats_freshest_only_training_under_serve_staleness():
    n_entities, n_history = 600, 40
    history_df, y, label_t = _make_entity_history(n_entities, n_history, seed=1)
    train_entities = np.arange(0, 400)
    test_entities = np.arange(400, 600)
    aggs = {"value": ["mean"]}
    serve_lag = 15.0

    as_of_train_fresh = pd.DataFrame({"entity": train_entities, "as_of": label_t})
    X_train_baseline = leakage_safe_aggregate(history_df, "entity", "t", as_of_train_fresh, aggs).set_index("entity").reindex(train_entities)

    as_of_train_randomized = randomize_as_of_lag(as_of_train_fresh, "as_of", max_lag=serve_lag, min_lag=0.0, random_state=0)
    X_train_randomized = leakage_safe_aggregate(history_df, "entity", "t", as_of_train_randomized, aggs).set_index("entity").reindex(train_entities)

    # Test always served at the TRUE fixed staleness -- the realistic production scenario.
    as_of_test = pd.DataFrame({"entity": test_entities, "as_of": label_t - serve_lag})
    X_test = leakage_safe_aggregate(history_df, "entity", "t", as_of_test, aggs).set_index("entity").reindex(test_entities)

    y_train, y_test = y[train_entities], y[test_entities]

    baseline_model = LinearRegression().fit(X_train_baseline.to_numpy(), y_train)
    mse_baseline = mean_squared_error(y_test, baseline_model.predict(X_test.to_numpy()))

    randomized_model = LinearRegression().fit(X_train_randomized.to_numpy(), y_train)
    mse_randomized = mean_squared_error(y_test, randomized_model.predict(X_test.to_numpy()))

    improvement = 1.0 - mse_randomized / mse_baseline
    assert improvement > 0.3, f"expected >30% MSE reduction vs. freshest-only training under a fixed serving staleness, got {improvement:.4f} (baseline={mse_baseline:.2f}, randomized={mse_randomized:.2f})"


def test_randomize_as_of_lag_shifts_cutoff_within_bounds():
    as_of = pd.DataFrame({"entity": range(500), "as_of": 100.0})
    shifted = randomize_as_of_lag(as_of, "as_of", max_lag=10.0, min_lag=2.0, random_state=0)
    offsets = 100.0 - shifted["as_of"].to_numpy()
    assert (offsets >= 2.0).all() and (offsets <= 10.0).all()
    assert not np.allclose(offsets, offsets[0])  # genuinely per-row random, not a constant shift


def test_randomize_as_of_lag_does_not_mutate_input():
    as_of = pd.DataFrame({"entity": [1, 2, 3], "as_of": [10.0, 20.0, 30.0]})
    original = as_of.copy()
    randomize_as_of_lag(as_of, "as_of", max_lag=5.0, random_state=0)
    pd.testing.assert_frame_equal(as_of, original)


def test_randomize_as_of_lag_datetime_cutoff():
    as_of = pd.DataFrame({"entity": [1, 2], "as_of": pd.to_datetime(["2024-01-10", "2024-01-10"])})
    shifted = randomize_as_of_lag(as_of, "as_of", max_lag=pd.Timedelta(days=5), min_lag=pd.Timedelta(days=1), random_state=0)
    deltas = as_of["as_of"] - shifted["as_of"]
    assert (deltas >= pd.Timedelta(days=1)).all()
    assert (deltas <= pd.Timedelta(days=5)).all()
