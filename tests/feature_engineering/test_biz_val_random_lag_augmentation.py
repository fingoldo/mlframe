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
    assert improvement > 0.3, (
        f"expected >30% MSE reduction vs. freshest-only training under a fixed serving staleness, got {improvement:.4f} (baseline={mse_baseline:.2f}, randomized={mse_randomized:.2f})"
    )


def test_biz_val_randomize_as_of_lag_histogram_beats_uniform_under_skewed_serving_lag():
    """Real serving-pipeline staleness is rarely uniform -- most requests hit a fast-refreshed cache
    (short lag) and a minority hit a slow/stale path (long tail). Training with ``randomize_as_of_lag``'s
    default UNIFORM draw over the same ``[min_lag, max_lag]`` range oversamples the rare long-lag regime
    relative to its true frequency, biasing the learned coefficient away from the short-lag-dominated
    regime the model actually sees served. Training with the opt-in empirical ``lag_histogram_edges``/
    ``lag_histogram_counts`` mode -- shaped to match the true observed staleness histogram -- should
    generalize better to test rows whose serving lag is drawn from that same skewed distribution."""
    n_entities, n_history = 600, 40
    history_df, y, label_t = _make_entity_history(n_entities, n_history, seed=2)
    train_entities = np.arange(0, 400)
    test_entities = np.arange(400, 600)
    aggs = {"value": ["mean"]}

    # Empirical staleness histogram: mostly short lags (fast refresh path), occasional long tail (stale path).
    hist_edges = [0.0, 1.0, 2.0, 5.0, 15.0]
    hist_counts = [70.0, 15.0, 10.0, 5.0]

    as_of_train_fresh = pd.DataFrame({"entity": train_entities, "as_of": label_t})

    as_of_train_uniform = randomize_as_of_lag(as_of_train_fresh, "as_of", max_lag=15.0, min_lag=0.0, random_state=0)
    X_train_uniform = leakage_safe_aggregate(history_df, "entity", "t", as_of_train_uniform, aggs).set_index("entity").reindex(train_entities)

    as_of_train_hist = randomize_as_of_lag(
        as_of_train_fresh, "as_of", max_lag=15.0, min_lag=0.0, random_state=0, lag_histogram_edges=hist_edges, lag_histogram_counts=hist_counts
    )
    X_train_hist = leakage_safe_aggregate(history_df, "entity", "t", as_of_train_hist, aggs).set_index("entity").reindex(train_entities)

    # Test rows' true serving lag is drawn from the SAME skewed histogram -- the realistic production scenario.
    as_of_test_fresh = pd.DataFrame({"entity": test_entities, "as_of": label_t})
    as_of_test = randomize_as_of_lag(
        as_of_test_fresh, "as_of", max_lag=15.0, min_lag=0.0, random_state=99, lag_histogram_edges=hist_edges, lag_histogram_counts=hist_counts
    )
    X_test = leakage_safe_aggregate(history_df, "entity", "t", as_of_test, aggs).set_index("entity").reindex(test_entities)

    y_train, y_test = y[train_entities], y[test_entities]

    uniform_model = LinearRegression().fit(X_train_uniform.to_numpy(), y_train)
    mse_uniform = mean_squared_error(y_test, uniform_model.predict(X_test.to_numpy()))

    hist_model = LinearRegression().fit(X_train_hist.to_numpy(), y_train)
    mse_hist = mean_squared_error(y_test, hist_model.predict(X_test.to_numpy()))

    improvement = 1.0 - mse_hist / mse_uniform
    assert improvement > 0.15, (
        f"expected >15% MSE reduction vs. uniform-lag training under skewed serving staleness, got {improvement:.4f} (uniform={mse_uniform:.2f}, histogram={mse_hist:.2f})"
    )


def test_randomize_as_of_lag_histogram_requires_both_edges_and_counts():
    as_of = pd.DataFrame({"entity": range(10), "as_of": 100.0})
    try:
        randomize_as_of_lag(as_of, "as_of", max_lag=10.0, lag_histogram_edges=[0.0, 5.0, 10.0])
        raise AssertionError("expected ValueError when only lag_histogram_edges is supplied")
    except ValueError:
        pass
    try:
        randomize_as_of_lag(as_of, "as_of", max_lag=10.0, lag_histogram_counts=[1.0, 1.0])
        raise AssertionError("expected ValueError when only lag_histogram_counts is supplied")
    except ValueError:
        pass


def test_randomize_as_of_lag_histogram_shifts_within_bin_bounds():
    as_of = pd.DataFrame({"entity": range(2000), "as_of": 100.0})
    shifted = randomize_as_of_lag(
        as_of, "as_of", max_lag=15.0, lag_histogram_edges=[0.0, 1.0, 2.0, 5.0, 15.0], lag_histogram_counts=[70.0, 15.0, 10.0, 5.0], random_state=0
    )
    offsets = 100.0 - shifted["as_of"].to_numpy()
    assert (offsets >= 0.0).all() and (offsets <= 15.0).all()
    # Skewed toward the short-lag bin -- most offsets should land under 1.0 given a 70% weight there.
    assert (offsets < 1.0).mean() > 0.5


def test_randomize_as_of_lag_datetime_cutoff_matches_default_when_histogram_omitted():
    as_of = pd.DataFrame({"entity": [1, 2, 3], "as_of": 100.0})
    default = randomize_as_of_lag(as_of, "as_of", max_lag=15.0, min_lag=0.0, random_state=0)
    explicit_none = randomize_as_of_lag(as_of, "as_of", max_lag=15.0, min_lag=0.0, random_state=0, lag_histogram_edges=None, lag_histogram_counts=None)
    pd.testing.assert_frame_equal(default, explicit_none)


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
