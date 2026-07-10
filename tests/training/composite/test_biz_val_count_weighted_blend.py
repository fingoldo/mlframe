"""biz_value test for ``training.composite.CountWeightedBlendEnsemble``.

The win: entities have highly skewed observation counts (most entities have 1-3 training rows, a few have
60-100). Each entity has a real, stable effect, but a single noisy observation is a poor (high-variance)
estimate of it -- a purely entity-specific model (one-hot per-entity intercept) overfits to that noise for
sparse entities, while a metadata-only global model ignores the entity effect entirely. Blending the two by
per-entity observation count should recover a materially lower error than either extreme alone -- mirroring
the KKBox Music Recommendation 1st place's "rely more on user-embedding if data sufficient, more on metadata
if not" technique.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from mlframe.training.composite import CountWeightedBlendEnsemble


def _make_skewed_entity_dataset(n_entities: int, seed: int, noise_scale: float = 12.0):
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, 4, n_entities)
    counts[:20] = rng.integers(60, 100, 20)  # a few well-observed entities
    entity_effect = rng.normal(scale=8.0, size=n_entities)
    w = 2.0

    rows = []
    for e in range(n_entities):
        n_obs = counts[e] + 5  # extra rows so every entity has at least a few held-out test rows too
        x = rng.normal(size=n_obs)
        y = x * w + entity_effect[e] + rng.normal(scale=noise_scale, size=n_obs)
        for i in range(n_obs):
            rows.append({"entity": e, "x": x[i], "y": y[i], "is_train_pool": i < counts[e]})
    df = pd.DataFrame(rows)
    train_df = df[df["is_train_pool"]].reset_index(drop=True)
    test_df = df[~df["is_train_pool"]].reset_index(drop=True)
    return train_df[["entity", "x"]], train_df["y"].to_numpy(), test_df[["entity", "x"]], test_df["y"].to_numpy()


def _entity_pipeline():
    return make_pipeline(ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), ["entity"])], remainder="passthrough"), LinearRegression())


def test_biz_val_count_weighted_blend_beats_entity_only_and_global_only_mse():
    X_train, y_train, X_test, y_test = _make_skewed_entity_dataset(n_entities=300, seed=0)

    entity_only = _entity_pipeline().fit(X_train, y_train)
    mse_entity_only = mean_squared_error(y_test, entity_only.predict(X_test))

    global_only = LinearRegression().fit(X_train[["x"]], y_train)
    mse_global_only = mean_squared_error(y_test, global_only.predict(X_test[["x"]]))

    blend = CountWeightedBlendEnsemble(entity_estimator=_entity_pipeline(), global_estimator=LinearRegression(), entity_col="entity", metadata_cols=["x"], k=10.0)
    blend.fit(X_train, y_train)
    mse_blend = mean_squared_error(y_test, blend.predict(X_test))

    improvement_vs_entity = 1.0 - mse_blend / mse_entity_only
    improvement_vs_global = 1.0 - mse_blend / mse_global_only
    assert improvement_vs_entity > 0.1, f"expected >10% MSE reduction vs. the entity-only model, got {improvement_vs_entity:.4f}"
    assert improvement_vs_global > 0.05, f"expected >5% MSE reduction vs. the global-only model, got {improvement_vs_global:.4f}"


def test_count_weighted_blend_weight_increases_with_observation_count():
    X_train, y_train, X_test, y_test = _make_skewed_entity_dataset(n_entities=300, seed=1)
    blend = CountWeightedBlendEnsemble(entity_estimator=_entity_pipeline(), global_estimator=LinearRegression(), entity_col="entity", metadata_cols=["x"], k=10.0)
    blend.fit(X_train, y_train)

    weights = blend._blend_weight(X_test)
    counts = X_test["entity"].map(X_train["entity"].value_counts().to_dict()).fillna(0).to_numpy()
    dense_mask = counts > 30
    sparse_mask = counts <= 3
    assert weights[dense_mask].mean() > weights[sparse_mask].mean()


def test_count_weighted_blend_unseen_entity_gets_zero_weight():
    X_train, y_train, _, _ = _make_skewed_entity_dataset(n_entities=50, seed=2)
    blend = CountWeightedBlendEnsemble(entity_estimator=_entity_pipeline(), global_estimator=LinearRegression(), entity_col="entity", metadata_cols=["x"], k=10.0)
    blend.fit(X_train, y_train)

    X_unseen = pd.DataFrame({"entity": [99999.0], "x": [0.5]})
    weight = blend._blend_weight(X_unseen)
    assert weight[0] == 0.0
