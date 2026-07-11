"""biz_value test for ``feature_selection.drop_raw_after_embedding.drop_raw_after_embedding``.

Synthetic: a moderate-cardinality entity-id column's target-mean encoding already captures all the id-level
signal a linear model can use (each id maps to one smoothed encoding value shared by its whole group). One-hot
expanding the raw id ALONGSIDE that encoding adds hundreds of near-collinear, near-noise columns competing for
the same fixed L2 regularization budget -- diluting the penalty actually available to the two genuinely
informative features (``x``, the encoding) and degrading held-out performance versus dropping the raw id once
its encoding exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlframe.feature_selection.drop_raw_after_embedding import drop_raw_after_embedding


def _make_dataset(n_rows: int, n_entities: int, seed: int):
    rng = np.random.default_rng(seed)
    entity_id = rng.integers(0, n_entities, size=n_rows)
    x = rng.normal(size=n_rows)
    y = 2.0 * x + rng.normal(scale=0.5, size=n_rows)
    df = pd.DataFrame({"entity_id": entity_id.astype(str), "x": x})
    return df, y


def test_biz_val_drop_raw_after_embedding_reduces_overfit_from_high_cardinality_raw_id():
    df, y = _make_dataset(n_rows=6000, n_entities=2000, seed=0)
    df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=0)

    entity_mean_map = df_train.assign(y=y_train).groupby("entity_id")["y"].mean()
    global_mean = float(y_train.mean())
    df_train = df_train.assign(entity_id_target_enc=df_train["entity_id"].map(entity_mean_map).fillna(global_mean))
    df_test = df_test.assign(entity_id_target_enc=df_test["entity_id"].map(entity_mean_map).fillna(global_mean))

    raw_to_derived = {"entity_id": ["entity_id_target_enc"]}

    features_with_raw = ["entity_id", "x", "entity_id_target_enc"]
    X_train_with_raw = pd.get_dummies(df_train[features_with_raw], columns=["entity_id"])
    X_test_with_raw = pd.get_dummies(df_test[features_with_raw], columns=["entity_id"]).reindex(columns=X_train_with_raw.columns, fill_value=0)
    model_with_raw = Ridge(alpha=0.3, random_state=0)
    model_with_raw.fit(X_train_with_raw, y_train)
    rmse_with_raw = float(mean_squared_error(y_test, model_with_raw.predict(X_test_with_raw)) ** 0.5)

    df_train_dropped = drop_raw_after_embedding(df_train, raw_to_derived)
    df_test_dropped = drop_raw_after_embedding(df_test, raw_to_derived)
    assert "entity_id" not in df_train_dropped.columns
    assert "entity_id_target_enc" in df_train_dropped.columns

    feature_cols_dropped = ["x", "entity_id_target_enc"]
    model_dropped = Ridge(alpha=0.3, random_state=0)
    model_dropped.fit(df_train_dropped[feature_cols_dropped], y_train)
    rmse_dropped = float(mean_squared_error(y_test, model_dropped.predict(df_test_dropped[feature_cols_dropped])) ** 0.5)

    assert rmse_dropped < rmse_with_raw * 0.9, f"expected dropping the raw high-cardinality id to reduce test RMSE by >=10% vs keeping it, got dropped={rmse_dropped:.4f} with_raw={rmse_with_raw:.4f}"


def test_drop_raw_after_embedding_keeps_raw_when_derived_missing():
    df = pd.DataFrame({"entity_id": ["a", "b"], "x": [1.0, 2.0]})
    out = drop_raw_after_embedding(df, raw_to_derived={"entity_id": ["entity_id_target_enc"]})
    assert "entity_id" in out.columns


def test_drop_raw_after_embedding_min_derived_present_threshold():
    df = pd.DataFrame({"entity_id": ["a", "b"], "enc_1": [0.1, 0.2], "x": [1.0, 2.0]})
    out = drop_raw_after_embedding(df, raw_to_derived={"entity_id": ["enc_1", "enc_2"]}, min_derived_present=2)
    assert "entity_id" in out.columns
    out2 = drop_raw_after_embedding(df, raw_to_derived={"entity_id": ["enc_1", "enc_2"]}, min_derived_present=1)
    assert "entity_id" not in out2.columns
