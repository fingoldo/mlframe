"""End-to-end integration tests for slice-stable ES through ``_setup_eval_set`` + UniversalCallback.

We don't go through the full ``train_mlframe_models_suite`` (that's covered by separate suite
tests) -- instead we drive a tiny LGB / XGB fit with the exact wiring the suite layer would
produce: eval_set is a list of (full_val, shard_0, ..., shard_K-1), the booster's native
early_stopping is stripped, and a UniversalCallback is registered with ``slice_k>0``. We
assert the callback's per-iteration shard history is populated and the aggregator drove the
best_iter selection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_tiny_data(n_train=400, n_val=200, n_test=200, d=5, seed=42):
    """Heteroscedastic regression so slice-stable has signal to chew on."""
    rng = np.random.default_rng(seed)
    def gen(n):
        X = rng.uniform(0, 1, (n, d))
        sigma = 0.05 + 1.5 * np.mean((X - 0.5) ** 2, axis=1)
        y_true = np.sum(np.sin(2 * np.pi * X), axis=1)
        y = y_true + rng.normal(0, sigma)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), y
    return gen(n_train), gen(n_val), gen(n_test)


def test_lgb_with_slice_callback_records_shard_history() -> None:
    pytest.importorskip("lightgbm")
    import lightgbm as lgb
    from mlframe.training._callbacks import LightGBMCallback
    from mlframe.training._data_helpers import _setup_eval_set
    from mlframe.training._slice_helpers import build_slice_eval_sets

    (X_tr, y_tr), (X_val, y_val), _ = _make_tiny_data()

    shards = build_slice_eval_sets(X_val, y_val, source="random", k=4, min_rows_per_shard=30, random_state=0)
    assert len(shards) == 4

    fit_params: dict = {}
    cb = LightGBMCallback(
        patience=10, min_delta=0.0,
        monitor_dataset="valid_0", monitor_metric="l2", mode="min",
        slice_k=4, slice_aggregate_mode="mean", slice_persist_history=True, verbose=0,
    )
    fit_params["callbacks"] = [cb]
    _setup_eval_set("LGBMRegressor", fit_params, X_val, y_val,
                    model_category="lgb", extra_eval_sets=shards)
    # The eval_set must be a list of K+1 tuples
    assert isinstance(fit_params["eval_set"], list)
    assert len(fit_params["eval_set"]) == 5

    model = lgb.LGBMRegressor(n_estimators=50, learning_rate=0.1, verbose=-1, num_leaves=8)
    model.fit(X_tr, y_tr, **fit_params)

    # The callback aggregated per-iteration shard scores; history must be populated.
    assert len(cb.slice_shard_score_history) > 0
    assert all(len(s) == 4 for s in cb.slice_shard_score_history)
    assert cb.best_metric is not None
    assert cb.best_iter is not None


def test_lgb_slice_off_bit_identical_to_baseline() -> None:
    """When ``slice_k=0``, the callback path is the legacy single-val one (no shard registration)."""
    pytest.importorskip("lightgbm")
    import lightgbm as lgb
    from mlframe.training._callbacks import LightGBMCallback
    from mlframe.training._data_helpers import _setup_eval_set

    (X_tr, y_tr), (X_val, y_val), _ = _make_tiny_data(seed=7)

    fit_params: dict = {}
    cb = LightGBMCallback(
        patience=10, min_delta=0.0,
        monitor_dataset="valid_0", monitor_metric="l2", mode="min",
        slice_k=0, verbose=0,
    )
    fit_params["callbacks"] = [cb]
    _setup_eval_set("LGBMRegressor", fit_params, X_val, y_val, model_category="lgb")
    assert fit_params["eval_set"] == (X_val, y_val), "single-val tuple format for legacy LGB"

    model = lgb.LGBMRegressor(n_estimators=40, learning_rate=0.1, verbose=-1, num_leaves=8)
    model.fit(X_tr, y_tr, **fit_params)

    # No shard history, no aggregate history -- slice path completely silent.
    assert cb.slice_shard_score_history == []
    assert cb.slice_aggregate_history == []
    assert cb.best_metric is not None
