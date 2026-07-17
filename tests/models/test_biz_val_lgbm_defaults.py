"""biz_value + unit tests for ``models.default_lgbm_params`` (``extra_trees=True`` by default).

The win: on a synthetic regression dataset with many correlated/noisy features and a large tree count —
exactly the regime Ubiquant's 7th place team reported the gain in ("steady improvement when the number of
trees goes large") — LightGBM's randomized-split-threshold ``extra_trees`` mode measurably reduces held-out
RMSE vs the LightGBM default (greedy-best-split trees), averaged across several seeds to avoid a single-run
coin flip.
"""

from __future__ import annotations

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from mlframe.models.lgbm_defaults import AUTO_EXTRA_TREES_MIN_N_ESTIMATORS, default_lgbm_params


def _make_correlated_noisy_regression(n_train: int, n_test: int, seed: int):
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    # 4 informative latent factors, each duplicated into 5 noisy correlated observed columns (20 cols
    # total) plus 15 pure-noise columns -- the "many correlated + noisy features" regime where greedy-
    # best-split trees repeatedly pick near-identical splits across trees (correlated ensemble).
    latents = rng.standard_normal((n, 4))
    correlated_cols = []
    for k in range(4):
        for _ in range(5):
            correlated_cols.append(latents[:, k] + 0.3 * rng.standard_normal(n))
    noise_cols = [rng.standard_normal(n) for _ in range(15)]
    X = np.column_stack(correlated_cols + noise_cols)
    y = latents[:, 0] + 0.5 * latents[:, 1] - 0.3 * latents[:, 2] + 0.2 * rng.standard_normal(n)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def test_default_lgbm_params_extra_trees_on_by_default():
    params = default_lgbm_params()
    assert params["extra_trees"] is True


def test_default_lgbm_params_overrides_win():
    params = default_lgbm_params(extra_trees=False, n_estimators=42)
    assert params["extra_trees"] is False
    assert params["n_estimators"] == 42


def test_default_lgbm_params_objective_forwarded():
    params = default_lgbm_params(objective="binary")
    assert params["objective"] == "binary"


def test_biz_val_extra_trees_reduces_holdout_rmse_at_large_tree_count():
    n_seeds = 6
    rmse_extra = []
    rmse_default = []
    for seed in range(n_seeds):
        X_train, y_train, X_test, y_test = _make_correlated_noisy_regression(1500, 1500, seed=seed)

        params_extra = default_lgbm_params(n_estimators=600, extra_trees=True, random_state=seed)
        model_extra = lgb.LGBMRegressor(**params_extra)
        model_extra.fit(X_train, y_train)
        pred_extra = model_extra.predict(X_test)
        rmse_extra.append(float(np.sqrt(np.mean((pred_extra - y_test) ** 2))))

        params_default = default_lgbm_params(n_estimators=600, extra_trees=False, random_state=seed)
        model_default = lgb.LGBMRegressor(**params_default)
        model_default.fit(X_train, y_train)
        pred_default = model_default.predict(X_test)
        rmse_default.append(float(np.sqrt(np.mean((pred_default - y_test) ** 2))))

    mean_rmse_extra = float(np.mean(rmse_extra))
    mean_rmse_default = float(np.mean(rmse_default))
    win_rate = float(np.mean(np.array(rmse_extra) < np.array(rmse_default)))

    assert mean_rmse_extra < mean_rmse_default, (
        f"extra_trees=True should reduce mean held-out RMSE at large n_estimators: extra_trees={mean_rmse_extra:.4f} default={mean_rmse_default:.4f}"
    )
    assert win_rate >= 0.5, f"extra_trees=True should win at least half of {n_seeds} seeds, got win_rate={win_rate:.2f}"


def test_biz_val_default_lgbm_params_auto_extra_trees_adapts_to_tree_count():
    """``auto_extra_trees=True`` must flip ``extra_trees`` off below the measured floor (where it actively
    hurts) and on at/above it (where it measurably helps), tracking real held-out RMSE rather than just the
    static flag -- reuses the same correlated/noisy-features regime as the static ``extra_trees`` win above.
    """
    n_seeds = 6
    small_n_estimators = 30  # well under AUTO_EXTRA_TREES_MIN_N_ESTIMATORS -- extra_trees measured to LOSE here.
    large_n_estimators = 300  # well over the floor -- extra_trees measured to WIN 6/6 seeds here.
    assert small_n_estimators < AUTO_EXTRA_TREES_MIN_N_ESTIMATORS <= large_n_estimators

    small_params = default_lgbm_params(n_estimators=small_n_estimators, auto_extra_trees=True, extra_trees=True)
    large_params = default_lgbm_params(n_estimators=large_n_estimators, auto_extra_trees=True, extra_trees=True)
    assert small_params["extra_trees"] is False, "adaptive rule must force extra_trees OFF below the measured floor"
    assert large_params["extra_trees"] is True, "adaptive rule must force extra_trees ON at/above the measured floor"

    rmse_small_auto, rmse_small_off, rmse_large_auto, rmse_large_static_off = [], [], [], []
    for seed in range(n_seeds):
        X_train, y_train, X_test, y_test = _make_correlated_noisy_regression(1500, 1500, seed=seed)

        def _rmse(params: dict) -> float:
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)  # noqa: B023 -- closure invoked 4x below, same iteration, never stored
            pred = model.predict(X_test)  # noqa: B023 -- closure invoked 4x below, same iteration, never stored
            return float(np.sqrt(np.mean((pred - y_test) ** 2)))  # noqa: B023 -- closure invoked 4x below, same iteration, never stored

        rmse_small_auto.append(_rmse(default_lgbm_params(n_estimators=small_n_estimators, auto_extra_trees=True, random_state=seed)))
        rmse_small_off.append(_rmse(default_lgbm_params(n_estimators=small_n_estimators, extra_trees=True, random_state=seed)))
        rmse_large_auto.append(_rmse(default_lgbm_params(n_estimators=large_n_estimators, auto_extra_trees=True, random_state=seed)))
        rmse_large_static_off.append(_rmse(default_lgbm_params(n_estimators=large_n_estimators, extra_trees=False, random_state=seed)))

    mean_small_auto = float(np.mean(rmse_small_auto))
    mean_small_off = float(np.mean(rmse_small_off))
    mean_large_auto = float(np.mean(rmse_large_auto))
    mean_large_static_off = float(np.mean(rmse_large_static_off))

    assert mean_small_auto < mean_small_off, (
        f"below the floor, auto_extra_trees (extra_trees forced OFF) should beat a static extra_trees=True: "
        f"auto={mean_small_auto:.4f} static_extra_trees_on={mean_small_off:.4f}"
    )
    assert mean_large_auto < mean_large_static_off, (
        f"at/above the floor, auto_extra_trees (extra_trees forced ON) should beat a static extra_trees=False: "
        f"auto={mean_large_auto:.4f} static_extra_trees_off={mean_large_static_off:.4f}"
    )


def test_default_lgbm_params_auto_extra_trees_is_opt_in():
    """Omitting ``auto_extra_trees`` must leave the static default (``extra_trees=True`` always) unchanged."""
    baseline_small = default_lgbm_params(n_estimators=30)
    baseline_large = default_lgbm_params(n_estimators=300)
    assert baseline_small["extra_trees"] is True
    assert baseline_large["extra_trees"] is True
    assert default_lgbm_params() == default_lgbm_params()  # bit-identical across calls, new params untouched.
