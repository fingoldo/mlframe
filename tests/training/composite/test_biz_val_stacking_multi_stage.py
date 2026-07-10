"""biz_value test for ``training.composite.MultiStageMetaFeatureStacker``.

The win: when a primary target depends linearly on a hidden factor ``z`` that is only recoverable
NONLINEARLY from the raw features, a linear stage-2 model trained directly on the raw features can't decode
``z``. An auxiliary target that also depends on ``z`` (with less noise) lets a nonlinear stage-1 model
recover ``z`` well; feeding its OOF prediction in as a meta-feature lets the linear stage-2 model pick it up
trivially, beating the same linear model trained on raw features alone by a wide margin. This mirrors the
MoA competition's non-scored-target-as-meta-feature technique.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlframe.training.composite import MultiStageMetaFeatureStacker


def _make_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    X = np.column_stack(
        [
            np.sin(z * 2) + rng.normal(scale=0.3, size=n),
            z**2 + rng.normal(scale=0.3, size=n),
            rng.normal(size=n),
            rng.normal(size=n),
        ]
    )
    y_aux = z + rng.normal(scale=0.2, size=n)
    y_primary = 5 * z + rng.normal(scale=0.5, size=n)
    return X, y_primary, y_aux


def test_biz_val_multi_stage_stacker_beats_raw_feature_baseline_mse():
    X, y_primary, y_aux = _make_dataset(n=3000, seed=0)
    X_train, X_test, y_train, y_test, y_aux_train, _ = train_test_split(X, y_primary, y_aux, test_size=0.3, random_state=0)

    baseline = LinearRegression()
    baseline.fit(X_train, y_train)
    baseline_mse = mean_squared_error(y_test, baseline.predict(X_test))

    stacker = MultiStageMetaFeatureStacker(
        stage1_estimator_factories={"aux": lambda: GradientBoostingRegressor(random_state=0, n_estimators=100)},
        stage2_estimator=LinearRegression(),
        n_splits=5,
        random_state=0,
    )
    stacker.fit(X_train, y_train, {"aux": y_aux_train})
    stacker_mse = mean_squared_error(y_test, stacker.predict(X_test))

    improvement = 1.0 - stacker_mse / baseline_mse
    assert improvement > 0.4, f"expected >40% MSE reduction vs. a raw-feature-only linear baseline, got {improvement:.4f} (baseline={baseline_mse:.2f}, stacker={stacker_mse:.2f})"


def test_multi_stage_stacker_requires_all_auxiliary_targets():
    X, y_primary, y_aux = _make_dataset(n=200, seed=1)
    stacker = MultiStageMetaFeatureStacker(
        stage1_estimator_factories={"aux": lambda: LinearRegression(), "missing": lambda: LinearRegression()},
        stage2_estimator=LinearRegression(),
    )
    try:
        stacker.fit(X, y_primary, {"aux": y_aux})
        assert False, "expected ValueError for missing auxiliary target"
    except ValueError as exc:
        assert "missing" in str(exc)


def test_multi_stage_stacker_works_with_pandas_and_ndarray_input():
    import pandas as pd

    X, y_primary, y_aux = _make_dataset(n=300, seed=2)
    X_df = pd.DataFrame(X, columns=[f"col{j}" for j in range(X.shape[1])])

    stacker_arr = MultiStageMetaFeatureStacker(
        stage1_estimator_factories={"aux": lambda: LinearRegression()}, stage2_estimator=LinearRegression(), n_splits=3,
    )
    stacker_arr.fit(X, y_primary, {"aux": y_aux})
    pred_arr = stacker_arr.predict(X)

    stacker_df = MultiStageMetaFeatureStacker(
        stage1_estimator_factories={"aux": lambda: LinearRegression()}, stage2_estimator=LinearRegression(), n_splits=3,
    )
    stacker_df.fit(X_df, y_primary, {"aux": y_aux})
    pred_df = stacker_df.predict(X_df)

    assert pred_arr.shape == pred_df.shape == (300,)
    np.testing.assert_allclose(pred_arr, pred_df, rtol=1e-6)
