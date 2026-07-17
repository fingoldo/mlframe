"""biz_value test for ``training.composite.MultiStageMetaFeatureStacker``.

The win: when a primary target depends linearly on a hidden factor ``z`` that is only recoverable
NONLINEARLY from the raw features, a linear stage-2 model trained directly on the raw features can't decode
``z``. An auxiliary target that also depends on ``z`` (with less noise) lets a nonlinear stage-1 model
recover ``z`` well; feeding its OOF prediction in as a meta-feature lets the linear stage-2 model pick it up
trivially, beating the same linear model trained on raw features alone by a wide margin. This mirrors the
MoA competition's non-scored-target-as-meta-feature technique.

A second win (``use_predict_proba``): when the auxiliary target is itself a binary classification label
whose UNDERLYING probability (not just its thresholded sign) carries the signal the primary target needs,
feeding stage 1's hard ``predict()`` label in as a meta-feature collapses that probability to two levels --
throwing away exactly the magnitude information the primary target depends on. Feeding the continuous
``predict_proba`` positive-class probability in instead preserves it.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
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
    assert improvement > 0.4, (
        f"expected >40% MSE reduction vs. a raw-feature-only linear baseline, got {improvement:.4f} (baseline={baseline_mse:.2f}, stacker={stacker_mse:.2f})"
    )


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
        stage1_estimator_factories={"aux": lambda: LinearRegression()},
        stage2_estimator=LinearRegression(),
        n_splits=3,
    )
    stacker_arr.fit(X, y_primary, {"aux": y_aux})
    pred_arr = stacker_arr.predict(X)

    stacker_df = MultiStageMetaFeatureStacker(
        stage1_estimator_factories={"aux": lambda: LinearRegression()},
        stage2_estimator=LinearRegression(),
        n_splits=3,
    )
    stacker_df.fit(X_df, y_primary, {"aux": y_aux})
    pred_df = stacker_df.predict(X_df)

    assert pred_arr.shape == pred_df.shape == (300,)
    np.testing.assert_allclose(pred_arr, pred_df, rtol=1e-6)


def _make_proba_dataset(n: int, seed: int):
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
    # Binary auxiliary label: sign of z plus noise -- its predict_proba carries |z|'s magnitude, its
    # thresholded predict() only carries the sign.
    y_aux = (z + rng.normal(scale=0.4, size=n) > 0).astype(np.float64)
    y_primary = 5 * z + rng.normal(scale=0.5, size=n)
    return X, y_primary, y_aux


def test_biz_val_multi_stage_stacker_use_predict_proba_beats_hard_label_meta_feature_mse():
    X, y_primary, y_aux = _make_proba_dataset(n=3000, seed=3)
    X_train, X_test, y_train, y_test, y_aux_train, _ = train_test_split(X, y_primary, y_aux, test_size=0.3, random_state=3)

    def aux_factory():
        return GradientBoostingClassifier(random_state=3, n_estimators=100)

    hard_label_stacker = MultiStageMetaFeatureStacker(
        stage1_estimator_factories={"aux": aux_factory},
        stage2_estimator=LinearRegression(),
        n_splits=5,
        random_state=3,
    )
    hard_label_stacker.fit(X_train, y_train, {"aux": y_aux_train})
    hard_label_mse = mean_squared_error(y_test, hard_label_stacker.predict(X_test))

    proba_stacker = MultiStageMetaFeatureStacker(
        stage1_estimator_factories={"aux": aux_factory},
        stage2_estimator=LinearRegression(),
        n_splits=5,
        random_state=3,
        use_predict_proba={"aux": True},
    )
    proba_stacker.fit(X_train, y_train, {"aux": y_aux_train})
    proba_mse = mean_squared_error(y_test, proba_stacker.predict(X_test))

    improvement = 1.0 - proba_mse / hard_label_mse
    assert improvement > 0.09, (
        f"expected >9% MSE reduction from use_predict_proba over the hard-label meta-feature, got "
        f"{improvement:.4f} (hard_label={hard_label_mse:.2f}, proba={proba_mse:.2f})"
    )


def test_multi_stage_stacker_use_predict_proba_default_off_matches_prior_behavior():
    """Regression guard: leaving ``use_predict_proba`` unset must be bit-identical to before it existed."""
    X, y_primary, y_aux = _make_dataset(n=400, seed=4)

    baseline = MultiStageMetaFeatureStacker(
        stage1_estimator_factories={"aux": lambda: GradientBoostingRegressor(random_state=0, n_estimators=20)},
        stage2_estimator=LinearRegression(),
        n_splits=3,
        random_state=0,
    )
    baseline.fit(X, y_primary, {"aux": y_aux})
    baseline_pred = baseline.predict(X)

    explicit_off = MultiStageMetaFeatureStacker(
        stage1_estimator_factories={"aux": lambda: GradientBoostingRegressor(random_state=0, n_estimators=20)},
        stage2_estimator=LinearRegression(),
        n_splits=3,
        random_state=0,
        use_predict_proba=None,
    )
    explicit_off.fit(X, y_primary, {"aux": y_aux})
    explicit_off_pred = explicit_off.predict(X)

    np.testing.assert_array_equal(baseline_pred, explicit_off_pred)
