"""Registry dispatch for the ``"bagging"`` / ``"composite_classification"`` ``mlframe_models`` string keys.

Batch E of the isolated-129 wiring effort: ``BaggedCompositeEstimator`` (bootstrap-bagged variance reduction
over a plain GBDT regressor) and ``CompositeClassificationEstimator`` (base-margin/init-score residual
composite for classification, auto-fitting its own ``LogisticRegression`` base margin when no
``base_margin_column`` is supplied) are both genuinely dataset-agnostic -- neither requires a caller-supplied
``base_column``/``segment_keys``/``regime_fn``/etc, unlike the other composite/ensemble estimators audited
and left opt-in (``SegmentedModelFactory``, ``GatedRegressionMixture``, ``RegimeSplitEnsemble``,
``CountWeightedBlendEnsemble``, the ``base_column``-family composites). Registered in
``configure_training_params`` (``_trainer_configure.py``) + ``MODEL_STRATEGIES``
(``training/strategies/__init__.py``) alongside ``gated_outlier``, explicit-allowlist-only (no auto-detection
heuristic -- see the inline rationale at each registration site).
"""

from __future__ import annotations

import numpy as np


def _regression_frame(seed=11, n=1200):
    import polars as pl

    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n).astype(np.float32)
    x1 = rng.normal(size=n).astype(np.float32)
    x2 = rng.normal(size=n).astype(np.float32)
    y = (2 * x0 - x1 + 0.5 * x2 + 0.3 * rng.normal(size=n)).astype(np.float32)
    return pl.DataFrame({"f0": x0, "f1": x1, "f2": x2, "target": y})


def _classification_frame(seed=13, n=1200):
    import polars as pl

    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n).astype(np.float32)
    x1 = rng.normal(size=n).astype(np.float32)
    x2 = rng.normal(size=n).astype(np.float32)
    logit = 1.5 * x0 - x1 + 0.5 * x2
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(np.int32)
    return pl.DataFrame({"f0": x0, "f1": x1, "f2": x2, "target": y})


def _run_suite(tmp_path, df, mlframe_models, regression):
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.configs import (
        PreprocessingBackendConfig,
        OutputConfig,
        TrainingBehaviorConfig,
        BaselineDiagnosticsConfig,
        DummyBaselinesConfig,
        ReportingConfig,
    )
    from mlframe.training._preprocessing_configs import TrainingSplitConfig
    from .shared import SimpleFeaturesAndTargetsExtractor

    return train_mlframe_models_suite(
        df=df,
        target_name="ce",
        model_name="ce_run",
        features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(target_column="target", regression=regression),
        mlframe_models=mlframe_models,
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        pipeline_config=PreprocessingBackendConfig(prefer_polarsds=False, categorical_encoding=None, scaler_name=None, imputer_strategy=None),
        split_config=TrainingSplitConfig(test_size=0.25, val_size=0.1),
        behavior_config=TrainingBehaviorConfig(prefer_gpu_configs=False),
        hyperparams_config={"iterations": 40},
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(honest_estimator_diagnostics=False),
        enable_target_distribution_analyzer=False,
        output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"),
        verbose=0,
    )


def _trained_entries(models):
    trained = [e for per_target in models.values() for entries in per_target.values() for e in entries]
    return [e[0] if isinstance(e, tuple) and e else e for e in trained]


# ---------------------------------------------------------------------------
# Unit tests: the estimators directly, no suite involved.
# ---------------------------------------------------------------------------


def test_unit_bagged_composite_estimator_fit_predict():
    from lightgbm import LGBMRegressor
    from mlframe.training.composite.bagging import BaggedCompositeEstimator

    df = _regression_frame().to_pandas()
    X, y = df[["f0", "f1", "f2"]], df["target"].to_numpy()

    est = BaggedCompositeEstimator(base_estimator=LGBMRegressor(n_estimators=50, num_leaves=15, verbose=-1, random_state=0), n_estimators=5)
    est.fit(X.iloc[:900], y[:900])
    preds = np.asarray(est.predict(X.iloc[900:]))
    assert preds.shape[0] == 300
    assert np.isfinite(preds).all()


def test_unit_composite_classification_estimator_fit_predict():
    from lightgbm import LGBMClassifier
    from mlframe.training.composite.classification import CompositeClassificationEstimator

    df = _classification_frame().to_pandas()
    X, y = df[["f0", "f1", "f2"]], df["target"].to_numpy()

    est = CompositeClassificationEstimator(base_estimator=LGBMClassifier(n_estimators=50, num_leaves=15, verbose=-1, random_state=0))
    est.fit(X.iloc[:900], y[:900])
    proba = np.asarray(est.predict_proba(X.iloc[900:]))
    assert proba.shape == (300, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    preds = np.asarray(est.predict(X.iloc[900:]))
    assert set(np.unique(preds)) <= {0, 1}


# ---------------------------------------------------------------------------
# Live end-to-end train_mlframe_models_suite tests.
# ---------------------------------------------------------------------------


def test_e2e_bagging_key_trains_and_predicts(tmp_path):
    from mlframe.training.composite.bagging import BaggedCompositeEstimator

    models, metadata = _run_suite(tmp_path, _regression_frame(), ["bagging"], regression=True)

    trained = _trained_entries(models)
    entries = [e for e in trained if isinstance(getattr(e, "model", None), BaggedCompositeEstimator)]
    assert entries, f"bagging key never trained; entries={[getattr(e, 'model_name', e) for e in trained]}"

    fitted = entries[0].model
    assert hasattr(fitted, "estimators_"), "BaggedCompositeEstimator did not fit its bootstrap members"

    import pandas as pd

    # The fitted feature set is NOT necessarily the raw ["f0","f1","f2"] -- default-ON preprocessing
    # extensions (row_wise_summary_stats_enabled / row_wise_extreme_columns_enabled) may have appended
    # engineered columns before the model saw the frame. Predict with whatever the fitted estimator
    # actually recorded at fit time, not a hardcoded raw column list.
    cols = list(getattr(fitted, "feature_names_in_", None) or range(getattr(fitted, "n_features_in_")))
    preds = np.asarray(fitted.predict(pd.DataFrame(np.zeros((5, len(cols)), dtype=np.float64), columns=cols)))
    assert preds.shape[0] == 5
    assert np.isfinite(preds).all()


def test_e2e_composite_classification_key_trains_and_predicts(tmp_path):
    from mlframe.training.composite.classification import CompositeClassificationEstimator

    models, metadata = _run_suite(tmp_path, _classification_frame(), ["composite_classification"], regression=False)

    trained = _trained_entries(models)
    entries = [e for e in trained if isinstance(getattr(e, "model", None), CompositeClassificationEstimator)]
    assert entries, f"composite_classification key never trained; entries={[getattr(e, 'model_name', e) for e in trained]}"

    fitted = entries[0].model
    assert hasattr(fitted, "estimator_"), "CompositeClassificationEstimator did not fit its inner booster"

    import pandas as pd

    # The fitted feature set is NOT necessarily the raw ["f0","f1","f2"] -- default-ON preprocessing
    # extensions (row_wise_summary_stats_enabled / row_wise_extreme_columns_enabled) may have appended
    # engineered columns before the model saw the frame. Predict with whatever the fitted estimator
    # actually recorded at fit time, not a hardcoded raw column list.
    cols = list(getattr(fitted, "feature_names_in_", None) or range(getattr(fitted, "n_features_in_")))
    preds = np.asarray(fitted.predict(pd.DataFrame(np.zeros((5, len(cols)), dtype=np.float64), columns=cols)))
    assert preds.shape[0] == 5


def test_existing_keys_unperturbed_by_new_registry_entries(tmp_path):
    """Regression guard: adding bagging/composite_classification must not change cb/lgb dispatch, and neither
    new key must be silently auto-included under the default allowlist (no auto-detection heuristic wired for
    either, unlike gated_outlier's point-mass gate)."""
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from mlframe.training.composite.bagging import BaggedCompositeEstimator
    from mlframe.training.composite.classification import CompositeClassificationEstimator

    models, metadata = _run_suite(tmp_path, _regression_frame(), ["cb", "lgb"], regression=True)

    trained = _trained_entries(models)
    fitted_models = [getattr(e, "model", None) for e in trained]
    assert any(isinstance(m, CatBoostRegressor) for m in fitted_models), f"cb key never trained; models={[type(m).__name__ for m in fitted_models]}"
    assert any(isinstance(m, LGBMRegressor) for m in fitted_models), f"lgb key never trained; models={[type(m).__name__ for m in fitted_models]}"
    assert not any(isinstance(m, BaggedCompositeEstimator) for m in fitted_models)
    assert not any(isinstance(m, CompositeClassificationEstimator) for m in fitted_models)


def test_default_allowlist_does_not_auto_include_new_keys(tmp_path):
    from mlframe.training.composite.bagging import BaggedCompositeEstimator

    models, metadata = _run_suite(tmp_path, _regression_frame(), None, regression=True)

    trained = _trained_entries(models)
    fitted_models = [getattr(e, "model", None) for e in trained]
    assert not any(isinstance(m, BaggedCompositeEstimator) for m in fitted_models)
