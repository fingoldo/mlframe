"""Business value of the suite's automatic HurdleRegressor on a zero-inflated target, measured on the test split.

A run with one zero-inflated amount (``charge``: ~70% exact zeros, lognormal magnitude on the event rows), one ordinary
regression target and one binary target. The suite must train the hurdle for ``charge`` only.

Against the plain LightGBM regressor the user asked for, measured on the test split over seeds 0/1/2: the hurdle ranks
rows better every time (Spearman +0.018 / +0.064 / +0.068), never predicts a negative amount while the plain model did
on two seeds (minimum -177 and -12, which also leaves RMSLE undefined), and is level on RMSE (within 1.5% either way).
MAE ranged from level to 20% better and is not asserted. The claim is therefore "a better-ranked, valid candidate at no
RMSE cost", not "wins every metric".
"""

from __future__ import annotations

import numpy as np
import pytest


def _frame(n=6000, seed=0):
    """Three targets over four features: zero-inflated ``charge``, Gaussian ``price``, binary ``churn``."""
    import polars as pl

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    happens = rng.random(n) < 1 / (1 + np.exp(-(1.6 * X[:, 0] - 1.2 * X[:, 1] - 0.9)))
    charge = np.where(happens, np.exp(3.5 + 0.9 * X[:, 2] - 0.6 * X[:, 3] + rng.normal(0, 1.2, n)), 0.0)
    price = 2 * X[:, 0] + X[:, 2] + rng.normal(size=n)
    churn = (X[:, 1] + rng.normal(size=n) > 0).astype(np.int8)
    return pl.DataFrame({**{f"f{i}": X[:, i] for i in range(4)}, "charge": charge, "price": price, "churn": churn})


@pytest.fixture(scope="module")
def suite_run(tmp_path_factory):
    """One suite call shared by the assertions below."""
    pytest.importorskip("lightgbm")
    from mlframe.training._preprocessing_configs import TrainingSplitConfig
    from mlframe.training.configs import (
        BaselineDiagnosticsConfig,
        DummyBaselinesConfig,
        OutputConfig,
        PreprocessingBackendConfig,
        ReportingConfig,
        TrainingBehaviorConfig,
    )
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

    return train_mlframe_models_suite(
        df=_frame(),
        target_name="hurdle_bv",
        model_name="hurdle_bv_run",
        features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(regression_targets=["charge", "price"], classification_targets=["churn"]),
        mlframe_models=["lgb"],
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        pipeline_config=PreprocessingBackendConfig(prefer_polarsds=False, categorical_encoding=None, scaler_name=None, imputer_strategy=None),
        split_config=TrainingSplitConfig(test_size=0.25, val_size=0.1),
        behavior_config=TrainingBehaviorConfig(prefer_gpu_configs=False),
        hyperparams_config={"iterations": 60},
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(honest_estimator_diagnostics=False),
        output_config=OutputConfig(data_dir=str(tmp_path_factory.mktemp("hurdle_bv")), models_dir="models"),
        verbose=0,
    )


def _by_class(entries):
    """``{estimator class name: entry}`` for one target's trained entries."""
    return {type(getattr(e, "model", e)).__name__: e for e in entries}


def test_hurdle_trains_only_on_the_zero_inflated_target(suite_run):
    """Not on the ordinary regression target, not on the classification target."""
    models, metadata = suite_run
    assert metadata["hurdle_for_zero_inflated"]["hurdle"]["targets"] == ["charge"]
    assert "HurdleRegressor" in _by_class(models["regression"]["charge"])
    assert "HurdleRegressor" not in _by_class(models["regression"]["price"])
    assert all("HurdleRegressor" not in _by_class(v) for v in models["binary_classification"].values())


def test_hurdle_ranks_better_stays_valid_and_matches_rmse(suite_run):
    """The value claim, on rows neither model saw."""
    from scipy.stats import spearmanr

    models, _ = suite_run
    by = _by_class(models["regression"]["charge"])
    hurdle, plain = by["HurdleRegressor"], by["LGBMRegressorWithDatasetReuse"]
    y = np.asarray(hurdle.test_target, dtype=np.float64).reshape(-1)
    p_h = np.asarray(hurdle.test_preds, dtype=np.float64).reshape(-1)
    p_l = np.asarray(plain.test_preds, dtype=np.float64).reshape(-1)
    rmse_h, rmse_l = np.sqrt(np.mean((y - p_h) ** 2)), np.sqrt(np.mean((y - p_l) ** 2))
    rho_h, rho_l = spearmanr(y, p_h)[0], spearmanr(y, p_l)[0]
    assert rho_h > rho_l, f"hurdle Spearman {rho_h:.3f} vs plain {rho_l:.3f}"
    assert rmse_h <= 1.05 * rmse_l, f"hurdle RMSE {rmse_h:.2f} vs plain {rmse_l:.2f}"
    assert p_h.min() >= 0.0, "a hurdle over a non-negative amount never predicts below its floor"


def test_suite_trained_hurdle_survives_pickling(suite_run):
    """The persisted artefact predicts exactly what the in-memory model does."""
    import pickle  # nosec B403 -- round-trip of a locally trained, trusted object

    models, _ = suite_run
    fitted = _by_class(models["regression"]["charge"])["HurdleRegressor"].model
    # Width from the fitted model: the suite adds engineered columns to the four raw features.
    X = np.random.default_rng(5).normal(size=(20, fitted.n_features_in_)).astype(np.float32)
    reloaded = pickle.loads(pickle.dumps(fitted))  # nosec B301 -- same trusted object
    np.testing.assert_allclose(np.asarray(fitted.predict(X)), np.asarray(reloaded.predict(X)))
