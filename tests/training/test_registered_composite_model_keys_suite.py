"""String-key dispatch of registered composite estimators through ``mlframe_models=[...]``.

``_trainer_configure.configure_training_params`` maps a handful of built-in string tags
("cb"/"lgb"/"xgb"/"hgb"/"mlp"/"ngb"/linear tags) to constructed model instances in ``models_params``.
``"gated_outlier"`` is additionally registered there (wrapping the suite's default LGBM regressor inside
``GatedOutlierEstimator``), regression-only, gated on lightgbm being importable. This test proves that
registration actually dispatches end-to-end through ``train_mlframe_models_suite`` (not just that the
underlying class works when constructed directly -- see ``tests/training/composite/test_biz_val_gated_outlier.py``
for that), and that adding it did not disturb dispatch of a pre-existing key ("lgb").
"""

from __future__ import annotations

import numpy as np
import pytest


def _zero_inflated_frame(seed=7, n=1500):
    import polars as pl

    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n).astype(np.float32)
    x1 = rng.normal(size=n).astype(np.float32)
    x2 = rng.normal(size=n).astype(np.float32)
    is_purchase = rng.random(n) < (0.1 + 0.8 * (x0 > 0))
    y = np.zeros(n, dtype=np.float32)
    y[is_purchase] = np.clip(100 + 20 * x1[is_purchase] + rng.normal(0, 5, is_purchase.sum()), 1, None).astype(np.float32)
    return pl.DataFrame({"f0": x0, "f1": x1, "f2": x2, "target": y})


def _train_suite(model_name, tmp_path, *, run_tag):
    pytest.importorskip("lightgbm")

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
        df=_zero_inflated_frame(),
        target_name="rk",
        model_name=run_tag,
        features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True),
        mlframe_models=[model_name],
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


def test_gated_outlier_string_key_dispatches_and_fits(tmp_path):
    from mlframe.training.composite import GatedOutlierEstimator

    models, _metadata = _train_suite("gated_outlier", tmp_path, run_tag="gated_outlier_run")

    trained = _trained_entries(models)
    gated_entries = [e for e in trained if isinstance(getattr(e, "model", None), GatedOutlierEstimator)]
    assert gated_entries, f"'gated_outlier' key never trained a GatedOutlierEstimator; entries={[type(getattr(e, 'model', None)).__name__ for e in trained]}"

    fitted = gated_entries[0].model
    assert hasattr(fitted, "regressor_") and hasattr(fitted, "classifier_"), "GatedOutlierEstimator did not fit its inner regressor/classifier"

    import pandas as pd

    cols = list(getattr(fitted, "feature_names_in_", ["f0", "f1", "f2"]))
    preds = np.asarray(fitted.predict(pd.DataFrame(np.zeros((5, len(cols)), dtype=np.float64), columns=cols)))
    assert preds.shape[0] == 5
    assert np.all(np.isfinite(preds))


def test_lgb_string_key_dispatch_unaffected_by_gated_outlier_registration(tmp_path):
    """Regression test: registering 'gated_outlier' in ``models_params`` must be purely additive."""
    from lightgbm import LGBMRegressor

    models, _metadata = _train_suite("lgb", tmp_path, run_tag="lgb_run")

    trained = _trained_entries(models)
    lgb_entries = [e for e in trained if isinstance(getattr(e, "model", None), LGBMRegressor)]
    assert lgb_entries, f"'lgb' key never trained an LGBMRegressor; entries={[type(getattr(e, 'model', None)).__name__ for e in trained]}"

    fitted = lgb_entries[0].model
    import pandas as pd

    cols = list(getattr(fitted, "feature_names_in_", ["f0", "f1", "f2"]))
    preds = np.asarray(fitted.predict(pd.DataFrame(np.zeros((5, len(cols)), dtype=np.float64), columns=cols)))
    assert preds.shape[0] == 5
