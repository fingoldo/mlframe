"""Registry dispatch for the ``"gated_outlier"`` composite-estimator string key.

``GatedOutlierEstimator`` (classifier gate + regression blend for zero-inflated/point-mass targets) is
registered in ``configure_training_params`` (``_trainer_configure.py``) alongside the built-in string tags
(cb/lgb/xgb/hgb/mlp/ngb/linear) so callers can request it via ``mlframe_models=["gated_outlier"]`` instead of
only reaching it through the generic estimator-instance path. Purely additive: existing string keys must keep
training identically after the registry edit (see ``test_existing_keys_unperturbed_by_new_registry_entry``).
"""

from __future__ import annotations

import numpy as np


def _zero_inflated_frame(seed=7, n=1600):
    import polars as pl

    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n).astype(np.float32)
    x1 = rng.normal(size=n).astype(np.float32)
    x2 = rng.normal(size=n).astype(np.float32)
    continuous = (2 * x0 - x1 + 0.5 * x2 + 0.3 * rng.normal(size=n)).astype(np.float32)
    is_point_mass = rng.random(n) < 0.35
    y = np.where(is_point_mass, 0.0, continuous).astype(np.float32)
    return pl.DataFrame({"f0": x0, "f1": x1, "f2": x2, "target": y})


def _run_suite(tmp_path, mlframe_models):
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
        target_name="ce",
        model_name="ce_run",
        features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True),
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


def test_e2e_gated_outlier_key_trains_and_predicts(tmp_path):
    from mlframe.training.composite.gated_outlier import GatedOutlierEstimator

    models, metadata = _run_suite(tmp_path, ["gated_outlier"])

    trained = _trained_entries(models)
    gated_entries = [e for e in trained if isinstance(getattr(e, "model", None), GatedOutlierEstimator)]
    assert gated_entries, f"gated_outlier key never trained; entries={[getattr(e, 'model_name', e) for e in trained]}"

    fitted = gated_entries[0].model
    assert hasattr(fitted, "classifier_") and hasattr(fitted, "regressor_"), "GatedOutlierEstimator did not fit its gate/regressor clones"

    import pandas as pd

    cols = ["f0", "f1", "f2"]
    preds = np.asarray(fitted.predict(pd.DataFrame(np.zeros((5, len(cols)), dtype=np.float64), columns=cols)))
    assert preds.shape[0] == 5
    assert np.isfinite(preds).all()


def test_existing_keys_unperturbed_by_new_registry_entry(tmp_path):
    """Regression guard: adding the gated_outlier registry entry must not change cb/lgb dispatch.

    Also covers the 2026-07-12 point-mass auto-detection flip: this data IS zero-inflated (35% point mass),
    but ``mlframe_models=["cb", "lgb"]`` is an EXPLICIT allowlist, so the auto-detection must stay off and
    gated_outlier must not be silently appended -- explicit ``mlframe_models`` still means exactly that list.
    """
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor

    models, metadata = _run_suite(tmp_path, ["cb", "lgb"])

    trained = _trained_entries(models)
    fitted_models = [getattr(e, "model", None) for e in trained]
    assert any(isinstance(m, CatBoostRegressor) for m in fitted_models), f"cb key never trained; models={[type(m).__name__ for m in fitted_models]}"
    assert any(isinstance(m, LGBMRegressor) for m in fitted_models), f"lgb key never trained; models={[type(m).__name__ for m in fitted_models]}"
    # No accidental cross-registration: requesting cb/lgb must not also train gated_outlier.
    from mlframe.training.composite.gated_outlier import GatedOutlierEstimator

    assert not any(isinstance(m, GatedOutlierEstimator) for m in fitted_models)


def test_e2e_default_allowlist_auto_includes_gated_outlier_on_point_mass_data(tmp_path):
    """2026-07-12 default flip: when ``mlframe_models`` is left at its top-level ``None`` default (resolved
    to the fixed ["cb","lgb","xgb","mlp","linear"] allowlist) AND the train target shows a genuine point
    mass (>=5% of rows share one value), ``gated_outlier`` is auto-added as an extra candidate model without
    requiring the caller to name it explicitly."""
    from mlframe.training.composite.gated_outlier import GatedOutlierEstimator

    models, metadata = _run_suite(tmp_path, None)

    trained = _trained_entries(models)
    fitted_models = [getattr(e, "model", None) for e in trained]
    assert any(isinstance(m, GatedOutlierEstimator) for m in fitted_models), (
        f"gated_outlier not auto-included for point-mass data under the default allowlist; "
        f"models={[type(m).__name__ for m in fitted_models]}"
    )
