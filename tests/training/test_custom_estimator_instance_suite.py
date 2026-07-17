"""Generic estimator-instance support in ``train_mlframe_models_suite`` (E3 layer 1).

Beyond the built-in string tags (cb/lgb/xgb/hgb/mlp/ngb/linear), ``mlframe_models`` may carry a
sklearn-compatible estimator INSTANCE (or a ``(name, estimator)`` tuple). ``get_strategy`` dispatches
instances MRO-based (composite/unknown -> LinearModelStrategy), ``configure_training_params`` builds a
minimal ``dict(model=est)`` params entry, and the per-target loop keys ``models_params`` / ``strategy_by_model``
by the entry object / ``id()``. This is the foundation E3 (distribution-driven estimator) trains on.
"""

from __future__ import annotations

import numpy as np


def _reg_frame(seed=11, n=1400):
    import polars as pl

    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n).astype(np.float32)
    x1 = rng.normal(size=n).astype(np.float32)
    x2 = rng.normal(size=n).astype(np.float32)
    y = (2 * x0 - x1 + 0.5 * x2 + 0.3 * rng.normal(size=n)).astype(np.float32)
    return pl.DataFrame({"f0": x0, "f1": x1, "f2": x2, "target": y})


def test_e2e_custom_sklearn_instance_trains_and_roundtrips(tmp_path):
    from sklearn.linear_model import Ridge

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

    custom = Ridge(alpha=0.7)

    models, _metadata = train_mlframe_models_suite(
        df=_reg_frame(),
        target_name="ce",
        model_name="ce_run",
        features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True),
        mlframe_models=[custom],
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

    # A trained entry whose .model is our Ridge instance must exist.
    trained = [e for per_target in models.values() for entries in per_target.values() for e in entries]
    trained = [e[0] if isinstance(e, tuple) and e else e for e in trained]
    ridge_entries = [e for e in trained if isinstance(getattr(e, "model", None), Ridge)]
    assert ridge_entries, f"custom Ridge instance never trained; entries={[getattr(e, 'model_name', e) for e in trained]}"

    fitted = ridge_entries[0].model
    import pandas as pd

    cols = list(getattr(fitted, "feature_names_in_", ["f0", "f1", "f2"]))
    preds = np.asarray(fitted.predict(pd.DataFrame(np.zeros((5, len(cols)), dtype=np.float64), columns=cols)))
    assert preds.shape[0] == 5

    # Round-trip the fitted estimator (suite persists models; verify the estimator itself pickles/loads).
    import pickle

    reloaded = pickle.loads(pickle.dumps(fitted))
    preds2 = np.asarray(reloaded.predict(pd.DataFrame(np.zeros((5, len(cols)), dtype=np.float64), columns=cols)))
    assert np.allclose(preds, preds2)
