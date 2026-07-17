"""Regression: MULTI_TARGET_REGRESSION must build a REGRESSOR end-to-end.

``select_target`` keyed the regressor-vs-classifier dispatch on
``use_regression = (target_type == TargetTypes.REGRESSION)``, which is False
for MULTI_TARGET_REGRESSION / QUANTILE_REGRESSION. ``configure_training_params``
therefore built a CatBoostClassifier for an MTR target; the MTR ``MultiRMSE``
loss is layered on later only when the model is a *Regressor*, so the classifier
kept a classification loss and CatBoost rejected the 2-D continuous target with
"Target Labels for MultiLogloss must be 0 or 1" at fit time.

The strategy-level helpers (supports_native_multi_target, MultiRMSE kwargs,
MultiOutputRegressor wrap) were already tested, but no test exercised the
suite-side model *construction* dispatch -- which is exactly where the bug
lived. This runs cb + MULTI_TARGET_REGRESSION through the full suite: pre-fix
it raised at fit, post-fix it trains and returns the MTR target bucket.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training import OutputConfig
from mlframe.training.configs import TargetTypes
from mlframe.training.core import train_mlframe_models_suite

from .shared import SimpleFeaturesAndTargetsExtractor, get_cpu_config, skip_if_dependency_missing


def test_cb_multi_target_regression_trains_without_classification_loss(temp_data_dir, common_init_params, fast_iterations):
    """CatBoost dispatches to a regressor (not MultiLogloss classifier) for an MTR target; pre-fix this crashed with CatBoostError."""
    skip_if_dependency_missing("cb")

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(rng.normal(size=(n, 4)), columns=["f0", "f1", "f2", "f3"])
    t0 = df["f0"].to_numpy() * 2.0 + 0.1 * rng.normal(size=n)
    t1 = df["f1"].to_numpy() * -1.0 + 0.1 * rng.normal(size=n)
    # MTR target: object column of per-row [t0, t1] vectors -> stacked to (N, 2).
    df["target"] = list(np.column_stack([t0, t1]).astype(np.float32))

    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", target_type=TargetTypes.MULTI_TARGET_REGRESSION)
    config_override = get_cpu_config("cb", fast_iterations)

    # Pre-fix this raised CatBoostError("Target Labels for MultiLogloss must be
    # 0 or 1") because cb was built as a CLASSIFIER for the MTR target.
    models, _metadata = train_mlframe_models_suite(
        df=df,
        target_name="mtr_target",
        model_name="cb_mtr",
        features_and_targets_extractor=fte,
        mlframe_models=["cb"],
        hyperparams_config=config_override,
        reporting_config=common_init_params,
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=temp_data_dir, models_dir="models"),
        verbose=0,
    )

    # Reaching here at all means the classification-loss crash is gone. The MTR
    # bucket must be present and populated (the model trained as a regressor).
    assert TargetTypes.MULTI_TARGET_REGRESSION in models, f"MTR bucket missing; got target types {list(models.keys())}"
    mtr_entries = models[TargetTypes.MULTI_TARGET_REGRESSION]
    assert mtr_entries, "MTR target bucket is empty -- the cb model did not train"

    # No fitted estimator anywhere in the MTR entries may be a CLASSIFIER.
    def _walk_estimators(obj, depth=0):
        """Recursively visits nested MTR model entries to assert none of the fitted estimators is a classifier."""
        if depth > 6 or obj is None:
            return
        cls = type(obj).__name__
        if "Classifier" in cls or "Classification" in cls:
            raise AssertionError(f"MTR target produced a classifier estimator: {cls}")
        for attr in ("estimator", "estimator_", "model", "regressor", "base_estimator"):
            inner = getattr(obj, attr, None)
            if inner is not None and inner is not obj:
                _walk_estimators(inner, depth + 1)

    for entry in mtr_entries.values() if isinstance(mtr_entries, dict) else []:
        for v in entry.values() if isinstance(entry, dict) else [entry]:
            model_obj = v.get("model") if isinstance(v, dict) else v
            _walk_estimators(model_obj)
