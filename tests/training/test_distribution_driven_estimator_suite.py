"""E3: distribution-driven composite estimator is actually trained in the suite.

When the target-distribution analyzer flags a heavy-tail regression target and
``behavior_config.distribution_driven_estimator=True``, the suite appends the recommended
``TailCompositeEstimator`` to the model list (auto base_column + light GBM base) so it trains/evaluates
alongside the requested models, and stamps the decision into ``metadata["distribution_driven_estimator"]``.
"""

from __future__ import annotations

import numpy as np
import pytest


def _heavy_tail_frame(seed=7, n=1600):
    import polars as pl

    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n).astype(np.float32)
    x1 = rng.normal(size=n).astype(np.float32)
    # Heavy-tailed (Student-t, df=2.5 -> high excess kurtosis) noise so the analyzer flags heavy_tail;
    # x0 carries the body signal so the auto-picked base_column is meaningful.
    # df=2 Student-t scaled x3 -> excess kurtosis ~60 (>> analyzer's 5.0 heavy_tail threshold); x0 carries body signal.
    noise = 3.0 * rng.standard_t(2.0, size=n)
    y = (2.0 * x0 - x1 + noise).astype(np.float32)
    return pl.DataFrame({"f0": x0, "f1": x1, "target": y})


def test_e2e_distribution_driven_estimator_trains_and_roundtrips(tmp_path):
    pytest.importorskip("lightgbm")
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.composite.extremes import TailCompositeEstimator
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

    models, metadata = train_mlframe_models_suite(
        df=_heavy_tail_frame(),
        target_name="dd",
        model_name="dd_run",
        features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True),
        mlframe_models=["xgb"],
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        pipeline_config=PreprocessingBackendConfig(prefer_polarsds=False, categorical_encoding=None, scaler_name=None, imputer_strategy=None),
        split_config=TrainingSplitConfig(test_size=0.25, val_size=0.1),
        behavior_config=TrainingBehaviorConfig(prefer_gpu_configs=False, distribution_driven_estimator=True),
        hyperparams_config={"iterations": 40, "xgb_kwargs": {"device": "cpu"}},
        baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
        dummy_baselines_config=DummyBaselinesConfig(enabled=False),
        reporting_config=ReportingConfig(honest_estimator_diagnostics=False),
        enable_target_distribution_analyzer=True,
        output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"),
        verbose=0,
    )

    assert "distribution_driven_estimator" in metadata, (
        "E3 did not fire; pathologies="
        + str((metadata.get("target_distribution_report") or {}).get("pathologies"))
    )
    stamp = metadata["distribution_driven_estimator"]
    assert "TailCompositeEstimator" in stamp
    assert stamp["TailCompositeEstimator"]["base_column"] in ("f0", "f1")

    trained = [e for per_target in models.values() for entries in per_target.values() for e in entries]
    trained = [e[0] if isinstance(e, tuple) and e else e for e in trained]
    tail_entries = [e for e in trained if isinstance(getattr(e, "model", None), TailCompositeEstimator)]
    assert tail_entries, f"TailComposite never trained; entries={[getattr(e, 'model_name', e) for e in trained]}"

    fitted = tail_entries[0].model
    import pandas as pd
    import pickle

    cols = list(getattr(fitted, "feature_names_in_", ["f0", "f1"]))
    X = pd.DataFrame(np.zeros((5, len(cols)), dtype=np.float64), columns=cols)
    preds = np.asarray(fitted.predict(X)).reshape(-1)
    assert preds.shape[0] == 5

    reloaded = pickle.loads(pickle.dumps(fitted))
    preds2 = np.asarray(reloaded.predict(X)).reshape(-1)
    assert np.allclose(preds, preds2, equal_nan=True)
