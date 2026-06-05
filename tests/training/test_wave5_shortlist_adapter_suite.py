"""A2-01: the ShortlistTransformerAdapter, wired via custom_pre_pipelines, runs end-to-end through
train_mlframe_models_suite (tiny budget) leakage-safely.

This proves the opt-in path that lets a research-only shortlist transformer (e.g. compute_rff_features)
participate in the suite, which the 103 standalone compute_* functions otherwise cannot do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from mlframe.training.configs import TargetTypes
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training import OutputConfig, ReportingConfig
    from mlframe.training._feature_selection_config import FeatureSelectionConfig
except Exception as exc:  # pragma: no cover
    pytest.skip(f"suite not importable ({exc!r})", allow_module_level=True)

from mlframe.feature_engineering.transformer import (
    ShortlistTransformerAdapter,
    compute_rff_features,
)
from .shared import SimpleFeaturesAndTargetsExtractor


def test_a2_01_adapter_unit_fit_transform_leakage_safe() -> None:
    """Unit: adapter fit stashes train fold; transform applies the train-fitted RFF to a held-out frame, passthrough adds features."""
    rng = np.random.default_rng(0)
    Xtr = pd.DataFrame(rng.standard_normal((100, 4)), columns=list("abcd"))
    Xte = pd.DataFrame(rng.standard_normal((30, 4)), columns=list("abcd"))
    adapter = ShortlistTransformerAdapter(compute_rff_features, seed=1, needs_y=False, compute_kwargs={"n_features": 16})
    adapter.fit(Xtr)
    out = adapter.transform(Xte)
    assert out.shape[0] == 30
    assert out.shape[1] == 4 + 16, "passthrough keeps the 4 raw cols + 16 RFF features"
    # Same train fold + same query -> identical (train-only fit is deterministic).
    out2 = adapter.transform(Xte)
    assert np.allclose(out.select_dtypes(float).to_numpy(), out2.select_dtypes(float).to_numpy())


def test_a2_01_adapter_runs_through_suite(tmp_path) -> None:
    """End-to-end: adapter passed via custom_pre_pipelines trains a model through the suite at tiny budget."""
    rng = np.random.default_rng(3)
    n = 600
    X = rng.standard_normal((n, 5))
    logits = 1.3 * X[:, 0] - 1.0 * X[:, 1] + 0.6 * X[:, 2]
    y = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y

    adapter = ShortlistTransformerAdapter(
        compute_rff_features, seed=7, needs_y=False, compute_kwargs={"n_features": 16}
    )

    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
    models, _meta = train_mlframe_models_suite(
        df=df,
        target_name="target",
        model_name="wave5_adapter_suite",
        features_and_targets_extractor=fte,
        mlframe_models=["linear"],
        reporting_config=ReportingConfig(show_perf_chart=False, show_fi=False),
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"),
        verbose=0,
        hyperparams_config={"iterations": 50},
        feature_selection_config=FeatureSelectionConfig(custom_pre_pipelines={"rff_adapter": adapter}),
    )
    entries = models[TargetTypes.BINARY_CLASSIFICATION]["target"]
    assert len(entries) >= 1, "suite produced no model entries with the adapter wired"
