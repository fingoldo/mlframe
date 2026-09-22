"""Two gaps the disk-serving entry point had against its in-memory twin.

PRD-11: every per-model load/predict error was logged and skipped, and when ALL of them failed the function returned
an empty, success-shaped result (`ensemble_predictions=None`) instead of raising as `predict_from_models` does.

PRD-07: the GBM-safe column rename training applies right after the suite pipeline transform was missing on the disk
path, so an engineered name like `mul(log(f2),sin(f3))` made CatBoost raise, the handler swallowed it, and the model
silently left the ensemble.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import pytest

from mlframe.training import OutputConfig
from mlframe.training.core import predict_mlframe_models_suite, train_mlframe_models_suite

from tests.training.shared import SimpleFeaturesAndTargetsExtractor


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    x = rng.randn(n, 3)
    df = pd.DataFrame(x, columns=[f"f_{i}" for i in range(3)])
    # A categorical column makes the suite persist a pipeline (an encoder), which is where the rename applies.
    df["cat"] = rng.choice(["a", "b", "c"], size=n)
    df["target"] = x @ np.array([1.5, -1.0, 0.5]) + rng.normal(0, 0.1, n)
    return df


@pytest.fixture(scope="module")
def trained_suite(tmp_path_factory):
    data_dir = str(tmp_path_factory.mktemp("suite"))
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)
    train_mlframe_models_suite(
        df=_frame(300, 0), target_name="t", model_name="m", features_and_targets_extractor=fte,
        mlframe_models=["linear"], use_ordinary_models=True, use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=data_dir, models_dir="models"), verbose=0,
    )
    models_path = f"{data_dir}/models/t/m"
    assert glob.glob(os.path.join(models_path, "**", "*.dump"), recursive=True), "the fixture saved no model"
    return models_path, fte


def test_every_model_failing_raises_instead_of_returning_an_empty_success(trained_suite, monkeypatch):
    models_path, fte = trained_suite
    import mlframe.training.core._predict_main_suite as pms

    def _broken(*a, **k):
        raise OSError("simulated version-drift unpickle failure")

    monkeypatch.setattr(pms, "load_mlframe_model", _broken)
    with pytest.raises(RuntimeError, match="failed to load or predict"):
        predict_mlframe_models_suite(df=_frame(50, 1), models_path=models_path, features_and_targets_extractor=fte, verbose=0)


def test_a_healthy_bundle_still_predicts(trained_suite):
    models_path, fte = trained_suite
    out = predict_mlframe_models_suite(df=_frame(50, 1), models_path=models_path, features_and_targets_extractor=fte, verbose=0)
    assert out["predictions"], "a working bundle must produce predictions"


def test_the_disk_path_sanitises_names_after_the_pipeline(trained_suite, monkeypatch):
    """Spied at the sanitiser: the disk path must pass the post-pipeline frame through it, as training and the
    in-memory path do. Before the fix it was never called on this path at all."""
    models_path, fte = trained_suite
    import mlframe.training._feature_name_sanitize as fns

    calls = []
    real = fns.sanitize_frame_columns

    def spy(df):
        calls.append(tuple(str(c) for c in getattr(df, "columns", [])))
        return real(df)

    monkeypatch.setattr(fns, "sanitize_frame_columns", spy)
    predict_mlframe_models_suite(df=_frame(50, 1), models_path=models_path, features_and_targets_extractor=fte, verbose=0)
    assert calls, "the disk path never sanitised the post-pipeline frame"
