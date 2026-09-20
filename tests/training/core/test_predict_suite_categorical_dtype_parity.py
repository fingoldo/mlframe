"""The disk-served suite must coerce categorical dtypes exactly as the in-memory path does.

`_coerce_cat_dtype_for_lgb_xgb` (with the persisted `enum_domains`) was called only by `predict_from_models`, so a
suite served from disk handed LightGBM a categorical column as float64/object instead of pandas `category`: its
predict-time categorical auto-detection then disagreed with the booster's fit-time spec and unseen categories were not
mapped through the train-time domain. The same bundle predicted differently in-memory and from disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training import OutputConfig
from mlframe.training.core import predict_from_models, predict_mlframe_models_suite, train_mlframe_models_suite

from tests.training.shared import SimpleFeaturesAndTargetsExtractor


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    x = rng.randn(n, 3)
    cat = rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2])
    logits = x @ np.array([1.5, -1.0, 0.8]) + np.where(cat == "a", 1.0, -0.5)
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
    df = pd.DataFrame(x, columns=[f"f_{i}" for i in range(3)])
    df["cat_low"] = cat
    df["target"] = y
    return df


def test_the_two_entry_points_agree_on_a_categorical_frame(tmp_path):
    pytest.importorskip("lightgbm")
    train_df = _frame(400, seed=0)
    test_df = _frame(120, seed=1)
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
    data_dir = str(tmp_path / "data")

    models, metadata = train_mlframe_models_suite(
        df=train_df,
        target_name="test_target",
        model_name="lgb_cat_parity",
        features_and_targets_extractor=fte,
        mlframe_models=["lgb"],
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=data_dir, models_dir="models"),
        verbose=0,
        hyperparams_config={"iterations": 10},
    )

    in_memory = predict_from_models(
        df=test_df, models=models, metadata=metadata,
        features_and_targets_extractor=fte, return_probabilities=True, verbose=0,
    )
    from_disk = predict_mlframe_models_suite(
        df=test_df, models_path=f"{data_dir}/models/test_target/lgb_cat_parity",
        features_and_targets_extractor=fte, return_probabilities=True, verbose=0,
    )

    # The two entry points key their probability dicts differently (by target vs by model name), which is not what
    # this test is about: with a single model there is exactly one array on each side and they must be the same
    # numbers, because the same booster is being served the same rows.
    mem_probs, disk_probs = in_memory.get("probabilities", {}), from_disk.get("probabilities", {})
    assert len(mem_probs) == 1 and len(disk_probs) == 1, f"expected one model per side, got {list(mem_probs)} / {list(disk_probs)}"
    mem = np.asarray(next(iter(mem_probs.values())))
    disk = np.asarray(next(iter(disk_probs.values())))
    assert disk.shape == mem.shape
    np.testing.assert_allclose(disk, mem, rtol=1e-10, atol=1e-12,
                               err_msg="the disk-served suite must reproduce the in-memory prediction")
