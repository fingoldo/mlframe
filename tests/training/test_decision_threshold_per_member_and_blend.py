"""Decision thresholds must be tuned on the probabilities they are applied to.

ENS-05: one threshold per target was tuned on the FIRST member that exposed val probabilities and then applied to the
blend and to every other member, so a flat linear member's threshold was used on the sharper blend.
ENS-06: tuning sat after the "fewer than two members" early return, so a lone model on an imbalanced target was never
tuned and kept 0.5.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest

from mlframe.training import OutputConfig
from mlframe.training.core import train_mlframe_models_suite

from tests.training.shared import SimpleFeaturesAndTargetsExtractor


def _imbalanced(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    x = rng.randn(n, 4)
    df = pd.DataFrame(x, columns=[f"f_{i}" for i in range(4)])
    df["target"] = ((x[:, 0] - x[:, 1] + rng.normal(0, 0.5, n)) > 2.2).astype(int)  # ~5% positives
    return df


def _train(models, ensembles):
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
    _, metadata = train_mlframe_models_suite(
        df=_imbalanced(), target_name="t", model_name="m", features_and_targets_extractor=fte,
        mlframe_models=models, use_ordinary_models=True, use_mlframe_ensembles=ensembles,
        output_config=OutputConfig(data_dir=tempfile.mkdtemp(), models_dir="models"),
        hyperparams_config={"iterations": 30}, verbose=0,
    )
    return metadata


def test_a_single_model_on_an_imbalanced_target_is_tuned():
    md = _train(["lgb"], ensembles=False)
    paths = md.get("decision_threshold_paths", {})
    target_keys = [k for k in paths if k.count("|") == 1]
    assert target_keys and all(paths[k] == "tuned" for k in target_keys), paths


def test_members_and_the_blend_get_their_own_thresholds():
    md = _train(["linear", "lgb"], ensembles=True)
    thresholds = md.get("decision_thresholds", {})
    target_keys = [k for k in thresholds if k.count("|") == 1]
    member_keys = [k for k in thresholds if k.count("|") == 2]
    assert len(target_keys) == 1, thresholds
    assert len(member_keys) == 2, f"each member needs its own threshold: {thresholds}"
    assert md["decision_threshold_paths"][target_keys[0]] == "tuned"
