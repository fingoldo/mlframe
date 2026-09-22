"""Two disk-path predict contracts.

PRD-10: a `model_names` filter matching nothing fell back to loading every model - a different, unrequested
prediction served on a WARN.
PRD-09: the suite-wide ensemble labelled at a fixed 0.5 with a strict `>`, while the per-target ensemble beside it
used the tuned threshold and `>=`.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest

from mlframe.training import OutputConfig
from mlframe.training.core import predict_mlframe_models_suite, train_mlframe_models_suite

from tests.training.shared import SimpleFeaturesAndTargetsExtractor


@pytest.fixture(scope="module")
def suite():
    rng = np.random.RandomState(0)
    n = 1500
    x = rng.randn(n, 4)
    df = pd.DataFrame(x, columns=[f"f_{i}" for i in range(4)])
    df["target"] = ((x[:, 0] - x[:, 1] + rng.normal(0, 0.5, n)) > 2.2).astype(int)
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
    data_dir = tempfile.mkdtemp()
    _, md = train_mlframe_models_suite(
        df=df, target_name="t", model_name="m", features_and_targets_extractor=fte, mlframe_models=["linear", "lgb"],
        use_ordinary_models=True, use_mlframe_ensembles=True, output_config=OutputConfig(data_dir=data_dir, models_dir="models"),
        hyperparams_config={"iterations": 30}, verbose=0,
    )
    return f"{data_dir}/models/t/m", fte, df, md


def test_a_filter_that_matches_nothing_raises(suite):
    models_path, fte, df, _ = suite
    with pytest.raises(ValueError, match="matched none"):
        predict_mlframe_models_suite(df=df.head(50), models_path=models_path, features_and_targets_extractor=fte, model_names=["no_such_model"], verbose=0)


def test_the_suite_wide_labels_use_the_tuned_threshold(suite):
    models_path, fte, df, md = suite
    out = predict_mlframe_models_suite(df=df.head(400), models_path=models_path, features_and_targets_extractor=fte, verbose=0)
    key = next(k for k in md["decision_thresholds"] if k.count("|") == 1)
    thr = md["decision_thresholds"][key]
    probs = np.asarray(out["ensemble_probabilities"])[:, 1]
    np.testing.assert_array_equal(np.asarray(out["ensemble_predictions"]), (probs >= thr).astype(int))
