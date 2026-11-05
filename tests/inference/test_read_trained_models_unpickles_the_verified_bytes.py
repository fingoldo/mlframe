"""``read_trained_models`` unpickles the same bytes it verified against the model's sha256 sidecar.

It used to verify the file by path and then reopen it by path to unpickle, so a payload replaced in between passed the check and was loaded.
"""

from __future__ import annotations

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mlframe.inference import predict as predict_mod
from mlframe.inference.predict import read_trained_models


def _fit(seed):
    """A logistic regression fitted on data drawn from ``seed``, so two calls give visibly different coefficients."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(60, 2)), columns=["a", "b"])
    y = (X["a"] + rng.normal(scale=0.5, size=60) > 0).astype(int)
    return LogisticRegression().fit(X, y), X


def test_model_is_unpickled_from_the_bytes_that_were_verified(tmp_path, monkeypatch):
    """A payload swapped on disk after the check is not what gets loaded."""
    fdir = tmp_path / "fs1"
    fdir.mkdir()
    feats = fdir / "features.dump.json"
    feats.write_text(json.dumps(["a", "b"]))
    (fdir / "features.dump.json.sha256").write_text(hashlib.sha256(feats.read_bytes()).hexdigest())
    original, X = _fit(0)
    impostor, _ = _fit(1)
    model_path = fdir / "model_0.joblib"
    joblib.dump(original, model_path)
    (fdir / "model_0.joblib.sha256").write_text(hashlib.sha256(model_path.read_bytes()).hexdigest())

    real_load = predict_mod.safe_joblib_load

    def swap_then_load(path):
        """Replace the payload on disk, then load the path, as an attacker racing the check would."""
        joblib.dump(impostor, path)
        return real_load(path)

    monkeypatch.setattr(predict_mod, "safe_joblib_load", swap_then_load)
    models, _ = read_trained_models("fs1", X, features_file_name="features.dump", inference_folder=str(tmp_path))
    assert "model_0" in models
    assert np.array_equal(models["model_0"].coef_, original.coef_)
