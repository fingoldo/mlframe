"""When the library-version check on a model's sidecar raises, the load goes ahead and the failure is logged as a warning.

The version check is what warns about library drift. If it raises, that warning never runs, so a debug-level line hid a drift risk from every
normal run.
"""

from __future__ import annotations

import hashlib
import orjson
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mlframe.inference import predict as predict_mod
from mlframe.inference.predict import read_trained_models


def _write_model(fdir, name, model):
    """Dump a fitted model with its sha256 sidecar, the way the save path writes it."""
    path = fdir / name
    joblib.dump(model, path)
    (fdir / f"{name}.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())


def test_failing_version_check_is_logged_as_a_warning(tmp_path, monkeypatch, caplog):
    """A raising version check still loads the model, and the failure appears at WARNING, not only at DEBUG."""
    fdir = tmp_path / "fs1"
    fdir.mkdir()
    feats = fdir / "features.dump.json"
    feats.write_bytes(orjson.dumps(["a", "b"]))
    (fdir / "features.dump.json.sha256").write_text(hashlib.sha256(feats.read_bytes()).hexdigest())
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 2)), columns=["a", "b"])
    y = (X["a"] > 0).astype(int)
    _write_model(fdir, "model_0.joblib", LogisticRegression().fit(X, y))

    def _broken_check(path, strict=False):
        """Stand-in for a version check that cannot read its sidecar."""
        raise ValueError("malformed meta sidecar")

    monkeypatch.setattr("mlframe.training.io.validate_load_meta_sidecar", _broken_check)
    with caplog.at_level(logging.WARNING, logger=predict_mod.logger.name):
        models, _ = read_trained_models("fs1", X, features_file_name="features.dump", inference_folder=str(tmp_path))
    assert "model_0" in models
    assert any(r.levelno == logging.WARNING and "malformed meta sidecar" in r.getMessage() for r in caplog.records)
