"""``read_trained_models`` says when it serves fewer models than the featureset holds.

Each skipped model is logged on its own, but nothing summarised the count, so an ensemble missing members looked complete to its caller.
"""

from __future__ import annotations

import hashlib
import json
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


def test_short_ensemble_is_reported_with_the_served_and_total_counts(tmp_path, caplog):
    """Two model files, one with a tampered sidecar: one is served and the warning says one of two."""
    fdir = tmp_path / "fs1"
    fdir.mkdir()
    feats = fdir / "features.dump.json"
    feats.write_text(json.dumps(["a", "b"]))
    (fdir / "features.dump.json.sha256").write_text(hashlib.sha256(feats.read_bytes()).hexdigest())
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 2)), columns=["a", "b"])
    y = (X["a"] > 0).astype(int)
    _write_model(fdir, "model_0.joblib", LogisticRegression().fit(X, y))
    _write_model(fdir, "model_1.joblib", LogisticRegression().fit(X, y))
    (fdir / "model_1.joblib.sha256").write_text("0" * 64)
    with caplog.at_level(logging.WARNING, logger=predict_mod.logger.name):
        models, _ = read_trained_models("fs1", X, features_file_name="features.dump", inference_folder=str(tmp_path))
    assert set(models) == {"model_0"}
    assert any("served 1 of 2" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
