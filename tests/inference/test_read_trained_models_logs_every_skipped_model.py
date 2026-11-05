"""Every model file that read_trained_models skips is logged, not only the first few per process.

The skip messages went through log_throttle with one fixed key per failure kind, so after five corrupt model files across the process every
later one was dropped silently and the ensemble quietly lost members. A skipped model is rare and each one changes the prediction, so each
skip now logs.
"""

from __future__ import annotations

import hashlib
import json
import logging

import pandas as pd

from mlframe.inference.predict import read_trained_models


def test_each_sha256_mismatch_is_logged_after_five_are_already_logged(tmp_path, caplog):
    """Seven model files with a wrong sidecar are all reported, not only the first five."""
    featureset = "fs1"
    fdir = tmp_path / featureset
    fdir.mkdir()
    features = fdir / "features.dump.json"
    features.write_text(json.dumps(["a", "b"]))
    (fdir / "features.dump.json.sha256").write_text(hashlib.sha256(features.read_bytes()).hexdigest())
    for i in range(7):
        (fdir / f"model_{i}.joblib").write_bytes(b"not really a model")
        (fdir / f"model_{i}.joblib.sha256").write_text("0" * 64)
    X = pd.DataFrame({"a": [1.0], "b": [2.0]})
    with caplog.at_level(logging.ERROR, logger="mlframe.inference.predict"):
        models, _ = read_trained_models(featureset, X, features_file_name="features.dump", inference_folder=str(tmp_path))
    assert models == {}
    skipped = [r for r in caplog.records if "sha256 mismatch" in r.getMessage()]
    assert len(skipped) == 7
