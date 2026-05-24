"""Regression sensor for w2b-percol-scattered inference/predict.py ``.values.tolist()`` -> ``.to_list()`` (finding #37).

Migrated paths must continue to return a plain python list[str] for downstream column lookups.
"""
from __future__ import annotations

import hashlib
import os
import sys
import json
import joblib

import numpy as np
import pandas as pd


def test_load_features_file_json_path_returns_list(tmp_path):
    from mlframe.inference.predict import _load_features_file

    feats = ["a", "b", "c"]
    features_path = tmp_path / "features.dump"
    json_path = str(features_path) + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(feats, f)
    loaded = _load_features_file(str(features_path))
    assert loaded == feats


def test_load_features_file_index_returns_python_list(tmp_path):
    """When the joblib payload is a pandas Index, the loader must return a plain list[str] (the .to_list() path) -- not a numpy.ndarray view."""
    from mlframe.inference.predict import _load_features_file

    idx = pd.Index(["x", "y", "z"])
    features_path = tmp_path / "features.dump"
    joblib.dump(idx, str(features_path))
    # Generate the .sha256 sidecar so the strict loader accepts the dump (mirrors the production load contract).
    with open(features_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    with open(str(features_path) + ".sha256", "w", encoding="utf-8") as f:
        f.write(digest + "  features.dump\n")
    loaded = _load_features_file(str(features_path))
    assert isinstance(loaded, list)
    assert loaded == ["x", "y", "z"]
