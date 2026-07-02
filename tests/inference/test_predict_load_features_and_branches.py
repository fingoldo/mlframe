"""Edge-case coverage for ``mlframe.inference.predict`` branches OTHER than the
path-traversal guard (already covered by ``test_read_trained_models_traversal_guard.py``).

Covers ``_load_features_file`` (json list / non-list / invalid / absent-with-unverified-dump),
``read_trained_models`` (missing dir, feature-name mismatch skip, features-file absent fallback,
disallowed vs explicitly-allowed extension), and ``get_models_raw_predictions`` (binary
positive-column, multiclass full matrix, regressor fallback).
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from mlframe.inference.predict import (
    _load_features_file,
    read_trained_models,
    get_models_raw_predictions,
)
from mlframe.utils.safe_pickle import write_sidecar


class DummyModel:
    """Module-level (picklable) stand-in exposing ``feature_names_in_`` + ``predict_proba``."""

    def __init__(self, feats):
        self.feature_names_in_ = np.array(feats)

    def predict_proba(self, X):
        return np.tile([0.4, 0.6], (len(X), 1))


# ---------------------------------------------------------------------------
# _load_features_file
# ---------------------------------------------------------------------------


def test_load_features_json_list(tmp_path):
    fp = tmp_path / "features.dump"
    (tmp_path / "features.dump.json").write_text(json.dumps(["a", "b", "c"]))
    assert _load_features_file(str(fp)) == ["a", "b", "c"]


def test_load_features_json_non_list_returns_none(tmp_path):
    fp = tmp_path / "features.dump"
    (tmp_path / "features.dump.json").write_text(json.dumps({"x": 1}))
    assert _load_features_file(str(fp)) is None


def test_load_features_invalid_json_raises(tmp_path):
    # Malformed JSON surfaces as a ValueError subclass (orjson.JSONDecodeError /
    # json.JSONDecodeError). read_trained_models wraps this call in try/except; at the
    # helper level the parse error propagates rather than being swallowed.
    fp = tmp_path / "features.dump"
    (tmp_path / "features.dump.json").write_text("{not valid json")
    with pytest.raises(ValueError):
        _load_features_file(str(fp))


def test_load_features_dump_without_sidecar_refused(tmp_path):
    # No JSON twin + a joblib dump lacking its .sha256 sidecar -> default-strict refusal -> None.
    fp = tmp_path / "features.dump"
    joblib.dump(["p", "q"], str(fp))
    assert _load_features_file(str(fp)) is None


def test_load_features_dump_with_sidecar_loads(tmp_path):
    fp = tmp_path / "features.dump"
    joblib.dump(["p", "q"], str(fp))
    write_sidecar(str(fp))
    assert _load_features_file(str(fp)) == ["p", "q"]


# ---------------------------------------------------------------------------
# read_trained_models
# ---------------------------------------------------------------------------


@pytest.fixture
def X():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})


def _make_featureset(tmp_path, name, feats_json, model_fname, model_feats):
    infer = tmp_path / "infer"
    fsdir = infer / name
    fsdir.mkdir(parents=True)
    if feats_json is not None:
        (fsdir / "features.dump.json").write_text(json.dumps(feats_json))
    if model_fname is not None:
        mp = fsdir / model_fname
        joblib.dump(DummyModel(model_feats), str(mp))
        write_sidecar(str(mp))
    return str(infer)


def test_read_missing_dir_returns_empty_and_unchanged_X(tmp_path, X):
    models, Xo = read_trained_models("nonexistent", X, inference_folder=str(tmp_path / "infer"))
    assert models == {}
    assert Xo is X


def test_read_feature_mismatch_skips_model(tmp_path, X):
    infer = _make_featureset(tmp_path, "fs1", ["a", "b"], "model.pkl", ["z"])
    models, Xo = read_trained_models("fs1", X, inference_folder=infer)
    assert models == {}, "model whose feature_names_in_ differs from the featureset must be skipped"
    assert list(Xo.columns) == ["a", "b"]


def test_read_feature_match_loads_model(tmp_path, X):
    infer = _make_featureset(tmp_path, "fs2", ["a", "b"], "model.pkl", ["a", "b"])
    models, Xo = read_trained_models("fs2", X, inference_folder=infer)
    assert list(models.keys()) == ["model"]
    assert isinstance(models["model"], DummyModel)
    assert list(Xo.columns) == ["a", "b"]


def test_read_disallowed_extension_skipped_but_allowed_when_whitelisted(tmp_path, X):
    infer = _make_featureset(tmp_path, "fs3", ["a", "b"], "model.txt", ["a", "b"])
    # .txt is outside the default allow-list -> skipped.
    models, _ = read_trained_models("fs3", X, inference_folder=infer)
    assert models == {}
    # explicit allow-list including .txt -> the same file now loads.
    models2, _ = read_trained_models("fs3", X, inference_folder=infer, allowed_extensions=[".txt"])
    assert list(models2.keys()) == ["model"]


def test_read_features_file_absent_falls_back_to_X_columns(tmp_path, X):
    infer = _make_featureset(tmp_path, "fs4", None, "model.pkl", ["a", "b"])
    models, Xo = read_trained_models("fs4", X, inference_folder=infer)
    # No features sidecar -> features taken from X.columns; the matching model still loads.
    assert list(models.keys()) == ["model"]
    assert list(Xo.columns) == ["a", "b"]


# ---------------------------------------------------------------------------
# get_models_raw_predictions
# ---------------------------------------------------------------------------


class _BinClf:
    def predict_proba(self, X):
        return np.tile([0.3, 0.7], (len(X), 1))


class _MultiClf:
    def predict_proba(self, X):
        return np.tile([0.1, 0.2, 0.7], (len(X), 1))


class _Reg:
    def predict(self, X):
        return np.arange(len(X), dtype=float)


def test_raw_predictions_binary_multiclass_and_regressor(X):
    preds = get_models_raw_predictions({"bin": _BinClf(), "multi": _MultiClf(), "reg": _Reg()}, X, None)
    # Binary -> positive-class column only.
    assert preds["bin"].shape == (3,)
    np.testing.assert_allclose(preds["bin"], 0.7)
    # Multiclass (>2 columns) -> full probability matrix.
    assert preds["multi"].shape == (3, 3)
    # Regressor (no predict_proba) -> predict output.
    assert preds["reg"].tolist() == [0.0, 1.0, 2.0]
