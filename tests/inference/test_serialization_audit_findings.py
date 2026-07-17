"""Regression sensors for the serialization/inference audit findings (P1-1, P1-2, P2-1).

Pre-fix issues:
- P1-1: ``get_models_raw_predictions`` performed zero internal validation of its own -- a caller that
  built ``trained_models``/``X`` without going through ``read_trained_models`` (skipping its
  ``feature_names_in_``/``feature_names_`` check) got no protection against a column-order/name
  mismatch, for any model type.
- P1-2: ``_load_features_file``'s JSON-sidecar path (``features.json`` / ``features.dump.json``)
  bypassed sha256 verification entirely -- only the joblib fallback path was gated.
- P2-1: ``read_trained_models``/``_load_features_file`` loaded via plain ``joblib.load``, which uses
  the unrestricted stdlib unpickler -- no module/class allowlist, unlike ``training/io.py``'s
  ``_SafeUnpickler`` used for training-side bundles.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression


def _sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _write_sidecar(path: Path) -> None:
    (path.parent / (path.name + ".sha256")).write_text(_sha256_hex(path) + "  " + path.name + "\n", encoding="utf-8")


class _FakeModelWrongFeatures:
    """Stands in for a fitted estimator trained on different features than X provides."""

    feature_names_in_ = np.array(["b", "a"])

    def predict(self, X):
        return np.zeros(len(X))


def test_get_models_raw_predictions_rejects_column_mismatch_without_read_trained_models():
    """P1-1: a caller that builds trained_models/X directly (bypassing read_trained_models) must still
    be protected -- get_models_raw_predictions itself must reject a model/X feature mismatch rather than
    silently predicting on misaligned columns.
    """
    from mlframe.inference.predict import get_models_raw_predictions

    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    trained_models = {"m1": _FakeModelWrongFeatures()}

    with pytest.raises(ValueError, match="different features"):
        get_models_raw_predictions(trained_models, X, Y=None)


def test_get_models_raw_predictions_allows_matching_features():
    """Happy path: matching feature_names_in_ must not be rejected."""
    from mlframe.inference.predict import get_models_raw_predictions

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [3.0, 4.0, 5.0, 6.0]})
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression().fit(X, y)
    predictions = get_models_raw_predictions({"m1": model}, X, Y=y)
    assert "m1" in predictions
    assert len(predictions["m1"]) == len(X)


def test_load_features_file_json_sidecar_rejects_missing_sha256(tmp_path: Path, monkeypatch):
    """P1-2: a JSON features sidecar with NO matching .sha256 companion must be refused (fail-closed),
    not silently trusted. Pre-fix, the JSON path never checked _verify_sidecar at all.
    """
    from mlframe.inference.predict import _load_features_file

    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    features_file = tmp_path / "features.dump"
    json_sidecar = tmp_path / "features.dump.json"
    json_sidecar.write_bytes(b'["a", "b"]')

    result = _load_features_file(str(features_file))
    assert result is None, "JSON sidecar with no .sha256 companion must fail-closed, matching the joblib path's contract"


def test_load_features_file_json_sidecar_rejects_corrupt_digest(tmp_path: Path, monkeypatch):
    """P1-2: a JSON features sidecar whose .sha256 companion does NOT match must be refused."""
    from mlframe.inference.predict import _load_features_file

    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    features_file = tmp_path / "features.dump"
    json_sidecar = tmp_path / "features.dump.json"
    json_sidecar.write_bytes(b'["a", "b"]')
    (tmp_path / "features.dump.json.sha256").write_text("0" * 64 + "  features.dump.json\n", encoding="utf-8")

    result = _load_features_file(str(features_file))
    assert result is None, "JSON sidecar with a corrupt digest must fail-closed"


def test_load_features_file_json_sidecar_loads_on_matching_digest(tmp_path: Path, monkeypatch):
    """P1-2 happy path: a JSON sidecar with a CORRECT .sha256 companion must still load successfully."""
    from mlframe.inference.predict import _load_features_file

    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    features_file = tmp_path / "features.dump"
    json_sidecar = tmp_path / "features.dump.json"
    json_sidecar.write_bytes(b'["a", "b"]')
    _write_sidecar(json_sidecar)

    result = _load_features_file(str(features_file))
    assert result == ["a", "b"]


class _EvilReduce:
    """Pickles to a call of builtins.eval -- the classic RCE gadget an unrestricted unpickler executes."""

    def __reduce__(self):
        return (eval, ("print('PWNED-BY-UNSAFE-UNPICKLE')",))


def test_safe_joblib_load_blocks_eval_gadget(tmp_path: Path):
    """P2-1: safe_joblib_load must block a builtins.eval reduce-gadget the same way _SafeUnpickler does
    for dill bundles. Plain joblib.load (pre-fix predict.py behaviour) would execute it.
    """
    import dill
    from mlframe.training.io import safe_joblib_load

    p = tmp_path / "evil.pkl"
    with open(p, "wb") as f:
        pickle.dump(_EvilReduce(), f)

    with pytest.raises(dill.UnpicklingError, match="Unsafe builtin"):
        safe_joblib_load(str(p))


def test_safe_joblib_load_roundtrips_sklearn_model(tmp_path: Path):
    """safe_joblib_load must still correctly load a real, benign joblib-dumped sklearn model."""
    import joblib
    from mlframe.training.io import safe_joblib_load

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [3.0, 4.0, 5.0, 6.0]})
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression().fit(X, y)
    p = tmp_path / "model.pkl"
    joblib.dump(model, str(p))

    loaded = safe_joblib_load(str(p))
    assert isinstance(loaded, LogisticRegression)
    np.testing.assert_allclose(loaded.coef_, model.coef_)


def test_read_trained_models_uses_safe_joblib_load(tmp_path: Path, monkeypatch):
    """P2-1: read_trained_models must route model loading through safe_joblib_load (blocking the eval
    gadget), not plain joblib.load, so infer/ models get the same RCE-surface restriction dill bundles do.
    """
    from mlframe.inference.predict import read_trained_models

    featureset = "fset"
    fdir = tmp_path / "infer" / featureset
    fdir.mkdir(parents=True)
    model_path = fdir / "evil.dump"
    with open(model_path, "wb") as f:
        pickle.dump(_EvilReduce(), f)
    _write_sidecar(model_path)

    X = pd.DataFrame({"a": [1.0], "b": [2.0]})
    models, _ = read_trained_models(featureset, X, inference_folder=str(tmp_path / "infer"))
    assert models == {}, "a model whose pickle payload is blocked by the allowlist must not appear in the loaded models dict"
