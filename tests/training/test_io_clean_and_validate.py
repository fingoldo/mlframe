"""Unit tests for `training.io.clean_mlframe_model` + `validate_load_meta_sidecar`.

Existing sidecar / fsync tests don't exercise the two un-tested helpers. This
file pins:
- `clean_mlframe_model` strips the documented inference-irrelevant fields and
  preserves the rest;
- `validate_load_meta_sidecar` returns None on missing sidecar, returns the
  parsed payload when present, and warns / raises on library-version drift.
"""

from __future__ import annotations

import orjson
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.training.io import (
    _LEAN_STRIP_FIELDS,
    _meta_sidecar_path,
    atomic_write_bytes,
    clean_mlframe_model,
    load_save_meta_sidecar,
    validate_load_meta_sidecar,
)


# ----- clean_mlframe_model -----------------------------------------------


def test_clean_strips_documented_inference_irrelevant_fields():
    # Build a namespace with every documented strip-field populated.
    model = SimpleNamespace()
    for field in _LEAN_STRIP_FIELDS:
        setattr(model, field, np.zeros(10, dtype=np.float32))
    # Plus some fields that MUST survive.
    model.feature_names = ["a", "b", "c"]
    model.model_class = "lgb"
    model.inner = object()
    cleaned = clean_mlframe_model(model)
    # Every documented field stripped.
    for field in _LEAN_STRIP_FIELDS:
        assert not hasattr(cleaned, field), f"{field} should have been stripped"
    # Other fields preserved.
    assert cleaned.feature_names == ["a", "b", "c"]
    assert cleaned.model_class == "lgb"
    assert cleaned.inner is model.inner


def test_clean_is_safe_when_no_strip_fields_present():
    # A namespace that never had any strip-field returns unchanged.
    model = SimpleNamespace(feature_names=["x"], threshold=0.5)
    cleaned = clean_mlframe_model(model)
    assert cleaned.feature_names == ["x"]
    assert cleaned.threshold == 0.5


def test_clean_modifies_in_place_returns_same_object():
    model = SimpleNamespace(test_preds=np.zeros(5))
    cleaned = clean_mlframe_model(model)
    # Returned namespace is the same instance (in-place mutation).
    assert cleaned is model
    assert not hasattr(cleaned, "test_preds")


def test_clean_partial_strip_subset():
    # Only some strip-fields populated; clean removes those, leaves the rest absent.
    model = SimpleNamespace(test_preds=np.zeros(2), val_preds=np.zeros(2))
    clean_mlframe_model(model)
    assert not hasattr(model, "test_preds")
    assert not hasattr(model, "val_preds")
    # Fields never set stay absent (clean does not create them).
    assert not hasattr(model, "train_preds")


# ----- validate_load_meta_sidecar ----------------------------------------


def _write_sidecar(bundle_path: str, payload: dict) -> str:
    sidecar = _meta_sidecar_path(bundle_path)
    data = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    atomic_write_bytes(sidecar, lambda f: f.write(data), fsync=False)
    return sidecar


def test_validate_load_meta_sidecar_missing_returns_none(tmp_path):
    # No sidecar -> None, no warning, no raise.
    bundle_path = str(tmp_path / "model.bundle")
    result = validate_load_meta_sidecar(bundle_path)
    assert result is None


def test_validate_load_meta_sidecar_corrupt_json_returns_none(tmp_path, caplog):
    bundle_path = str(tmp_path / "model.bundle")
    sidecar = _meta_sidecar_path(bundle_path)
    # Half-written / invalid JSON.
    atomic_write_bytes(sidecar, lambda f: f.write(b"{ not valid json"), fsync=False)
    with caplog.at_level(logging.WARNING, logger="mlframe.training.io"):
        result = validate_load_meta_sidecar(bundle_path)
    assert result is None
    # WARN logged on corrupt payload.
    assert any("failed to read" in r.message for r in caplog.records)


def test_validate_load_meta_sidecar_non_object_returns_none(tmp_path, caplog):
    # JSON that parses but isn't a dict (e.g. a JSON list) -> None, WARN.
    bundle_path = str(tmp_path / "model.bundle")
    sidecar = _meta_sidecar_path(bundle_path)
    atomic_write_bytes(sidecar, lambda f: f.write(b"[1, 2, 3]"), fsync=False)
    with caplog.at_level(logging.WARNING, logger="mlframe.training.io"):
        result = validate_load_meta_sidecar(bundle_path)
    assert result is None
    assert any("not a JSON object" in r.message for r in caplog.records)


def test_validate_load_meta_sidecar_matching_versions_returns_payload(tmp_path, caplog):
    bundle_path = str(tmp_path / "model.bundle")
    # Use a library that's guaranteed installed and matches its live version.
    import numpy

    payload = {
        "sidecar_version": 1,
        "saved_at_utc": "2026-05-24T00:00:00Z",
        "lib_versions": {"numpy": numpy.__version__},
    }
    _write_sidecar(bundle_path, payload)
    with caplog.at_level(logging.WARNING, logger="mlframe.training.io"):
        result = validate_load_meta_sidecar(bundle_path)
    assert result is not None
    assert result["lib_versions"]["numpy"] == numpy.__version__
    # No drift -> no WARN about version mismatch.
    drift_warns = [r for r in caplog.records if "library-version drift" in r.message]
    assert drift_warns == []


def test_validate_load_meta_sidecar_drift_warns_by_default(tmp_path, caplog):
    bundle_path = str(tmp_path / "model.bundle")
    payload = {
        "sidecar_version": 1,
        "saved_at_utc": "2026-05-24T00:00:00Z",
        "lib_versions": {"numpy": "99.99.99-fake"},  # impossible version => drift
    }
    _write_sidecar(bundle_path, payload)
    with caplog.at_level(logging.WARNING, logger="mlframe.training.io"):
        result = validate_load_meta_sidecar(bundle_path, strict=False)
    # Payload still returned (drift only WARNs by default).
    assert result is not None
    # WARN logged.
    assert any("library-version drift" in r.message for r in caplog.records)


def test_validate_load_meta_sidecar_strict_raises_on_drift(tmp_path):
    bundle_path = str(tmp_path / "model.bundle")
    payload = {
        "sidecar_version": 1,
        "saved_at_utc": "2026-05-24T00:00:00Z",
        "lib_versions": {"numpy": "99.99.99-fake"},
    }
    _write_sidecar(bundle_path, payload)
    with pytest.raises(ValueError, match="library-version drift"):
        validate_load_meta_sidecar(bundle_path, strict=True)


def test_validate_load_meta_sidecar_missing_live_library_flagged(tmp_path, caplog):
    bundle_path = str(tmp_path / "model.bundle")
    payload = {
        "sidecar_version": 1,
        "saved_at_utc": "2026-05-24T00:00:00Z",
        "lib_versions": {
            "definitely_not_installed_pkg_xyz_42": "1.2.3",
        },
    }
    _write_sidecar(bundle_path, payload)
    with caplog.at_level(logging.WARNING, logger="mlframe.training.io"):
        validate_load_meta_sidecar(bundle_path)
    # Missing-from-live is treated as drift.
    assert any("NOT-INSTALLED" in r.message for r in caplog.records)


# ----- load_save_meta_sidecar (the underlying loader) --------------------


def test_load_save_meta_sidecar_returns_dict_when_present(tmp_path):
    bundle_path = str(tmp_path / "m.bundle")
    payload = {"sidecar_version": 1, "lib_versions": {"numpy": "1.0.0"}}
    _write_sidecar(bundle_path, payload)
    out = load_save_meta_sidecar(bundle_path)
    assert out == payload


def test_load_save_meta_sidecar_returns_none_when_missing(tmp_path):
    bundle_path = str(tmp_path / "missing.bundle")
    assert load_save_meta_sidecar(bundle_path) is None
