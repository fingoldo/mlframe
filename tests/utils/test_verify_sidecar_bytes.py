"""A sha256 sidecar is verified over the bytes that get deserialized, never over a second read of the path.

Verifying by path and then opening the path again lets another process replace the file in between: the digest check passes on the old bytes
and the loader unpickles the new ones. Each loader below reads the file once, verifies those bytes, and loads from them; the tests swap the
file on disk right after the verification and expect the originally verified content.
"""

from __future__ import annotations

import hashlib
import pickle

import orjson
import pytest

from mlframe.utils.safe_pickle import UNVERIFIED_ENV_VAR, verify_sidecar_payload


def _write_with_sidecar(path, payload: bytes) -> None:
    """write with sidecar."""
    path.write_bytes(payload)
    (path.parent / (path.name + ".sha256")).write_text(hashlib.sha256(payload).hexdigest() + "  " + path.name + "\n", encoding="utf-8")


def test_verify_sidecar_payload_accepts_the_matching_digest(tmp_path):
    """Test verify sidecar bytes accepts the matching digest."""
    f = tmp_path / "a.bin"
    _write_with_sidecar(f, b"payload")
    assert verify_sidecar_payload(str(f), b"payload") is True


def test_verify_sidecar_payload_hashes_the_given_bytes_not_the_file(tmp_path):
    """Test verify sidecar bytes hashes the given bytes not the file."""
    f = tmp_path / "a.bin"
    _write_with_sidecar(f, b"payload")
    assert verify_sidecar_payload(str(f), b"other bytes") is False


@pytest.mark.parametrize("sidecar_text", ["", "not-a-digest\n", "é" * 64 + "\n"])
def test_verify_sidecar_payload_rejects_a_malformed_sidecar(tmp_path, sidecar_text):
    """Test verify sidecar bytes rejects a malformed sidecar."""
    f = tmp_path / "a.bin"
    f.write_bytes(b"payload")
    (tmp_path / "a.bin.sha256").write_text(sidecar_text, encoding="utf-8")
    assert verify_sidecar_payload(str(f), b"payload") is False


def test_verify_sidecar_payload_missing_sidecar_is_refused_unless_opted_in(tmp_path, monkeypatch):
    """Test verify sidecar bytes missing sidecar is refused unless opted in."""
    f = tmp_path / "a.bin"
    f.write_bytes(b"payload")
    monkeypatch.delenv(UNVERIFIED_ENV_VAR, raising=False)
    assert verify_sidecar_payload(str(f), b"payload") is False
    monkeypatch.setenv(UNVERIFIED_ENV_VAR, "1")
    assert verify_sidecar_payload(str(f), b"payload") is True


def test_metadata_loader_unpickles_the_verified_bytes_when_the_file_is_swapped(tmp_path, monkeypatch):
    """Test metadata loader unpickles the verified bytes when the file is swapped."""
    from mlframe.training.core import _metadata_loader as loader

    f = tmp_path / "metadata.pkl"
    _write_with_sidecar(f, pickle.dumps({"which": "verified"}))
    real_verify = loader.verify_sidecar_payload

    def verify_then_swap(path, payload):
        """Verify then swap."""
        ok = real_verify(path, payload)
        f.write_bytes(pickle.dumps({"which": "swapped"}))
        return ok

    monkeypatch.setattr(loader, "verify_sidecar_payload", verify_then_swap)
    assert loader.load_metadata_file(str(f), "pkl", "test") == {"which": "verified"}


def test_features_loader_reads_the_verified_json_when_the_file_is_swapped(tmp_path, monkeypatch):
    """Test features loader reads the verified json when the file is swapped."""
    from mlframe.inference import predict

    f = tmp_path / "features.json"
    _write_with_sidecar(f, orjson.dumps(["a", "b"]))
    real_verify = predict._verify_sidecar_payload

    def verify_then_swap(path, payload):
        """Verify then swap."""
        ok = real_verify(path, payload)
        f.write_bytes(orjson.dumps(["swapped"]))
        return ok

    monkeypatch.setattr(predict, "_verify_sidecar_payload", verify_then_swap)
    assert predict._load_features_file(str(tmp_path / "features")) == ["a", "b"]


def test_features_loader_refuses_a_file_whose_bytes_do_not_match_the_sidecar(tmp_path):
    """Test features loader refuses a file whose bytes do not match the sidecar."""
    from mlframe.inference import predict

    f = tmp_path / "features.json"
    _write_with_sidecar(f, orjson.dumps(["a", "b"]))
    f.write_bytes(orjson.dumps(["tampered"]))
    assert predict._load_features_file(str(tmp_path / "features")) is None


def test_model_loader_unpickles_the_verified_bundle_when_the_file_is_swapped(tmp_path, monkeypatch):
    """Test model loader unpickles the verified bundle when the file is swapped."""
    from types import SimpleNamespace

    from mlframe.training import io as mlio

    good, other = tmp_path / "good.dump", tmp_path / "other.dump"
    mlio.save_mlframe_model(SimpleNamespace(which="verified"), str(good), verbose=0)
    mlio.save_mlframe_model(SimpleNamespace(which="swapped"), str(other), verbose=0)
    swapped_bytes = other.read_bytes()
    real_verify = mlio._verify_bundle_digest

    def verify_then_swap(path, handle=None):
        """Verify then swap."""
        real_verify(path, handle)
        good.write_bytes(swapped_bytes)

    monkeypatch.setattr(mlio, "_verify_bundle_digest", verify_then_swap)
    mlio._load_model_cache_clear()
    loaded = mlio.load_mlframe_model(str(good))
    assert loaded is not None and loaded.which == "verified"
