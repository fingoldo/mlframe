"""Cluster 28 regression: ``_open_verified_bundle`` hands back the verified bytes, or refuses (audit 2026-11-20, 28-F76, part)."""

import pytest

from mlframe.training import io as tio


def _write_bundle(tmp_path, payload=b"model-bytes", sidecar=True):
    """Write a bundle file and, when asked, its save sidecar with the real digest."""
    path = tmp_path / "m.bundle"
    path.write_bytes(payload)
    if sidecar:
        tio._write_save_meta_sidecar(str(path))
    return str(path)


def test_missing_file_returns_none(tmp_path):
    """A bundle that does not exist yields None so the loader reports it (28-F76)."""
    assert tio._open_verified_bundle(str(tmp_path / "absent.bundle")) is None


def test_matching_digest_returns_the_exact_bytes(tmp_path):
    """With a sidecar digest that matches, the returned buffer holds the bundle's bytes (28-F76)."""
    path = _write_bundle(tmp_path)
    buf = tio._open_verified_bundle(path)
    assert buf.read() == b"model-bytes"


def test_changed_bytes_after_the_sidecar_are_refused(tmp_path):
    """Overwriting the bundle after its sidecar was written makes the digest mismatch and raises (28-F76)."""
    path = _write_bundle(tmp_path)
    with open(path, "wb") as fh:
        fh.write(b"tampered")
    with pytest.raises(tio.UnverifiedBundleError, match="does not match"):
        tio._open_verified_bundle(path)


def test_bundle_without_a_recorded_digest_is_refused_unless_opted_out(tmp_path, monkeypatch):
    """No sidecar: refused by default; loaded when the explicit opt-out environment flag is set (28-F76)."""
    path = _write_bundle(tmp_path, sidecar=False)
    monkeypatch.delenv(tio.UNVERIFIED_ENV_VAR, raising=False)
    with pytest.raises(tio.UnverifiedBundleError, match="no sha256 recorded"):
        tio._open_verified_bundle(path)
    monkeypatch.setenv(tio.UNVERIFIED_ENV_VAR, "1")
    assert tio._open_verified_bundle(path).read() == b"model-bytes"
