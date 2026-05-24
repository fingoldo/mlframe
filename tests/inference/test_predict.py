"""Smoke tests for mlframe.inference.predict helpers.

Covers small file-level helpers that don't require a trained model:
  - _sha256_of_file
  - _verify_sidecar (matching + mismatching + absent sidecar)
"""
from __future__ import annotations

import hashlib

import pytest

predict = pytest.importorskip("mlframe.inference.predict")


@pytest.mark.fast
def test_sha256_of_file_matches_hashlib(tmp_path):
    """_sha256_of_file must match the same digest hashlib produces directly."""
    p = tmp_path / "blob.bin"
    payload = b"hello mlframe inference predict\n" * 1024
    p.write_bytes(payload)

    expected = hashlib.sha256(payload).hexdigest()
    actual = predict._sha256_of_file(str(p))
    assert actual == expected


@pytest.mark.fast
def test_verify_sidecar_absent_default_strict_returns_false(tmp_path, monkeypatch):
    """Default-strict mode: missing sidecar -> False (RCE-bypass guard).

    Prior contract was fail-OPEN (return True with WARN); flipped to
    fail-CLOSED in S72 fix so an attacker who plants a pickle without a
    matching sidecar cannot bypass content verification. Legacy un-sidecar'd
    bundles can still be loaded via the MLFRAME_ALLOW_UNVERIFIED_PICKLE=1
    escape hatch -- covered by the next test.
    """
    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    p = tmp_path / "model.pkl"
    p.write_bytes(b"x")
    assert predict._verify_sidecar(str(p)) is False


@pytest.mark.fast
def test_verify_sidecar_absent_with_env_opt_in_returns_true(tmp_path, monkeypatch):
    """MLFRAME_ALLOW_UNVERIFIED_PICKLE=1 restores legacy pass-through with WARN."""
    monkeypatch.setenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", "1")
    p = tmp_path / "model.pkl"
    p.write_bytes(b"x")
    assert predict._verify_sidecar(str(p)) is True


@pytest.mark.fast
def test_verify_sidecar_match(tmp_path):
    """Matching sha256 sidecar -> True."""
    p = tmp_path / "model.pkl"
    p.write_bytes(b"trained model bytes")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path / "model.pkl.sha256").write_text(digest + "  model.pkl\n", encoding="utf-8")
    assert predict._verify_sidecar(str(p)) is True


@pytest.mark.fast
def test_verify_sidecar_mismatch(tmp_path):
    """Mismatching sha256 sidecar -> False (catches tampering)."""
    p = tmp_path / "model.pkl"
    p.write_bytes(b"trained model bytes")
    (tmp_path / "model.pkl.sha256").write_text("0" * 64 + "  model.pkl\n", encoding="utf-8")
    assert predict._verify_sidecar(str(p)) is False
