"""Regression sensor for S72: pickle/joblib RCE surface only partially mitigated.

Pre-fix issues:
- ``inference.predict._verify_sidecar`` returned True with a WARN when no
  ``.sha256`` sidecar existed. An attacker who plants a pickle file in
  the inference folder without a sidecar bypassed verification entirely
  (the only check was extension allow-listing).
- ``training.io._write_save_meta_sidecar`` left ``bundle_sha256``
  unfilled (literal ``"...", # not yet`` placeholder docstring) so
  operators had no payload-content fingerprint to compare against
  even when the meta sidecar was present.

Post-fix contract:
- ``_verify_sidecar`` is fail-CLOSED by default when no sidecar exists.
- ``MLFRAME_ALLOW_UNVERIFIED_PICKLE=1`` env var opens an opt-in escape
  for legacy deploys that haven't generated sidecars yet (with WARN).
- A sidecar with a CORRECT digest still loads successfully.
- A sidecar with a CORRUPT digest still returns False (pre-fix
  behaviour preserved -- the existing check is the load-time signal).
- ``_write_save_meta_sidecar`` populates ``bundle_sha256`` with the
  actual blake2b/sha256 of the bundle file (no more placeholder).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def tmp_pickle(tmp_path: Path):
    """Plant a dummy bundle file (not really pickle bytes, since
    _verify_sidecar only hashes the file -- digest match is what we
    test, not deserialisation). Returns the path."""
    p = tmp_path / "model.dump"
    p.write_bytes(b"FAKE-BUNDLE-BYTES-FOR-DIGEST-TEST" * 16)
    return p


def _sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def test_verify_sidecar_fails_closed_when_no_sidecar(tmp_pickle: Path, monkeypatch):
    """Default mode (no env opt-in): missing sidecar -> verification fails.

    Pre-fix: returned True with WARN (RCE-bypass surface).
    """
    from mlframe.inference.predict import _verify_sidecar

    # Ensure escape hatch is OFF.
    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    assert not _verify_sidecar(str(tmp_pickle)), (
        "Default-strict mode must REFUSE pickle load when .sha256 sidecar "
        "is missing (RCE-bypass guard). Pre-fix returned True with WARN; "
        "fix is to return False so the caller skips load."
    )


def test_verify_sidecar_opt_in_escape_for_legacy_deploys(tmp_pickle: Path, monkeypatch):
    """``MLFRAME_ALLOW_UNVERIFIED_PICKLE=1`` opens the legacy escape
    hatch with WARN so existing un-sidecar'd bundles keep loading
    during the migration window.
    """
    from mlframe.inference.predict import _verify_sidecar

    monkeypatch.setenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", "1")
    assert _verify_sidecar(str(tmp_pickle)), (
        "MLFRAME_ALLOW_UNVERIFIED_PICKLE=1 must restore legacy "
        "pass-through behaviour (no sidecar -> True with WARN)."
    )


def test_verify_sidecar_passes_on_matching_digest(tmp_pickle: Path, monkeypatch):
    """Sidecar present + correct digest -> True (the happy path)."""
    from mlframe.inference.predict import _verify_sidecar

    sidecar = tmp_pickle.with_suffix(tmp_pickle.suffix + ".sha256")
    sidecar.write_text(_sha256_hex(tmp_pickle) + "  model.dump\n", encoding="utf-8")
    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    assert _verify_sidecar(str(tmp_pickle))


def test_verify_sidecar_rejects_corrupt_digest(tmp_pickle: Path, monkeypatch):
    """Sidecar present + wrong digest -> False (pre-fix behaviour preserved)."""
    from mlframe.inference.predict import _verify_sidecar

    sidecar = tmp_pickle.with_suffix(tmp_pickle.suffix + ".sha256")
    # Wrong digest.
    sidecar.write_text("0" * 64 + "  model.dump\n", encoding="utf-8")
    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    assert not _verify_sidecar(str(tmp_pickle))


def test_write_save_meta_sidecar_populates_real_bundle_sha256(tmp_pickle: Path):
    """``_write_save_meta_sidecar`` must compute and store the actual
    bundle SHA-256 (no more ``"..."  # not yet`` placeholder).
    """
    from mlframe.training.io import _write_save_meta_sidecar, _meta_sidecar_path

    _write_save_meta_sidecar(str(tmp_pickle))
    meta_path = Path(_meta_sidecar_path(str(tmp_pickle)))
    assert meta_path.is_file(), "meta sidecar must be written next to the bundle"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "bundle_sha256" in payload, "bundle_sha256 key must be present"
    expected = _sha256_hex(tmp_pickle)
    assert payload["bundle_sha256"] == expected, (
        f"bundle_sha256 must be the actual SHA-256 of the bundle file. "
        f"Got {payload['bundle_sha256']!r}, expected {expected!r}."
    )
    # Defence-in-depth: the placeholder string ``...`` must not leak through.
    assert payload["bundle_sha256"] != "...", "bundle_sha256 must not be a placeholder"
    assert len(payload["bundle_sha256"]) == 64, "SHA-256 hex digest is 64 chars"
