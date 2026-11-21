"""Audit 2026-11-20 finding 28-F25: a save whose bundle cannot be digested fails, rather than writing an unverifiable sidecar.

Before the fix the sidecar was written without bundle_sha256, so the save reported success and every later load raised. The test makes the
digest fail and checks the sidecar writer raises and writes nothing.
"""

import pytest

from mlframe.training import io as io_module


def test_unreadable_bundle_fails_the_save_and_writes_no_sidecar(tmp_path, monkeypatch):
    """With the digest unavailable the writer raises RuntimeError and leaves no sidecar file behind."""
    bundle = tmp_path / "model.bundle"
    bundle.write_bytes(b"payload")
    monkeypatch.setattr(io_module, "_bundle_sha256", lambda path: None)
    with pytest.raises(RuntimeError, match="sha256"):
        io_module._write_save_meta_sidecar(str(bundle))
    assert not any(p.name.endswith(".meta.json") or "meta" in p.name for p in tmp_path.iterdir() if p != bundle)
