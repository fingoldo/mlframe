"""A joblib metadata file is unpickled from the bytes its sidecar was checked against.

The loader used to verify the file by path and then reopen it by path, so a payload replaced between the two passed the check.
"""

from __future__ import annotations

import hashlib

import joblib

from mlframe.training.core import _metadata_loader as loader


def test_joblib_metadata_is_the_verified_payload_even_if_swapped_on_disk(tmp_path, monkeypatch):
    """A payload replaced after the check is not what gets loaded."""
    path = tmp_path / "meta.joblib"
    joblib.dump({"ok": 1}, path)
    (tmp_path / "meta.joblib.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    real_loader = loader.safe_joblib_load_bytes

    def swap_then_load(payload, filename):
        """Replace the file on disk at the moment the loader would reopen it, then load the bytes it was handed."""
        joblib.dump({"ok": 2}, filename)
        return real_loader(payload, filename)

    monkeypatch.setattr(loader, "safe_joblib_load_bytes", swap_then_load)
    assert loader.load_metadata_file(str(path), "joblib", "test") == {"ok": 1}
