"""When the HuggingFace lock directory cannot be created, the lock lives in the per-user private directory, not the shared temp root.

A lock path directly in the shared temp root is predictable from the model name, so another local user could pre-create it.
"""

from __future__ import annotations

import os
import tempfile

from mlframe.training.feature_handling import hf_provider


def test_unwritable_lock_root_falls_back_to_the_private_directory(monkeypatch, tmp_path):
    """With the lock directory refused, the lock path sits inside a private ``mlframe-hf-locks`` directory, not in the temp root."""
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    real_makedirs = os.makedirs

    def _refuse_lock_dir(path, *args, **kwargs):
        """Refuse the lock directory only, as a read-only home would."""
        if str(path).endswith(".mlframe-fhc-locks"):
            raise OSError("read-only")
        return real_makedirs(path, *args, **kwargs)

    monkeypatch.setattr(hf_provider.os, "makedirs", _refuse_lock_dir)
    lock = hf_provider._hf_cache_lock_path("org/model", None)
    lock_dir = os.path.dirname(lock)
    assert lock_dir != tempfile.gettempdir()
    assert "mlframe-hf-locks" in lock_dir
    assert os.path.basename(lock).startswith("hf_")
