"""The Hugging Face load lock lives in the user's cache, not the shared temp directory by default (PRV3-2).

A predictable lock path under the shared temp dir can be pre-created by another local user. The default root is now under the home directory.
"""

import os
import tempfile

import pytest

from mlframe.training.feature_handling import hf_provider


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point the home directory at a temp dir and clear the HF cache variables that would override the default."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    return tmp_path


def test_default_lock_path_is_under_the_user_home_not_the_shared_temp_dir(isolated_home):
    """With no HF cache variable set, the lock file is created under the user's home cache."""
    path = hf_provider._hf_cache_lock_path("some-model", None)
    assert path.startswith(str(isolated_home))
    assert not path.startswith(os.path.join(tempfile.gettempdir(), "mlframe-hf-locks"))
