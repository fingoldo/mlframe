"""The bootstrap-rank tuning sweep puts back the device override a user set, instead of removing it.

The sweep forces the device path while it times both walks. It used to pop the variable on exit, so a user's ``=0`` was gone for the rest of the
process.
"""

from __future__ import annotations

import pytest

from mlframe.feature_selection.filters import _bootstrap_rank_mi_gpu as gpu_mod

_NAME = "MLFRAME_BOOTSTRAP_RANK_DEVICE"


def test_forced_env_restores_a_value_the_user_had_set(monkeypatch):
    """A value set before the sweep is the value after it, not unset."""
    monkeypatch.setenv(_NAME, "0")
    with gpu_mod._forced_env(_NAME, "1"):
        assert gpu_mod.os.environ[_NAME] == "1"
    assert gpu_mod.os.environ[_NAME] == "0"


def test_forced_env_removes_the_variable_when_it_was_unset(monkeypatch):
    """A variable that was unset before the sweep is unset again after it."""
    monkeypatch.delenv(_NAME, raising=False)
    with gpu_mod._forced_env(_NAME, "1"):
        pass
    assert _NAME not in gpu_mod.os.environ


def test_forced_env_restores_on_an_exception(monkeypatch):
    """A failing timing run still puts the user's value back."""
    monkeypatch.setenv(_NAME, "0")
    with pytest.raises(RuntimeError):
        with gpu_mod._forced_env(_NAME, "1"):
            raise RuntimeError("boom")
    assert gpu_mod.os.environ[_NAME] == "0"
