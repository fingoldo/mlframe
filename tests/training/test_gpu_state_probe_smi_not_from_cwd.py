"""``nvidia-smi`` is never taken from the working directory (SUB2-02).

On Windows a bare command name is searched in the current directory first, so a planted ``nvidia-smi.exe`` there ran on every GPU probe.
The lookup now uses absolute PATH entries only.
"""

import os

import pytest

from mlframe.training import _gpu_state_probe as probe


def test_planted_smi_in_the_working_directory_is_not_resolved(tmp_path, monkeypatch):
    """With only a relative PATH entry available, the lookup refuses rather than returning the planted binary."""
    planted = tmp_path / ("nvidia-smi.exe" if os.name == "nt" else "nvidia-smi")
    planted.write_text("#!/bin/sh\necho planted\n", encoding="utf-8")
    planted.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", os.pathsep.join(["", "relative_dir"]))
    with pytest.raises(FileNotFoundError):
        probe._resolve_smi()


def test_absolute_path_entry_is_still_searched(tmp_path, monkeypatch):
    """A real binary in an absolute PATH directory is found, so the probe still works where nvidia-smi is installed."""
    tool = tmp_path / ("nvidia-smi.exe" if os.name == "nt" else "nvidia-smi")
    tool.write_text("#!/bin/sh\n", encoding="utf-8")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert os.path.samefile(probe._resolve_smi(), str(tool))
