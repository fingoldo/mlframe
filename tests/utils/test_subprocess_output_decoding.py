"""Child-process output is decoded as UTF-8 with replacement, not with the machine's code page (tracker 2026-11-02, ENC7-1, ENC7-2)."""

import subprocess

from mlframe.training import _gpu_state_probe as gp
from mlframe.utils import native_import_probe as nip


def test_probe_child_survives_a_child_that_writes_utf8_names(monkeypatch):
    """A child importing a module with a Cyrillic name writes UTF-8 bytes; decoding them with a legacy code page used to raise UnicodeDecodeError."""
    monkeypatch.setenv("PYTHONUTF8", "1")
    done = nip._probe_child("ИИ_no_such_module_wave7")
    assert done is not None
    assert done.returncode != 0
    assert "no_such_module_wave7" in done.stderr


def test_the_smi_queries_decode_with_replacement(monkeypatch):
    """Both nvidia-smi queries are decoded as UTF-8 with errors replaced, so a non-ASCII or odd byte in a process path cannot raise."""
    seen = []

    def fake_run(cmd, **kwargs):
        """Record the decoding arguments and return an empty result."""
        seen.append((kwargs.get("encoding"), kwargs.get("errors")))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(gp, "_resolve_smi", lambda: "nvidia-smi")
    monkeypatch.setattr(gp.subprocess, "run", fake_run)
    gp._probe_smi()
    assert seen == [("utf-8", "replace"), ("utf-8", "replace")]
