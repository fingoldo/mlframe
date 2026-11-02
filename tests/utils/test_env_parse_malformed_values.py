"""Malformed MLFRAME_* numeric variables fall back to their defaults instead of crashing (tracker 2026-11-02 wave 10, ENV10-1..10)."""

import subprocess
import sys

import pytest


def test_family_gate_imports_with_a_malformed_cache_size():
    """MLFRAME_PERM_BLOCK_CACHE_MB='1g' used to raise ValueError while the module was imported."""
    import os

    env = {"MLFRAME_PERM_BLOCK_CACHE_MB": "1g"}

    done = subprocess.run(
        [sys.executable, "-c", "import mlframe.feature_selection.filters._family_maxt_gate as m; print(m._PERM_BLOCK_CACHE_BYTES)"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env={**os.environ, **env}, timeout=300,
    )
    assert done.returncode == 0, done.stderr[-500:]
    assert done.stdout.strip().splitlines()[-1] == str(1024 * 1024 * 1024)


def test_pysr_limit_rejects_malformed_and_non_positive(monkeypatch):
    """A float string or a non-positive value gives the default budget."""
    from mlframe.feature_engineering import bruteforce as bf

    for bad in ("2e8", "200MB", "0", "-5"):
        monkeypatch.setenv("MLFRAME_PYSR_INPUT_BYTES_LIMIT", bad)
        assert bf._pysr_bytes_limit() == bf._DEFAULT_PYSR_INPUT_BYTES_LIMIT
    monkeypatch.setenv("MLFRAME_PYSR_INPUT_BYTES_LIMIT", "12345")
    assert bf._pysr_bytes_limit() == 12345


def test_lazy_prebin_floor_falls_back(monkeypatch):
    """A non-numeric floor no longer crashes composite discovery."""
    from mlframe.training.composite.discovery import _fit

    monkeypatch.setenv("MLFRAME_DISCOVERY_LAZY_PREBIN_MIN_N", "lots")
    assert _fit._lazy_prebin_floor() == 50000


def test_vram_floor_honours_the_documented_shared_variable(monkeypatch):
    """MLFRAME_FE_GPU_MIN_FREE_MB now reaches the gpu-resident gate; the longer name still overrides it."""
    from mlframe.feature_selection.filters import _gpu_resident_shared as g

    monkeypatch.delenv("MLFRAME_FE_GPU_MIN_FREE_VRAM_MB", raising=False)
    monkeypatch.setenv("MLFRAME_FE_GPU_MIN_FREE_MB", "4096")
    assert g._gpu_min_free_vram_mb() == 4096
    monkeypatch.setenv("MLFRAME_FE_GPU_MIN_FREE_VRAM_MB", "512")
    assert g._gpu_min_free_vram_mb() == 512


@pytest.mark.parametrize("value,expected", [("0", False), ("false", False), ("", False), ("1", True), ("yes", True)])
def test_force_full_refit_parses_a_boolean_string(monkeypatch, value, expected):
    """'0' and 'false' used to read as True because bool() of a non-empty string is True."""
    from mlframe.feature_selection.filters import _usability_greedy_steps as u

    monkeypatch.setenv("_USAB_FORCE_FULL_REFIT", value)
    assert u._force_full_refit() is expected
