"""Environment switches take effect when they are set after import.

These gates were read once at import, so exporting the variable after ``mlframe`` was imported, as a notebook or a launcher often does, changed
nothing. Each helper now reads its variable per call, and the module attribute stays available as a test override.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mlframe.feature_selection.filters import _neural_mi as neural_mi
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_extra_basis_fe_helpers as helpers


def test_infonet_cache_dir_follows_the_environment_set_after_import(monkeypatch, tmp_path):
    """Exporting ``MLFRAME_INFONET_CACHE`` after import moves the checkpoint directory."""
    monkeypatch.setattr(neural_mi, "_INFONET_CACHE_DIR", None)
    monkeypatch.setenv("MLFRAME_INFONET_CACHE", str(tmp_path))
    assert neural_mi._infonet_cache_dir() == Path(tmp_path)


@pytest.mark.parametrize("raw, expected", [("0", False), ("1", True)])
def test_coarse_gemv_gate_follows_the_environment_set_after_import(monkeypatch, raw, expected):
    """``MLFRAME_FOURIER_COARSE_GEMV=0`` switches the coarse GEMV path off when set after import."""
    monkeypatch.setattr(helpers, "_COARSE_GEMV", None)
    monkeypatch.setenv("MLFRAME_FOURIER_COARSE_GEMV", raw)
    assert helpers._coarse_gemv_enabled() is expected
