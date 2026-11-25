"""Cluster 40 regression: the dimensionality-reduction builder carries no unused importlib.util import (audit 2026-11-20, finding 40-F32).

Source-level pin: the UMAP branch of _build_dim_reducer no longer imports importlib.util as _ilu, which nothing used.
"""

from pathlib import Path

from mlframe.training import pipeline as pl

_SRC = Path(pl.__file__).resolve()


def test_unused_ilu_import_is_removed():
    """The unused _ilu import is gone from the pipeline package (40-F32)."""
    assert "import importlib.util as _ilu" not in _SRC.read_text(encoding="utf-8")


def test_build_dim_reducer_still_present():
    """Control: the builder remains and is callable (40-F32)."""
    assert callable(pl._build_dim_reducer)
