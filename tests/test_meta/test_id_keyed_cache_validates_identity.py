"""A cache keyed by id() must validate the object identity on a hit, since ids are reused after garbage collection."""
from __future__ import annotations

from pathlib import Path

from py_ci_shared.id_keyed_cache_validates_identity import assert_id_keyed_cache_validates_identity

SRC = Path(__file__).resolve().parents[2] / "src" / "mlframe"


def test_id_keyed_caches_validate_identity():
    """Every id()-keyed cache in the package checks identity (weakref or the object itself) on lookup."""
    assert_id_keyed_cache_validates_identity(SRC, min_files=500)
