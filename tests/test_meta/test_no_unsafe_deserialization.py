"""Unpickling and other code-executing loads in src/mlframe must be allowlisted, verified or marked."""
from __future__ import annotations

from pathlib import Path

from py_ci_shared.unsafe_deserialization import assert_unsafe_deserialization

SRC = Path(__file__).resolve().parents[2] / "src" / "mlframe"


def test_no_unsafe_deserialization_in_package():
    """No unguarded pickle/dill/torch.load/yaml.load in the package (benchmarks excluded)."""
    assert_unsafe_deserialization(SRC, min_files=500, exclude=("__pycache__", "/_benchmarks/"))
