"""Regression test for the training-core cProfile harness.

Asserts that running ``profile_training_core.profile`` materialises a ``.prof``
file the audit/profile review tooling can open. The actual hotspot content is
not validated here (cProfile attribution is noisy); only the artefact contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HARNESS_DIR = Path(__file__).parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

profile_training_core = pytest.importorskip("profile_training_core")


def test_profile_training_core_writes_prof_artifact(tmp_path):
    """Profile training core writes prof artifact."""
    out = tmp_path / "training_core.prof"
    written = profile_training_core.profile(n_rows=500, output_path=out, top=5)
    assert written == out
    assert out.exists(), f"cProfile harness did not write {out}"
    assert out.stat().st_size > 0, f"cProfile dump at {out} is empty"
