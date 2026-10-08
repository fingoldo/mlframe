"""A failed hardware probe must not be persisted: a transient driver fault would pin the host to CPU forever."""
from __future__ import annotations

from pathlib import Path

from py_ci_shared.persisted_negative_probe import assert_persisted_negative_probe

SRC = Path(__file__).resolve().parents[2] / "src" / "mlframe"


def test_no_persisted_negative_hardware_probe():
    """No broad except handler persists a negative GPU/device probe result."""
    assert_persisted_negative_probe(SRC, min_files=500)
