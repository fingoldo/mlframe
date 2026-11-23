"""Cluster 25 regression: the MI docstrings describe the bin-code range the validator accepts, not a fixed 0-14 (audit 2026-11-20, finding 25-F55).

Source-level pin: no docstring claims bin codes are 0-14, and the corrected wording that names the [0, 127] validation is present.
"""

from pathlib import Path

from mlframe.feature_selection import mi as mi_mod

_SRC = Path(mi_mod.__file__).resolve()


def test_no_docstring_claims_a_fixed_0_14_range():
    """The stale 'already binned 0-14' wording is gone from the MI module (25-F55)."""
    assert "already binned 0-14" not in _SRC.read_text(encoding="utf-8")


def test_docstrings_name_the_validated_range():
    """The corrected docstrings state the [0, 127] validation (25-F55)."""
    assert "validated against [0, 127], 25-F55" in _SRC.read_text(encoding="utf-8")
