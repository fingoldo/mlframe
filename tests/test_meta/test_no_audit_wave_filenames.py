"""Test files are named after what they cover, not after the audit wave that produced them.

`py_ci_shared.audit_wave_filenames` holds the shared patterns; git history keeps the process metadata a
name like `test_wave97_*` would otherwise carry.
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.audit_wave_filenames import assert_no_new_audit_wave_filenames

TESTS_DIR = Path(__file__).resolve().parents[1]

# Process tags seen only in this repository, on top of the shared set.
_EXTRA_PATTERNS = (r"^test_jolly_wishing_deer_", r"^test_low_polish_core_low_polish_core")


def test_no_audit_wave_filenames() -> None:
    """No test file is named after the audit wave that produced it."""
    assert_no_new_audit_wave_filenames(TESTS_DIR, extra_patterns=_EXTRA_PATTERNS)
