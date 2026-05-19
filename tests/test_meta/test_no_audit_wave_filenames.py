"""Meta-linter: forbid audit-wave filenames in tests/.

Filenames like ``test_audit_2026_05_18_loop_47_*`` or
``test_round17_valonly_null_detection`` carry process metadata that belongs
in git history, not on disk. See feedback_no_audit_phase_in_comments.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parent.parent

# Any test_*.py whose stem matches one of these patterns is rejected.
FORBIDDEN_PATTERNS = [
    re.compile(r"^test_audit_\d{4}_"),
    re.compile(r"^test_round\d+_"),
    re.compile(r"^test_prod_log_fixes_\d{4}_"),
    re.compile(r"^test_jolly_wishing_deer_"),
]


def _iter_test_files() -> list[Path]:
    return [
        p
        for p in TESTS_ROOT.rglob("test_*.py")
        if "__pycache__" not in p.parts and p.name != Path(__file__).name
    ]


def test_no_audit_wave_filenames() -> None:
    offenders: list[str] = []
    for path in _iter_test_files():
        stem = path.stem
        for pat in FORBIDDEN_PATTERNS:
            if pat.match(stem):
                offenders.append(str(path.relative_to(TESTS_ROOT)))
                break
    assert not offenders, (
        "Audit-wave filenames must be renamed to topic-canonical names. "
        f"Offenders ({len(offenders)}): {offenders[:10]}"
    )
