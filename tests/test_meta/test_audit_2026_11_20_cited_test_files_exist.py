"""Every test file the 2026-11-20 audit's tracker or dispositions name must exist (audit 2026-11-20, a gap found by a rebase).

The shared disposition checker reads backticked names and ``tests/...py`` paths only. A bare ``test_something.py`` in a table cell
slipped past it: a disposition cited a test file that was never on master. This gate reads the bare names too.
"""

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_AUDIT = _REPO / "audits" / "2026-11-20"
_CITED = re.compile(r"(?<![\w/])(test_\w+\.py)(?!\w)")


def test_bare_test_file_names_in_the_2026_11_20_audit_exist():
    """A tracker or disposition row naming ``test_x.py`` must have that file under tests/ (catches the 35-F38 class of error)."""
    names = {p.name for p in (_REPO / "tests").rglob("test_*.py")}
    missing = []
    for doc in (_AUDIT / "TRACKER.md", _AUDIT / "DISPOSITIONS.md"):
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").split("\n"), 1):
            for m in _CITED.finditer(line):
                if m.group(1) not in names:
                    missing.append(f"{doc.name}:{lineno}: {m.group(1)}")
    assert not missing, "audit rows cite test files that do not exist:\n  " + "\n  ".join(missing)
