"""Ratchet: the MRMR feature-selection code must not gain explicit float64 uses.

The dtype of the feature-selection path comes from the top-level setting (the FE dtype, float32 by default). An explicit float64
literal fixes it, so this meta-test counts the code lines that name float64 without an ``Intentional float64`` marker, per file,
and fails if any file's count grows past the recorded baseline. The baseline only ever goes down: when a file is cleaned up, its
count is lowered in ``data/float64_baseline.json``. A family that needs float64 for correctness stays, marked on its line or on the
line above with the words ``Intentional float64`` and a reason (the test that needs it).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection"
BASELINE = Path(__file__).resolve().parent / "data" / "float64_baseline.json"
MARKER = "Intentional float64"
FLOAT64 = re.compile(r"float64")


def _counted_lines(path: Path) -> int:
    """Code lines of ``path`` that name float64 and carry no ``Intentional float64`` marker."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    n = 0
    for i, line in enumerate(lines):
        code = line.strip()
        if code.startswith("#") or not FLOAT64.search(line):
            continue
        marked = MARKER in line or (i > 0 and MARKER in lines[i - 1])
        if not marked:
            n += 1
    return n


def _current_counts() -> dict[str, int]:
    """Counts per file, keyed by the path relative to ``src/mlframe``, for the feature-selection tree (benchmarks excluded)."""
    base = ROOT.parent
    out: dict[str, int] = {}
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(base).as_posix()
        if "_benchmarks" in rel or "/tests/" in rel:
            continue
        c = _counted_lines(path)
        if c:
            out[rel] = c
    return out


def test_no_file_gains_explicit_float64_uses():
    """Each feature-selection file keeps at most its recorded count of unmarked float64 lines."""
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    current = _current_counts()
    grew = {f: (baseline.get(f, 0), c) for f, c in current.items() if c > baseline.get(f, 0)}
    assert not grew, f"explicit float64 grew in: {grew}. Use the caller's dtype, or mark an intentional family with '{MARKER}' and a reason."


def test_baseline_is_not_stale():
    """A file whose count dropped must have its baseline lowered, so the ratchet keeps tightening."""
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    current = _current_counts()
    stale = {f: (b, current.get(f, 0)) for f, b in baseline.items() if current.get(f, 0) < b}
    assert not stale, f"lower the baseline for these files (cleaned up): {stale}"
