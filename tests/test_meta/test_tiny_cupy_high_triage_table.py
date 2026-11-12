"""The high-severity tiny-cupy triage table (``audits/2026-11-09/tiny_cupy_high_triage.md``) stays in step with the detector.

Every function the detector reports at high severity has exactly one row, no row is left open, and a row whose loop has been fixed (it is no longer reported) is
removed with the baseline entry, so the table never lists a stale or unread loop.
"""

from __future__ import annotations

from pathlib import Path

from tests.test_meta.test_no_tiny_cupy_ops_in_python_loops import _current_hits

TABLE = Path(__file__).resolve().parents[2] / "audits" / "2026-11-09" / "tiny_cupy_high_triage.md"
_DISPOSITIONS = ("FALSE POSITIVE", "FALLBACK", "REAL", "LEGACY")


def _rows() -> list:
    """``(file, function, lines, disposition)`` of every table row."""
    rows = []
    for line in TABLE.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| file") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append((cells[0], cells[1], cells[2], "|".join(cells[3:]).strip()))
    return rows


def test_every_row_has_a_disposition():
    """No row is open, and each disposition starts with one of the four kinds."""
    bad = [f"{f}::{fn}: {d[:60]}" for f, fn, _ln, d in _rows() if not d.startswith(_DISPOSITIONS)]
    assert not bad, f"rows without a disposition: {bad}"


def test_table_rows_are_the_current_high_severity_hits():
    """The rows are exactly the detector's high-severity functions: a new high hit needs a row, a fixed one must leave the table."""
    table = {f"src/mlframe/{f}::{fn}" for f, fn, _ln, _d in _rows()}
    current = {key for key, severity in _current_hits().items() if severity == "high"}
    assert table == current, f"missing rows: {sorted(current - table)}; stale rows: {sorted(table - current)}"
