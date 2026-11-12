"""The tiny-cupy triage tables (``audits/2026-11-09/tiny_cupy_{high,low}_triage.md``) stay in step with the detector.

Every function the detector reports at a severity has exactly one row in that severity's table, no row is left open, and a row whose loop has been fixed (it is no longer
reported) is removed with the baseline entry, so a table never lists a stale or unread loop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_meta.test_no_tiny_cupy_ops_in_python_loops import _current_hits

_AUDIT = Path(__file__).resolve().parents[2] / "audits" / "2026-11-09"
_TABLES = (("high", _AUDIT / "tiny_cupy_high_triage.md"), ("low", _AUDIT / "tiny_cupy_low_triage.md"))
_DISPOSITIONS = ("FALSE POSITIVE", "FALLBACK", "REAL", "LEGACY")


def _rows(table: Path) -> list:
    """``(file, function, lines, disposition)`` of every row of one table."""
    rows = []
    for line in table.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| file") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append((cells[0], cells[1], cells[2], "|".join(cells[3:]).strip()))
    return rows


@pytest.mark.parametrize("severity,table", _TABLES)
def test_every_row_has_a_disposition(severity, table):
    """No row is open, and each disposition starts with one of the four kinds."""
    bad = [f"{f}::{fn}: {d[:60]}" for f, fn, _ln, d in _rows(table) if not d.startswith(_DISPOSITIONS)]
    assert not bad, f"rows without a disposition: {bad}"


@pytest.mark.parametrize("severity,table", _TABLES)
def test_table_rows_are_the_current_hits_of_its_severity(severity, table):
    """The rows are exactly the detector's functions of that severity: a new hit needs a row, a fixed one must leave the table."""
    rows = {f"src/mlframe/{f}::{fn}" for f, fn, _ln, _d in _rows(table)}
    current = {key for key, sev in _current_hits().items() if sev == severity}
    assert rows == current, f"missing rows: {sorted(current - rows)}; stale rows: {sorted(rows - current)}"
