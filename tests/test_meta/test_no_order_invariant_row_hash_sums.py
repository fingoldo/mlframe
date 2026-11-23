"""No digest may sum per-row hashes: the sum is order-invariant, so row-permuted frames share a cache key (audit 2026-11-20, finding 40-F20).

Found while fixing the pipeline frame hash. The fix hashes the ordered per-row hash bytes; this gate stops the sum from
returning in any cache key.
"""

import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
_ORDER_INVARIANT_SUM = re.compile(r"hash_rows\(\)\.sum\(\)|\brow_hash\w*\.sum\(\)")


def test_no_source_line_sums_per_row_hashes():
    """A ``hash_rows()`` result or a row-hash variable must not be summed into a digest (40-F20)."""
    offenders = []
    for path in sorted(_SRC.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            if _ORDER_INVARIANT_SUM.search(line):
                offenders.append(f"{path.relative_to(_SRC.parent)}:{lineno}: {line.strip()}")
    assert not offenders, "per-row hashes summed into a digest (order-invariant):\n  " + "\n  ".join(offenders)
