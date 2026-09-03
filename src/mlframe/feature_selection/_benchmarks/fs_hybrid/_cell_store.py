"""Resumable JSONL result store: one object per cell, append-only, durable.

Every cell -- including a failed one -- writes exactly one record. A cell is never silently skipped: a
crash writes its status and its traceback tail, so `reliability` can be computed later from the file
itself rather than from the absence of rows (complete-case aggregation over a grid where the hardest
scenarios kill the weakest arms is textbook survivorship bias).

Durability matters because the file *is* the resume state. Each append is followed by `flush()` and
`os.fsync()`, so a killed process loses at most the record it was mid-write on, and the loader tolerates
that one truncated trailing line.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Union

logger = logging.getLogger(__name__)

__all__ = ["JsonlCellStore"]

PathLike = Union[str, "os.PathLike[str]"]


class JsonlCellStore:
    """Append-only JSONL store keyed by `cell_key`, supporting resume and crash-tolerant reads."""

    def __init__(self, path: PathLike) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        """Yield every well-formed record in the file, skipping a truncated trailing line."""
        if not self.path.exists():
            return
        text = self.path.read_bytes().decode("utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                logger.warning("dropping unparseable JSONL line %d of %s (partial write?)", lineno, self.path)
                continue
            if isinstance(obj, dict):
                yield obj

    def load(self) -> List[Dict[str, Any]]:
        """Return every well-formed record as a list."""
        return list(self.iter_records())

    def completed_keys(self, statuses: Optional[Set[str]] = None) -> Set[str]:
        """Return the `cell_key`s already present, restricted to `statuses` when given.

        Resume defaults to skipping *any* recorded cell, failures included: re-running a cell that
        deterministically crashes only re-pays its cost. Pass `statuses={"ok"}` to retry failures.
        """
        out: Set[str] = set()
        for rec in self.iter_records():
            key = rec.get("cell_key")
            if not isinstance(key, str):
                continue
            if statuses is not None and rec.get("status") not in statuses:
                continue
            out.add(key)
        return out

    def append(self, record: Dict[str, Any]) -> None:
        """Append one record and force it to disk before returning."""
        with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record, sort_keys=True, separators=(",", ":"), default=str))
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
