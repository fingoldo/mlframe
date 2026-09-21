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

import portalocker

logger = logging.getLogger(__name__)

__all__ = ["JsonlCellStore", "SCHEMA_VERSION", "SchemaVersionMismatchError"]

PathLike = Union[str, "os.PathLike[str]"]

#: Stamped on every record. Bump it when the MEANING of a field changes -- a renamed key, a metric that
#: starts being computed differently, a status that stops meaning what it did. Adding a new key does not
#: need a bump: a reader that does not know the key ignores it, and an older record simply lacks it.
SCHEMA_VERSION = 1

#: A record written before versioning existed carries the shape version 1 describes, so it reads as 1
#: rather than as unknown. That is a claim about this repository's own history, not a general rule, and it
#: is why the constant is not simply defaulted at read time in the mismatch check below.
_UNVERSIONED_IS = 1

#: Generous, because the contended side of this lock is one short append: a wait longer than this means a
#: holder died without releasing rather than a queue, and failing the cell records that loudly.
_LOCK_TIMEOUT_S = 60.0


class SchemaVersionMismatchError(RuntimeError):
    """Raised when one results file mixes record schema versions.

    Resume FAILS here rather than warning. The two halves of such a file disagree about what a field
    means, so every aggregate over it is a silent average of two different quantities -- and the resume
    path is precisely where nobody is watching the output.
    """


class JsonlCellStore:
    """Append-only JSONL store keyed by `cell_key`, supporting resume and crash-tolerant reads."""

    def __init__(self, path: PathLike) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def schema_versions(self) -> Set[int]:
        """Return every schema version present in the file, unversioned records counting as version 1."""
        seen: Set[int] = set()
        for rec in self.iter_records():
            raw = rec.get("schema_version", _UNVERSIONED_IS)
            try:
                seen.add(int(raw))
            except (TypeError, ValueError):
                raise SchemaVersionMismatchError(f"{self.path} contains a record whose schema_version is not an integer: {raw!r}") from None
        return seen

    def assert_single_schema_version(self) -> int:
        """Return the file's one schema version, or raise when it mixes several.

        Raises:
            SchemaVersionMismatchError: When the file holds more than one version, or a version this code does
                not know how to read. Both cases make every aggregate over the file a mixture, so they
                stop the run instead of annotating it.
        """
        seen = self.schema_versions()
        if not seen:
            return SCHEMA_VERSION
        if len(seen) > 1:
            raise SchemaVersionMismatchError(f"{self.path} mixes record schema versions {sorted(seen)}; a resume across a schema change would average two different quantities")
        only = next(iter(seen))
        if only > SCHEMA_VERSION:
            raise SchemaVersionMismatchError(f"{self.path} was written by schema version {only}, but this code reads at most {SCHEMA_VERSION}")
        return int(only)

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
        """Return one record per cell, last write winning.

        The file is append-only, so a retried cell appears twice -- once failed, once succeeded. Returning
        both would let a cell be counted twice by anything that aggregates over records: reliability would
        read 20 of 28 instead of 20 of 20, charging an arm for a failure that was subsequently fixed.
        """
        latest: Dict[str, Dict[str, Any]] = {}
        ordered: List[Dict[str, Any]] = []
        for rec in self.iter_records():
            key = rec.get("cell_key")
            if not isinstance(key, str):
                ordered.append(rec)
                continue
            if key not in latest:
                ordered.append(rec)
            latest[key] = rec
        return [latest.get(str(r.get("cell_key")), r) for r in ordered]

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
        """Append one record, stamped with the schema version, and force it to disk before returning.

        The stamp is applied to a COPY: a caller reusing its record dict for a retry would otherwise find
        the version silently added to the object it still holds, which is the kind of action at a distance
        that makes a retried cell differ from a fresh one for no stated reason.
        """
        record = {**record, "schema_version": int(record.get("schema_version", SCHEMA_VERSION))}
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str) + "\n"
        # Locked for the whole write, because the runner's worker pool has several processes appending to
        # one file. A single `write` of a few kilobytes is not atomic on Windows, and two interleaved
        # records produce one unparseable line, which `iter_records` then drops as a partial write. That
        # loses a COMPLETED cell silently and makes it look like it was never run.
        with portalocker.Lock(self.path, mode="a", encoding="utf-8", newline="\n", flags=portalocker.LOCK_EX | portalocker.LOCK_NB, timeout=_LOCK_TIMEOUT_S) as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
