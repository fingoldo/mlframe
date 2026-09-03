"""Cell identity and status vocabulary for the Phase 0 pre-registered protocol.

A *cell* is one `(scenario, arm, dataset_seed, cv_seed)` unit of work. Its identity is a sha256 over a
canonical JSON encoding of the cell's full specification plus the protocol version, so that a resumed run
skips exactly the cells it already finished and re-runs everything whose definition moved.

`json.dumps(..., sort_keys=True)` is load-bearing: without it the digest depends on dict insertion order,
two processes building the same spec in different orders disagree, and a resume silently re-runs the grid.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

__all__ = [
    "PROTOCOL_VERSION",
    "CELL_STATUSES",
    "TERMINAL_FAILURE_STATUSES",
    "CellSpec",
    "canonical_json",
    "cell_key",
    "classify_exception",
]

PROTOCOL_VERSION = "phase0-v1"

# `ok` means every declared downstream score was produced. The four failure statuses are kept distinct
# because the pre-registration scores them as informative, not as missing-at-random: an arm that dies on
# `p > n` is a worse arm than one that returns garbage, and `oom`/`timeout` are properties of the arm's
# resource behaviour rather than of its ranking.
CELL_STATUSES: Tuple[str, ...] = ("ok", "error", "timeout", "crashed", "oom")
TERMINAL_FAILURE_STATUSES: Tuple[str, ...] = ("error", "timeout", "crashed", "oom")


def canonical_json(payload: Any) -> str:
    """Encode `payload` as JSON with sorted keys and no incidental whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class CellSpec:
    """The full identity of one benchmark cell; everything that can change its result belongs here."""

    scenario: str
    arm: str
    dataset_seed: int
    cv_seed: int
    protocol_version: str = PROTOCOL_VERSION
    config: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return the spec as a plain dict suitable for JSON encoding."""
        return asdict(self)

    def key(self) -> str:
        """Return this cell's stable sha256 identity."""
        return cell_key(self.as_dict())


def cell_key(spec: Dict[str, Any]) -> str:
    """Return the sha256 hex digest of the canonical JSON encoding of `spec`."""
    return hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()


def classify_exception(exc: BaseException) -> str:
    """Map a raised exception onto one of `CELL_STATUSES` (never `ok`)."""
    if isinstance(exc, MemoryError):
        return "oom"
    if isinstance(exc, TimeoutError):
        return "timeout"
    text = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in text or "cannot allocate" in text or "paging file" in text or "winerror 1455" in text:
        return "oom"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if isinstance(exc, (SystemError, OSError)) and "access violation" in text:
        return "crashed"
    return "error"
