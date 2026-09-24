"""Which consumer read which named row set, and in what role: a ledger for split-role contracts.

Several defects compared numbers measured on the rows that had chosen them: the "never-touched" honest holdout both
dropped specs and reported the survivors' gain, the composite-vs-raw verdict was decided on the val split discovery had
selected on, and the discovery chart mixed test rows into a train-time diagnostic. Each consumer that reads a named row
set notes it here with its role (``fit``, ``select``, ``report``, ``verdict``, ``plot``); a contract test checks that the
roles never collide. The ledger records nothing unless ``MLFRAME_ROW_ROLE_LEDGER=1`` or a test sets ``_FORCED``, so
production pays one boolean check per annotated call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from mlframe.utils.env_flags import env_flag

ROLES = frozenset({"fit", "select", "report", "verdict", "plot"})


@dataclass(frozen=True)
class RowRead:
    """One annotated read: the named row set, the role it played, the consumer, and the row positions when known."""

    row_set: str
    role: str
    consumer: str
    rows: Optional[np.ndarray] = None


_LOG: list[RowRead] = []
_FORCED = False


def ledger_enabled() -> bool:
    """True when a test forced it on or ``MLFRAME_ROW_ROLE_LEDGER=1``."""
    return _FORCED or env_flag("MLFRAME_ROW_ROLE_LEDGER")


def note_rows(row_set: str, role: str, consumer: str, rows: Any = None) -> None:
    """Record that ``consumer`` read ``row_set`` in ``role``; a no-op unless the ledger is enabled."""
    if not ledger_enabled():
        return
    if role not in ROLES:
        raise ValueError(f"unknown row role {role!r}; expected one of {sorted(ROLES)}")
    _LOG.append(RowRead(str(row_set), role, str(consumer), None if rows is None else np.asarray(rows).copy()))
