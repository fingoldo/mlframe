"""A fixed workload timed next to every cell, so wall-clock means something across machines and months.

A wall-clock second is not a unit. It depends on the host, on how many of this machine's sixteen cores
another session is using, and on what the CPU's boost clock happened to be doing. Two numbers measured a
month apart are not comparable, and on this host -- which routinely runs over a hundred python processes --
two numbers measured an hour apart often are not either.

The anchor is the cheapest available fix: run a FIXED, deterministic, dependency-free workload immediately
next to the arm and record how long it took. The ratio ``wall_time_s / anchor_s`` is then in units of "this
machine right now", and a report can compare ratios where it must refuse to compare seconds.

What the workload has to be, and why this one:

* **deterministic** -- the same instructions every time, so a difference in the anchor is a difference in
  the machine and never in the work. Fixed shapes, a fixed seed, no data-dependent branching.
* **single-threaded** -- otherwise the anchor measures thread availability rather than speed, and it would
  compete with the very fit it is trying to calibrate. BLAS is explicitly avoided for this reason: a matmul
  silently fans out across cores and its thread count is set by whichever library loaded first.
* **short** -- paid once per cell, so tens of milliseconds, not seconds.
* **not cached or memoized** -- it must actually run each time.

The anchor is advisory in exactly the way wall-clock is advisory. It corrects for machine speed, not for
contention *during* the arm's own fit: an arm that ran while eight other processes woke up is still
mis-timed, and no single measurement before it can know that. What the anchor gives is the ability to say
so, by recording the machine's state at a known instant rather than assuming it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ANCHOR_VERSION", "AnchorReading", "measure_anchor"]

#: Bump when the workload itself changes. Anchors from two different versions measure different amounts of
#: work, so a ratio computed across a bump is meaningless -- and an aggregate that silently mixed them
#: would look like a hardware change.
ANCHOR_VERSION = 1

#: Sized so one pass lands in the tens of milliseconds on a current desktop core. Small enough that paying
#: it on every cell is negligible next to any arm, large enough that the clock's own resolution and the
#: interpreter's dispatch overhead are not what is being measured.
_ROWS = 20_000
_COLS = 24
_PASSES = 3


def _workload() -> float:
    """Run the fixed workload and return a checksum, which exists only to defeat dead-code elimination.

    Sorting and a cumulative reduction rather than a matmul: both stay on one thread under every BLAS this
    repository might be installed against, and both are memory-bound in the way a real fit's inner loops
    are, which makes the anchor track the thing that actually varies between hosts.
    """
    rng = np.random.default_rng(0)
    data = rng.standard_normal((_ROWS, _COLS))
    checksum = 0.0
    for _ in range(_PASSES):
        ordered = np.sort(data, axis=0)
        checksum += float(np.cumsum(ordered[:, 0])[-1])
        data = ordered * 1.0000001
    return checksum


@dataclass(frozen=True)
class AnchorReading:
    """One anchor measurement, carried alongside a cell's own timings."""

    anchor_s: float
    anchor_process_s: float
    anchor_version: int = ANCHOR_VERSION

    def as_dict(self) -> Dict[str, Any]:
        """Return the record shape stored on a cell."""
        return {"anchor_s": round(self.anchor_s, 4), "anchor_process_s": round(self.anchor_process_s, 4), "anchor_version": int(self.anchor_version)}


def measure_anchor() -> AnchorReading:
    """Run the fixed workload once and return how long it took.

    Both clocks are recorded because they disagree in the informative direction: ``perf_counter`` includes
    time this process spent descheduled while other work ran, and ``process_time`` does not. A cell whose
    wall anchor is far above its process anchor was measured on a busy machine, and that is exactly the
    cell whose own wall-clock should not be quoted.
    """
    wall0, proc0 = time.perf_counter(), time.process_time()
    _workload()
    return AnchorReading(anchor_s=time.perf_counter() - wall0, anchor_process_s=time.process_time() - proc0)
