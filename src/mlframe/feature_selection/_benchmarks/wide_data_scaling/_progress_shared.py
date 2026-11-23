"""Shared progress-log checkpoint helper for the wide_data_scaling H1/H2 bench family: independently
duplicated across those scripts, consolidated here so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

import os
import time

# Written under MLFRAME_BENCH_OUT when set, otherwise under _results next to this package; the hard-coded D: path
# aborted the first checkpoint on any host without a D: drive and created a stray "D:" directory on POSIX (13-F32).
PROG = os.path.join(os.environ.get("MLFRAME_BENCH_OUT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_results"), "synergy_scale_bench", "progress.txt")


def ck(msg: str) -> None:
    """Append a timestamped ``msg`` to the shared synergy-scale-bench progress log and echo it to stdout."""
    os.makedirs(os.path.dirname(PROG), exist_ok=True)
    with open(PROG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " | " + msg + "\n")
    print(msg, flush=True)
