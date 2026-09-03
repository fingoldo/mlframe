"""Shared progress-log checkpoint helper for the wide_data_scaling H1/H2 bench family: independently
duplicated across those scripts, consolidated here so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

import os
import time

PROG = r"D:/Temp/synergy_scale_bench/progress.txt"


def ck(msg: str) -> None:
    """Append a timestamped ``msg`` to the shared synergy-scale-bench progress log and echo it to stdout."""
    os.makedirs(os.path.dirname(PROG), exist_ok=True)
    with open(PROG, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " | " + msg + "\n")
    print(msg, flush=True)
