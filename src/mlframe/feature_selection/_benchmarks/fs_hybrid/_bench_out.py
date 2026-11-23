"""Output location for the fs_hybrid bench scripts (audit 2026-11-20, finding 12-F98).

Results used to be written to a hard-coded ``D:/Temp`` folder, which fails or misplaces output on any other host. ``bench_out(name)``
returns ``<MLFRAME_BENCH_OUT or _results next to this package>/<name>``, creating the folder on first use.
"""

from __future__ import annotations

import os

__all__ = ["bench_out"]


def bench_out(name: str) -> str:
    """Return the path for result file ``name`` inside the bench output folder, creating the folder if needed."""
    base = os.environ.get("MLFRAME_BENCH_OUT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, name)
