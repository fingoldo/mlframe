"""Idle windows of a ``rss_scale_fit`` timeline: the stretches where the CPU and the GPU are both under-used, with the step the main thread was in.

Run ``python -m mlframe.feature_selection._benchmarks.profiling.timeline_idle_windows LOG [LOG2]`` on logs written with ``RSS_TICK_S`` set (a log of a run the RAM guard stopped has no timeline). A window is a run of
consecutive ticks with ``cpu <= CPU_MAX`` and ``gpu <= GPU_MAX`` (defaults 40 and 30 percent); the report lists its start, length and the most frequent step. With two logs the windows of the second are matched to the first by step,
so a fix that closed a window shows as a shorter or missing one.
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter
from pathlib import Path

_TICK = re.compile(r"^\s*tick t=\s*([\d.]+)s\s*\+\s*(-?\d+) MB\s+cpu\s+(\d+)%\s+gpu\s+(\d+)%\s+(\S+)")
CPU_MAX = int(os.environ.get("IDLE_CPU_MAX", "40"))
GPU_MAX = int(os.environ.get("IDLE_GPU_MAX", "30"))


def read_ticks(path: Path) -> list[tuple[float, int, int, int, str]]:
    """``(t, rss_growth_mb, cpu, gpu, step)`` of every tick line of the log."""
    ticks = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _TICK.match(line)
        if m:
            ticks.append((float(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)))
    return ticks


def idle_windows(ticks: list[tuple[float, int, int, int, str]]) -> list[dict]:
    """Runs of consecutive idle ticks: start, end, number of ticks and the most frequent step."""
    windows: list[dict] = []
    run: list[tuple[float, int, int, int, str]] = []
    for tick in [*ticks, (float("inf"), 0, 100, 100, "")]:
        if tick[2] <= CPU_MAX and tick[3] <= GPU_MAX:
            run.append(tick)
            continue
        if run:
            step, _ = Counter(t[4].split("/")[-1] for t in run).most_common(1)[0]
            windows.append({"start": run[0][0], "end": run[-1][0], "ticks": len(run), "step": step})
            run = []
    return windows


def main(argv: list[str]) -> int:
    """Print the idle windows of each log given."""
    for arg in argv[1:3]:
        ticks = read_ticks(Path(arg))
        wins = idle_windows(ticks)
        total = sum(w["end"] - w["start"] for w in wins)
        print(f"{arg}: {len(ticks)} ticks, {len(wins)} idle windows (cpu <= {CPU_MAX}%, gpu <= {GPU_MAX}%), about {total:.0f} s in them")
        for w in sorted(wins, key=lambda w: -(w["end"] - w["start"]))[:10]:
            print(f"  t={w['start']:7.1f}..{w['end']:7.1f} s ({w['ticks']} ticks)  {w['step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
