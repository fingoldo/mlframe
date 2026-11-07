"""Idle-time ranking of the main-thread steps of a fit from the timeline ``rss_scale_fit.py`` prints with ``RSS_TICK_S``.

Usage: tick_utilization.py LOGFILE [TOP]. Each tick line is ``tick t=..s +.. MB  cpu NN% gpu NN% file:function:line``; the lines are grouped by module and ranked by
``ticks * (200 - cpu - gpu) / 200``: the share of the sampled wall a step spent with neither the CPUs nor the GPU busy. The steps at the top are where serial host work or
device waits sit; read the module's calls with a cProfile of the host side at the same row count before changing anything.
"""
import collections
import re
import sys

TICK = re.compile(r"\s*tick t=\s*([\d.]+)s\s+\+\s*\d+ MB\s+cpu\s+(\d+)%\s+gpu\s+(\S+)%\s+(\S+)")


def main(path: str, top: int = 14) -> None:
    """Print the ranking for ``path``."""
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = TICK.match(line)
        if m:
            rows.append((float(m.group(1)), int(m.group(2)), m.group(3), m.group(4)))
    agg: "dict[str, list[float]]" = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for _t, cpu, gpu, where in rows:
        mod = where.split(":")[0].split("/")[-1]
        a = agg[mod]
        a[0] += 1
        a[1] += cpu
        a[2] += int(gpu) if gpu.isdigit() else 0
    print(f"{len(rows)} ticks; ticks  avgCPU avgGPU  idle-score  module")
    ranked = sorted(agg.items(), key=lambda kv: -kv[1][0] * (200 - kv[1][1] / kv[1][0] - kv[1][2] / kv[1][0]))
    for mod, (n, cpu, gpu) in ranked[:top]:
        print(f"{n:4d} {cpu / n:6.0f} {gpu / n:6.0f}  {n * (200 - cpu / n - gpu / n) / 200:6.1f}  {mod}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 14)
