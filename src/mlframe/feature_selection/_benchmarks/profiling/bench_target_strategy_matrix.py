"""Grid of one MRMR fit per (target kind x nbins_strategy), each in a fresh process, sequentially, with the wall, the peak RSS growth and the selected columns of every cell.

Usage: ``python bench_target_strategy_matrix.py ROWS COLS [--targets a,b] [--strategies x,y] [--out FILE.tsv]``. Every cell runs ``rss_scale_fit.py ROWS COLS all`` with ``BENCH_TARGET`` and ``BENCH_NBINS`` set, so
the numbers are those of the scale benchmark. The default grid is every target kind of the benchmark against every production ``nbins_strategy`` (``knuth`` and ``blocks`` are research-only and left out).
Nothing else may run on the machine while it does; one cell is one measurement, so read differences below ~15 % as noise unless a cell is repeated.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

TARGETS = ("binary", "multiclass", "regression", "count", "multilabel", "multitarget", "heavy_tail", "zero_inflated")
STRATEGIES = (
    "mdlp", "mdlp_validated", "fayyad_irani", "fayyad_irani_validated", "optimal_joint", "cv", "mah", "mah_sci", "sci", "marx",
    "auto", "sturges", "freedman_diaconis", "fd", "qs", "quantile", "uniform",
)
HERE = os.path.dirname(os.path.abspath(__file__))


def run_cell(rows: int, cols: int, target: str, strategy: str, timeout_s: float) -> dict:
    """One fit in a fresh process: ``{"wall_s", "status", "growth_mb", "selected"}``."""
    env = dict(os.environ, BENCH_TARGET=target, BENCH_NBINS=strategy, PYTHONUNBUFFERED="1")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, os.path.join(HERE, "rss_scale_fit.py"), str(rows), str(cols), "all"], env=env, capture_output=True, text=True, timeout=timeout_s, check=False)
        out = proc.stdout + proc.stderr
        status = "ok" if proc.returncode == 0 else f"rc{proc.returncode}"
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        status = "timeout"
    wall = time.perf_counter() - t0
    sel = re.search(r"selected (\d+) columns, sum (\d+)", out)
    growth = re.search(r"peak RSS growth[^0-9]*([0-9.]+)\s*MB", out)
    for marker in ("ABORTED", "Traceback", "ValueError", "KeyError"):
        if marker in out and status == "ok":
            status = marker.lower()
    eng = re.search(r"engineered (\d+) features", out)
    return {"wall_s": round(wall, 1), "status": status, "growth_mb": growth.group(1) if growth else "", "selected": f"{sel.group(1)}/{sel.group(2)}" if sel else "", "engineered": eng.group(1) if eng else ""}


def main() -> None:
    """Run the grid and print one tab-separated line per cell."""
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", type=int)
    ap.add_argument("cols", type=int)
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    lines = ["target\tstrategy\twall_s\tstatus\tgrowth_mb\tselected(count/sum)	engineered"]
    print(lines[0], flush=True)
    for target in args.targets.split(","):
        for strategy in args.strategies.split(","):
            r = run_cell(args.rows, args.cols, target, strategy, args.timeout_s)
            line = f"{target}\t{strategy}\t{r['wall_s']}\t{r['status']}\t{r['growth_mb']}\t{r['selected']}	{r['engineered']}"
            print(line, flush=True)
            lines.append(line)
            if args.out:
                with open(args.out, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
