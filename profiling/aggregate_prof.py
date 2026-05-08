"""Aggregate ``.prof`` files from multiple ``profile_one_combo`` runs
into a cross-run hotspot report.

Why
---
``profile_one_combo --save-stats`` writes one ``.prof`` per run.
After several iterations of the optimize loop, you have N files.
This script collapses cumulative time across them so cross-combo
hotspots float to the top, AND shows per-file breakdowns so you
can see which combos dominate a given function.

Usage:
    python profiling/aggregate_prof.py --dir D:/Temp/profruns --top 30
    python profiling/aggregate_prof.py *.prof --top 50 --filter 'mlframe'
"""

from __future__ import annotations

import argparse
import glob
import os
import pstats
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


def _load_prof_files(paths: List[str]) -> List[Tuple[str, pstats.Stats]]:
    out: List[Tuple[str, pstats.Stats]] = []
    for p in paths:
        try:
            s = pstats.Stats(p)
            out.append((p, s))
        except Exception as e:
            print(f"!! failed to load {p}: {e}", file=sys.stderr)
    return out


def _aggregate(
    runs: List[Tuple[str, pstats.Stats]],
    name_filter: str = "",
) -> Tuple[
    Dict[Tuple[str, int, str], Tuple[float, float, int, Dict[str, float]]],
    int,
]:
    """Return (aggregated, n_runs) where aggregated maps function-key
    to (total_cumtime, total_tottime, total_ncalls, per_file_cumtime).
    """
    agg: Dict[Tuple[str, int, str], Tuple[float, float, int, Dict[str, float]]] = {}
    for path, stats in runs:
        for func, (cc, nc, tt, ct, _callers) in stats.stats.items():
            fname = func[0] or ""
            if name_filter and name_filter not in fname:
                continue
            if func in agg:
                ct_total, tt_total, nc_total, per_file = agg[func]
                per_file[path] = per_file.get(path, 0.0) + ct
                agg[func] = (ct_total + ct, tt_total + tt, nc_total + nc, per_file)
            else:
                agg[func] = (ct, tt, nc, {path: ct})
    return agg, len(runs)


def _print_aggregate(
    agg: Dict[Tuple[str, int, str], Tuple[float, float, int, Dict[str, float]]],
    n_runs: int, top: int = 30, show_per_file: bool = False,
):
    rows = sorted(agg.items(), key=lambda kv: kv[1][0], reverse=True)[:top]
    print(f"\nTop {top} cross-run hotspots (cumtime summed across {n_runs} runs)")
    print("=" * 100)
    print(f"{'Σ cumtime':>12} {'Σ tottime':>12} {'Σ ncalls':>12}  function")
    print("-" * 100)
    for func, (ct, tt, nc, per_file) in rows:
        fname = f"{os.path.basename(func[0] or '')}:{func[1]}({func[2]})"
        print(f"{ct:>12.3f} {tt:>12.3f} {nc:>12}  {fname}")
        if show_per_file:
            for path, ct_in_file in sorted(per_file.items(),
                                           key=lambda kv: kv[1], reverse=True):
                print(f"  {os.path.basename(path):>40s}  {ct_in_file:>10.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="*", help="Explicit .prof paths.")
    p.add_argument("--dir", type=str, default=None,
                   help="Directory containing *.prof files (recurses).")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--filter", type=str, default="",
                   help="Substring to require in the function path "
                        "(e.g. 'mlframe' to drop sklearn / numpy / numba noise).")
    p.add_argument("--per-file", action="store_true",
                   help="Show per-input-file cumtime under each hotspot.")
    args = p.parse_args()

    paths: List[str] = list(args.paths)
    if args.dir:
        paths.extend(glob.glob(os.path.join(args.dir, "**", "*.prof"),
                               recursive=True))
    if not paths:
        print("No .prof files supplied.", file=sys.stderr)
        sys.exit(1)

    runs = _load_prof_files(paths)
    if not runs:
        print("No .prof files loaded successfully.", file=sys.stderr)
        sys.exit(1)
    agg, n_runs = _aggregate(runs, name_filter=args.filter)
    _print_aggregate(agg, n_runs, top=args.top, show_per_file=args.per_file)


if __name__ == "__main__":
    main()
