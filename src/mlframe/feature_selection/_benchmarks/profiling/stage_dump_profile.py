"""What the MainThread of a fit was doing, by stage: the dumps of ``dump_main_thread.sh`` laid over the windows of the stage ledger (``BENCH_LEDGER`` of ``rss_scale_fit.py``).

``py-spy dump`` of the MainThread every 2 s (``dump_main_thread.sh PID OUT 2``) gives one line per sample: the epoch seconds, then the mlframe frames innermost first joined by `` > ``. The stage ledger
(``<prefix>.ledger.jsonl``) has the epoch window of every stage. Laying one over the other answers "which frame is the MainThread in while stage X runs" without a profiler's 20 to 30 percent overhead:

* ``python stage_dump_profile.py LEDGER.jsonl DUMP.txt --stage stage:screen_predictors`` - the innermost frames and the commonest short call chains inside the windows of one stage;
* ``python stage_dump_profile.py LEDGER.jsonl DUMP.txt --whole`` - the same over the whole fit (what the MainThread waited for the most);
* ``python stage_dump_profile.py LEDGER.jsonl DUMP.txt --between 398 415`` - the window of seconds since the start of the ledger (a stretch with no stage open).

Dumps are 2 s apart: a frame seen in k dumps accounts for about 2k seconds, with the noise of a sampler. A worker thread of a pool is not in the dump (its stack does not contain the caller): the MainThread's wait for
the pool shows as the pool's ``map`` frame.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from collections.abc import Iterable


def read_windows(ledger: str, stage: "str | None") -> "list[tuple[float, float]]":
    """The ``(t0, t1)`` epoch windows of the ledger rows named ``stage`` (every window of positive length when ``stage`` is None)."""
    windows = []
    with open(ledger, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if (stage is None or row["name"] == stage) and row["t1"] > row["t0"]:
                windows.append((row["t0"], row["t1"]))
    return windows


def ledger_start(ledger: str) -> float:
    """The epoch second of the first ledger row (the origin of the ``--between`` seconds)."""
    with open(ledger, encoding="utf-8") as handle:
        return min(json.loads(line)["t0"] for line in handle if line.strip())


def read_dumps(path: str) -> "list[tuple[float, list[str]]]":
    """``(epoch seconds, frames innermost first)`` of every dump line of ``path`` that has frames."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(" ", 1)
            try:
                t = float(parts[0])
            except ValueError:
                continue
            frames = [f.strip() for f in (parts[1] if len(parts) > 1 else "").split(" > ") if f.strip()]
            if frames:
                out.append((t, [re.sub(r"\s*\((.*?)[\\/]?([^\\/]+:\d+)\)", r" (\2)", f)[:90] for f in frames]))
    return out


def profile(dumps: "Iterable[tuple[float, list[str]]]", windows: "list[tuple[float, float]] | None", top: int) -> str:
    """The report of the dumps inside ``windows`` (all dumps when it is None): the innermost frames and the commonest chains of the three innermost frames."""
    inner: collections.Counter = collections.Counter()
    chains: collections.Counter = collections.Counter()
    hit = 0
    for t, frames in dumps:
        if windows is not None and not any(a <= t <= b for a, b in windows):
            continue
        hit += 1
        inner[frames[0]] += 1
        chains[" < ".join(frames[:3])] += 1
    lines = [f"{hit} dumps (2 s apart: about {2 * hit} s)"]
    lines += [f"  {v:3d}  {k}" for k, v in inner.most_common(top)]
    lines.append("")
    lines += [f"  {v:3d}  {k[:230]}" for k, v in chains.most_common(5)]
    return "\n".join(lines)


def main(argv: "list[str] | None" = None) -> int:
    """Command line entry: print the report for a stage, a stretch of the ledger or the whole fit."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("ledger")
    parser.add_argument("dump")
    parser.add_argument("--stage", help="the ledger name of a stage, e.g. stage:fe_step")
    parser.add_argument("--between", nargs=2, type=float, metavar=("FROM_S", "TO_S"), help="seconds since the first ledger row")
    parser.add_argument("--whole", action="store_true", help="every dump of the file")
    parser.add_argument("--top", type=int, default=14)
    args = parser.parse_args(argv)
    dumps = read_dumps(args.dump)
    if args.whole:
        windows = None
    elif args.between:
        origin = ledger_start(args.ledger)
        windows = [(origin + args.between[0], origin + args.between[1])]
    elif args.stage:
        windows = read_windows(args.ledger, args.stage)
    else:
        parser.error("give --stage, --between or --whole")
    print(profile(dumps, windows, args.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
