"""Stage walls of two runs side by side: ``python ledger_compare.py OLD.ledger.jsonl NEW.ledger.jsonl [TOP]`` (the ``BENCH_LEDGER`` files of ``rss_scale_fit.py``).

The wall of a 500000 x 1000 fit moves about 15 s between two runs of equal code, so a change is judged by the walls of the stages it touches, and the stages that moved without it are the noise of the pair.
Each line is ``new - old``, the two walls, and the name; a name that occurs in several windows (a family called by many decisions) is summed.
"""

from __future__ import annotations

import collections
import json
import sys


def stage_walls(path: str) -> "dict[str, float]":
    """``{name: seconds}`` summed over the windows of every ledger row of ``path``."""
    walls: dict[str, float] = collections.defaultdict(float)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                walls[row["name"]] += row["t1"] - row["t0"]
    return dict(walls)


def compare(old: "dict[str, float]", new: "dict[str, float]", top: int) -> "list[str]":
    """The ``top`` names whose wall moved the most, as report lines."""
    rows = sorted(((new.get(k, 0.0) - old.get(k, 0.0), k, old.get(k, 0.0), new.get(k, 0.0)) for k in set(old) | set(new)), key=lambda r: -abs(r[0]))
    return [f"{delta:+7.1f}  {name:55s} {a:7.1f} -> {b:7.1f}" for delta, name, a, b in rows[:top]]


def main(argv: "list[str] | None" = None) -> int:
    """Command line entry."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 2:
        print(__doc__)
        return 2
    top = int(args[2]) if len(args) > 2 else 25
    print("\n".join(compare(stage_walls(args[0]), stage_walls(args[1]), top)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
