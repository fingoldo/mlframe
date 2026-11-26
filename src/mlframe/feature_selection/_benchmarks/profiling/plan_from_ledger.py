"""The scheduling instance of a measured fit, and what the planner of ``filters/_fe_plan`` makes of it.

``python plan_from_ledger.py PREFIX [--cores 8] [--physical 4]`` reads ``PREFIX.ledger.jsonl`` (the stage ledger of ``rss_scale_fit.py``) and ``PREFIX.timeline.jsonl`` (its one-second CPU / GPU samples) and
builds the instance of the fit's top-level stages:

* the four FE cascades (early A, early B, mid A, mid B) read the same step input and are independent of each other (the code runs them together when memory allows and merges them in order), so the
  planner may overlap them; each holds the cores and the GPU share the timeline shows inside its window, and a share of the interpreter lock (``--gil``, the part of its time its Python code holds it);
* the discretisation warm-up (the supervised edge search of the raw columns) is a background task that must end before the real discretisation;
* the accuracy gate, the discretisation, the first screen, the FE step, the second screen and the tail follow each other.

It prints the sequential wall (what ran), the lower bound and the plan's makespan for several interpreter-lock shares: the planner's answer to "what would overlap buy on this machine", under the
measured demands and without interference beyond the declared resources. The measured demands are those of a window in which other work may have overlapped; the numbers bound what is possible, they do not
promise it.
"""

from __future__ import annotations

import argparse
import json
import sys

from mlframe.feature_selection.filters._fe_plan import Instance, Mode, Task, plan

CASCADES = ("stage:cascade_early_a", "stage:cascade_early_b", "stage:cascade_mid_a", "stage:cascade_mid_b")
CHAIN = ("stage:accuracy_gate", "stage:discretize_inputs", "stage:screen_predictors", "stage:fe_step", "stage:screen_predictors#2")


def read_jsonl(path: str) -> list:
    """The records of a JSON-lines file."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stage_windows(ledger: list) -> "dict[str, tuple[float, float]]":
    """The ``(t0, t1)`` window of every named top-level stage on the MainThread; a stage that occurs twice gets ``#2`` on its second window."""
    seen: dict[str, int] = {}
    out: dict[str, tuple[float, float]] = {}
    for row in sorted(ledger, key=lambda r: r["t0"]):
        name = row["name"]
        if row.get("thread_name") not in ("MainThread", None) and name not in CASCADES:
            continue
        if name not in CASCADES + CHAIN[:-1] + ("stage:discretize-warmup",) and name != "stage:categorize_dataset":
            continue
        seen[name] = seen.get(name, 0) + 1
        key = name if seen[name] == 1 else f"{name}#{seen[name]}"
        out.setdefault(key, (row["t0"], row["t1"]))
    return out


def demand_of(samples: list, window: "tuple[float, float]", logical: int) -> "tuple[float, float]":
    """``(cores busy, GPU share)`` averaged over the samples inside ``window``."""
    inside = [s for s in samples if window[0] <= s["t"] <= window[1]]
    if not inside:
        return 1.0, 0.0
    cores = sum(s["cpu"] for s in inside) / len(inside) / 100.0 * logical
    gpu = sum((s["gpus"][0]["util"] if s.get("gpus") else 0.0) for s in inside) / len(inside) / 100.0
    return cores, gpu


def build_instance(ledger: list, samples: list, *, logical: int, capacity_cores: float, gil: float) -> "tuple[Instance, float, float]":
    """``(instance, sequential wall, ledger span)`` of the fit; ``gil`` is the interpreter-lock share the cascades hold while they run."""
    windows = stage_windows(ledger)
    span = max(r["t1"] for r in ledger) - min(r["t0"] for r in ledger)
    tasks = []
    for name in CASCADES:
        if name in windows:
            cores, share = demand_of(samples, windows[name], logical)
            tasks.append(Task(name, (Mode("host", windows[name][1] - windows[name][0], {"cores": cores, "gpu": share, "gil": gil}),)))
    previous = tuple(t.name for t in tasks)
    for name in CHAIN:
        if name in windows:
            cores, share = demand_of(samples, windows[name], logical)
            tasks.append(Task(name, (Mode("host", windows[name][1] - windows[name][0], {"cores": min(cores, capacity_cores), "gpu": share, "gil": 1.0}),), previous))
            previous = (name,)
    named = sum(t.modes[0].duration for t in tasks)
    rest = max(0.0, span - named)
    tasks.append(Task("rest", (Mode("host", rest, {"cores": 1.0, "gil": 1.0}),), previous))
    return Instance(tuple(tasks), {"cores": capacity_cores, "gpu": 1.0, "gil": 1.0}), named + rest, span


def main(argv: "list[str] | None" = None) -> int:
    """Print the sequential wall, the lower bound and the plan's makespan for several interpreter-lock shares of the cascades."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("prefix")
    parser.add_argument("--cores", type=int, default=8, help="logical cores of the host")
    parser.add_argument("--physical", type=float, default=4.0, help="physical cores: the CPU capacity the planner may use")
    args = parser.parse_args(argv)
    ledger = read_jsonl(args.prefix + ".ledger.jsonl")
    samples = read_jsonl(args.prefix + ".timeline.jsonl")
    print(f"{'GIL share':>10} {'sequential':>11} {'lower bound':>12} {'plan':>8} {'gain':>8}  status")
    for gil in (0.0, 0.3, 0.5, 0.7, 1.0):
        instance, sequential, _span = build_instance(ledger, samples, logical=args.cores, capacity_cores=args.physical * 1.5, gil=gil)
        result = plan(instance, time_limit_s=1.0)
        print(f"{gil:10.1f} {sequential:11.1f} {result.lower_bound:12.1f} {result.makespan:8.1f} {sequential - result.makespan:8.1f}  {result.status} ({result.solver})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
