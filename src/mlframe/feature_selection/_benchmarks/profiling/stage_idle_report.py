"""For every stage of an MRMR fit: how the machine was loaded while it ran, which devices stood idle with enough free memory in the window before it started, and which stage it followed.

Usage: ``python -m mlframe.feature_selection._benchmarks.profiling.stage_idle_report LEDGER.jsonl TIMELINE.jsonl [--min-stage-s 1.0] [--lookback 10] [--top 40]``. ``LEDGER`` is written by the stage ledger
(``MLFRAME_STAGE_LEDGER=<path>``, see ``filters/_stage_ledger.py``), ``TIMELINE`` by ``hardware_timeline.TimelineSampler``; both use the epoch clock.

The question each row answers (owner, 2026-11-24): was there a device with low load and enough free memory to hold the stage's data BEFORE the stage started, and if so why did the stage not start earlier?
A row lists ``idle_before`` (the devices that were idle in the window: ``cpu`` with its idle core count, ``gpuN`` with its utilisation, each with the free memory against the stage's input size) and ``after`` (the
stage that ended last before this one started: the dependency to look at first). A stage with an idle device before it and no predecessor that ended just before is the candidate for an earlier start; a
stage that follows another one at once is waiting on it. The analysis functions are pure (events and samples in, rows out) so they are unit-tested."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional

#: A core is idle below this load, a GPU below this utilisation (percent).
CORE_IDLE_PCT = 20.0
GPU_IDLE_PCT = 20.0
#: Both devices count as idle for the timeline summary below these (percent of all cores / of the busiest GPU).
BOTH_IDLE_CPU_PCT = 25.0
BOTH_IDLE_GPU_PCT = 10.0


def load_jsonl(path: str) -> "list[dict[str, Any]]":
    """The JSON lines of ``path`` (lines that do not parse are skipped: concurrent appends can corrupt one)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def window_stats(samples: "list[dict[str, Any]]", t0: float, t1: float) -> "Optional[dict[str, Any]]":
    """Load in ``[t0, t1]``: mean busy cores, core count, per-GPU mean utilisation and minimum free VRAM, minimum free RAM; ``None`` when no sample falls inside."""
    inside = [s for s in samples if t0 <= s["t"] <= t1]
    if not inside:
        return None
    n_cores = max(len(s["cores"]) for s in inside)
    busy = sum(sum(s["cores"]) / 100.0 for s in inside) / len(inside)
    gpus: "dict[int, dict[str, float]]" = {}
    for s in inside:
        for g in s.get("gpus", []):
            slot = gpus.setdefault(int(g["index"]), {"util_sum": 0.0, "n": 0.0, "free_gb": float("inf")})
            slot["util_sum"] += g["util"]
            slot["n"] += 1
            slot["free_gb"] = min(slot["free_gb"], g["mem_free_gb"])
    return {
        "n_samples": len(inside),
        "n_cores": n_cores,
        "busy_cores": busy,
        "idle_cores": n_cores - busy,
        "ram_free_gb": min(s["ram_free_gb"] for s in inside),
        "gpus": {i: {"util": v["util_sum"] / v["n"], "free_gb": v["free_gb"]} for i, v in gpus.items()},
    }


def classify(stats: "Optional[dict[str, Any]]") -> str:
    """``cpu``, ``gpu``, ``cpu+gpu`` or ``wait`` for the load measured while a stage ran (a stage that used under 1.5 cores and no GPU is waiting)."""
    if stats is None:
        return "unknown"
    cpu = stats["busy_cores"] >= 1.5
    gpu = any(g["util"] >= 30.0 for g in stats["gpus"].values())
    return "cpu+gpu" if cpu and gpu else "gpu" if gpu else "cpu" if cpu else "wait"


def idle_before(stats: "Optional[dict[str, Any]]", in_gb: Optional[float]) -> "list[str]":
    """The devices that were idle over a window with their free memory against the stage's input size (``in_gb``; ``None`` counts as 0)."""
    if stats is None:
        return []
    need = float(in_gb or 0.0)
    out = []
    if stats["idle_cores"] >= max(2.0, 0.25 * stats["n_cores"]) and stats["ram_free_gb"] >= need:
        out.append(f"cpu {stats['idle_cores']:.1f}/{stats['n_cores']} cores idle, RAM free {stats['ram_free_gb']:.1f} GB")
    for index, g in sorted(stats["gpus"].items()):
        if g["util"] < GPU_IDLE_PCT and g["free_gb"] >= need:
            out.append(f"gpu{index} {g['util']:.0f}% busy, VRAM free {g['free_gb']:.1f} GB")
    return out


def analyse(
    events: "list[dict[str, Any]]", samples: "list[dict[str, Any]]", *, min_stage_s: float = 1.0, lookback_s: float = 10.0, follow_gap_s: float = 1.0
) -> "list[dict[str, Any]]":
    """One row per stage of at least ``min_stage_s`` seconds, in start order: its load, the idle devices of the ``lookback_s`` before it and the stage it followed."""
    rows = []
    ordered = sorted(events, key=lambda e: e["t0"])
    for ev in ordered:
        dur = ev["t1"] - ev["t0"]
        if dur < min_stage_s:
            continue
        during = window_stats(samples, ev["t0"], ev["t1"])
        before = window_stats(samples, ev["t0"] - lookback_s, ev["t0"] - 1e-6)  # the sample AT the start belongs to the stage
        predecessors = [p for p in ordered if p is not ev and ev["t0"] - follow_gap_s <= p["t1"] <= ev["t0"] + 0.1]
        after = max(predecessors, key=lambda p: p["t1"])["name"] if predecessors else None
        rows.append(
            {
                "name": ev["name"],
                "t0": ev["t0"],
                "wall_s": dur,
                "thread": ev.get("thread_name") or ev.get("thread"),
                "device": ev.get("device") or classify(during),
                "in_gb": ev.get("in_gb"),
                "busy_cores": None if during is None else during["busy_cores"],
                "gpu_util": None if during is None else (max((g["util"] for g in during["gpus"].values()), default=0.0)),
                "idle_before": idle_before(before, ev.get("in_gb")),
                "after": after,
            }
        )
    return rows


def idle_timeline(events: "list[dict[str, Any]]", samples: "list[dict[str, Any]]") -> "dict[str, Any]":
    """The share of the samples with the CPU below ``BOTH_IDLE_CPU_PCT`` percent AND every GPU below ``BOTH_IDLE_GPU_PCT``, and the seconds of those samples by the shortest stage that covered them."""
    if not samples:
        return {"n_samples": 0, "idle_share": 0.0, "by_stage": []}
    step = (samples[-1]["t"] - samples[0]["t"]) / max(len(samples) - 1, 1) if len(samples) > 1 else 0.0
    idle = [s for s in samples if s["cpu"] < BOTH_IDLE_CPU_PCT and all(g["util"] < BOTH_IDLE_GPU_PCT for g in s.get("gpus", []))]
    by_stage: "dict[str, float]" = {}
    for s in idle:
        covering = [e for e in events if e["t0"] <= s["t"] <= e["t1"]]
        name = min(covering, key=lambda e: e["t1"] - e["t0"])["name"] if covering else "(no stage)"
        by_stage[name] = by_stage.get(name, 0.0) + step
    return {"n_samples": len(samples), "idle_share": len(idle) / len(samples), "by_stage": sorted(by_stage.items(), key=lambda kv: -kv[1])}


def render(rows: "list[dict[str, Any]]", timeline: "dict[str, Any]", top: int = 40) -> str:
    """The report as text: the stage table (longest first) and the idle-timeline summary."""
    lines = [f"{'stage':36s} {'wall s':>7s} {'device':>8s} {'cores':>6s} {'gpu%':>5s} {'in GB':>6s}  after / idle before"]
    for r in sorted(rows, key=lambda r: -r["wall_s"])[:top]:
        cores = "-" if r["busy_cores"] is None else f"{r['busy_cores']:.1f}"
        gpu = "-" if r["gpu_util"] is None else f"{r['gpu_util']:.0f}"
        gb = "-" if r["in_gb"] is None else f"{r['in_gb']:.3f}"
        lines.append(f"{r['name']!s:36.36} {r['wall_s']:7.1f} {r['device']!s:>8} {cores:>6s} {gpu:>5s} {gb:>6s}  after {r['after']}; idle before: {'; '.join(r['idle_before']) or 'none'}")
    lines.append(f"\nsamples {timeline['n_samples']}; both devices idle (cpu < {BOTH_IDLE_CPU_PCT:.0f}% and every gpu < {BOTH_IDLE_GPU_PCT:.0f}%): {100.0 * timeline['idle_share']:.1f}% of the time")
    lines += [f"  {name:36s} {seconds:7.1f} s" for name, seconds in timeline["by_stage"][:15]]
    return "\n".join(lines)


def main(argv: "Optional[list[str]]" = None) -> int:
    """Command line entry: print the report of a ledger and a timeline."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ledger")
    ap.add_argument("timeline")
    ap.add_argument("--min-stage-s", type=float, default=1.0)
    ap.add_argument("--lookback", type=float, default=10.0)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args(argv)
    events, samples = load_jsonl(args.ledger), load_jsonl(args.timeline)
    print(render(analyse(events, samples, min_stage_s=args.min_stage_s, lookback_s=args.lookback), idle_timeline(events, samples), args.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
