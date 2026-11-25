"""List scheduling for the multi-mode resource-constrained instance: the fallback when no exact solver is installed and the starting point of the one that is.

The serial schedule generation scheme takes the tasks one at a time in a priority order that respects the precedences, and places each in the mode and at the earliest start where it fits the
resources next to what is already placed. Different priority orders give different schedules; the search tries the order of the longest remaining chain first, then a few perturbed orders, and keeps the
shortest. It is deterministic for a seed.
"""

from __future__ import annotations

import random

from ._bounds import lower_bound, tails
from ._model import EPS, Instance, Placement, Plan, Task, topological_order


class _Profile:
    """The placed intervals and what they hold, and the earliest start where one more fits."""

    def __init__(self, capacity: dict[str, float]) -> None:
        self.capacity = capacity
        self.items: list[tuple[float, float, dict[str, float]]] = []

    def fits(self, start: float, end: float, demand: dict[str, float]) -> bool:
        """Whether holding ``demand`` over ``[start, end)`` keeps every resource within its capacity."""
        overlapping = [(s, e, d) for s, e, d in self.items if s < end - EPS and e > start + EPS]
        points = sorted({start} | {s for s, _, _ in overlapping if s > start})
        for t in points:
            for resource, amount in demand.items():
                used = sum(d.get(resource, 0.0) for s, e, d in overlapping if s <= t + EPS < e - EPS)
                if used + amount > float(self.capacity.get(resource, 0.0)) + 1e-9:
                    return False
        return True

    def earliest(self, ready: float, duration: float, demand: dict[str, float]) -> float:
        """The first start at or after ``ready`` where the demand fits for ``duration`` (candidate starts are ``ready`` and the ends of the placed intervals)."""
        candidates = sorted({ready} | {e for _, e, _ in self.items if e > ready})
        for c in candidates:
            if self.fits(c, c + duration, demand):
                return c
        return max(candidates[-1], ready)  # the last candidate is after every placed interval: the demand fits there when it fits the capacity at all

    def add(self, start: float, end: float, demand: dict[str, float]) -> None:
        """Record a placed interval."""
        self.items.append((start, end, dict(demand)))


def _ready_time(instance: Instance, task: Task, mode_machine: str, placed: dict[str, Placement]) -> float:
    """The earliest start of ``task`` on ``mode_machine`` from the ends of its predecessors and the transfer delays."""
    return max((placed[a].end + instance.delay(placed[a].machine, mode_machine) for a in task.after), default=0.0)


def _place(instance: Instance, task: Task, profile: _Profile, placed: dict[str, Placement]) -> Placement:
    """The placement of ``task`` that ends earliest over its feasible modes (ties: the smaller area, then the lower mode index)."""
    best: "tuple[float, float, int, float] | None" = None
    for i in instance.feasible_modes(task):
        mode = task.modes[i]
        demand = {r: float(a) for r, a in mode.demand.items() if float(a) > 0.0}
        start = profile.earliest(_ready_time(instance, task, mode.machine, placed), mode.duration, demand)
        key = (start + mode.duration, mode.duration * sum(demand.values()), i, start)
        if best is None or key[:3] < best[:3]:
            best = key
    assert best is not None  # validate_structure guarantees a feasible mode
    end, _area, index, start = best
    mode = task.modes[index]
    profile.add(start, end, {r: float(a) for r, a in mode.demand.items() if float(a) > 0.0})
    return Placement(task=task.name, mode=index, machine=mode.machine, start=start, end=end)


def _schedule(instance: Instance, rank: dict[str, float]) -> dict[str, Placement]:
    """One serial schedule for the priorities ``rank`` (higher first among the tasks whose predecessors are all placed)."""
    tasks = instance.by_name()
    placed: dict[str, Placement] = {}
    profile = _Profile({r: float(c) for r, c in instance.capacity.items()})
    remaining = set(tasks)
    while remaining:
        available = [n for n in remaining if all(a in placed for a in tasks[n].after)]
        name = max(available, key=lambda n: (rank[n], n))
        placed[name] = _place(instance, tasks[name], profile, placed)
        remaining.discard(name)
    return placed


def _makespan(placed: dict[str, Placement]) -> float:
    """The latest end of a schedule."""
    return max((p.end for p in placed.values()), default=0.0)


def list_schedule(instance: Instance, *, restarts: int = 40, seed: int = 0) -> Plan:
    """The best of ``restarts`` serial schedules: the longest-remaining-chain order first, then orders whose priorities are perturbed by up to 30 percent."""
    instance.validate_structure()
    if topological_order(instance) is None:
        raise ValueError("the precedences contain a cycle")
    base = tails(instance)
    rng = random.Random(seed)
    best = _schedule(instance, base)
    best_span = _makespan(best)
    for _ in range(max(0, restarts - 1)):
        noisy = {n: v * (1.0 + 0.3 * (rng.random() - 0.5)) + 1e-9 * rng.random() for n, v in base.items()}
        candidate = _schedule(instance, noisy)
        span = _makespan(candidate)
        if span < best_span - 1e-12:
            best, best_span = candidate, span
    bound = lower_bound(instance)
    status = "optimal" if best_span <= bound + 1e-9 else "heuristic"
    return Plan(placements=best, makespan=best_span, lower_bound=bound, status=status, solver="list-scheduling")
