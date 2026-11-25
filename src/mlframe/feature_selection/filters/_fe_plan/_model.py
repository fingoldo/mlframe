"""The scheduling instance of a fit: tasks with alternative modes, cumulative resources, precedences and transfer delays, and the plan that solves it.

A fit is a set of stages (tasks). Each stage can run in one or more *modes*: on the CPU, on one GPU, on another GPU. A mode has a duration and holds an amount of each resource while it runs: CPU cores,
the share of the interpreter lock, the compute share of a device, host memory, the memory of a card. The resources are cumulative (the sum of what the running modes hold never exceeds the capacity), which
is how "two stages that each use a quarter of the GPU can overlap, a third that needs the rest cannot" and "a stage whose working set does not fit the card is not a candidate for it" are written. A
stage starts after its predecessors end, plus a transfer delay when the two ran on different machines. The objective is the makespan.

This is the multi-mode resource-constrained project scheduling problem; the module holds only the data of an instance and of a plan, so that the solvers, the bounds and the validity check share one
vocabulary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

#: Times are compared to this tolerance (seconds) when a plan is validated.
EPS = 1e-6


@dataclass(frozen=True)
class Mode:
    """One way to run a task: the machine it runs on, how long it takes there and what it holds while it runs.

    ``demand`` maps a resource name to the amount held for the whole duration; a resource that is not named is not used."""

    machine: str
    duration: float
    demand: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Task:
    """A stage: its alternative modes (at least one) and the names of the tasks that must end before it starts."""

    name: str
    modes: tuple[Mode, ...]
    after: tuple[str, ...] = ()


@dataclass(frozen=True)
class Instance:
    """The tasks, the capacity of every resource and the transfer delay between machines.

    ``transfer[(a, b)]`` is the delay, in seconds, between the end of a task that ran on machine ``a`` and the start of its successor on machine ``b`` (absent or zero for the same machine)."""

    tasks: tuple[Task, ...]
    capacity: Mapping[str, float]
    transfer: Mapping[tuple[str, str], float] = field(default_factory=dict)

    def by_name(self) -> dict[str, Task]:
        """The tasks by name."""
        return {t.name: t for t in self.tasks}

    def feasible_modes(self, task: Task) -> list[int]:
        """Indices of the modes of ``task`` whose demand fits the capacity of every resource it names (a mode that cannot ever fit is not a candidate)."""
        return [i for i, m in enumerate(task.modes) if all(float(a) <= float(self.capacity.get(r, 0.0)) + EPS for r, a in m.demand.items())]

    def delay(self, machine_from: str, machine_to: str) -> float:
        """The transfer delay of an edge between two machines."""
        if machine_from == machine_to:
            return 0.0
        return float(self.transfer.get((machine_from, machine_to), 0.0))

    def validate_structure(self) -> None:
        """Raise ``ValueError`` for an instance no solver can answer: a duplicate name, an unknown predecessor, a cycle, a task with no mode that fits."""
        names = [t.name for t in self.tasks]
        if len(set(names)) != len(names):
            raise ValueError("duplicate task names")
        known = set(names)
        for t in self.tasks:
            missing = [a for a in t.after if a not in known]
            if missing:
                raise ValueError(f"task {t.name!r} follows unknown tasks {missing}")
            if not self.feasible_modes(t):
                raise ValueError(f"task {t.name!r} has no mode that fits the capacities")
        if topological_order(self) is None:
            raise ValueError("the precedences contain a cycle")


@dataclass(frozen=True)
class Placement:
    """Where and when a task runs in a plan."""

    task: str
    mode: int
    machine: str
    start: float
    end: float


@dataclass
class Plan:
    """A schedule: a placement per task, its makespan, a lower bound on the optimal makespan and how it was obtained.

    ``status`` is ``"optimal"`` (proved), ``"feasible"`` (the best found within the time limit, the gap is ``makespan - lower_bound``) or ``"heuristic"`` (list scheduling only)."""

    placements: dict[str, Placement]
    makespan: float
    lower_bound: float
    status: str
    solver: str
    notes: list[str] = field(default_factory=list)

    @property
    def gap(self) -> float:
        """Relative distance of the makespan to the lower bound (0 when the plan is proved optimal or meets the bound)."""
        if self.makespan <= 0.0:
            return 0.0
        return max(0.0, (self.makespan - self.lower_bound) / self.makespan)

    def order(self) -> list[Placement]:
        """The placements by start time (ties by end time and name)."""
        return sorted(self.placements.values(), key=lambda p: (p.start, p.end, p.task))


def topological_order(instance: Instance) -> "list[str] | None":
    """The task names in an order where every task follows its predecessors, or ``None`` when the precedences contain a cycle."""
    indegree = {t.name: len(t.after) for t in instance.tasks}
    successors: dict[str, list[str]] = {t.name: [] for t in instance.tasks}
    for t in instance.tasks:
        for a in t.after:
            if a in successors:
                successors[a].append(t.name)
    ready = [n for n, d in indegree.items() if d == 0]
    out: list[str] = []
    while ready:
        n = ready.pop()
        out.append(n)
        for s in successors[n]:
            indegree[s] -= 1
            if indegree[s] == 0:
                ready.append(s)
    return out if len(out) == len(instance.tasks) else None
