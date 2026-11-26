"""Lower bounds on the makespan of an instance: what no schedule, on any devices, can beat."""

from __future__ import annotations

from ._model import Instance, Task, topological_order


def _fastest(instance: Instance, task: Task) -> float:
    """The shortest duration among the modes of ``task`` that fit the capacities."""
    return min(task.modes[i].duration for i in instance.feasible_modes(task))


def critical_path(instance: Instance) -> float:
    """The longest chain of precedences with every task in its fastest feasible mode and no transfer: the makespan with unlimited resources."""
    order = topological_order(instance)
    if order is None:
        raise ValueError("the precedences contain a cycle")
    tasks = instance.by_name()
    finish: dict[str, float] = {}
    for name in order:
        t = tasks[name]
        start = max((finish[a] for a in t.after), default=0.0)
        finish[name] = start + _fastest(instance, t)
    return max(finish.values(), default=0.0)


def tails(instance: Instance) -> dict[str, float]:
    """For every task the length of the longest chain from its start to the end of the project, itself included, in fastest modes: the priority of the list scheduler."""
    order = topological_order(instance)
    if order is None:
        raise ValueError("the precedences contain a cycle")
    tasks = instance.by_name()
    successors: dict[str, list[str]] = {n: [] for n in tasks}
    for t in instance.tasks:
        for a in t.after:
            successors[a].append(t.name)
    tail: dict[str, float] = {}
    for name in reversed(order):
        tail[name] = _fastest(instance, tasks[name]) + max((tail[s] for s in successors[name]), default=0.0)
    return tail


def energy_bound(instance: Instance) -> float:
    """The largest, over the resources, of the least total ``duration x demand`` of the tasks divided by the capacity: a resource cannot do more work than its capacity a second."""
    best = 0.0
    for resource, cap in instance.capacity.items():
        if float(cap) <= 0.0:
            continue
        work = 0.0
        for t in instance.tasks:
            work += min(t.modes[i].duration * float(t.modes[i].demand.get(resource, 0.0)) for i in instance.feasible_modes(t))
        best = max(best, work / float(cap))
    return best


def lower_bound(instance: Instance) -> float:
    """The larger of the critical path and the energy bound."""
    return max(critical_path(instance), energy_bound(instance))
