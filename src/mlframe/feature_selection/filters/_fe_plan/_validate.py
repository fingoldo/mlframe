"""An independent check of a plan: it never trusts the solver that made it."""

from __future__ import annotations

from ._model import EPS, Instance, Plan


def violations(instance: Instance, plan: Plan) -> list[str]:
    """Every way ``plan`` breaks ``instance``: a missing task, a mode that does not fit, a wrong duration, an early start after a predecessor, an overloaded resource, a wrong makespan. Empty when it is valid."""
    problems: list[str] = []
    tasks = instance.by_name()
    if set(plan.placements) != set(tasks):
        problems.append("the plan does not place exactly the tasks of the instance")
        return problems
    for name, p in plan.placements.items():
        problems.extend(_task_problems(instance, tasks[name], p, plan))
    problems.extend(_resource_problems(instance, plan))
    latest = max((p.end for p in plan.placements.values()), default=0.0)
    if abs(latest - plan.makespan) > 1e-4:
        problems.append(f"makespan {plan.makespan} is not the latest end {latest}")
    return problems


def _task_problems(instance: Instance, task, placement, plan: Plan) -> list[str]:
    """The problems of one placement: its mode, its duration, its start against the ends of the predecessors plus the transfer."""
    out: list[str] = []
    if not 0 <= placement.mode < len(task.modes) or placement.mode not in instance.feasible_modes(task):
        return [f"{task.name}: mode {placement.mode} is not a feasible mode"]
    mode = task.modes[placement.mode]
    if placement.machine != mode.machine:
        out.append(f"{task.name}: placed on {placement.machine}, the mode runs on {mode.machine}")
    if abs((placement.end - placement.start) - mode.duration) > 1e-4:
        out.append(f"{task.name}: runs {placement.end - placement.start}, the mode lasts {mode.duration}")
    if placement.start < -EPS:
        out.append(f"{task.name}: starts before 0")
    for pred in task.after:
        q = plan.placements[pred]
        if placement.start + 1e-4 < q.end + instance.delay(q.machine, placement.machine):
            out.append(f"{task.name}: starts at {placement.start} before {pred} ends at {q.end} plus the transfer")
    return out


def _resource_problems(instance: Instance, plan: Plan) -> list[str]:
    """The overloads of the cumulative resources, checked at every start time (the usage only rises at a start)."""
    out: list[str] = []
    tasks = instance.by_name()
    items = [(p, tasks[p.task].modes[p.mode]) for p in plan.placements.values() if p.mode < len(tasks[p.task].modes)]
    for t in sorted({p.start for p, _ in items}):
        for resource, cap in instance.capacity.items():
            used = sum(float(m.demand.get(resource, 0.0)) for p, m in items if p.start <= t + 1e-9 < p.end - 1e-9)
            if used > float(cap) + 1e-6:
                out.append(f"{resource}: {used:.4g} > {float(cap):.4g} at t={t:.4g}")
    return out
