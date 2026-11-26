"""The exact solver of the instance: a CP-SAT model with one optional interval per mode, cumulative constraints per resource and transfer delays on the edges that cross machines.

Times are integers in the model (milliseconds), demands too (hundredths of a unit); a duration or an amount is rounded UP so that a plan feasible in the model is feasible in the instance. The list
schedule is given as a hint. ``ortools`` is optional: :func:`available` says whether it is installed, and the caller falls back to the list scheduler when it is not.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ._bounds import lower_bound
from ._model import Instance, Placement, Plan

logger = logging.getLogger(__name__)

_TIME_SCALE = 1000  # model time unit: one millisecond
_DEMAND_SCALE = 100  # model resource unit: one hundredth


def available() -> bool:
    """Whether ``ortools`` CP-SAT can be imported."""
    try:
        from ortools.sat.python import cp_model  # noqa: F401
    except ImportError:
        return False
    return True


def _ceil_units(value: float, scale: int) -> int:
    """``value`` in model units, rounded up (and at least one unit for a positive value)."""
    units = int(-(-round(float(value) * scale * 1_000_000) // 1_000_000))
    return max(units, 1) if value > 0.0 else 0


def solve_cpsat(instance: Instance, *, time_limit_s: float, hint: "Plan | None" = None, workers: "int | None" = None) -> "Plan | None":
    """The best plan CP-SAT finds within ``time_limit_s`` seconds, or ``None`` when ortools is missing or no schedule is found in time."""
    if not available():
        return None
    from ortools.sat.python import cp_model

    instance.validate_structure()
    model = cp_model.CpModel()
    horizon = sum(max(m.duration for m in t.modes) for t in instance.tasks) + sum(abs(float(v)) for v in instance.transfer.values()) * max(1, len(instance.tasks))
    horizon_u = _ceil_units(horizon, _TIME_SCALE) + 1
    starts, ends, presence, intervals = _declare(model, instance, horizon_u)
    _add_precedences(model, instance, starts, ends, presence)
    _add_resources(model, instance, intervals)
    makespan = model.new_int_var(0, horizon_u, "makespan")
    for name in starts:
        model.add(makespan >= ends[name])
    model.minimize(makespan)
    if hint is not None:
        _add_hint(model, instance, hint, starts, presence)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.05, float(time_limit_s))
    solver.parameters.num_workers = int(workers if workers is not None else min(8, os.cpu_count() or 1))
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return _extract(instance, solver, status == cp_model.OPTIMAL, starts, presence)


def _declare(model: Any, instance: Instance, horizon_u: int) -> tuple:
    """Start and end variables of every task, one presence literal per feasible mode (exactly one holds) and the optional interval each mode makes."""
    starts: dict[str, Any] = {}
    ends: dict[str, Any] = {}
    presence: dict[tuple[str, int], Any] = {}
    intervals: dict[tuple[str, int], Any] = {}
    for t in instance.tasks:
        starts[t.name] = model.new_int_var(0, horizon_u, f"s_{t.name}")
        ends[t.name] = model.new_int_var(0, horizon_u, f"e_{t.name}")
        literals = []
        for i in instance.feasible_modes(t):
            lit = model.new_bool_var(f"p_{t.name}_{i}")
            d = _ceil_units(t.modes[i].duration, _TIME_SCALE)
            presence[(t.name, i)] = lit
            intervals[(t.name, i)] = model.new_optional_interval_var(starts[t.name], d, ends[t.name], lit, f"i_{t.name}_{i}")
            model.add(ends[t.name] == starts[t.name] + d).only_enforce_if(lit)
            literals.append(lit)
        model.add_exactly_one(literals)
    return starts, ends, presence, intervals


def _add_precedences(model: Any, instance: Instance, starts: dict, ends: dict, presence: dict) -> None:
    """Finish-to-start edges; an edge whose two modes sit on different machines also waits for the transfer delay."""
    tasks = instance.by_name()
    for t in instance.tasks:
        for a in t.after:
            model.add(starts[t.name] >= ends[a])
            for i in instance.feasible_modes(tasks[a]):
                for j in instance.feasible_modes(t):
                    delay = instance.delay(tasks[a].modes[i].machine, t.modes[j].machine)
                    if delay > 0.0:
                        model.add(starts[t.name] >= ends[a] + _ceil_units(delay, _TIME_SCALE)).only_enforce_if([presence[(a, i)], presence[(t.name, j)]])


def _add_resources(model: Any, instance: Instance, intervals: dict) -> None:
    """One cumulative constraint per resource over the optional intervals of the modes that hold it."""
    tasks = instance.by_name()
    for resource, cap in instance.capacity.items():
        shown: list[Any] = []
        demands: list[int] = []
        cap_units = round(float(cap) * _DEMAND_SCALE)
        for (name, i), interval in intervals.items():
            amount = float(tasks[name].modes[i].demand.get(resource, 0.0))
            if amount > 0.0:
                shown.append(interval)
                demands.append(min(_ceil_units(amount, _DEMAND_SCALE), cap_units))
        if shown:
            model.add_cumulative(shown, demands, cap_units)


def _add_hint(model: Any, instance: Instance, hint: Plan, starts: dict, presence: dict) -> None:
    """Start the search from a known schedule (modes and start times)."""
    for name, p in hint.placements.items():
        model.add_hint(starts[name], _ceil_units(p.start, _TIME_SCALE))
        for i in instance.feasible_modes(instance.by_name()[name]):
            model.add_hint(presence[(name, i)], 1 if i == p.mode else 0)


def _extract(instance: Instance, solver: Any, optimal: bool, starts: dict, presence: dict) -> Plan:
    """The plan of a solved model, with the lower bound the solver proved (never below the instance's own)."""
    placements: dict[str, Placement] = {}
    for t in instance.tasks:
        mode = next(i for i in instance.feasible_modes(t) if solver.value(presence[(t.name, i)]))
        s = solver.value(starts[t.name]) / _TIME_SCALE
        placements[t.name] = Placement(task=t.name, mode=mode, machine=t.modes[mode].machine, start=s, end=s + t.modes[mode].duration)
    span = max((p.end for p in placements.values()), default=0.0)
    bound = max(lower_bound(instance), float(solver.best_objective_bound) / _TIME_SCALE)
    return Plan(placements=placements, makespan=span, lower_bound=min(bound, span), status="optimal" if optimal else "feasible", solver="cp-sat")
