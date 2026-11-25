"""The planner entry point: the exact solver when it is installed and the instance is worth it, the list scheduler otherwise, always with a lower bound and a validity check."""

from __future__ import annotations

import logging
import time

from ._cpsat import available, solve_cpsat
from ._heuristic import list_schedule
from ._model import Instance, Plan
from ._validate import violations

logger = logging.getLogger(__name__)

#: An instance with at most this many tasks and one mode each, one resource and no precedence slack is solved by the list scheduler alone when it meets the lower bound.
_SOLVER_MIN_TASKS = 2


def plan(instance: Instance, *, time_limit_s: float = 0.5, prefer: str = "auto", seed: int = 0) -> Plan:
    """The schedule of ``instance`` with the shortest makespan found in ``time_limit_s`` seconds, its lower bound and its status.

    ``prefer="auto"`` runs the list scheduler first (milliseconds) and hands its schedule to CP-SAT as a hint when ortools is installed, the list schedule does not already meet the lower bound and the
    instance has at least two tasks; ``"heuristic"`` stops after the list scheduler. The returned plan has been checked by :func:`violations`; a CP-SAT plan that fails the check is dropped for the
    list schedule (and the failure is logged), so a solver defect cannot reach the executor."""
    started = time.perf_counter()
    base = list_schedule(instance, seed=seed)
    if prefer == "heuristic" or base.status == "optimal" or len(instance.tasks) < _SOLVER_MIN_TASKS or not available():
        return _checked(instance, base, base)
    left = max(0.05, float(time_limit_s) - (time.perf_counter() - started))
    exact = solve_cpsat(instance, time_limit_s=left, hint=base)
    if exact is None:
        base.notes.append("cp-sat found nothing in time or is unavailable; the list schedule is kept")
        return _checked(instance, base, base)
    best = exact if exact.makespan <= base.makespan + 1e-9 else base
    best.lower_bound = max(best.lower_bound, exact.lower_bound)
    if best.makespan <= best.lower_bound + 1e-9:
        best.status = "optimal"
    return _checked(instance, best, base)


def _checked(instance: Instance, candidate: Plan, fallback: Plan) -> Plan:
    """``candidate`` when it is valid, else ``fallback`` (the list schedule), with the problem noted."""
    problems = violations(instance, candidate)
    if not problems:
        return candidate
    logger.warning("the %s plan is invalid (%s); the list schedule is used", candidate.solver, problems[0])
    fallback.notes.append(f"{candidate.solver} plan rejected: {problems[0]}")
    return fallback
