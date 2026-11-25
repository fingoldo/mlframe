"""The planner of the fit as an operation-research problem (owner requirement of 2026-11-25, section 2a of ``audits/2026-11-23/PLAN_speed_and_device_utilization.md``).

An :class:`Instance` is a set of tasks, each with alternative modes (a machine, a duration, the resources it holds), precedences, cumulative resource capacities and transfer delays between machines.
:func:`plan` returns a :class:`Plan` with the shortest makespan found within a time limit, a lower bound, a status and a gap; :func:`violations` checks any plan against its instance. The modules:

* ``_model``: the data of an instance and of a plan;
* ``_bounds``: the critical path and the resource-energy lower bounds;
* ``_heuristic``: the list scheduler (serial schedule generation with restarts);
* ``_cpsat``: the exact CP-SAT model (optional dependency ``ortools``);
* ``_validate``: the independent validity check;
* ``_solve``: the entry point.

Nothing in the fit calls this package yet: it is the solver and its tests; the instance of a fit is built from the stage ledger by ``_benchmarks/profiling/plan_from_ledger.py``.
"""

from ._bounds import critical_path, energy_bound, lower_bound
from ._cpsat import available as cpsat_available
from ._heuristic import list_schedule
from ._model import Instance, Mode, Placement, Plan, Task
from ._solve import plan
from ._validate import violations

__all__ = [
    "Instance",
    "Mode",
    "Placement",
    "Plan",
    "Task",
    "cpsat_available",
    "critical_path",
    "energy_bound",
    "list_schedule",
    "lower_bound",
    "plan",
    "violations",
]
