"""Run the independent FE stage cascades (early_b, mid_a, mid_b) concurrently, each on a private copy of the estimator and in a device slot of its own, and fold the results in the fixed stage order.

The cascades read one shared step-input frame and return it plus their own columns (see ``_fe_stage_cascade_run``); the recipes of different stages are different families, so they never collide. What the
stages DO share is the estimator's attributes: the lists every stage appends to (``hybrid_orth_features_``, ``_fe_rejection_records_``) and the per-family ``*_features_`` lists that one stage creates empty and
another fills. Each worker therefore gets a copy of the estimator with its own containers, and after all workers finished the changes are merged into the real estimator in the order early_b, mid_a, mid_b, which is
the order a sequential run would have produced them in:

* a list that grew by appending contributes only the appended items, concatenated in stage order;
* a dict contributes its new or changed keys, a set its new members;
* a new attribute that several stages create (an empty list in one, the filled list in another) is the concatenation of their lists in stage order;
* any other attribute that changed is taken from the last stage that changed it.

EXPERIMENTAL AND NOT SAFE ON A GPU HOST: measured at 100000 x 100, the stages in threads gained nothing (151-155 s against 154 s sequential, the stage time is GIL-bound Python), and a later run with the newest tree
raised ``cudaErrorIllegalAddress`` thousands of times and corrupted the CUDA context (cause not found: the suspects are device arrays that the process-global caches share between a stage's private pool and the
other threads, and numba.cuda kernels launched from several threads). Do not enable it without a device-side reproduction and a fix.

Off by default (``MLFRAME_FE_STAGE_CONCURRENCY=1`` turns it on); it also needs a device plan from ``_fe_device_scheduler`` (every stage gets at least the minimum VRAM budget), else the caller runs the cascades one by one.
"""

from __future__ import annotations

import copy
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence

from .._fe_deadline import _state as _deadline_state
from .._fe_deadline import fe_deadline_scope

logger = logging.getLogger(__name__)

#: Relative costs of early_b, mid_a and mid_b used to place them on devices (measured walls at 100000 x 100, every family).
STAGE_WEIGHTS = (14.0, 34.0, 18.0)


def concurrency_enabled() -> bool:
    """Whether ``MLFRAME_FE_STAGE_CONCURRENCY`` turns the concurrent cascade on."""
    return os.environ.get("MLFRAME_FE_STAGE_CONCURRENCY", "").strip() == "1"


def private_copy(estimator: Any) -> Any:
    """A shallow copy of ``estimator`` whose lists, dicts and sets are copies too, so a stage that mutates them does not touch the original or its siblings."""
    clone = copy.copy(estimator)
    for name, value in vars(clone).items():
        if isinstance(value, (list, dict, set)):
            setattr(clone, name, type(value)(value))
    return clone


def snapshot(estimator: Any) -> "dict[str, Any]":
    """The attributes of ``estimator`` as they are now, containers copied, for the later comparison."""
    return {name: (type(value)(value) if isinstance(value, (list, dict, set)) else value) for name, value in vars(estimator).items()}


def merge_stage_state(target: Any, base: "dict[str, Any]", stage_copies: Sequence[Any]) -> None:
    """Fold the changes of ``stage_copies`` (in stage order) relative to ``base`` into ``target``."""
    created: "dict[str, Any]" = {}
    for stage in stage_copies:
        for name, value in vars(stage).items():
            if name not in base:
                if isinstance(value, list):
                    created.setdefault(name, []).extend(value)
                elif isinstance(value, dict):
                    created.setdefault(name, {}).update(value)
                elif isinstance(value, set):
                    created.setdefault(name, set()).update(value)
                else:
                    created[name] = value
                continue
            before = base[name]
            if isinstance(value, list) and isinstance(before, list):
                if value[: len(before)] == before:
                    getattr(target, name).extend(value[len(before) :])  # appended: only the new items
                elif value != before:
                    setattr(target, name, list(value))  # rebuilt: the last stage wins
            elif isinstance(value, dict) and isinstance(before, dict):
                changed = {k: v for k, v in value.items() if k not in before or (before[k] is not v and before[k] != v)}
                getattr(target, name).update(changed)
            elif isinstance(value, set) and isinstance(before, set):
                getattr(target, name).update(value - before)
            elif value is not before and value != before:
                setattr(target, name, value)
    for name, value in created.items():
        setattr(target, name, value)


def run_stages_concurrently(estimator: Any, stages: Sequence[Callable[[Any], Any]], weights: Sequence[float] = STAGE_WEIGHTS) -> "list[Any] | None":
    """Results of ``stages`` (each called with a private copy of ``estimator``) run together, one per device slot, or ``None`` when no plan exists (the caller then runs them in order).

    The estimator's attributes are merged back in stage order before returning. A stage's exception is raised after all workers ended, for the first failing stage in order.
    """
    from .._fe_device_scheduler import plan_stage_slots, stage_device_slot

    slots = plan_stage_slots(list(weights))
    if slots is None or len(slots) != len(stages):
        logger.info("FE stage concurrency: no device plan (a stage would get too little VRAM); running the cascades one by one")
        return None
    base = snapshot(estimator)
    copies = [private_copy(estimator) for _ in stages]
    deadline = getattr(_deadline_state, "deadline", None)

    def _run(i: int) -> Any:
        """One stage on its copy, in its device slot, with the fit's deadline published to this worker thread."""
        with fe_deadline_scope(deadline), stage_device_slot(slots[i]):
            return stages[i](copies[i])

    with ThreadPoolExecutor(max_workers=len(stages)) as pool:
        futures = [pool.submit(_run, i) for i in range(len(stages))]
    outcomes: "list[Any]" = []
    for f in futures:
        outcomes.append(f.result())  # raises the first failure in stage order
    merge_stage_state(estimator, base, copies)
    return outcomes
