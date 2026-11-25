"""Run the independent FE stage cascades (early_b, mid_a, mid_b) concurrently, each on a private copy of the estimator and in a device slot of its own, and fold the results in the fixed stage order.

The cascades read one shared step-input frame and return it plus their own columns (see ``_fe_stage_cascade_run``); the recipes of different stages are different families, so they never collide. What the
stages DO share is the estimator's attributes: the lists every stage appends to (``hybrid_orth_features_``, ``_fe_rejection_records_``) and the per-family ``*_features_`` lists that one stage creates empty and
another fills. Each worker therefore gets a copy of the estimator with its own containers, and after all workers finished the changes are merged into the real estimator in the order early_b, mid_a, mid_b, which is
the order a sequential run would have produced them in:

* a list that grew by appending contributes only the appended items, concatenated in stage order;
* a dict contributes its new or changed keys, a set its new members;
* a new attribute that several stages create (an empty list in one, the filled list in another) is the concatenation of their lists in stage order;
* any other attribute that changed is taken from the last stage that changed it.

CUDA-context corruption (``cudaErrorIllegalAddress``, thousands per fit) that stopped the first attempt was caused by the slot pool being destroyed when its slot ended: cupy frees the in-use chunks of a destroyed pool,
and the process-global caches kept arrays allocated inside the slot, so the next kernel that read one hit freed memory. The slot pools now live for the whole process (``_fe_device_scheduler``); with that the
30000 x 40 fit has no device errors and the same selection as the sequential run.

On by default since 2026-11-03 (``MLFRAME_FE_STAGE_CONCURRENCY=0`` turns it off): at 100000 x 100 with every family, one process, alternating order, concurrent 118.0 / 124.4 / 143.1 s against sequential 129.4 / 171.6 s, the same six features in every run and no device errors (after the day's GPU and CPU changes the stages no longer just queue on the device, unlike the 300k-row measurement in ``_fe_stage_cascade_run``). It also needs a device plan from ``_fe_device_scheduler`` (every stage gets at least the minimum VRAM budget), else the caller runs the cascades one by one.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence

import numpy as np

from .._fe_deadline import _state as _deadline_state
from .._fe_family_timing import close_thread_laps
from .. import _stage_ledger as stage_ledger
from .._fe_deadline import fe_deadline_scope
from mlframe.system import host_memory
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

#: Relative costs of early_b, mid_a and mid_b used to place them on devices (measured walls at 100000 x 100, every family).
STAGE_WEIGHTS = (14.0, 34.0, 18.0)
#: The same with early_a in front (its families - the orthogonal bases, triplets / quadruplets, the auto-scorer - measured about 25 s at 100000 x 100).
STAGE_WEIGHTS_WITH_EARLY_A = (25.0, 14.0, 34.0, 18.0)
#: Peak device-pool use of the stages (MB, sequential fit at 50000 x 100, every family): early_b 461, mid_a 274, mid_b 110 (early_a 486). They split a device's
#: memory among its stages; splitting it by the walls above gave early_b, the shortest stage with the largest need, the smallest slot (out-of-memory fallbacks).
STAGE_VRAM_WEIGHTS = (461.0, 274.0, 110.0)
STAGE_VRAM_WEIGHTS_WITH_EARLY_A = (486.0, 461.0, 274.0, 110.0)


def concurrency_enabled() -> bool:
    """Whether the cascades run concurrently: on by default, ``MLFRAME_FE_STAGE_CONCURRENCY=0`` runs them one by one."""
    return os.environ.get("MLFRAME_FE_STAGE_CONCURRENCY", "1").strip() != "0"


#: The concurrent cascades hold about this many frames of working set together (measured at 100000 x 1000 with every family: 14 frames of growth concurrent against 9.6 serial, 500000 x 1000: 2.4).
_CONCURRENT_FRAME_FACTOR = 4.0


def memory_allows_concurrency(n_rows: int, n_cols: int) -> bool:
    """Whether the cascades may run side by side: the float32 frame times ``_CONCURRENT_FRAME_FACTOR`` (``MLFRAME_FE_STAGE_CONCURRENCY_MEM_FACTOR`` overrides) must fit the available host RAM.

    Measured on 100000 x 1000 with every family: concurrent 238.0 s and 5585 MB of growth, one after the other 254.1 s (+6.8 percent) and 3845 MB (-31 percent); on a host where the working set of four
    stages does not fit, the slower order is the one that finishes (1M x 1000 on a 16 GB host). An unreadable RAM figure keeps the concurrent order."""
    try:
        _raw_factor = os.environ.get("MLFRAME_FE_STAGE_CONCURRENCY_MEM_FACTOR", "").strip()
        factor = float(_raw_factor) if _raw_factor else _CONCURRENT_FRAME_FACTOR
        available = float(host_memory()[0])
    except Exception as exc:
        log_throttle(
            logger,
            "_fe_stage_concurrency.mem_probe",
            logging.INFO,
            "stage concurrency: available RAM unreadable (%r); the cascades keep their concurrent order",
            exc,
        )
        return True
    need = factor * float(n_rows) * float(n_cols) * 4.0
    allowed = need <= available
    if not allowed:
        log_throttle(
            logger,
            "_fe_stage_concurrency.serialised",
            logging.WARNING,
            "stage concurrency off for this fit: %d x %d needs about %.1f GB for concurrent cascades (%.1fx the frame) against %.1f GB available; the cascades run one after the other",
            int(n_rows),
            int(n_cols),
            need / 2**30,
            factor,
            available / 2**30,
        )
    return allowed


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


def _same(a: Any, b: Any) -> bool:
    """Whether ``a`` and ``b`` hold the same value. ``==`` alone cannot be used: on numpy / cupy arrays (the weights of a weighted fit, cached codes) it
    returns an array, and ``if`` on that raised "truth value ... is ambiguous" in every weighted fit whose stages ran concurrently. A comparison that
    fails counts as a change, so the stage's value is kept."""
    if a is b:
        return True
    if hasattr(a, "shape") and hasattr(b, "shape") and hasattr(a, "dtype") and hasattr(b, "dtype"):
        try:
            if type(a) is not type(b) or a.shape != b.shape or a.dtype != b.dtype:
                return False
            if type(a).__module__.startswith("cupy"):
                import cupy as cp

                return bool(cp.array_equal(a, b))
            return bool(np.array_equal(a, b))
        except Exception as exc:  # best-effort: a comparison that raises is reported as unequal, the conservative answer
            log_throttle(logger, "fe_stage_concurrency.same", logging.WARNING, "stage-state comparison raised %s: %s; reported as unequal", type(exc).__name__, exc)
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception as exc:  # best-effort: a comparison that raises is reported as unequal, the conservative answer
        log_throttle(logger, "fe_stage_concurrency.same", logging.WARNING, "stage-state comparison raised %s: %s; reported as unequal", type(exc).__name__, exc)
        return False


def _synchronize_device(device_id: int) -> None:
    """Wait for all work queued on ``device_id`` so a timer stopped after it measures completion; a host without cupy has nothing to wait for."""
    try:
        import cupy as cp
    except ImportError:  # best-effort: no cupy means no device work to wait for
        return
    with cp.cuda.Device(device_id):
        cp.cuda.Device().synchronize()


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
                if _same(value[: len(before)], before):
                    getattr(target, name).extend(value[len(before) :])  # appended: only the new items
                elif not _same(value, before):
                    setattr(target, name, list(value))  # rebuilt: the last stage wins
            elif isinstance(value, dict) and isinstance(before, dict):
                changed = {k: v for k, v in value.items() if k not in before or not _same(before[k], v)}
                getattr(target, name).update(changed)
            elif isinstance(value, set) and isinstance(before, set):
                getattr(target, name).update(value - before)
            elif not _same(value, before):
                setattr(target, name, value)
    for name, value in created.items():
        setattr(target, name, value)


#: Rows of the fit the stage peaks above were measured at; the peaks grow about linearly with the rows.
STAGE_VRAM_MEASURED_ROWS = 50_000


def _stages_fit_device(vram_weights: "Sequence[float]", n_rows: int) -> bool:
    """Whether the stages' peak pool use, scaled from the 50000-row measurement to ``n_rows``, fits the device memory the fit can still use (free device memory
    plus the default pool's cached blocks). Concurrent stages each get a slice of it; at 500000 rows the scaled peaks (8.4 GB) far exceed a 4 GB card, and the
    slots' out-of-memory errors sent stages to the host paths, so the cascades run one by one there and each stage has the whole card."""
    # MLFRAME_FE_STAGE_VRAM_MAX_ROWS caps the row count the peaks are scaled to: the stages stream their column blocks under the pool limit, so their peak stops growing with the rows, and the
    # linear scaling made every fit above ~200000 rows run its cascades one after the other (host-bound and device-bound stages then never overlap)
    cap = os.environ.get("MLFRAME_FE_STAGE_VRAM_MAX_ROWS", "").strip()
    scaled_rows = min(float(n_rows), float(cap)) if cap.isdigit() and int(cap) > 0 else float(n_rows)
    need = sum(float(w) for w in vram_weights) * (1 << 20) * max(1.0, scaled_rows / STAGE_VRAM_MEASURED_ROWS)
    try:
        import cupy as cp

        pool = cp.get_default_memory_pool()
        avail = int(cp.cuda.Device().mem_info[0]) + int(pool.free_bytes())
        limit = int(pool.get_limit())
        if limit > 0:
            # the fit caps the pool (``fe_gpu_pool``): memory above the cap is not usable by the stages even when the device has it free
            avail = min(avail, max(0, limit - int(pool.used_bytes())))  # the cap less what is in use: the pool's cached blocks are reusable inside it
    except ImportError:
        return True  # no cupy: the stages are host stages, there is no device memory to protect
    except Exception as exc:
        # an unreadable device is not a roomy one: the check exists to keep concurrent stages from overrunning the card, and running the cascades one by one is always correct
        log_throttle(
            logger, "_fe_stage_concurrency.except_152", logging.WARNING, "stage concurrency: device memory unreadable (%r); the cascades run one by one", exc
        )
        return False
    if need > avail:
        logger.info(
            "FE stage concurrency: the stages' scaled peak VRAM (%.1f GB at %d rows) exceeds the %.1f GB available; running the cascades one by one",
            need / 2**30,
            n_rows,
            avail / 2**30,
        )
        return False
    return True


def run_stages_concurrently(
    estimator: Any,
    stages: Sequence[Callable[[Any], Any]],
    weights: Sequence[float] = STAGE_WEIGHTS,
    merge_count: "Callable[[list], int] | None" = None,
    vram_weights: "Sequence[float] | None" = None,
    n_rows: "int | None" = None,
) -> "list[Any] | None":
    """Results of ``stages`` (each called with a private copy of ``estimator``) run together, one per device slot, or ``None`` when no plan exists (the caller then runs them in order).

    The estimator's attributes are merged back in stage order before returning. A stage's exception is raised after all workers ended, for the first failing stage in order.
    ``merge_count(outcomes)`` (called with the first stage's result, the rest ``None``) may keep only the first stages' state, as a sequential run that stops after
    them (a dominant candidate) would; the dropped stages' results come back as ``None`` and their failures are ignored, since a sequential run never ran them.
    """
    from .._fe_device_scheduler import plan_stage_slots, stage_device_slot
    from .._fit_context import capture_fit_context, fit_context_scope

    if vram_weights is None:
        vram_weights = STAGE_VRAM_WEIGHTS if len(stages) == 3 else (STAGE_VRAM_WEIGHTS_WITH_EARLY_A if len(stages) == 4 else weights)
    if n_rows is not None and not _stages_fit_device(vram_weights, int(n_rows)):
        return None
    slots = plan_stage_slots(list(weights), vram_weights=list(vram_weights))
    if slots is None or len(slots) != len(stages):
        logger.info("FE stage concurrency: no device plan (a stage would get too little VRAM); running the cascades one by one")
        return None
    base = snapshot(estimator)
    copies = [private_copy(estimator) for _ in stages]
    deadline = getattr(_deadline_state, "deadline", None)
    context = capture_fit_context()
    walls = [0.0] * len(stages)
    input_gb = stage_ledger.frame_gb(int(n_rows), int(getattr(estimator, "n_features_in_", 0) or 0)) if n_rows is not None else None

    def _run(i: int) -> Any:
        """One stage on its copy, in its device slot, with the fit's deadline and scoring settings published to this worker thread."""
        t0 = time.perf_counter()
        epoch0 = time.time() if stage_ledger.enabled() else 0.0
        stage_failed = True
        try:
            with fe_deadline_scope(deadline), fit_context_scope(context), stage_device_slot(slots[i], i):
                out = stages[i](copies[i])
            stage_failed = False
            return out
        finally:
            try:
                close_thread_laps()  # the stage's last family lap ends with the stage, on this worker thread
                if epoch0:  # the stage ledger: the cascade, the device slot it ran in and the step-input frame it read
                    stage_ledger.record(f"fe_stage_{i}", epoch0, time.time(), device=f"gpu:{slots[i].device_id}", in_gb=input_gb)
                _synchronize_device(slots[i].device_id)  # the wall is the stage's work, not the launch of its last kernel
            except Exception as cleanup_exc:
                if not stage_failed:
                    raise
                # the stage's own exception (it names the failing family) must reach the caller, not the sticky device error the cleanup then hits
                log_throttle(
                    logger,
                    "_fe_stage_concurrency.cleanup_after_failure",
                    logging.WARNING,
                    "stage %d: cleanup after the stage's own failure also failed (%r)",
                    i,
                    cleanup_exc,
                )
            finally:
                walls[i] = time.perf_counter() - t0
                stage_finished()  # the cascades still running take over this one's share of the edge workers

    from .._edge_pool import concurrent_stages, stage_finished

    with concurrent_stages(len(stages)), ThreadPoolExecutor(max_workers=len(stages)) as pool:
        futures = [pool.submit(_run, i) for i in range(len(stages))]
    errors = [f.exception() for f in futures]
    # the slowest stage sets the section's wall: these show which cascade to shorten or rebalance
    logger.debug("FE stage concurrency: stage walls %s s", [round(w, 2) for w in walls])
    if merge_count is None:
        keep = len(stages)
    else:
        if errors[0] is not None:
            raise errors[0]
        # merge_count sees the first stage's result (the stages it may drop have not been looked at yet)
        keep = max(1, min(len(stages), int(merge_count([futures[0].result()] + [None] * (len(stages) - 1)))))
    for failed_at, err in enumerate(errors[:keep]):
        if err is not None:
            # a sequential run has the state of the stages BEFORE the failing one on the estimator when the failure surfaces (a caller that survives it keeps their engineered columns and recipes); the
            # concurrent run merged nothing before raising, so those stages' results were lost. The stages after the failing one are not merged: a sequential run never reached them.
            merge_stage_state(estimator, base, copies[:failed_at])
            raise err  # the first failure in stage order among the kept stages; a dropped stage would not have run sequentially
    outcomes: "list[Any]" = [f.result() for f in futures[:keep]]
    merge_stage_state(estimator, base, copies[:keep])
    return outcomes + [None] * (len(stages) - keep)
