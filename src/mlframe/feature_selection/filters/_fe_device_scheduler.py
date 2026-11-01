"""Assign independent FE stages to GPUs and give each a VRAM budget of its own.

Running the independent stage cascades concurrently on one card exhausted the process-wide cupy pool (every stage allocates from the same pool, whose cap is a share of the card): the log filled with
``Out of memory allocating ...`` and the exact-path fallbacks. Here every stage gets a *slot*: a device and a byte budget. Inside :func:`stage_device_slot` the thread allocates from a private
``MemoryPool`` limited to that budget (cupy's ``using_allocator`` is thread-local), so a stage that outgrows its budget raises ``OutOfMemoryError`` inside its own pool, where the device paths already fall
back to the host, instead of taking memory from its neighbours.

The plan follows the hardware it finds: any number of devices with different free memory and different SM counts. Stages are placed heaviest first on the device whose projected finish time
(load / SM count) is smallest, and each device splits its usable memory (free memory minus the cushion) between the stages placed on it in proportion to their weights. A plan that cannot give every
stage the minimum budget is refused (``None``) and the caller keeps the sequential cascade.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

logger = logging.getLogger(__name__)

#: Smallest budget a stage may be given; below it the stage's device paths would mostly fall back to the host.
MIN_STAGE_BUDGET_BYTES = 256 << 20


@dataclass(frozen=True)
class DeviceSlot:
    """A device and the VRAM budget one stage may use on it."""

    device_id: int
    budget_bytes: int


def _device_facts(cp: Any, device_ids: "Sequence[int] | None") -> "list[tuple[int, int, int]]":
    """``[(device id, free bytes, SM count)]`` of the usable devices."""
    ids = list(device_ids) if device_ids is not None else list(range(int(cp.cuda.runtime.getDeviceCount())))
    facts = []
    for d in ids:
        with cp.cuda.Device(d):
            free_b, _total = cp.cuda.runtime.memGetInfo()
            sms = int(cp.cuda.runtime.getDeviceProperties(d)["multiProcessorCount"])
        facts.append((int(d), int(free_b), max(1, sms)))
    return facts


def plan_stage_slots(
    weights: Sequence[float],
    *,
    device_ids: "Sequence[int] | None" = None,
    cushion_bytes: int = 1 << 30,
    min_budget_bytes: int = MIN_STAGE_BUDGET_BYTES,
    facts: "Sequence[tuple[int, int, int]] | None" = None,
) -> "list[DeviceSlot] | None":
    """One :class:`DeviceSlot` per stage (same order as ``weights``), or ``None`` when some stage would get less than ``min_budget_bytes``.

    ``weights`` are the relative costs of the stages (their measured walls, say). ``facts`` overrides the probe with ``[(device id, free bytes, SM count)]``, for tests and for hosts that probed already.
    """
    if not weights:
        return []
    if facts is None:
        try:
            import cupy as cp

            facts = _device_facts(cp, device_ids)
        except Exception as exc:
            logger.debug("stage scheduler: no usable device (%s: %s)", type(exc).__name__, exc)
            return None
    if not facts:
        return None
    order = sorted(range(len(weights)), key=lambda i: -float(weights[i]))  # heaviest first
    load = {d: 0.0 for d, _f, _s in facts}
    placed: "dict[int, list[int]]" = {d: [] for d, _f, _s in facts}
    sms = {d: s for d, _f, s in facts}
    for i in order:
        best = min(load, key=lambda d: (load[d] + float(weights[i])) / sms[d])
        load[best] += float(weights[i])
        placed[best].append(i)
    slots: "list[DeviceSlot | None]" = [None] * len(weights)
    for d, free_b, _s in facts:
        usable = int(free_b) - int(cushion_bytes)
        group = placed[d]
        if not group:
            continue
        total_w = sum(max(float(weights[i]), 1e-9) for i in group)
        for i in group:
            budget = int(usable * max(float(weights[i]), 1e-9) / total_w)
            if budget < min_budget_bytes:
                return None
            slots[i] = DeviceSlot(device_id=d, budget_bytes=budget)
    return [s for s in slots if s is not None]


#: Pools of the stage slots, kept alive for the whole process: an array allocated inside a slot can outlive the slot (the process-global caches keep device columns), and cupy frees the in-use chunks of a pool when
#: the pool object is destroyed, so a dropped pool turned those cached arrays into dangling pointers (``cudaErrorIllegalAddress`` in the next kernel that read one).
_SLOT_POOLS: "dict[tuple[int, int], Any]" = {}
_SLOT_POOLS_LOCK = threading.Lock()


def _slot_pool(slot: DeviceSlot, key: int) -> Any:
    """The persistent pool of stage ``key`` on the slot's device, its limit set to the slot's budget."""
    import cupy as cp

    with _SLOT_POOLS_LOCK:
        pool = _SLOT_POOLS.get((slot.device_id, key))
        if pool is None:
            pool = cp.cuda.MemoryPool()
            _SLOT_POOLS[(slot.device_id, key)] = pool
    pool.set_limit(size=int(slot.budget_bytes))
    return pool


@contextlib.contextmanager
def stage_device_slot(slot: DeviceSlot, key: int = 0) -> "Iterator[Any]":
    """Run the calling thread on ``slot.device_id`` with the persistent memory pool of stage ``key``, limited to ``slot.budget_bytes``; the pool's unused blocks are released on exit (its in-use blocks stay valid for the arrays that outlive the slot)."""
    import cupy as cp

    pool = _slot_pool(slot, key)
    try:
        with cp.cuda.Device(slot.device_id), cp.cuda.using_allocator(pool.malloc):
            yield pool
    finally:
        pool.free_all_blocks()
