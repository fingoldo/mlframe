"""Waves of the FE cascades when the three of them do not fit the host RAM together.

``memory_allows_concurrency`` is all or nothing: early_b, mid_a and mid_b run side by side when the frame times ``_CONCURRENT_FRAME_FACTOR`` fits the available RAM, and one
after the other otherwise. Between the two lie the partitions of the three stages into consecutive groups (waves): the stages of a wave run side by side, the waves one
after the other. A wave of two stages holds about two thirds of the working set of three, so a host that cannot hold the three can often hold two.

The plan is the exact optimum over the order-keeping partitions (there are ``2 ** (k - 1)`` of them for ``k`` stages, four for three): each wave must fit the RAM
(a single stage always does, as in the sequential order), the makespan is the sum over the waves of the longest stage of the wave, and the shortest makespan wins
(fewer waves on a tie). Waves keep the stage order, so the stages' state is merged into the estimator in the fixed cascade order, as in the other two paths; a stage
reads the step-input frame in every path. A wave whose stages the device cannot hold (``run_stages_concurrently`` returns ``None``) runs its stages one by one.

The durations are the relative walls of ``STAGE_WEIGHTS``; the planner of ``filters/_fe_plan`` takes over when the stage set grows beyond what enumeration covers.
"""

from __future__ import annotations

import logging
import os
from itertools import product
from typing import Any, Callable, Optional, Sequence

from mlframe.system import host_memory

from ._fe_stage_concurrency import _CONCURRENT_FRAME_FACTOR, STAGE_VRAM_WEIGHTS, STAGE_WEIGHTS, run_stages_concurrently

logger = logging.getLogger(__name__)


def _partitions(k: int) -> "list[list[tuple[int, ...]]]":
    """Every split of ``0 .. k-1`` into consecutive groups, in order."""
    out = []
    for cuts in product((False, True), repeat=max(0, k - 1)):
        groups, current = [], [0]
        for i, cut in enumerate(cuts, start=1):
            if cut:
                groups.append(tuple(current))
                current = [i]
            else:
                current.append(i)
        groups.append(tuple(current))
        out.append(groups)
    return out


def plan_stage_waves(
    n_rows: int, n_cols: int, weights: Sequence[float] = STAGE_WEIGHTS, available: Optional[float] = None, factor: Optional[float] = None
) -> "list[tuple[int, ...]] | None":
    """The waves of the fastest feasible order-keeping partition of the stages, or ``None`` when it is one stage after another (nothing to run side by side).

    A stage of a wave holds ``factor / k`` frames of working set (``factor`` is the measured working set of the ``k`` stages together, ``_CONCURRENT_FRAME_FACTOR``
    unless ``MLFRAME_FE_STAGE_CONCURRENCY_MEM_FACTOR`` says otherwise); ``available`` is the host RAM free now."""
    k = len(weights)
    if k < 2:
        return None
    if factor is None:
        raw = os.environ.get("MLFRAME_FE_STAGE_CONCURRENCY_MEM_FACTOR", "").strip()
        factor = float(raw) if raw else _CONCURRENT_FRAME_FACTOR
    if available is None:
        try:
            available = float(host_memory()[0])
        except Exception as exc:  # an unreadable RAM figure: the sequential order is always correct
            logger.info("stage waves: available RAM unreadable (%r); the cascades run one after the other", exc)
            return None
    per_stage = float(factor) / k * float(n_rows) * float(n_cols) * 4.0
    best: "tuple[float, int, list[tuple[int, ...]]] | None" = None
    for waves in _partitions(k):
        if any(len(w) > 1 and len(w) * per_stage > available for w in waves):
            continue
        makespan = sum(max(float(weights[i]) for i in w) for w in waves)
        key = (makespan, len(waves), waves)
        if best is None or key[:2] < best[:2]:
            best = key
    if best is None or all(len(w) == 1 for w in best[2]):
        return None
    return best[2]


def run_stage_waves(
    estimator: Any,
    stages: Sequence[Callable[[Any], Any]],
    waves: "Sequence[tuple[int, ...]]",
    n_rows: int,
    weights: Sequence[float] = STAGE_WEIGHTS,
    vram_weights: Sequence[float] = STAGE_VRAM_WEIGHTS,
) -> list:
    """The stages' results in stage order: the waves one after the other, the stages of a wave side by side (each on a private copy of the estimator, its state merged back
    in stage order), a single-stage wave or a wave the device cannot hold on the estimator itself, one stage after another."""
    outs: list = [None] * len(stages)
    for wave in waves:
        together = None
        if len(wave) > 1:
            together = run_stages_concurrently(
                estimator, [stages[i] for i in wave], [weights[i] for i in wave], vram_weights=[vram_weights[i] for i in wave], n_rows=n_rows
            )
        if together is None:
            for i in wave:
                outs[i] = stages[i](estimator)
        else:
            for i, out in zip(wave, together):
                outs[i] = out
    return outs


__all__ = ["plan_stage_waves", "run_stage_waves"]
