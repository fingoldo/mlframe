"""Device barrier for wall-clock timings that may cover GPU work.

CUDA launches are asynchronous: a cupy / torch / numba.cuda call returns once the work is queued,
so ``perf_counter()`` around it measures launch overhead, not compute. Timing helpers call
``synchronize_gpu_if_available()`` right before reading the stop timer (and before the start timer
when earlier async work may still be in flight) so the interval covers the device work.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def synchronize_gpu_if_available() -> None:
    """Block until pending cupy / torch / numba.cuda work on the current device completes.

    Only libraries already present in ``sys.modules`` are synchronized: a library that was never
    imported cannot have queued device work, and importing it here would add seconds of import cost
    to CPU-only timings. Never raises - a missing driver or device makes this a no-op.
    """
    cp = sys.modules.get("cupy")
    if cp is not None:
        try:
            cp.cuda.Device().synchronize()
        except Exception as e:  # nosec B110 - no usable device means nothing to wait for
            logger.debug("cupy synchronize skipped: %s", e)
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
        except Exception as e:  # nosec B110
            logger.debug("torch synchronize skipped: %s", e)
    numba_cuda = sys.modules.get("numba.cuda")
    if numba_cuda is not None:
        try:
            if numba_cuda.is_available():
                numba_cuda.synchronize()
        except Exception as e:  # nosec B110
            logger.debug("numba.cuda synchronize skipped: %s", e)
