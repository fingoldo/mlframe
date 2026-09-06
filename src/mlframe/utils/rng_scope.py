"""Save/restore scopes for the global RNG state numpy, numba and cupy each keep separately.

Seeding is contagious in a way that is easy to miss: ``np.random.seed(s)`` inside an ``@njit`` body sets
*numba's* per-thread global stream, not numpy's, and numba exposes no portable ``get_state``. A helper that
seeds for its own determinism therefore repositions the stream of every later njit kernel on that thread,
and the caller has no way to notice.

``feature_selection/filters/screen.py`` worked this out first and restores numba and cupy by drawing fresh
entropy-derived seeds on entry and re-seeding with them on exit -- a snapshot in spirit, since an
unpredictable new position is as good as the old one for any consumer that is not itself seeded. This module
is that block, extracted so the other call sites can use it instead of rediscovering the hazard.
"""

from __future__ import annotations

import logging
import os
import struct
from contextlib import contextmanager
from typing import Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)


# numba types the seed argument as int64, so anything at or above 2**63 raises OverflowError on the way in.
# A full 64-bit draw crosses that line about half the time, and because the restore is best-effort (it must
# not mask whatever the guarded block did), the exception lands in a debug log and the stream is left exactly
# where the block put it -- the failure this scope exists to prevent, occurring silently on half the calls.
_SEED_MASK = (1 << 63) - 1


def _fresh_seed() -> int:
    """An unpredictable seed that numba can actually accept, for leaving a stream where the caller did not choose."""
    return int(struct.unpack("<Q", os.urandom(8))[0]) & _SEED_MASK


@contextmanager
def numba_rng_scope(seed: Optional[int] = None) -> Iterator[None]:
    """Leave numba's global RNG stream unrepositioned by whatever runs inside.

    With ``seed``, the block is seeded for its own determinism; on exit the stream is moved to a fresh
    entropy-derived position rather than left where the block's draws ended. Without ``seed``, nothing is
    seeded on entry and the exit re-seed is skipped, matching ``screen.py``'s rule that an unseeded run
    never touches a stream it did not set.

    numpy is untouched here -- it has a real ``get_state``/``set_state`` pair, so callers that need it
    should use :func:`numpy_rng_scope`, or both.
    """
    from pyutilz.data.numbalib import set_numba_random_seed

    if seed is None:
        yield
        return

    restore_seed = _fresh_seed()
    set_numba_random_seed(int(seed))
    try:
        yield
    finally:
        try:
            set_numba_random_seed(int(restore_seed))
        except Exception as exc:  # nosec B110 - best-effort restoration must not mask the block's own outcome
            logger.debug("numba RNG restoration seed failed: %s", exc)


@contextmanager
def numpy_rng_scope(seed: Optional[int] = None) -> Iterator[None]:
    """Restore numpy's global RNG state exactly, whether or not the block seeded it."""
    snapshot = np.random.get_state()
    try:
        if seed is not None:
            np.random.seed(seed)
        yield
    finally:
        np.random.set_state(snapshot)
