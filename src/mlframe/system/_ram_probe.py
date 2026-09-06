"""One psutil available-RAM probe, shared by the callers that size work against free memory.

Two modules carried their own ``_available_ram_bytes`` -- a filters one and a shap-proxy one -- with the
same body and DIFFERENT failure sentinels: ``-1`` meaning "no cap" and ``None`` meaning "drop the RAM term
from the auto-size". Merging them into one sentinel would have changed both contracts, so the probe is
shared and each caller keeps the sentinel its own callers already branch on.

The part worth sharing is the probe itself: the psutil import, the exception envelope, and the decision
that a failed probe is never fatal to the work it sizes. That is what drifts when a copy is edited.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def available_ram_bytes(caller: str = "caller") -> Optional[int]:
    """Host available RAM in bytes, or ``None`` when psutil is absent or the probe fails.

    Never raises: a sizing hint that cannot be read must leave the caller free to proceed uncapped rather
    than take down the work it was meant to bound.
    """
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception as exc:
        logger.debug("available_ram_bytes: psutil unavailable or failed (%s); %s proceeds without the RAM term", exc, caller)
        return None
