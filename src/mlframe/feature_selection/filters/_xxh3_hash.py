"""The xxh3 64-bit hash used as a fast content fingerprint, or ``None`` when xxhash is unusable.

The caller decides what a missing hash means for its own cache key or duplicate test; this module only resolves the
function once and reports why it could not, through the caller's logger so the message stays under that module's name.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, cast

__all__ = ["resolve_xxh3_64"]


def resolve_xxh3_64(log: logging.Logger, fallback: str) -> Optional[Callable]:
    """Return ``xxhash.xxh3_64_intdigest``, or ``None`` when it cannot be imported or fails to load.

    ``fallback`` names what the caller does instead (for example ``"falls back to pandas hash_array"``); it is quoted in the
    log record for either failure, so an unusable install is never silent.
    """
    try:
        import xxhash as _xxhash

        return cast(Callable[..., Any], _xxhash.xxh3_64_intdigest)
    except ImportError as e:
        log.debug("xxhash unavailable, %s: %s", fallback, e)
        return None
    except Exception as e:
        # Resolved once per process, so a broken install would otherwise pin every fit onto the slower path without a trace.
        log.warning("xxhash is installed but unusable (%s: %s); %s", type(e).__name__, e, fallback)
        return None
