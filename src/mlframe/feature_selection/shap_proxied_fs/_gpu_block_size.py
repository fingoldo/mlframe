"""One kernel_tuning_cache block-size lookup, shared by the shap-proxy GPU kernels.

Three modules carried their own ``_block_size`` differing only in the cache key, the entry field and the
fallback constant. A copy-pasted lookup keeps working, so nothing forces the copies to stay in step: a fix
to the swallowed-exception list, or to what counts as a usable entry, reaches whichever module the author
happened to open. The parameters are what actually differed, so they are arguments now.
"""

from __future__ import annotations

import logging
from typing import Any, cast

logger = logging.getLogger(__name__)


def block_size_from_tuning_cache(kernel_key: str, entry_field: str, default: int) -> int:
    """Hardware-tuned CUDA block size for ``kernel_key``, or ``default``.

    Falls back to ``default`` when the cache is unavailable, has no entry for this kernel, or the entry
    carries no usable ``entry_field`` -- a tuning miss must never be fatal to the kernel it sizes.
    """
    try:
        from mlframe.feature_selection.filters import get_kernel_tuning_cache

        ktc = get_kernel_tuning_cache()
        if ktc is not None:
            entry = cast(Any, ktc).lookup(kernel_key)  # may be absent -> fall through to the default
            if isinstance(entry, dict) and entry.get(entry_field):
                return int(entry[entry_field])
    except (ImportError, KeyError, ValueError, TypeError) as exc:
        logger.debug("block-size tuning-cache lookup failed for %s/%s, using default %d: %s", kernel_key, entry_field, default, exc)
    return int(default)
