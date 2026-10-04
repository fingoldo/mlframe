"""Gated polars -> pandas conversion for the opt-in paths whose consumers only accept pandas."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

PANDAS_BRIDGE_WARN_BYTES = 2 * 1024**3


def polars_to_pandas_gated(X: Any, site: str) -> pd.DataFrame:
    """Convert a polars frame to pandas through the Arrow split-blocks bridge, warning when the frame is at least ``PANDAS_BRIDGE_WARN_BYTES``.

    Frame-format conversion is the caller's decision, made once at the suite boundary; these call sites are opt-in paths whose consumer cannot read polars,
    so the frame is converted here but the cost is made visible (the default ``to_pandas`` also consolidates blocks, a second transient copy).
    """
    nbytes = int(X.estimated_size()) if hasattr(X, "estimated_size") else 0
    if nbytes >= PANDAS_BRIDGE_WARN_BYTES:
        log_throttle(
            logger, f"polars_to_pandas_gate_{site}", logging.WARNING,
            "%s: converting a %.1f GB polars frame to pandas; pass a pandas frame (or disable this stage) to avoid the copy.", site, nbytes / 1024**3,
        )
    try:
        return X.to_pandas(split_blocks=True)
    except TypeError:
        return X.to_pandas()
