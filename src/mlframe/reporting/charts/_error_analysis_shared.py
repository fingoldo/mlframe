"""Constants and array helpers shared by ``error_analysis`` and its carved-out sibling modules.

These live here rather than in either module so both import ONE definition. A second copy of the overlay bin
count or the drift z-quantile would drift silently, and the two modules would then disagree about the same
chart's binning -- the exact failure the renderer clusters' duplicated constants produced.
"""

from __future__ import annotations

import numpy as np

# Histogram resolution for per-feature / target overlays; above ~60 the density curves turn into noisy combs
# at the row counts this subsystem sees.
DEFAULT_OVERLAY_BINS: int = 40
# Two-sided 95% normal quantile, used to size the target-drift bar from each split's own sampling error.
_DRIFT_Z: float = 1.96


def _as_float_1d(a: np.ndarray) -> np.ndarray:
    """Coerce ``a`` to a flat float64 array."""
    return np.asarray(a, dtype=np.float64).ravel()


__all__ = ["DEFAULT_OVERLAY_BINS", "_DRIFT_Z", "_as_float_1d"]
