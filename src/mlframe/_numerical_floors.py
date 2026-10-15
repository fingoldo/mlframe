"""Shared numerical floors for variance and scale guards."""

# Below this, a variance is treated as zero: it is float64 rounding noise on a constant column, not signal.
VARIANCE_FLOOR: float = 1e-24
