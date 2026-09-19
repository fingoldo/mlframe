"""Right-tail compressors are wrong for a left-skewed target.

``log_y`` / ``cbrt_y`` / ``box_cox_y`` / ``signed_power_y`` (exponent in (0, 1]) all compress the RIGHT tail. A target
skewed to the LEFT -- a 0..5 feedback score piled up at 5 with a tail toward 0 (production skew -9.6 / -11.3) -- gets its
skew deepened by them, yet discovery auto-enabled them on those targets because the heavy-tail detector only looks at
|skew| and kurtosis. ``yeo_johnson_y`` is not in the list: its fitted lambda goes above 1 for a left tail, which is the
correct remedy.
"""

from __future__ import annotations

import numpy as np

RIGHT_TAIL_COMPRESSORS = frozenset({"log_y", "cbrt_y", "box_cox_y", "signed_power_y"})

# Skew below this counts as left-skewed enough to rule the compressors out.
LEFT_SKEW_THRESHOLD = -1.0


def left_skewed_right_tail_skips(y: np.ndarray) -> frozenset:
    """Transform names to skip for ``y``: the right-tail compressors when ``y`` is left-skewed, else nothing."""
    arr = np.asarray(y, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 30 or np.ptp(arr) == 0:
        return frozenset()
    if arr.size > 200_000:
        arr = arr[np.random.default_rng(0).choice(arr.size, size=200_000, replace=False)]
    from scipy.stats import skew

    return RIGHT_TAIL_COMPRESSORS if float(skew(arr)) < LEFT_SKEW_THRESHOLD else frozenset()
