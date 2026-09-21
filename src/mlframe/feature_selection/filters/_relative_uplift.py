"""The relative-uplift ratio every orthogonal FE family ranks its candidates by.

``engineered_mi / baseline_mi`` answers "how much more does the engineered column tell us about the target than its source did". The ratio
is only defined when the source told us something: a baseline of zero means the question has no answer, not that the answer is huge. Padding
the denominator (``mi / (baseline + 1e-12)``) answers it anyway, with a number around ``mi * 1e12`` that clears every ``min_uplift`` gate in
the codebase and sorts first, so a column built on a noise source outranks one built on a genuinely improved signal. An undefined uplift is
reported as NaN instead, which every consumer already handles the way this finding wants: ``sort_values(ascending=False)`` puts NaN last, and
``uplift >= min_uplift`` is False for NaN.
"""

from __future__ import annotations

import math

import numpy as np

_UNDEFINED = float("nan")


def relative_uplift(engineered_mi, baseline_mi) -> float:
    """``engineered_mi / baseline_mi``, or NaN when the baseline is missing, non-finite or not positive."""
    if baseline_mi is None:
        return _UNDEFINED
    baseline = float(baseline_mi)
    if not math.isfinite(baseline) or baseline <= 0.0:
        return _UNDEFINED
    return float(engineered_mi) / baseline


def relative_uplift_array(engineered_mi, baseline_mi):
    """The elementwise form of :func:`relative_uplift`, for a whole replicate of engineered columns at once."""
    baseline = np.asarray(baseline_mi, dtype=np.float64)
    usable = np.isfinite(baseline) & (baseline > 0.0)
    return np.where(usable, np.asarray(engineered_mi, dtype=np.float64) / np.where(usable, baseline, 1.0), _UNDEFINED)
