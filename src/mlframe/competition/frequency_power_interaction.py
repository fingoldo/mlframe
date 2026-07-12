"""COMPETITION/EXPLORATORY-ONLY utility. NOT for production use.

Implements the "frequency/count encoding with power interaction" trick documented in
``MLFRAME_IDEAS_competitions.md`` (5th place, santander-customer-transaction-prediction):
"create 200 extra features representing the count of unique values for all numerical
features ... using minmaxscaler of (-4,4) on the original features and then doing
``Xn**countn`` where the count is clipped between 1 and 3."

The winning variant is ``X_scaled ** clip(count, 1, 3)`` (raw feature raised to a power
derived from its own per-value occurrence count), NOT ``count ** X``. This exploits the
specific synthetic/anonymized-independent-feature structure of Santander-style
competitions where the per-value occurrence count itself carries categorical-like
signal for otherwise-continuous features (e.g. certain values happen to repeat exactly
N times more often when the row's target is positive). On real continuous production
features this interaction is unmotivated and is very likely to just inject noise.

Never import this module from production mlframe code paths and never wire it into
any default pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import MinMaxScaler


@dataclass
class FrequencyPowerInteractionResult:
    """Container for the outputs of ``frequency_power_interaction``.

    Attributes:
        scaled_feature: The raw feature after MinMax scaling to ``feature_range``.
        counts: Per-value occurrence count of the *raw* (unscaled) feature within
            the input array, one entry per row, aligned with ``x``.
        clipped_counts: ``counts`` clipped to ``count_clip_range`` (the exponent
            actually used).
        interaction_feature: ``scaled_feature ** clipped_counts`` — the new
            engineered feature.
    """

    scaled_feature: npt.NDArray[np.float64]
    counts: npt.NDArray[np.int64]
    clipped_counts: npt.NDArray[np.float64]
    interaction_feature: npt.NDArray[np.float64]


def frequency_power_interaction(
    x: npt.ArrayLike,
    *,
    feature_range: tuple[float, float] = (-4.0, 4.0),
    count_clip_range: tuple[float, float] = (1.0, 3.0),
) -> FrequencyPowerInteractionResult:
    """COMPETITION-ONLY. Not for production use.

    Compute the "value^count" power-interaction feature for a single numerical
    feature array: per-value occurrence count -> MinMax-scale the raw feature to
    ``feature_range`` -> raise the scaled feature to the power of its own
    (clipped) per-value count.

    This assumes ``x`` is a 1-D array of a single numerical feature (call once per
    feature; the source competition applied this to ~200 numerical features
    independently). Counts are computed over the values passed in ``x`` — for the
    "reduce time-dependence" competition variant, pass the concatenation of
    train+test values (see the related ``synthetic_row_detector`` module for the
    caveats of doing that with organizer-injected synthetic rows); for a plain
    single-split usage just pass the feature values of that split.

    Args:
        x: 1-D array-like of a single numerical feature.
        feature_range: Target range for MinMax scaling of the raw feature, default
            ``(-4, 4)`` per the source write-up.
        count_clip_range: ``(low, high)`` clip bounds applied to the per-value
            occurrence count before using it as the exponent, default ``(1, 3)``
            per the source write-up (an unclipped count exponent can blow up for
            highly-repeated values and small ``feature_range`` magnitudes).

    Returns:
        A ``FrequencyPowerInteractionResult`` with the scaled feature, raw and
        clipped counts, and the final interaction feature.

    Raises:
        ValueError: if ``x`` is not 1-D, is empty, or contains non-finite values.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("x must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x must contain only finite values (no NaN/inf)")

    low, high = count_clip_range
    if not (low <= high):
        raise ValueError(f"count_clip_range must have low <= high, got {count_clip_range}")

    uniques, inverse, raw_counts = np.unique(arr, return_inverse=True, return_counts=True)
    counts = raw_counts[inverse].astype(np.int64)

    scaler = MinMaxScaler(feature_range=feature_range)
    scaled = scaler.fit_transform(arr.reshape(-1, 1)).ravel()

    clipped_counts = np.clip(counts.astype(np.float64), low, high)
    interaction = np.power(scaled, clipped_counts)

    return FrequencyPowerInteractionResult(
        scaled_feature=scaled,
        counts=counts,
        clipped_counts=clipped_counts,
        interaction_feature=interaction,
    )
