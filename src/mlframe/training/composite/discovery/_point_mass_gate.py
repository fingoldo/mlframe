"""Curved y-compressors cannot reconstruct the spread of a target that is mostly one value.

When a large fraction of ``y`` sits on a single value -- the canonical case being a zero-inflated amount, where the
event simply did not happen for most rows -- a curved unary transform maps that whole mass onto one point in T. A
regression model then predicts close to that point for most rows, and the CONVEX inverse squeezes what little spread
the T-scale predictions had back into a narrow band. The reconstructed prediction is near-constant, which is not a
modelling failure the fit can recover from: it is what the transform pair does to a point mass.

A production run demonstrated it twice on ``target_total_charge`` (median 0, mean 85.65, so more than half the rows
were exactly 0):

    [regression-collapse-sensor:std-collapse] ... target_total_charge-logY  pred_std=6.44 (1.8% of target_std=349)
    [regression-collapse-sensor:std-collapse] ... target_total_charge-cbrtY pred_std=4.49 (1.3% of target_std=349)

The collapse sensor caught both -- after 2.6 and 5.6 minutes of GPU each. The point mass is visible in the target's
own summary statistics before discovery starts, so the check belongs here, mirroring the left-skew gate in
``_skew_gate.py`` that already refuses right-tail compressors on the wrong-shaped target.

Only the curved compressors are refused. Clipping-style transforms (``y_quantile_clip``) keep a piecewise-linear
inverse and survive a point mass fine -- in the same production run that one was the single composite of the three
that beat raw y.
"""

from __future__ import annotations

import numpy as np

CURVED_Y_COMPRESSORS = frozenset({"log_y", "cbrt_y", "box_cox_y", "signed_power_y", "yeo_johnson_y"})
"""Unary y-transforms with a curved (convex) inverse. Unlike the left-skew gate, ``yeo_johnson_y`` IS included: its
lambda fit does nothing about a point mass, which is a problem of concentration rather than of tail direction."""

POINT_MASS_FRACTION_THRESHOLD = 0.5
"""Fraction of rows sharing one value above which a curved inverse is refused. At half the sample the conditional
mean in T-space is dominated by the mass and the reconstruction has no spread left to recover."""

MIN_ROWS_FOR_POINT_MASS_CHECK = 100
"""Below this the modal fraction is too noisy to act on."""


def point_mass_fraction(y: np.ndarray) -> float:
    """Largest fraction of finite rows in ``y`` sharing a single exact value; 0.0 when undecidable.

    Exact equality on purpose: the failure mode comes from a genuine atom in the distribution (an amount that is
    exactly zero because nothing happened), not from a dense continuous region.
    """
    arr = np.asarray(y, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < MIN_ROWS_FOR_POINT_MASS_CHECK:
        return 0.0
    if arr.size > 200_000:
        arr = arr[np.random.default_rng(0).choice(arr.size, size=200_000, replace=False)]
    counts = np.unique(arr, return_counts=True)[1]
    return float(counts.max() / arr.size)


def point_mass_curved_inverse_skips(y: np.ndarray, threshold: float = POINT_MASS_FRACTION_THRESHOLD) -> frozenset:
    """Transform names to skip for ``y``: the curved y-compressors when ``y`` carries a dominant point mass."""
    return CURVED_Y_COMPRESSORS if point_mass_fraction(y) >= threshold else frozenset()
