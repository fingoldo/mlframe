"""Regression: the KS reference-CDF plotting positions must reach ~1.0.

The pre-fix reference levels ``(arange(k)+1)/(k+1)`` capped the train-side CDF at
``k/(k+1) < 1``, leaving a constant ~``1/(k+1)`` positive bias on EVERY KS statistic ->
false drift alerts even on a no-drift batch. Midpoint positions ``(arange(k)+0.5)/k``
span (0,1) and reach ~1.0 at the top knot, so a no-drift batch scores ~0.
"""

import numpy as np

from mlframe.training.composite.monitoring import _ks_statistic


def test_no_drift_batch_has_near_zero_ks():
    """KS against a reference sketch built from the batch's own quantile knots sits at the intrinsic knot-quantization floor, not inflated by formula bias."""
    rng = np.random.default_rng(0)
    batch = rng.normal(size=200_000)
    # The cleanest no-drift condition: the reference sketch is the batch's OWN quantile
    # knots, so any KS above the intrinsic ~1/k knot-quantization floor is pure
    # reference-CDF-formula bias, not sampling noise between two independent draws. With
    # many knots the floor shrinks (~1/k); the pre-fix (i+1)/(k+1) levels keep the train
    # CDF below 1.0 and inflate KS by an extra ~1/(k+1) offset on top of the floor, while
    # the post-fix (i+0.5)/k midpoints sit right at the floor.
    k = 512
    ref_knots = np.quantile(batch, (np.arange(k) + 0.5) / k)
    ks = _ks_statistic(ref_knots, batch)

    # Floor (post-fix) ~0.00196; pre-fix ~0.00292. Threshold separates the two.
    assert ks < 0.0025, f"self-referenced no-drift KS should sit at the knot floor, got {ks}"


def test_real_drift_still_detected():
    """A genuinely shifted batch still produces a clearly elevated KS above the no-drift knot floor."""
    rng = np.random.default_rng(1)
    k = 64
    ref_knots = np.quantile(rng.normal(size=20_000), (np.arange(k) + 0.5) / k)
    shifted = rng.normal(loc=2.0, size=20_000)
    assert _ks_statistic(ref_knots, shifted) > 0.5
