"""Audit 2026-11-20 finding 29-F20: smoothed rank percentiles stay in the band the hard mode uses, [0.5/n, (n-0.5)/n].

The stacker was trained on hard percentiles in that band, so a smoothed value of exactly 0 or 1 was outside its training scale. The test
checks the smoothed OOF and test percentiles lie inside the band, including for a test value beyond the OOF range.
"""

import numpy as np

from mlframe.votenrank.rank_percentile_stacking import rank_percentile_transform


def test_smoothed_percentiles_are_inside_the_hard_band():
    """Every smoothed percentile is at least 0.5/n and at most (n-0.5)/n, for OOF and out-of-range test values."""
    oof = np.linspace(0.0, 1.0, 20)
    test = np.array([-5.0, 0.5, 9.0])
    oof_p, test_p = rank_percentile_transform(oof, test, smoothing=0.1)
    n = len(oof)
    lo, hi = 0.5 / n, (n - 0.5) / n
    assert oof_p.min() >= lo - 1e-12 and oof_p.max() <= hi + 1e-12
    assert test_p.min() >= lo - 1e-12 and test_p.max() <= hi + 1e-12
