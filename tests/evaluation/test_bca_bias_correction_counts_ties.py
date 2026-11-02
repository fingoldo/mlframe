"""BCa's bias correction counts resamples tied with the point estimate as half below (tracker 2026-11-02 wave 12, STAT12-3)."""

import numpy as np

from mlframe.evaluation._bootstrap_jackknife import _ci_from_samples


def test_symmetric_discrete_distribution_gives_a_symmetric_interval():
    """400 below, 200 equal, 400 above the point: z0 must be 0, so with a zero acceleration the BCa interval equals the percentile one."""
    samples = np.concatenate([np.full(400, 0.4), np.full(200, 0.5), np.full(400, 0.6)])
    rng = np.random.default_rng(0)
    samples = samples + rng.normal(scale=1e-3, size=1000) * (samples != 0.5)
    jackknife = np.array([0.49, 0.5, 0.51, 0.5, 0.5])
    lo, hi = _ci_from_samples(samples, 0.5, 0.05, "bca", jackknife)
    plo, phi = _ci_from_samples(samples, 0.5, 0.05, "percentile")
    assert abs(lo - plo) < 0.02 and abs(hi - phi) < 0.02
