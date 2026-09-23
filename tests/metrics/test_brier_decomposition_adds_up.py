"""The debiased Brier decomposition must add up to the Brier it is printed beside.

REL was clamped at 0 per bin while RES was debiased independently and not clamped, so whenever a bin clamped,
`REL - RES + UNC` rose above the plug-in binned Brier: on a well-calibrated model with small bins nearly every bin
clamps, and the report's `RL..+U..-RS..` token visibly failed to sum to the `BR` next to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.calibration._calibration_metrics import (
    compute_brier_decomposition_debiased,
    compute_ece_and_brier_decomposition,
)


@pytest.mark.parametrize("n, nbins, seed", [(300, 20, 0), (5000, 10, 1), (120, 50, 2)])
def test_the_identity_holds_and_matches_the_plugin_brier(n, nbins, seed):
    """REL - RES + UNC must equal the Brier printed beside it, on the well-calibrated data where bins clamp."""
    rng = np.random.default_rng(seed)
    p = rng.random(n)
    y = (rng.random(n) < p).astype(np.float64)  # well calibrated: the regime where bins clamp
    rel, res, unc, bb = compute_brier_decomposition_debiased(y, p, nbins)
    plug = compute_ece_and_brier_decomposition(y, p, nbins)
    assert rel - res + unc == pytest.approx(bb, abs=1e-12)
    assert bb == pytest.approx(plug[4], abs=1e-12)
    assert rel >= 0.0


def test_without_a_clamp_resolution_is_the_textbook_debiased_value():
    """Large bins and a strongly miscalibrated model: no bin clamps, so RES must equal RES_plugin - sum w*Var."""
    rng = np.random.default_rng(3)
    n, nbins = 50_000, 5
    p = rng.random(n)
    y = (rng.random(n) < np.clip(p * 0.5, 0, 1)).astype(np.float64)
    _rel, res, _unc, _bb = compute_brier_decomposition_debiased(y, p, nbins)
    idx = np.minimum((p * nbins).astype(int), nbins - 1)
    base = y.mean()
    res_expected = 0.0
    for k in range(nbins):
        m = idx == k
        nk = m.sum()
        acc = y[m].mean()
        res_expected += nk / n * ((acc - base) ** 2 - acc * (1 - acc) / (nk - 1))
    assert res == pytest.approx(res_expected, rel=1e-9)
