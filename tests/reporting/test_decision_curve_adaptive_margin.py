"""The decision-curve usefulness verdict must scale its bar with sample size.

Net benefit is a difference of per-row rates, so its sampling error shrinks as ``1/sqrt(n)``. The verdict
used a FLAT ``1e-3`` margin at every n, and at n=2000 a purely random score wanders ~0.01 above the
reference envelope on noise alone -- ten times that bar -- so random predictions were stamped "USEFUL" on
the figure, which is the single most consequential thing this chart says.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts.decision_curve import _MIN_USEFULNESS_MARGIN, _usefulness_margin, build_decision_curve_spec


def test_margin_shrinks_with_n_and_never_goes_below_the_floor():
    """The bar must track 1/sqrt(n) in the regime that matters, then stop at the numerical floor."""
    assert _usefulness_margin(2_000) > _usefulness_margin(200_000)
    # 1/sqrt(n): a 100x larger sample should relax the bar by ~10x.
    assert _usefulness_margin(2_000) / _usefulness_margin(200_000) == pytest.approx(10.0, rel=0.05)
    # An enormous n must not drive the bar to zero, where FP rounding alone would read as a gain.
    assert _usefulness_margin(10**9) == _MIN_USEFULNESS_MARGIN
    # Degenerate input must not raise or produce a negative bar.
    assert _usefulness_margin(0) == _MIN_USEFULNESS_MARGIN


@pytest.mark.parametrize("n", [500, 2000, 20000])
def test_random_scores_are_never_called_useful(n):
    """A score with no signal must fail the verdict at every sample size.

    Under the previous flat bar this failed at n=2000: the measured advantage of a random score was 0.0100
    against a 0.001 threshold.
    """
    rng = np.random.default_rng(0)
    verdicts = []
    for _ in range(8):
        y = rng.integers(0, 2, n)
        verdicts.append(build_decision_curve_spec(y, rng.random(n)).useful)
    assert not any(verdicts), f"random score declared useful at n={n}: {verdicts}"


@pytest.mark.parametrize("n", [500, 2000, 20000])
def test_genuinely_informative_scores_still_pass(n):
    """Raising the bar must not cost real detections -- the verdict has to stay useful for a real model."""
    rng = np.random.default_rng(1)
    verdicts = []
    for _ in range(8):
        y = rng.integers(0, 2, n)
        score = np.clip(0.35 * y + 0.65 * rng.random(n), 0, 1)
        verdicts.append(build_decision_curve_spec(y, score).useful)
    assert all(verdicts), f"informative score rejected at n={n}: {verdicts}"
