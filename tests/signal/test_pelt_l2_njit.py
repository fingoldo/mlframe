"""Coverage for signal._pelt_l2_njit.pelt_l2, previously untested (only the module's own docstring
claim -- 13x faster than ruptures.Pelt(model="l2") with bit-identical breakpoints -- existed
unverified by any test)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.signal._pelt_l2_njit import pelt_l2

pytestmark = pytest.mark.fast


def _mean_shift_signal(seed=0):
    """A clean 3-segment mean-shift signal with an obvious changepoint at 100 and 200."""
    rng = np.random.default_rng(seed)
    seg1 = rng.normal(0.0, 0.1, 100)
    seg2 = rng.normal(5.0, 0.1, 100)
    seg3 = rng.normal(-3.0, 0.1, 100)
    return np.concatenate([seg1, seg2, seg3])


def test_pelt_l2_detects_obvious_mean_shifts():
    """A 3-segment mean-shift signal must yield breakpoints near the true 100/200 boundaries."""
    y = _mean_shift_signal()
    bps = pelt_l2(y, min_size=5, penalty=10.0)
    assert isinstance(bps, list)
    assert len(bps) >= 2
    # Sorted ascending, within a small window of the true boundaries.
    assert bps == sorted(bps)
    assert any(abs(b - 100) <= 5 for b in bps)
    assert any(abs(b - 200) <= 5 for b in bps)


def test_pelt_l2_no_changepoints_on_constant_signal():
    """A flat (zero-variance) signal has no genuine mean shift; a high penalty should find none."""
    y = np.zeros(50)
    bps = pelt_l2(y, min_size=5, penalty=1000.0)
    assert bps == []


def test_pelt_l2_higher_penalty_finds_fewer_or_equal_breakpoints():
    """Penalty monotonicity: a stricter (higher) penalty never finds MORE breakpoints than a looser one."""
    y = _mean_shift_signal(seed=1)
    bps_loose = pelt_l2(y, min_size=5, penalty=1.0)
    bps_strict = pelt_l2(y, min_size=5, penalty=1000.0)
    assert len(bps_strict) <= len(bps_loose)


def test_pelt_l2_matches_ruptures_reference():
    """Bit-identical breakpoints against ruptures.Pelt(model="l2"), per the module's own performance-parity claim."""
    ruptures = pytest.importorskip("ruptures")
    y = _mean_shift_signal(seed=2)
    ours = pelt_l2(y, min_size=5, penalty=10.0)
    ref_algo = ruptures.Pelt(model="l2", min_size=5, jump=1).fit(y)
    ref_bps = ref_algo.predict(pen=10.0)
    # ruptures includes the trailing n as its final "breakpoint"; ours does not.
    ref_bps_no_tail = [b for b in ref_bps if b != len(y)]
    assert ours == ref_bps_no_tail
