"""The streaming change-point scan must find the same break whatever constant the data carries.

The split scan built prefix sums over RAW products and then recovered each segment's variance by
subtraction, cancelling twice. On data carrying a large offset -- an epoch timestamp, a price level -- the
segment SSE came apart: measured against an offset-free truth of 0.985561, the raw form returned 100.0 at
offset 1e7 and EXACTLY 0.0 at epoch scale. Zero is the damaging one: it takes the `best_sse <= 0.0` early
return, so a real regime change is reported as `found=False`, indistinguishable from a stationary buffer,
and the caller then refits across the dead and live regimes together.

Every OLS here carries its own intercept, so adding a constant to `base` and `y` shifts only that intercept
and leaves the residuals -- hence every SSE and the F statistic -- unchanged. These tests assert exactly that
invariance, which is a property of the statistic rather than a frozen number.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.streaming import _detect_change_point

# No offset, a large one, and a real epoch-second timestamp -- the regime that returned 0.0.
_OFFSETS = (0.0, 1.0e3, 1.0e7, 1.7e9)


def _planted_break(n: int = 400, k: int = 200, offset: float = 0.0):
    """A clean slope/level break at row `k`, with `offset` added to both series."""
    b0 = np.linspace(0.0, 1.0, n)
    signal = np.where(np.arange(n) < k, 2.0 * b0, 2.0 * b0 + 3.0)
    noise = np.random.default_rng(0).normal(scale=0.05, size=n)
    return b0 + offset, signal + noise + offset


@pytest.mark.parametrize("offset", _OFFSETS, ids=[f"offset_{o:.0e}" for o in _OFFSETS])
def test_a_planted_break_is_found_at_every_offset(offset: float):
    """The break exists in the residuals, which no constant shift can touch."""
    base, y = _planted_break(offset=offset)
    result = _detect_change_point(y, base)
    assert result["found"], f"offset={offset:.0e}: a planted break was reported as absent (f_stat={result['f_stat']}, sse_split={result['sse_split']})"


@pytest.mark.parametrize("offset", _OFFSETS, ids=[f"offset_{o:.0e}" for o in _OFFSETS])
def test_the_split_sse_is_offset_invariant(offset: float):
    """Adding a constant shifts each segment's intercept and nothing else."""
    base0, y0 = _planted_break(offset=0.0)
    expected = _detect_change_point(y0, base0)["sse_split"]

    base, y = _planted_break(offset=offset)
    got = _detect_change_point(y, base)["sse_split"]

    assert got == pytest.approx(expected, rel=1e-6), f"offset={offset:.0e}: split SSE {got:.6f} against the offset-free {expected:.6f}"


def test_the_f_statistic_is_offset_invariant():
    """`f_stat` divides sse_full by sse_split; both are offset-invariant, so their ratio is too.

    This is the assertion that would have caught the original defect most directly: `sse_full` was already
    computed on centred data while the split scan was not, so only one side of the ratio was stable.
    """
    base0, y0 = _planted_break(offset=0.0)
    expected = _detect_change_point(y0, base0)["f_stat"]

    base, y = _planted_break(offset=1.7e9)
    got = _detect_change_point(y, base)["f_stat"]

    assert got == pytest.approx(expected, rel=1e-6), f"F statistic moved from {expected:.4f} to {got:.4f} purely from an epoch-scale offset"


def test_a_stationary_buffer_still_reports_no_break():
    """The rewrite must not turn the scan into something that finds a break in everything."""
    n = 400
    base = np.linspace(0.0, 1.0, n) + 1.7e9
    y = 2.0 * np.linspace(0.0, 1.0, n) + np.random.default_rng(1).normal(scale=0.05, size=n) + 1.7e9

    assert not _detect_change_point(y, base)["found"], "a stationary buffer was reported as containing a change point"
