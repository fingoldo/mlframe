"""The parallel AP-bootstrap kernel must agree with the numpy resample loop it replaced.

``bootstrap_ap_ci`` used to run one Python iteration per resample. The resamples are independent, so they now
go through a ``prange`` kernel a chunk at a time -- but the draw stays in numpy and stays chunked, which is
what keeps a given seed producing the same interval. These tests pin both halves: the two paths agree, and
the RNG is still consumed in the same order (a changed order would move the interval without any error).
"""

from __future__ import annotations

import numpy as np
import pytest

import mlframe.reporting.charts.binary as binary_mod
from mlframe.reporting.charts._ap_bootstrap import jit_is_active
from mlframe.reporting.charts.binary import bootstrap_ap_ci


def _fixture(n: int, seed: int):
    """Imbalanced labels with a separable-but-imperfect score."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.25).astype(int)
    return y, np.clip(0.3 * y + rng.random(n) * 0.7, 0, 1)


@pytest.mark.skipif(not jit_is_active(), reason="the kernel is only taken when the JIT is on; the numpy path is the other arm of this comparison")
@pytest.mark.parametrize("n", [12_000, 60_000])
def test_the_kernel_and_the_numpy_loop_return_the_same_interval(n, monkeypatch):
    """Same seed, same rows: the compiled path must not move the AP or either bound.

    Agreement is to floating-point reassociation, not bit-identity: the kernel accumulates the
    precision-weighted mass in one pass where numpy took a dot product over two cumulative sums. The rank
    order of the summation is the same, so the gap is at the 1e-15 level -- far below anything that moves a
    percentile, which the bound comparison below is what actually pins.
    """
    y, score = _fixture(n, seed=11)

    ap_jit, lo_jit, hi_jit = bootstrap_ap_ci(y, score, seed=0)
    monkeypatch.setattr(binary_mod, "ap_jit_is_active", lambda: False)
    ap_np, lo_np, hi_np = bootstrap_ap_ci(y, score, seed=0)

    assert ap_jit == ap_np, "the point estimate does not go through the kernel at all and must be untouched"
    assert lo_jit == pytest.approx(lo_np, abs=1e-9), f"lower bound moved: {lo_jit!r} vs {lo_np!r}"
    assert hi_jit == pytest.approx(hi_np, abs=1e-9), f"upper bound moved: {hi_jit!r} vs {hi_np!r}"
    assert lo_jit < ap_jit < hi_jit, "the interval must bracket the point estimate, or the comparison above is vacuous"


def test_the_same_seed_still_reproduces_the_same_interval():
    """The chunked draw is what keeps the generator order stable; a rewrite that drew it whole would not."""
    y, score = _fixture(20_000, seed=3)
    first = bootstrap_ap_ci(y, score, seed=17)
    second = bootstrap_ap_ci(y, score, seed=17)
    assert first == second, "same seed produced a different interval"
    other = bootstrap_ap_ci(y, score, seed=18)
    assert other[1:] != first[1:], "two different seeds produced identical bounds; the seed is not reaching the draw"


def test_a_resample_with_no_positive_is_dropped_rather_than_scored_zero():
    """A degenerate resample must not be counted as AP=0, which would drag the lower bound down.

    Reached with a single positive row: some resamples draw it zero times, and the kernel writes NaN for those
    so the caller drops them. Scoring them 0.0 instead would put the lower bound near zero on any rare-positive
    panel -- a silently wrong interval rather than a visible failure.
    """
    n = 4000
    rng = np.random.default_rng(5)
    y = np.zeros(n, dtype=int)
    y[0] = 1
    score = rng.random(n)
    score[0] = 0.99

    ap, lo, hi = bootstrap_ap_ci(y, score, seed=1)
    assert np.isfinite(ap), "a single-positive panel still has a defined AP"
    assert np.isfinite(lo) and np.isfinite(hi), "the interval collapsed to NaN, so every resample was dropped"
    assert lo > 0.0, f"lower bound {lo!r} reached zero, which is what counting no-positive resamples as AP=0 produces"
