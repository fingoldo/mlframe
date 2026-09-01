"""`optimal_threshold_bootstrap_ci` had no test at all, and it was wrong.

The kernel sweeps each resample maintaining incremental counts, deriving `tn = N - fp` and `fn = P - tp`. Those
are confusion counts only while the walk is monotone in score -- every row seen so far must be a row scoring at
or above the current cut. The caller drew `idx` as random POSITIONS into the descending-sorted arrays and handed
them over unsorted, so each resample was swept in random order and the counts corresponded to no threshold.

The kernel's docstring asserted the opposite ("resampling positions keeps every resample in sorted order for
free"), which is what let it stand. Measured against a reference bootstrap that re-solves the threshold from
scratch per resample, the shipped interval was 8.7x too wide on a well-identified threshold; it is 1.01x now.

The number matters because this interval is printed as `95% CI [lo, hi]` beside the tuned threshold, and it is
what an operator reads to decide whether the tuning is trustworthy at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.classification._threshold_optimization import optimal_threshold, optimal_threshold_bootstrap_ci


def _thr(y, s, metric="f1"):
    """The point estimate, unwrapped from whatever shape `optimal_threshold` returns."""
    r = optimal_threshold(y, s, metric=metric)
    return float(r[0]) if isinstance(r, tuple) else float(r)


def _overlapping(n: int = 4000, seed: int = 7, pos_rate: float = 0.30):
    """Classes that overlap, so the optimal threshold is well identified and ties cannot dominate."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(np.int64)
    s = np.clip(np.where(y == 1, rng.normal(0.62, 0.16, n), rng.normal(0.38, 0.16, n)), 0.0, 1.0)
    return y, s


def _reference_ci(y, s, n_boot=200, seed=0, metric="f1"):
    """The honest bootstrap: re-solve the threshold from scratch on each resample."""
    rng = np.random.default_rng(seed)
    thresholds = [_thr(y[i], s[i], metric) for i in rng.integers(0, len(y), size=(n_boot, len(y)))]
    lo, hi = np.quantile(thresholds, [0.025, 0.975])
    return float(lo), float(hi)


class TestTheIntervalMatchesAnHonestBootstrap:
    """The discriminating property: agreement with a re-solve-per-resample reference."""

    def test_the_width_is_not_inflated(self):
        """Pre-fix this ratio was 8.7 on exactly this fixture."""
        y, s = _overlapping()
        lo, hi = optimal_threshold_bootstrap_ci(y, s, metric="f1", n_boot=300, random_state=0)
        r_lo, r_hi = _reference_ci(y, s, n_boot=300, seed=1)
        ratio = (hi - lo) / (r_hi - r_lo)
        assert 0.5 < ratio < 2.0, f"interval width is {ratio:.1f}x the honest bootstrap's ([{lo:.4f}, {hi:.4f}] vs [{r_lo:.4f}, {r_hi:.4f}])"

    def test_the_endpoints_track_the_reference(self):
        """A correct interval sits where the honest one sits, not merely at a similar width."""
        y, s = _overlapping()
        lo, hi = optimal_threshold_bootstrap_ci(y, s, metric="f1", n_boot=300, random_state=0)
        r_lo, r_hi = _reference_ci(y, s, n_boot=300, seed=1)
        span = r_hi - r_lo
        assert abs(lo - r_lo) < 0.5 * span, f"lower endpoint {lo:.4f} is far from the reference {r_lo:.4f}"
        assert abs(hi - r_hi) < 0.5 * span, f"upper endpoint {hi:.4f} is far from the reference {r_hi:.4f}"

    def test_the_point_estimate_lies_inside(self):
        """A percentile interval that excludes its own point estimate is not describing the same quantity."""
        y, s = _overlapping()
        lo, hi = optimal_threshold_bootstrap_ci(y, s, metric="f1", n_boot=300, random_state=0)
        assert lo <= _thr(y, s) <= hi

    def test_more_data_narrows_it(self):
        """A bootstrap interval must shrink with n; a random-order sweep had no reason to."""
        small = optimal_threshold_bootstrap_ci(*_overlapping(n=600, seed=3), metric="f1", n_boot=200, random_state=0)
        large = optimal_threshold_bootstrap_ci(*_overlapping(n=8000, seed=3), metric="f1", n_boot=200, random_state=0)
        assert (large[1] - large[0]) < (small[1] - small[0]), f"width did not shrink with n: {small} -> {large}"


class TestTheContract:
    """The documented edges, none of which had coverage."""

    def test_it_is_reproducible_from_the_seed(self):
        """The draw is made once up front precisely so the result is seed-reproducible."""
        y, s = _overlapping(n=1500)
        a = optimal_threshold_bootstrap_ci(y, s, n_boot=100, random_state=11)
        b = optimal_threshold_bootstrap_ci(y, s, n_boot=100, random_state=11)
        assert a == b

    def test_an_empty_input_gives_nan(self):
        """Documented: both endpoints are nan for an empty input."""
        lo, hi = optimal_threshold_bootstrap_ci(np.array([], dtype=np.int64), np.array([], dtype=np.float64))
        assert np.isnan(lo) and np.isnan(hi)

    @pytest.mark.parametrize("metric", ["f1", "youden"])
    def test_each_metric_produces_an_ordered_interval(self, metric):
        """Whatever the metric, `lo <= hi` and both are real cut points or the documented +inf."""
        y, s = _overlapping(n=1200)
        lo, hi = optimal_threshold_bootstrap_ci(y, s, metric=metric, n_boot=100, random_state=0)
        assert lo <= hi

    def test_an_unknown_metric_raises(self):
        """The guard exists; nothing exercised it."""
        y, s = _overlapping(n=200)
        with pytest.raises(ValueError, match="metric must be one of"):
            optimal_threshold_bootstrap_ci(y, s, metric="not_a_metric")
