"""The worm plot's shape verdict was inoperative: every non-normal residual read as "light tails".

Its HEAVY-TAILS branch required `_rt > 0 and _lt < 0`, a combination no real departure produces. Standardising
by the sample sd -- which the outliers themselves inflate -- pulls BOTH tails of a heavy-tailed sample inside the
normal quantiles, giving `_rt < 0, _lt > 0`. Measured on six known distributions at n=300k, every non-normal one
fell through to the final `else`: a Student-t(3) with excess kurtosis +52 and a lognormal with skew +5.5 were
both reported as light-tailed, in a default-on panel whose whole purpose is to tell an operator whether the
RMSE and prediction-interval assumptions hold.

The two tail medians also cannot separate heavy from light by themselves: uniform residuals (excess kurtosis
-1.2) give the same signs as t(8) (+1.4) and a larger magnitude. They separate ASYMMETRY cleanly, so they now
decide skew, and excess kurtosis decides the tails.

This file is a truth table over distributions whose shape is known by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts.regression import _worm_panel

N = 300_000


def _verdict(resid: np.ndarray) -> str:
    """The shape clause of the panel title -- what an operator actually reads."""
    return _worm_panel(resid, np.zeros_like(resid)).title.splitlines()[0].split("? ")[1]


@pytest.fixture(scope="module")
def rng():
    """One generator, so the fixtures are reproducible across the whole table."""
    return np.random.default_rng(0)


class TestTheVerdictMatchesTheKnownShape:
    """Each case's shape is a property of the distribution it is drawn from, not of the fixture."""

    def test_gaussian_reads_as_normal(self, rng):
        """The only case the previous table got right."""
        assert "normal" in _verdict(rng.normal(0, 1, N))

    @pytest.mark.parametrize("df,label", [(3, "extreme"), (8, "mild")])
    def test_student_t_reads_as_heavy_tailed(self, rng, df, label):
        """Both were reported as LIGHT tails before -- the verdict was inverted, not merely vague."""
        assert "HEAVY TAILS" in _verdict(rng.standard_t(df, N)), f"t({df}) ({label} excess kurtosis) not flagged as heavy"

    def test_a_lognormal_reads_as_right_skewed(self, rng):
        """Skew +5.5; the old RIGHT-SKEW branch needed both medians positive and never fired."""
        assert "RIGHT-SKEW" in _verdict(rng.lognormal(0, 1, N))

    def test_a_mirrored_lognormal_reads_as_left_skewed(self, rng):
        """The mirror case must land on the mirror verdict, not on the same one."""
        assert "LEFT-SKEW" in _verdict(-rng.lognormal(0, 1, N))

    def test_a_uniform_reads_as_light_tailed(self, rng):
        """The one distribution that genuinely IS light-tailed."""
        assert "LIGHT TAILS" in _verdict(rng.uniform(-1, 1, N))

    def test_no_case_falls_through_to_a_catch_all(self, rng):
        """The failure mode was a table whose branches were unreachable, so every case hit the else."""
        verdicts = {
            _verdict(rng.normal(0, 1, N)),
            _verdict(rng.standard_t(3, N)),
            _verdict(rng.lognormal(0, 1, N)),
            _verdict(-rng.lognormal(0, 1, N)),
            _verdict(rng.uniform(-1, 1, N)),
        }
        assert len(verdicts) == 5, f"distinct distributions produced only {len(verdicts)} distinct verdicts: {verdicts}"


class TestTheVerdictDoesNotDependOnRowCount:
    """The decimation is a no-op below its cap, so a size-sensitive verdict is a defect by itself."""

    @pytest.mark.parametrize("n", [1_500, 50_000, 300_000])
    def test_a_gaussian_is_normal_at_every_size(self, rng, n):
        """Same distribution, same answer."""
        assert "normal" in _verdict(rng.normal(0, 1, n))

    @pytest.mark.parametrize("n", [5_000, 300_000])
    def test_a_heavy_tail_is_heavy_at_every_size(self, rng, n):
        """The decimation keeps both tails verbatim, which used to bias the excursion count with n."""
        assert "HEAVY TAILS" in _verdict(rng.standard_t(3, n))


class TestTheDegenerateInputsStillShortCircuit:
    """The panel's existing guards must survive the rewrite."""

    def test_constant_residuals_are_refused(self):
        """Zero variance has no shape to report."""
        panel = _worm_panel(np.zeros(1000), np.zeros(1000))
        assert "constant" in getattr(panel, "text", "") or "constant" in getattr(panel, "title", "")
