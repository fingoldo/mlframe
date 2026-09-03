"""The noise band multiplied an ESTIMATED standard error by a normal quantile.

`cv_score_equivalence_band(method="sem")` -- the default -- returned `z_{1-alpha/2} * sem`. The SE is estimated
from `n_folds` values, so the correct multiplier is `t_{n-1, 1-alpha/2}`. At the canonical k=5 that is 2.776
against 1.960: the shipped band was 29% too narrow and covered ~86%, not the documented 95%.

The direction matters. A band that is too narrow ACCEPTS noise, and every consumer -- `is_within_noise_band`,
`triage_cv_delta`, `CVDeltaHistory.pooled_band` -- exists to stop a selection loop doing exactly that.
`calibration/policy._heldout_ece_ci` had already made this correction for the same reason.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from mlframe.evaluation.cv_delta_triage import CVDeltaHistory
from mlframe.evaluation.noise_band import cv_score_equivalence_band, is_within_noise_band

FOLDS = np.array([0.80, 0.82, 0.79, 0.83, 0.81])


class TestTheMultiplierIsStudentT:
    """Stated as the exact number, since "wider" alone would pass for any inflation."""

    def test_the_band_is_the_t_half_width(self):
        """`t_{4,0.975} * sem`, not `1.96 * sem`."""
        sem = FOLDS.std(ddof=1) / np.sqrt(5)
        assert cv_score_equivalence_band(FOLDS) == pytest.approx(stats.t.ppf(0.975, 4) * sem)

    def test_it_is_wider_than_the_normal_band(self):
        """The regression direction: never silently narrow again."""
        sem = FOLDS.std(ddof=1) / np.sqrt(5)
        assert cv_score_equivalence_band(FOLDS) > stats.norm.ppf(0.975) * sem

    @pytest.mark.parametrize("k,expected", [(3, 4.303), (5, 2.776), (10, 2.262), (30, 2.045)])
    def test_the_quantile_tracks_the_fold_count(self, k, expected):
        """A z-based band is constant in k; a t-based one shrinks toward z as the SE estimate improves."""
        rng = np.random.default_rng(0)
        scores = 0.8 + rng.normal(0, 0.02, k)
        sem = scores.std(ddof=1) / np.sqrt(k)
        assert cv_score_equivalence_band(scores) / sem == pytest.approx(expected, abs=0.01)

    def test_it_converges_to_z_at_large_k(self):
        """t is the right answer at every k, including where it stops mattering."""
        rng = np.random.default_rng(1)
        scores = 0.8 + rng.normal(0, 0.02, 2000)
        sem = scores.std(ddof=1) / np.sqrt(2000)
        assert cv_score_equivalence_band(scores) / sem == pytest.approx(stats.norm.ppf(0.975), abs=0.005)


class TestTheConsumersInherit:
    """A wider band has to reach the functions that decide."""

    def test_a_delta_between_the_two_bands_is_now_called_noise(self):
        """The concrete false accept: a delta the z band cleared and the t band does not."""
        sem = FOLDS.std(ddof=1) / np.sqrt(5)
        delta = 0.5 * (stats.norm.ppf(0.975) + stats.t.ppf(0.975, 4)) * sem
        assert is_within_noise_band(0.80, 0.80 + delta, FOLDS), "a delta inside the honest band was accepted as real"

    def test_a_genuinely_large_delta_is_still_actionable(self):
        """Widening must not swallow real improvements."""
        assert not is_within_noise_band(0.80, 0.95, FOLDS)

    def test_the_std_method_is_untouched(self):
        """Only the `"sem"` branch applies a quantile."""
        assert cv_score_equivalence_band(FOLDS, method="std") == pytest.approx(FOLDS.std(ddof=1))

    def test_bonferroni_still_widens(self):
        """The multiple-comparisons correction composes with the new quantile rather than replacing it."""
        assert cv_score_equivalence_band(FOLDS, n_comparisons=20) > cv_score_equivalence_band(FOLDS)


class TestPooledBandUsesItsOwnDof:
    """`pooled_band` pools variance across history, so it must pool the degrees of freedom too."""

    def test_a_single_update_gets_the_small_sample_quantile(self):
        """One 5-fold update carries 4 dof. The SE is that of a DIFFERENCE of two means, hence sqrt(2/n)."""
        h = CVDeltaHistory()
        h.update(FOLDS)
        se = h.pooled_std * np.sqrt(2 / 5)
        assert h.pooled_band(5) == pytest.approx(stats.t.ppf(0.975, 4) * se)

    def test_accumulated_history_converges_toward_z(self):
        """The payoff for tracking dof: a long history is not penalised with a k=5 quantile forever."""
        h = CVDeltaHistory()
        rng = np.random.default_rng(2)
        for _ in range(200):
            h.update(0.8 + rng.normal(0, 0.02, 5))
        assert h.pooled_band(5) / (h.pooled_std * np.sqrt(2 / 5)) == pytest.approx(stats.norm.ppf(0.975), abs=0.01)

    def test_the_pooled_band_is_never_narrower_than_normal(self):
        """t >= z always; a pooled band below the normal half-width would mean the dof wiring is inverted."""
        h = CVDeltaHistory()
        h.update(FOLDS)
        assert h.pooled_band(5) >= stats.norm.ppf(0.975) * (h.pooled_std * np.sqrt(2 / 5))

    def test_no_history_still_returns_none(self):
        """Unchanged contract."""
        assert CVDeltaHistory().pooled_band(5) is None


class TestTheBandBracketsADifferenceOfTwoMeans:
    """The second, independent way the band was too narrow.

    `triage_cv_delta` and `is_within_noise_band` both test a DIFFERENCE of two fold-score means against a band
    derived as if it bracketed ONE mean. The difference of two independent means has standard error sqrt(2)
    times either one's, so every such comparison was held to a bar up to 29% below the quantity being tested.
    Combined with the z-vs-t defect above, the measured false-positive rate on a null delta was 12-26% against
    a nominal 5%.
    """

    def test_the_two_sample_band_is_wider_than_the_one_mean_band(self):
        """The direct statement."""
        from mlframe.evaluation.noise_band import two_sample_score_band

        other = FOLDS + 0.01
        assert two_sample_score_band(FOLDS, other) > cv_score_equivalence_band(FOLDS)

    def test_it_is_the_classic_two_sample_t_half_width(self):
        """Pinned as the formula, not just as "bigger"."""
        from mlframe.evaluation.noise_band import two_sample_score_band

        b = np.array([0.78, 0.84, 0.80, 0.79, 0.85])
        se = np.sqrt((FOLDS.var(ddof=1) + b.var(ddof=1)) / 5)
        assert two_sample_score_band(FOLDS, b) == pytest.approx(stats.t.ppf(0.975, 8) * se)

    def test_it_does_not_collapse_when_the_delta_is_constant_across_folds(self):
        """The reason this is not the paired-difference band: a uniform shift would give a zero band, making
        every delta actionable and zeroing out any multiplier applied to it downstream."""
        from mlframe.evaluation.noise_band import two_sample_score_band

        assert two_sample_score_band(FOLDS, FOLDS + 0.02) > 0.0

    def test_triage_holds_the_nominal_rate_on_a_null_delta(self):
        """The end-to-end property. Under the old one-mean band this rate was ~0.26."""
        from mlframe.evaluation.cv_delta_triage import triage_cv_delta

        rng = np.random.default_rng(42)
        hits = sum(
            triage_cv_delta(0.7 + rng.normal(0, 0.01, 4), 0.7 + rng.normal(0, 0.01, 4), change_source="feature_engineering")["actionable"] for _ in range(400)
        )
        assert hits / 400 <= 0.10, f"false-positive rate on a null delta is {hits / 400}, nominal alpha is 0.05"

    def test_a_real_improvement_is_still_actionable(self):
        """Widening the band must not make the triage useless."""
        from mlframe.evaluation.cv_delta_triage import triage_cv_delta

        rng = np.random.default_rng(3)
        base = 0.70 + rng.normal(0, 0.005, 5)
        assert triage_cv_delta(base, base + 0.05 + rng.normal(0, 0.005, 5), change_source="feature_engineering")["actionable"]

    def test_the_opt_out_reproduces_the_one_mean_band(self):
        """`both_estimated=False` is for comparing an estimate against a fixed reference."""
        assert is_within_noise_band(0.80, 0.80 + cv_score_equivalence_band(FOLDS) * 1.05, FOLDS, both_estimated=False) is False
        assert is_within_noise_band(0.80, 0.80 + cv_score_equivalence_band(FOLDS) * 0.95, FOLDS, both_estimated=False) is True

    def test_mismatched_fold_counts_are_refused(self):
        """A silently-broadcast comparison would produce a meaningless band."""
        from mlframe.evaluation.noise_band import two_sample_score_band

        with pytest.raises(ValueError, match="same shape"):
            two_sample_score_band(FOLDS, FOLDS[:3])
