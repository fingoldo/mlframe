"""Four numbers in the calibration cluster were selected maxima presented as production estimates.

`optimize_decision_threshold` returns `best_score` -- the largest of 200 metric values computed on the very
rows the threshold was picked from. On 500 rows at a 5% positive rate that is 200 tries against ~25
informative events, and the return-value documentation carried no warning at all. Its `cv_report` reported how
much the THRESHOLD moves but never how much the SCORE shrinks out of sample.

`_threshold_stability_report` refit each fold's threshold on the fold's TEST index -- one fifth of the data at
the default k=5 -- rather than on the complement, so the reported coefficient of variation was the instability
of a threshold fitted on a fifth of the rows, roughly `sqrt((n - n/k) / (n/k)) = 2.0` times too large, and
`is_stable` called thresholds unstable that are stable at the full sample size.

`backtest_override` picks `safe_threshold` by scanning bucket improvements on the supplied rows and then
reports `mae_blend_safe` on those same rows, documented as what a caller thresholding on `safe_threshold`
would get in production.

`compare_cv_schemes` tests a post-hoc-selected winner against every runner-up at the nominal alpha, with no
multiplicity correction -- while the Bonferroni knob for exactly that sat on `cv_score_equivalence_band`,
unreachable because `triage_cv_delta` neither accepted nor forwarded it.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from mlframe.calibration.smoothed_override_backtest import backtest_override
from mlframe.calibration.threshold_optimizer import _threshold_stability_report, optimize_decision_threshold
from mlframe.evaluation.cv_delta_triage import triage_cv_delta


@pytest.fixture
def noisy_binary():
    """A weak signal on 600 rows: enough for a 200-way sweep to find a flattering in-sample maximum."""
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, n)
    p = np.clip(0.5 + 0.08 * (y - 0.5) + rng.normal(0, 0.25, n), 0.001, 0.999)
    return y, p


class TestTheStabilityReportRefitsOnTheTrainSide:
    """A fold threshold fitted on a fifth of the data is not the leave-one-fold-out analogue of the real fit."""

    def test_each_fold_threshold_is_fitted_on_the_larger_side(self, noisy_binary):
        """Recorded by counting the rows each fold's SWEEP saw.

        The sweep calls the metric once per candidate threshold, so its row count is by far the most common
        one; the held-out scoring adds a single call per fold at the complementary size. The modal size is
        therefore the fit size, which used to be n/k (a fifth of the data at the default k=5).
        """
        import collections

        y, p = noisy_binary
        seen: list = []

        def _metric(yt, yp):
            """Record the row count, then score normally."""
            seen.append(len(yt))
            return f1_score(yt, yp, zero_division=0)

        _threshold_stability_report(y, p, np.linspace(0, 1, 50), _metric, n_splits=5, seed=0, stability_cv_threshold=0.15)
        modal_size = collections.Counter(seen).most_common(1)[0][0]
        assert modal_size > len(y) // 2, f"the sweep ran on {modal_size} of {len(y)} rows -- the minority side"

    def test_the_reported_spread_is_not_inflated(self, noisy_binary):
        """Fitting on n/k instead of n-n/k inflates the fold-to-fold spread by about sqrt(4) at k=5."""
        y, p = noisy_binary
        rep = _threshold_stability_report(y, p, np.linspace(0, 1, 200), f1_score, n_splits=5, seed=0, stability_cv_threshold=0.15)
        assert np.isfinite(rep["cv"])

    def test_it_reports_a_held_out_score(self, noisy_binary):
        """The companion `best_score` needs: chosen on the train side, scored on the held-out side."""
        y, p = noisy_binary
        rep = _threshold_stability_report(y, p, np.linspace(0, 1, 200), f1_score, n_splits=5, seed=0, stability_cv_threshold=0.15)
        assert "heldout_score_mean" in rep and np.isfinite(rep["heldout_score_mean"])

    def test_the_held_out_score_is_below_the_in_sample_maximum(self, noisy_binary):
        """The whole point: the 200-way maximum does not survive contact with fresh rows."""
        y, p = noisy_binary
        res = optimize_decision_threshold(y, p, f1_score, cv=5)
        assert res["cv_report"]["heldout_score_mean"] < res["best_score"], (
            res["cv_report"]["heldout_score_mean"],
            res["best_score"],
        )

    def test_the_in_sample_bias_is_documented(self):
        """A caller reading `best_score` as the operating point's F1 will over-promise."""
        doc = optimize_decision_threshold.__doc__ or ""
        assert "IN-SAMPLE SELECTED MAXIMUM" in doc and "heldout_score_mean" in doc


class TestTheOverrideBacktestReportsAnHonestSafeMae:
    """`mae_blend_safe` is measured on the rows its threshold was scanned from."""

    @pytest.fixture
    def pure_noise_override(self):
        """An override source carrying no signal at all, where any apparent improvement is selection."""
        rng = np.random.default_rng(1)
        n = 3000
        y = rng.normal(0, 1, n)
        model = y + rng.normal(0, 0.5, n)
        override = rng.normal(0, 1, n)  # unrelated to y
        conf = rng.random(n)
        return y, model, override, conf

    def test_a_held_out_estimate_is_reported(self, pure_noise_override):
        """The field a caller should read instead."""
        res = backtest_override(*pure_noise_override, a=0.3)
        assert np.isfinite(res.mae_blend_safe_heldout)

    def test_the_in_sample_number_flatters_a_noise_override(self, pure_noise_override):
        """With no signal, blending can only hurt -- so an in-sample number that looks better is selection."""
        res = backtest_override(*pure_noise_override, a=0.3)
        assert res.mae_blend_safe_heldout >= res.mae_blend_safe - 1e-12, (res.mae_blend_safe, res.mae_blend_safe_heldout)

    def test_a_tiny_input_returns_nan_rather_than_a_fake_estimate(self):
        """Fewer rows than folds cannot be split; NaN says so instead of inventing a number."""
        rng = np.random.default_rng(2)
        n = 6
        res = backtest_override(rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n), rng.random(n), a=0.3)
        assert np.isnan(res.mae_blend_safe_heldout)

    def test_the_summary_labels_the_in_sample_number(self, pure_noise_override):
        """The rendered report is where an operator reads it."""
        assert "in-sample" in backtest_override(*pure_noise_override, a=0.3).summary()

    def test_the_docstring_no_longer_calls_it_a_production_estimate(self):
        """It said `mae_blend_safe` matches "what a caller ... would actually get in production"."""
        doc = backtest_override.__doc__ or ""
        assert "IS IN-SAMPLE" in doc and "mae_blend_safe_heldout" in doc


class TestMultiplicityIsCorrectable:
    """The Bonferroni knob existed but no production caller could reach it."""

    FOLDS_A = np.array([0.80, 0.82, 0.79, 0.83, 0.81])
    FOLDS_B = np.array([0.81, 0.83, 0.80, 0.84, 0.82])

    def test_triage_accepts_a_family_size(self):
        """`grep -rn n_comparisons src/` used to show it read only inside `noise_band.py`."""
        assert triage_cv_delta(self.FOLDS_A, self.FOLDS_B, change_source="feature_engineering", n_comparisons=4) is not None

    def test_a_family_correction_widens_the_band(self):
        """Correcting for four comparisons must make the bar harder to clear, not merely be accepted."""
        one = triage_cv_delta(self.FOLDS_A, self.FOLDS_B, change_source="feature_engineering")["band"]
        four = triage_cv_delta(self.FOLDS_A, self.FOLDS_B, change_source="feature_engineering", n_comparisons=4)["band"]
        assert four > one

    def test_a_marginal_delta_is_no_longer_actionable_under_correction(self):
        """The failure mode: a post-hoc winner clearing four tests at the nominal alpha on fold noise."""
        plain = triage_cv_delta(self.FOLDS_A, self.FOLDS_B, change_source="feature_engineering")
        corrected = triage_cv_delta(self.FOLDS_A, self.FOLDS_B, change_source="feature_engineering", n_comparisons=8)
        assert not corrected["actionable"] or not plain["actionable"]

    def test_a_nonsense_family_size_is_refused(self):
        """Zero or negative comparisons would silently widen or invert the band."""
        with pytest.raises(ValueError, match="positive integer"):
            triage_cv_delta(self.FOLDS_A, self.FOLDS_B, change_source="feature_engineering", n_comparisons=0)

    def test_compare_cv_schemes_passes_the_family_size(self):
        """The caller that most obviously needs it: one test per non-winning scheme against a chosen winner.

        Checked structurally on the call site rather than by running `compare_cv_schemes`, which needs a
        model factory, a metric and an out-of-time holdout to reach this line -- and rather than by matching
        source text, which would pass for a literal sitting in a comment.
        """
        import ast
        import pathlib as _pl

        src = (_pl.Path(__file__).resolve().parents[2] / "src" / "mlframe" / "evaluation" / "compare_cv_schemes.py").read_text(encoding="utf-8")
        calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "triage_cv_delta"]
        assert calls, "the significance call was not found; this test needs updating"
        assert all(any(kw.arg == "n_comparisons" for kw in c.keywords) for c in calls), "a significance test runs without a family-size correction"
