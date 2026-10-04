"""Four report verdicts stated something the numbers behind them did not support.

  * Every regression report recorded `regression_panels` in `charts["failed"]`, for a panel grid that does not
    exist. `binary_panels` defaults non-empty, so the guard opens for a regression report, the dispatcher
    returns None because no branch matched, and nothing raised -- so no `panel_exceptions` entry either. An
    operator reads FAILED and goes looking for a rendering bug, and a genuine failure in that slot is
    indistinguishable from the no-op.
  * The residual audit's mild-skew branch appended a rationale and fell through, so residuals with skew +2.0
    were reported as "residuals look ~Gaussian: |skew|=2.00 (< 0.3)" under the verdict "Gaussian
    (well-behaved)" with the advice that MSE is appropriate.
  * For a zero-mean regression target the analyzer substituted the ABSOLUTE std for `rel_std`, compared it
    against a relative threshold, printed it under the name `rel_std`, and early-returned -- so daily returns
    with std 8e-4 were labelled degenerate and every other detector was skipped.
  * The learning curve is computed once per REPORTED split and labelled its two series "train score" and
    "holdout score". On a test report both are scores on disjoint subsets of the test rows.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

REPORTING_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "mlframe" / "training" / "reporting" / "_reporting.py"


class TestAPanelGridThatDoesNotExistIsNotAFailure:
    """Skipped and failed have to be different buckets."""

    def test_the_no_op_branch_records_skipped_rather_than_failed(self):
        """A regression report with no panel grid and no exception lands in ``skipped`` and leaves ``failed`` empty."""
        from mlframe.training.reporting._reporting import _account_panel_render

        metrics: dict = {}
        _account_panel_render(metrics, "regression", None, [], "base")
        charts = metrics["charts"]
        assert not charts["failed"], f"a grid that does not exist was recorded as failed: {charts['failed']}"
        assert "regression_panels" in charts["skipped"]

    def test_a_real_panel_failure_still_lands_in_failed(self):
        """The bucket must keep working for the case it was meant for: a branch that matched and then raised."""
        from mlframe.training.reporting._reporting import _account_panel_render

        metrics: dict = {}
        _account_panel_render(metrics, "regression", None, ["boom"], "base")
        charts = metrics["charts"]
        assert charts["failed"] == ["regression_panels"]
        assert charts["panel_exceptions"] == ["boom"]
        assert not charts.get("skipped")


class TestTheResidualVerdictMatchesTheSkew:
    """A rationale that restates a threshold as though it held is worse than no rationale.

    The fixture is deliberately MILD. A strongly-skewed residual is caught by the formal normality test and
    reported as "Empirically non-Normal", so it never reaches the contradictory sentence. The gap the defect
    lives in is a skew above the 0.3 band that the K-squared test lacks the power to reject: at n=200 a
    standardised gamma(60) gives skew +0.32 and a formal test that does not reject. At n=400 the same shape is
    rejected by the formal test and never reaches the contradictory sentence, which is why the fixture is small.
    """

    def _diagnose(self, shape, n=200):
        """Standardised gamma residuals at the requested shape, with a Normal-band kurtosis."""
        from mlframe.training.targets.regression_residual_audit import audit_residuals

        rng = np.random.default_rng(7)
        resid = rng.gamma(shape=shape, scale=1.0, size=n)
        resid = (resid - resid.mean()) / resid.std()
        y_pred = rng.normal(0, 1, n)
        return audit_residuals(y_true=y_pred + resid, y_pred=y_pred)

    def test_a_skew_above_the_band_is_not_called_gaussian(self):
        """The headline defect: |skew| = 0.32 reported under "Gaussian (well-behaved)"."""
        report = self._diagnose(60)
        assert abs(report.skew) >= 0.3, f"the fixture drifted to skew {report.skew:+.3f}; it would prove nothing"
        assert report.hypothesis != "Gaussian (well-behaved)", f"skew {report.skew:+.3f} was reported as {report.hypothesis!r}"

    def test_the_rationale_does_not_claim_a_threshold_that_failed(self):
        """The sentence "|skew|=0.32 (< 0.3)" contradicts itself."""
        report = self._diagnose(60)
        assert "(< 0.3)" not in " ".join(report.rationale), report.rationale

    def test_the_verdict_names_the_asymmetry(self):
        """An operator has to be able to see WHY it is not the plain Gaussian verdict."""
        assert "skew" in self._diagnose(60).hypothesis.lower()

    def test_a_symmetric_residual_still_gets_the_gaussian_verdict(self):
        """The fix must not make the verdict unreachable. A large gamma shape is essentially Normal."""
        report = self._diagnose(100_000)
        assert abs(report.skew) < 0.3 and report.hypothesis == "Gaussian (well-behaved)", (report.skew, report.hypothesis)


class TestZeroMeanTargetsAreNotCalledDegenerate:
    """A relative test needs something to be relative to."""

    def _report(self, y):
        """Run the regression branch of the target-distribution analyzer."""
        from mlframe.training.targets._target_distribution_analyzer_target_fn import analyze_target_distribution

        return analyze_target_distribution(y)

    @staticmethod
    def _centred(scale, seed, n=5000):
        """A target with an EXACTLY zero mean, which is what a residual series is.

        A random draw's mean is ~1e-5, not <= 1e-9, so it takes the ratio branch and would exercise nothing.
        """
        y = np.random.default_rng(seed).normal(0, scale, n)
        return y - y.mean()

    def test_zero_mean_returns_are_not_near_constant(self):
        """Daily returns: mean ~0, std 8e-4, a perfectly healthy dispersion."""
        assert not any("near_constant" in p for p in self._report(self._centred(8e-4, 0)).pathologies)

    def test_the_other_detectors_still_run_for_a_zero_mean_target(self):
        """The early return suppressed heavy-tail, skew, multi-modal and AR detection for that target."""
        y = np.random.default_rng(1).standard_t(3, 5000) * 8e-4
        assert len(self._report(y - y.mean()).diagnostics) > 2

    def test_a_constant_zero_mean_target_is_still_flagged(self):
        """An all-zero target has no dispersion at all, and must not reach the downstream detectors."""
        report = self._report(np.zeros(5000))
        assert any("near_constant" in p or "single_class" in p for p in report.pathologies), report.pathologies

    def test_the_zero_mean_branch_reports_an_absolute_spread(self):
        """The printed name has to match the printed quantity, which `rel_std` did not."""
        d = self._report(self._centred(8e-4, 4)).diagnostics
        assert "rel_std" not in d and d["abs_std"] == pytest.approx(d["std"])

    def test_the_printed_name_matches_the_printed_quantity(self):
        """`rel_std` was the name on an absolute standard deviation."""
        d = self._report(self._centred(8e-4, 2)).diagnostics
        assert "rel_std" not in d or d["rel_std"] == pytest.approx(abs(d["std"]) / abs(d["mean"]), rel=1e-6)

    def test_a_nonzero_mean_target_still_uses_the_ratio(self):
        """The path that was correct must be untouched."""
        d = self._report(np.random.default_rng(3).normal(100.0, 1.0, 5000)).diagnostics
        assert d["rel_std"] == pytest.approx(abs(d["std"]) / abs(d["mean"]), rel=1e-6)


class TestTheLearningCurveNamesItsSplit:
    """Neither series is a train-split quantity when the curve runs on the reported split."""

    def _panel(self, source_split):
        """A learning-curve panel over a small ridge fit."""
        from sklearn.linear_model import Ridge
        from sklearn.metrics import get_scorer

        from mlframe.training.diagnostics import compute_learning_curve
        from mlframe.training.diagnostics.learning_curve import learning_curve_panel

        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (300, 3))
        y = X[:, 0] * 2 + rng.normal(0, 0.2, 300)
        res = compute_learning_curve(lambda: Ridge(), X, y, sizes=(0.3, 0.6, 1.0), scorer=get_scorer("r2"), scorer_name="r2", higher_is_better=True)
        return learning_curve_panel(res, source_split=source_split)

    def test_the_series_are_not_called_train_and_holdout(self):
        """They were, on a panel built entirely from test rows."""
        assert "train score" not in self._panel("test").panels[0][0].series_labels

    def test_the_series_name_the_split_they_came_from(self):
        """The honest form."""
        labels = self._panel("test").panels[0][0].series_labels
        assert all("test" in lbl for lbl in labels), labels

    def test_the_verdict_subtitle_names_it_too(self):
        """`data_starved` / `saturated` reads as a statement about training otherwise."""
        assert "test" in self._panel("test").panels[0][0].title

    def test_an_unnamed_split_still_renders(self):
        """The parameter is optional; the panel must not depend on it."""
        assert self._panel("").panels[0][0].series_labels == ("fit-subset score", "held-out score")


class TestTheLearningCurveScorerMatchesTheTask:
    """`roc_auc` was hardcoded for everything non-regression, so multiclass raised and was swallowed."""

    @pytest.mark.parametrize(
        "target_type,expected",
        [
            ("regression", "r2"),
            ("quantile_regression", "r2"),
            ("multiclass_classification", "roc_auc_ovr_weighted"),
            ("binary_classification", "roc_auc"),
        ],
    )
    def test_the_scorer_is_chosen_by_target_type(self, target_type, expected):
        """A scorer that raises for the task turns the whole diagnostic into a swallowed exception, so the curve is built end to end for each task."""
        import pandas as pd
        from sklearn.linear_model import LogisticRegression, Ridge

        from mlframe.training.diagnostics.learning_curve import LearningCurveConfig
        from mlframe.training.reporting._reporting_diagnostics import _build_learning_curve

        rng = np.random.default_rng(0)
        n = 600
        X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
        signal = 2.0 * X["a"].to_numpy() + rng.normal(scale=0.5, size=n)
        if expected == "r2":
            model, y = Ridge(), signal
        elif expected == "roc_auc_ovr_weighted":
            model, y = LogisticRegression(max_iter=200), np.digitize(signal, np.quantile(signal, [1 / 3, 2 / 3]))
        else:
            model, y = LogisticRegression(max_iter=200), (signal > 0).astype(int)
        metrics: dict = {}
        cfg = LearningCurveConfig(enabled=True, sizes=(0.5, 1.0), n_jobs=1)
        panel = _build_learning_curve(model, X, y, list(X.columns), target_type, cfg, metrics)
        assert panel is not None
        assert metrics["learning_curve"]["scorer_name"] == expected
        assert np.all(np.isfinite(metrics["learning_curve"]["holdout_scores"]))
        assert len(metrics["learning_curve"]["holdout_scores"]) == 2
