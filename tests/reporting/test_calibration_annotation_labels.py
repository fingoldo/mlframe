"""The reliability diagram printed three labels that did not say what they measured.

All three were read off a real production chart:

1. The header carried ``ECE=1.2%`` from the metrics layer and the line directly beneath it carried ``ECE=0.013``
   from the chart layer -- same label, different unit, different number. They are two different estimates over two
   different binnings, but nothing on the chart said so, so it read as a contradiction.
2. ``PR=..%, RE=..%, F1=..%`` sat beside ROC AUC and PR AUC. The two AUCs are threshold-free; the three beside them
   are not, and the threshold they were computed at appeared nowhere.
3. ``miscal. significant on 15% of range`` told the reader that some of the curve is genuinely off the diagonal but
   not which part or in which direction -- the one question a reliability diagram exists to answer.

These are labelling defects, not maths defects: every number was correct.
"""

from __future__ import annotations

import numpy as np

from mlframe.metrics.calibration import render_title_metric_token
from mlframe.reporting.charts.calibration import _significant_region_note, build_calibration_spec


def _token_args(**overrides):
    """Every keyword ``render_title_metric_token`` requires, zeroed, so a test states only what it exercises."""
    args = dict(
        ndigits=3,
        ice=0.0,
        brier_loss=0.0,
        ece=0.0,
        brier_reliability=0.0,
        brier_resolution=0.0,
        brier_uncertainty=0.0,
        calibration_mae=0.0,
        calibration_std=0.0,
        use_weights=True,
        calibration_coverage=0.0,
        nbins=10,
        ll=None,
        max_hits=0,
        min_hits=0,
        roc_auc=0.9,
        mean_group_roc_auc=None,
        pr_auc=0.8,
        mean_group_pr_auc=None,
        precision=0.75,
        recall=0.6,
        f1=0.667,
    )
    args.update(overrides)
    return args


def _calibrated(n=20_000, seed=0):
    """A well-calibrated binary problem: the label is drawn at exactly the predicted probability."""
    rng = np.random.default_rng(seed)
    score = rng.uniform(0.02, 0.98, size=n)
    return (rng.random(n) < score).astype(int), score


def _overconfident(n=20_000, seed=1):
    """Predictions pushed away from the base rate, so the observed rate sits BELOW the predicted one."""
    rng = np.random.default_rng(seed)
    score = rng.uniform(0.02, 0.98, size=n)
    true_p = np.clip(score - 0.18, 0.01, 0.99)
    return (rng.random(n) < true_p).astype(int), score


class TestTheChartECEIsDistinguishableFromTheHeaderECE:
    """Two ECE estimates may coexist; two identically-labelled ones in different units may not."""

    def test_the_chart_ece_names_its_basis(self):
        """``ECE=`` alone is the header's label; this one has to say which binning it came from."""
        from mlframe.metrics import fast_calibration_binning

        y, score = _calibrated()
        fp, ft, hits = fast_calibration_binning(y, score, nbins=15)
        title = build_calibration_spec(fp, ft, hits, plot_title="rel").panels[0][0].title
        assert "ECE (plotted bins)=" in title

    def test_the_chart_ece_is_rendered_in_the_same_unit_as_the_header(self):
        """The header renders percent; a bare 0.013 beneath a 1.2% is the whole confusion."""
        from mlframe.metrics import fast_calibration_binning

        y, score = _calibrated()
        fp, ft, hits = fast_calibration_binning(y, score, nbins=15)
        title = build_calibration_spec(fp, ft, hits, plot_title="rel").panels[0][0].title
        ece_line = next(line for line in title.splitlines() if "ECE" in line)
        assert "%" in ece_line

    def test_the_debiased_term_is_not_a_second_bare_ece_label(self):
        """It qualifies the same quantity, so it reads as ``debiased=``, not as another ``ECE=``."""
        from mlframe.metrics import fast_calibration_binning

        y, score = _calibrated()
        fp, ft, hits = fast_calibration_binning(y, score, nbins=15)
        title = build_calibration_spec(fp, ft, hits, plot_title="rel").panels[0][0].title
        assert "debiased=" in title
        assert "ECE_debiased=" not in title


class TestThresholdDependentMetricsNameTheirThreshold:
    """PR / RE / F1 change with the threshold; the two AUCs printed beside them do not."""

    def test_the_threshold_appears_in_the_fragment(self):
        """Without it the three numbers are unattributable."""
        out = render_title_metric_token("PR_AUC", binary_threshold=0.5, **_token_args())
        assert "PR@0.50=" in out

    def test_a_non_default_threshold_is_the_one_reported(self):
        """A tuned decision threshold is exactly the case where the reader must not assume 0.5."""
        out = render_title_metric_token("PR_AUC", binary_threshold=0.23, **_token_args())
        assert "PR@0.23=" in out

    def test_an_omitted_threshold_degrades_to_the_old_bare_label(self):
        """Callers that never computed one still render, rather than printing a made-up 0.50."""
        out = render_title_metric_token("PR_AUC", **_token_args())
        assert "PR=" in out
        assert "@" not in out

    def test_the_threshold_free_aucs_are_left_alone(self):
        """Attaching a threshold to ROC AUC would be the same defect in the opposite direction."""
        out = render_title_metric_token("ROC_AUC", binary_threshold=0.23, **_token_args())
        assert "@" not in out


class TestTheSignificanceClaimSaysWhereAndWhichWay:
    """A fraction of the range is a quantity; the reader needs the location."""

    def test_it_reports_the_span_that_excludes_the_diagonal(self):
        """The excluded grid points bound the region; nothing else has to be recomputed."""
        grid = np.linspace(0.0, 1.0, 11)
        lower = grid - 0.05
        upper = grid + 0.05
        upper[3:6] = grid[3:6] - 0.02  # band entirely below the diagonal on p in [0.3, 0.5]
        note = _significant_region_note(grid, lower, upper)
        assert "p in [0.30, 0.50]" in note

    def test_a_band_below_the_diagonal_reads_as_over_confident(self):
        """Observed rate under the predicted one is over-confidence, and that is the actionable half."""
        grid = np.linspace(0.0, 1.0, 11)
        lower, upper = grid - 0.05, grid + 0.05
        upper[3:6] = grid[3:6] - 0.02
        assert "over-confident" in _significant_region_note(grid, lower, upper)

    def test_a_band_above_the_diagonal_reads_as_under_confident(self):
        """The mirror case must not be reported with the same word."""
        grid = np.linspace(0.0, 1.0, 11)
        lower, upper = grid - 0.05, grid + 0.05
        lower[3:6] = grid[3:6] + 0.02
        assert "under-confident" in _significant_region_note(grid, lower, upper)

    def test_deviations_on_both_sides_are_not_reported_as_one_direction(self):
        """An S-shaped curve is over-confident at one end and under-confident at the other."""
        grid = np.linspace(0.0, 1.0, 11)
        lower, upper = grid - 0.05, grid + 0.05
        upper[1:3] = grid[1:3] - 0.02
        lower[8:10] = grid[8:10] + 0.02
        assert "mixed" in _significant_region_note(grid, lower, upper)

    def test_a_band_that_never_leaves_the_diagonal_adds_nothing(self):
        """No significant region means no location to name, and no empty ': ' left dangling."""
        grid = np.linspace(0.0, 1.0, 11)
        assert _significant_region_note(grid, grid - 0.05, grid + 0.05) == ""

    def test_the_rendered_title_carries_the_location(self):
        """End to end: a genuinely miscalibrated model gets a located claim, not a bare percentage."""
        from mlframe.metrics import fast_calibration_binning

        y, score = _overconfident()
        fp, ft, hits = fast_calibration_binning(y, score, nbins=15)
        title = build_calibration_spec(fp, ft, hits, raw_probs=score, raw_labels=y, plot_title="").panels[0][0].title
        assert "miscal. significant on" in title
        assert "p in [" in title
        assert "over-confident" in title
