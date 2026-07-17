"""Smoke + correctness sensors for the uncertainty / accuracy diagnostic plot builders
in ``composite/diagnostics.py``:

- ``plot_reliability_diagram`` (calibration; consumes the ``calibration_report`` shape OR
  raw ``y_true`` + ``proba``) -- asserts a Figure with calibration axes and that the
  annotated ECE matches a hand-computed value.
- ``plot_interval_coverage`` (conformal / CQR / Mondrian bands) -- asserts the coverage
  annotation matches a hand-computed empirical coverage + mean width.
- ``plot_interval_width_vs_x`` (adaptive CQR vs constant split width) -- asserts a Figure
  with the mean-width reference line.

All assertions run under the headless ``Agg`` backend; each builder returns a
``matplotlib.figure.Figure`` the caller owns.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from mlframe.training.composite.diagnostics import (
    plot_interval_coverage,
    plot_interval_width_vs_x,
    plot_reliability_diagram,
)


def _annotation_texts(fig: Figure) -> str:
    """Concatenate every text artist on the first axes (annotation boxes + legend)."""
    ax = fig.axes[0]
    return "\n".join(t.get_text() for t in ax.texts)


def test_reliability_diagram_from_report_has_axes_and_ece():
    """Reliability diagram from report has axes and ece."""
    report = {
        "bin_confidence": np.array([0.55, 0.85]),
        "bin_accuracy": np.array([0.50, 0.90]),
        "bin_count": np.array([100, 100]),
        "ece": 0.05,
    }
    fig = plot_reliability_diagram(report=report)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "mean predicted confidence"
    assert ax.get_ylabel() == "observed accuracy"
    assert "ECE = 0.0500" in _annotation_texts(fig)


def test_reliability_diagram_from_raw_matches_hand_computed_ece():
    # 4 rows, 2 classes, n_bins=2. Confidences {0.6,0.6,0.9,0.9}; the two 0.6 rows
    # land in bin1 [0.5,1.0)... build so binning is unambiguous and ECE checkable.
    # bin edges for n_bins=2: [0,0.5,1.0]. conf 0.6 -> bin1, conf 0.9 -> bin1.
    # Use n_bins=10 to spread: conf 0.6 -> bin5, 0.9 -> bin8.
    """Reliability diagram from raw matches hand computed ece."""
    proba = np.array(
        [
            [0.4, 0.6],  # pred class1, conf 0.6
            [0.4, 0.6],  # pred class1, conf 0.6
            [0.1, 0.9],  # pred class1, conf 0.9
            [0.9, 0.1],  # pred class0, conf 0.9
        ]
    )
    y_true = np.array([1, 0, 1, 1])  # bin5: rows0,1 -> acc 0.5; bin8: rows2,3 -> acc 0.5
    fig = plot_reliability_diagram(y_true=y_true, proba=proba, n_bins=10)
    # Hand ECE: bin5 has 2 rows conf=0.6 acc=0.5 -> |0.6-0.5|*(2/4)=0.05;
    #           bin8 has 2 rows conf=0.9 acc=0.5 -> |0.9-0.5|*(2/4)=0.20; total 0.25.
    assert "ECE = 0.2500" in _annotation_texts(fig)


def test_reliability_diagram_requires_inputs():
    """Reliability diagram requires inputs."""
    with pytest.raises(ValueError):
        plot_reliability_diagram()


def test_interval_coverage_annotation_matches_hand_value():
    # 5 rows. Intervals cover rows where lower<=y<=upper.
    """Interval coverage annotation matches hand value."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lower = np.array([0.0, 1.5, 5.0, 3.5, 4.0])  # row2 (y=3) NOT in [5,6]
    upper = np.array([2.0, 2.5, 6.0, 4.5, 6.0])
    # covered: rows 0,1,3,4 -> 4/5 = 0.8 coverage.
    # widths: 2,1,1,1,2 -> mean 1.4.
    fig = plot_interval_coverage(y_true, lower, upper)
    assert isinstance(fig, Figure)
    txt = _annotation_texts(fig)
    assert "empirical coverage = 0.8000" in txt
    assert "mean width = 1.4000" in txt
    ax = fig.axes[0]
    assert ax.get_ylabel() == "value"


def test_interval_coverage_shape_mismatch_raises():
    """Interval coverage shape mismatch raises."""
    with pytest.raises(ValueError):
        plot_interval_coverage(np.zeros(5), np.zeros(4), np.zeros(5))


def test_interval_width_vs_x_constant_vs_adaptive():
    """Interval width vs x constant vs adaptive."""
    np.random.default_rng(0)
    x = np.linspace(0, 10, 400)
    # Constant (split-like) band: width near-zero CV.
    lo_c, hi_c = -np.ones_like(x), np.ones_like(x)
    fig_c = plot_interval_width_vs_x(x, lo_c, hi_c)
    assert isinstance(fig_c, Figure)
    assert "constant (split-like)" in _annotation_texts(fig_c)
    # Adaptive (CQR-like) band: width grows with x.
    half = 0.2 + 0.5 * x
    fig_a = plot_interval_width_vs_x(x, -half, half)
    assert "adaptive (CQR-like)" in _annotation_texts(fig_a)
    assert fig_a.axes[0].get_ylabel() == "interval width (upper - lower)"


def test_interval_width_vs_x_subsamples_large_input():
    """Interval width vs x subsamples large input."""
    n = 50_000
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    half = np.abs(x) + 0.1
    fig = plot_interval_width_vs_x(x, -half, half, sample_n=3000)
    # Scatter collection should hold at most the cap, not all 50k points.
    coll = fig.axes[0].collections[0]
    assert coll.get_offsets().shape[0] <= 3000


def test_all_builders_return_distinct_figures():
    """All builders return distinct figures."""
    report = {
        "bin_confidence": np.array([0.6]),
        "bin_accuracy": np.array([0.6]),
        "bin_count": np.array([10]),
        "ece": 0.0,
    }
    f1 = plot_reliability_diagram(report=report)
    f2 = plot_interval_coverage(np.array([1.0]), np.array([0.0]), np.array([2.0]))
    f3 = plot_interval_width_vs_x(np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    assert len({id(f1), id(f2), id(f3)}) == 3
