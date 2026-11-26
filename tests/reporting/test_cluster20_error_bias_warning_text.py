"""Cluster 20 regression: a feature that cannot be binned is reported as skipped, not as a fallback (audit 2026-11-20, finding 20-F32).

The warning used to repeat copy-pasted fallback wording, which implied the feature was still drawn. The panel is now dropped, and the record
says so.
"""

import logging

import numpy as np

from mlframe.reporting.charts import _error_bias as eb

_GROUPS = ("OVER", "UNDER", "MAJORITY")


def _panel(col):
    """Build the error-group panel for one feature column, with the three error groups the chart uses."""
    n = col.size
    third = n // 3
    masks = {
        "OVER": np.arange(n) >= n - third,
        "UNDER": np.arange(n) < third,
        "MAJORITY": (np.arange(n) >= third) & (np.arange(n) < n - third),
    }
    return eb._error_bias_panel(
        col,
        "feature",
        masks=masks,
        resid_signed=np.linspace(-0.1, 0.1, n),
        nbins=10,
        group_colors={g: "red" for g in _GROUPS},
    )


def test_unbinnable_feature_is_skipped_with_a_warning_that_says_so(caplog):
    """A column whose range numpy cannot carve into bins yields no panel and one WARNING that names the skip (20-F32)."""
    col = np.array([0.0, 5e-324, 1e-323, 5e-324])
    with caplog.at_level(logging.WARNING, logger=eb.logger.name):
        panel = _panel(col)

    assert panel is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "skipped this feature: the histogram could not be binned" in warnings[0]
    assert "fell back" not in warnings[0]


def test_binnable_feature_produces_a_panel_and_no_warning(caplog):
    """Control: an ordinary column is binned, returns a panel and logs nothing (20-F32)."""
    col = np.linspace(0.0, 1.0, 50)
    with caplog.at_level(logging.WARNING, logger=eb.logger.name):
        panel = _panel(col)

    assert panel is not None
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
