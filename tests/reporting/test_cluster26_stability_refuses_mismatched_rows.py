"""Audit 2026-11-20 finding 26-F24: the prediction-stability chart is not drawn when target and member rows differ.

The old code truncated both to the shorter length and paired the rest by position, so each error was measured against another row's
target while the chart still rendered. It now refuses and says why.
"""

import logging

import numpy as np

from mlframe.reporting._diagnostics_dispatch_extra import render_prediction_stability_diagnostic


def test_mismatched_row_counts_refuse_to_render(tmp_path, caplog):
    """A 10-row prediction matrix against a 7-row target returns False and names both counts in a warning."""
    member_preds = np.random.default_rng(0).random((10, 3))
    with caplog.at_level(logging.WARNING):
        ok = render_prediction_stability_diagnostic(
            member_preds=member_preds,
            y_true=np.zeros(7),
            plot_outputs="png",
            base_path=str(tmp_path / "run"),
        )
    assert ok is False
    assert any("7 rows" in record.getMessage() or "target has 7" in record.getMessage() for record in caplog.records)
