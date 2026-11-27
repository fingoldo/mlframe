"""Cluster 12 regression: a failing downstream fit in the MRMR large-N campaign is logged as a WARNING (audit 2026-11-20, 12-F79)."""

import logging

import numpy as np

from mlframe.feature_selection._benchmarks.fs_quality import mrmr_largeN_campaign as camp


def test_failing_downstream_fit_warns_and_scores_none(monkeypatch, caplog):
    """A fit that raises gives None for the cell and a WARNING naming the error, instead of a DEBUG line (12-F79)."""
    monkeypatch.delenv("MRMR_CAMPAIGN_RAISE", raising=False)
    X = np.zeros((20, 3))
    y_bad = np.array(["a", "b"] * 10, dtype=object)  # regression on strings: the GBM fit raises
    with caplog.at_level(logging.WARNING, logger=camp.logger.name):
        out = camp._downstream_score(X, y_bad, X, y_bad, [0, 1], "regression")
    assert out is None
    assert any("campaign cell failed" in r.getMessage() for r in caplog.records)
