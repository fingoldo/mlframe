"""min_relevance_gain=0.0 is a valid threshold: non-best candidates with a positive gain must still be logged at verbose>2."""

from __future__ import annotations

import logging

from mlframe.feature_selection.filters import _evaluation_candidate as ec


def test_zero_min_relevance_gain_still_logs_positive_gain(caplog):
    """A positive non-best gain is logged when min_relevance_gain is exactly 0.0."""
    with caplog.at_level(logging.INFO, logger=ec.logger.name):
        ec.handle_best_candidate(
            current_gain=0.1, best_gain=0.5, X=(0,), best_candidate=(1,), factors_names=["a", "b"], verbose=3, min_relevance_gain=0.0,
        )
    assert any("current_gain=" in r.getMessage() for r in caplog.records)
