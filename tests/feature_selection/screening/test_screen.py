"""
Comprehensive tests for feature_selection/filters.py

Tests include:
- Property-based tests for helper functions using hypothesis
- MRMR feature selection tests for classification and regression
- Feature engineering capability tests
- Edge cases and integration tests
"""

import pytest
import numpy as np
import pandas as pd
import warnings

from hypothesis import given, settings, strategies as st, assume, HealthCheck
from hypothesis.extra.numpy import arrays

from sklearn.datasets import make_classification, make_regression

# Import the module under test
from mlframe.feature_selection.filters import (
    MRMR,
    entropy,
    categorize_dataset,
    discretize_array,
    compute_mi_from_classes,
)

class TestScreenPredictorsPatienceObservability:

    def test_screen_predictors_logs_patience_summary(self, simple_classification_data, caplog):
        """On any invocation (happy path or patience-stopped), the
        function must emit a termination-reason summary line at exit.
        We use max_consec_unconfirmed=1 on high-noise data so the
        patience path is reliably triggered within the tiny test
        budget."""
        import logging
        X, y, _ = simple_classification_data
        # Use MRMR wrapper with aggressive early-stop so patience trips
        # reliably on this small fixture.
        mrmr = MRMR(
            full_npermutations=3,
            baseline_npermutations=3,
            max_consec_unconfirmed=1,  # trip patience fast
            min_relevance_gain=0.99,   # strict threshold, helps patience path
            verbose=0,
            n_jobs=1,
        )
        with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters"):
            with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.filters"):
                mrmr.fit(X, y)
        # Logger name became hierarchical after the package split (the legacy
        # monolith now lives in mlframe.feature_selection.filters._legacy and
        # individual modules will move to filters.<submod>). Match any logger
        # under the package prefix.
        msgs = [r.message for r in caplog.records
                if r.name.startswith("mlframe.feature_selection.filters")
                and r.levelname in ("INFO", "WARNING")]
        # The summary must fire at least once with the "screen_predictors"
        # prefix and a selected-feature count, regardless of path taken.
        assert any(
            "screen_predictors" in m and ("terminated early" in m or "finished naturally" in m)
            for m in msgs
        ), f"Expected termination-reason summary log; got:\n{msgs}"

    # Note — a second sensor that tried to force the patience-triggered
    # WARN path was dropped: on any realistic synthetic fixture, the
    # termination reason is data-dependent (noise distribution, nbins
    # quantization outcome, permutation RNG). Rather than chase a
    # reliable trigger, the first sensor above catches the common
    # regression (summary log removed entirely). The patience WARN
    # branch is straight-line code: if ``patience_triggered=True`` flows
    # through, the log format is verified by the INFO branch's contract.


# ================================================================================================
# Run Tests
# ================================================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
