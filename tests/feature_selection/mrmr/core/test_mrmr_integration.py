"""
Comprehensive tests for feature_selection/filters.py

Tests include:
- Property-based tests for helper functions using hypothesis
- MRMR feature selection tests for classification and regression
- Feature engineering capability tests
- Edge cases and integration tests
"""

import pytest
import warnings



# Import the module under test
from mlframe.feature_selection.filters import (
    MRMR,
)


class TestMRMRIntegration:
    """Integration tests for MRMR with downstream tasks."""

    def test_pipeline_compatibility(self, simple_classification_data):
        """Test MRMR works in sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestClassifier

        X, y, _ = simple_classification_data

        pipeline = Pipeline(
            [
                ("feature_selection", MRMR(full_npermutations=3, baseline_npermutations=3, verbose=0, n_jobs=1)),
                ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X, y)

        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)

    def test_with_downstream_model(self, simple_classification_data):
        """Test that MRMR-selected features improve or maintain model performance."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        X, y, _ = simple_classification_data

        mrmr = MRMR(full_npermutations=5, baseline_npermutations=5, verbose=0, n_jobs=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)

        X_selected = mrmr.transform(X)

        # Train model on selected features
        clf = RandomForestClassifier(n_estimators=20, random_state=42)
        scores = cross_val_score(clf, X_selected, y, cv=3, scoring="accuracy")

        # Should achieve reasonable accuracy
        assert scores.mean() > 0.5, f"Mean accuracy too low: {scores.mean()}"


# ================================================================================================
# 2026-04-19 — MRMR patience-termination observability sensor
# ================================================================================================
# Pre-fix: screen_predictors early-exited on max_consec_unconfirmed
# patience with a log.info gated by verbose>=1. At default verbose=0
# (and the call from filters.py:2964 uses verbose=2 but the outer
# mlframe caller often uses 0), operators had no signal whether MRMR
# returned fewer features because "natural gain threshold reached"
# (done) vs "patience exhausted on noisy data" (try higher budget).
# Fix: unconditional summary log at function exit with the termination
# reason. WARN on patience-triggered exit.


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
