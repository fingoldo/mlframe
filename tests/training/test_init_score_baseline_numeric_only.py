"""A categorical feature ranked high on ablation and took the whole diagnostic down.

The production log: ``BaselineDiagnostics failed ... could not convert string to float: 'desktop_rjp'``, after
spending 2m15s on the ablation whose results were then thrown away. ``_fit_init_score_baseline`` takes the top-K
ablation features and casts them to float64 -- but an init score is a value ADDED to the prediction, so only a
numeric column can serve as one. A category's integer codes carry no additive meaning on that scale either.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.training.baselines._baseline_diagnostics_init_score import _is_numeric_column


@pytest.fixture
def frame():
    """One numeric column, one string column, one pandas category -- the three shapes the ablation can rank."""
    return pd.DataFrame({
        "prior_score": np.linspace(0.0, 1.0, 8),
        "job_post_device": ["desktop_rjp", "mobile"] * 4,
        "region": pd.Categorical(["a", "b"] * 4),
        "count": np.arange(8),
    })


class TestWhatCanCarryAnInitScore:
    """Only a column whose values live on the prediction scale."""

    @pytest.mark.parametrize("col", ["prior_score", "count"])
    def test_numeric_columns_qualify(self, frame, col):
        """Floats and ints are both addable to a logit."""
        assert _is_numeric_column(frame, col) is True

    def test_a_string_column_does_not(self, frame):
        """The exact column and value from the production crash."""
        assert _is_numeric_column(frame, "job_post_device") is False

    def test_a_category_column_does_not(self, frame):
        """Its codes are integers, but adding a code to a logit means nothing."""
        assert _is_numeric_column(frame, "region") is False

    def test_a_missing_column_is_not_numeric_rather_than_raising(self, frame):
        """This runs inside a diagnostic; an exception here would discard the ablation results too."""
        assert _is_numeric_column(frame, "no_such_column") is False

    def test_a_polars_frame_is_handled(self):
        """The frame arrives as polars on the native path."""
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({"num": [1.0, 2.0], "txt": ["a", "b"]})
        assert _is_numeric_column(df, "num") is True
        assert _is_numeric_column(df, "txt") is False


class TestTheBaselineSkipsNonNumericTopFeatures:
    """The behaviour that keeps the rest of the diagnostic alive."""

    def _ablation(self, *features):
        """Ablation entries ranked in the order given, all with a positive delta."""
        from types import SimpleNamespace

        return [SimpleNamespace(feature=f, delta_pct=10.0 - i) for i, f in enumerate(features)]

    def _run(self, frame, ablation, top_k=2):
        """Call the init-score fit on the REAL diagnostics object, with only the refit stubbed out.

        A hand-rolled stub kept discovering more attributes the function legitimately reads; using the real
        object keeps the test about the feature filter instead of about mirroring an internal surface.
        """
        from mlframe.training.baselines.diagnostics import BaselineDiagnostics
        from mlframe.training.configs import BaselineDiagnosticsConfig

        diag = BaselineDiagnostics(BaselineDiagnosticsConfig(init_score_top_k=top_k))
        diag._fit_quick_and_score = lambda *a, **k: (0.61, {})  # type: ignore[method-assign]
        y = np.array([0, 1] * 4)
        return diag._fit_init_score_baseline(
            frame, y, list(frame.columns), [], ablation, "roc_auc", True, 0.5,
            target_type="binary_classification",
        )

    def test_a_string_top_feature_no_longer_raises(self, frame):
        """One assertion for the production failure: the diagnostic must survive it."""
        self._run(frame, self._ablation("job_post_device", "prior_score"))

    def test_the_skip_is_reported(self, frame, caplog):
        """A silently dropped candidate reads as "no init-score baseline was possible"."""
        with caplog.at_level(logging.INFO, logger="mlframe.training.baselines._baseline_diagnostics_init_score"):
            self._run(frame, self._ablation("job_post_device", "prior_score"))
        assert "job_post_device" in " ".join(r.getMessage() for r in caplog.records)

    def test_an_all_categorical_top_k_returns_no_baseline(self, frame):
        """With nothing numeric left there is no init score to build, which is not an error."""
        assert self._run(frame, self._ablation("job_post_device", "region")) is None
