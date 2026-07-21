"""Direct unit coverage for ``_fe_linear_explainability.raws_linearly_explain_y`` (mrmr_audit_2026-07-20
test_coverage.md #11). Only exercised transitively via full MRMR fits before this file -- pins the
regression R^2 threshold, the classification always-False contract, and the degenerate-input
best-effort-False fallback directly."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._fe_linear_explainability import raws_linearly_explain_y


class TestRegressionKnownLinearSignal:
    """A cleanly linear regression target must be recovered as 'linearly explained' (True); a
    genuinely nonlinear target must not."""

    def test_clean_linear_target_returns_true(self):
        """y = 2*a + 3*b + tiny noise -- a plain linear fit should trivially clear the default threshold."""
        rng = np.random.default_rng(0)
        n = 500
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        y = 2.0 * a + 3.0 * b + 0.001 * rng.standard_normal(n)
        X = pd.DataFrame({"a": a, "b": b})
        assert raws_linearly_explain_y(X, pd.Series(y)) is True

    def test_nonlinear_target_returns_false(self):
        """y = sin(a) * b (genuinely nonlinear interaction) -- a plain linear fit should NOT clear
        the threshold, so nonlinear FE passes must still run."""
        rng = np.random.default_rng(1)
        n = 500
        a = rng.uniform(-3, 3, n)
        b = rng.standard_normal(n)
        y = np.sin(a) * b
        X = pd.DataFrame({"a": a, "b": b})
        assert raws_linearly_explain_y(X, pd.Series(y)) is False

    def test_threshold_is_respected(self):
        """A moderately-linear-but-noisy target with R^2 clearly below a strict threshold must return False."""
        rng = np.random.default_rng(2)
        n = 500
        a = rng.standard_normal(n)
        y = 1.0 * a + 5.0 * rng.standard_normal(n)  # heavy noise relative to signal -> low R^2
        X = pd.DataFrame({"a": a})
        assert raws_linearly_explain_y(X, pd.Series(y), thresh=0.92) is False


class TestClassificationAlwaysFalse:
    """Classification targets always return False regardless of how linearly-separable the raws are
    -- the R^2-style shortcut is regression-only by construction."""

    def test_binary_classification_returns_false_even_when_linearly_separable(self):
        """A perfectly linearly-separable binary target must still return False (operators must run)."""
        rng = np.random.default_rng(3)
        n = 500
        a = rng.standard_normal(n)
        y = (a > 0).astype(int)
        X = pd.DataFrame({"a": a})
        assert raws_linearly_explain_y(X, pd.Series(y)) is False

    def test_multiclass_classification_returns_false(self):
        """A multiclass integer-coded target also always returns False."""
        rng = np.random.default_rng(4)
        n = 500
        a = rng.standard_normal(n)
        y = np.digitize(a, bins=[-1, 0, 1])
        X = pd.DataFrame({"a": a})
        assert raws_linearly_explain_y(X, pd.Series(y)) is False


class TestDegenerateInputsReturnFalse:
    """Best-effort contract: any failure/degenerate input returns False (run the passes -- correctness
    over the optimisation), never raises."""

    def test_non_dataframe_x_returns_false(self):
        """A raw ndarray X (not a DataFrame) returns False without raising."""
        rng = np.random.default_rng(5)
        X = rng.standard_normal((100, 2))
        y = rng.standard_normal(100)
        assert raws_linearly_explain_y(X, y) is False

    def test_no_numeric_columns_returns_false(self):
        """A DataFrame with only non-numeric columns returns False."""
        X = pd.DataFrame({"cat": ["a", "b", "c"] * 10})
        y = np.arange(30, dtype=np.float64)
        assert raws_linearly_explain_y(X, y) is False

    def test_row_count_mismatch_returns_false(self):
        """len(y) != len(X) returns False rather than raising a shape error."""
        rng = np.random.default_rng(6)
        X = pd.DataFrame({"a": rng.standard_normal(100)})
        y = rng.standard_normal(50)
        assert raws_linearly_explain_y(X, y) is False

    def test_too_few_rows_returns_false(self):
        """n < 8 rows returns False (the probe's own minimum-rows guard)."""
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = np.array([1.0, 2.0, 3.0])
        assert raws_linearly_explain_y(X, y) is False

    def test_non_finite_y_returns_false(self):
        """A target containing NaN/Inf returns False rather than fitting on poisoned data."""
        rng = np.random.default_rng(7)
        n = 50
        X = pd.DataFrame({"a": rng.standard_normal(n)})
        y = rng.standard_normal(n)
        y[0] = np.inf
        assert raws_linearly_explain_y(X, y) is False

    def test_row_subsampling_does_not_crash_on_large_n(self):
        """n > max_rows triggers the row-subsampling branch -- must not crash and must still detect
        a clean linear signal (subsampling shouldn't destroy an obvious signal)."""
        rng = np.random.default_rng(8)
        n = 5000
        a = rng.standard_normal(n)
        y = 4.0 * a + 0.001 * rng.standard_normal(n)
        X = pd.DataFrame({"a": a})
        assert raws_linearly_explain_y(X, y, max_rows=200) is True
