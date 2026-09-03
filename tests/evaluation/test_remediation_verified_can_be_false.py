"""`remediation_verified` was True by construction, for every input, including a failed remediation.

`auto_remediate=True` recursed into the detector with a callback that ignored its `fit_df` argument and looked
the stitched values up by index instead. Both branches of the re-check were therefore handed element-for-element
identical arrays: the leaky and honest scores were equal every fold, the inflation was exactly 0.0, and the flag
could not return False. The docstring claimed it "proves the suggested recomputation boundary actually removes
the inflation rather than masking it" -- it proved nothing.

The verification now scores the caller-visible remediated array against an honest per-fold refit of the original
feature, and reports the residual gap it measured as `remediation_inflation`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

import mlframe.evaluation.expanding_window_leakage as ewl
from mlframe.evaluation.expanding_window_leakage import detect_expanding_window_feature_leakage

N = 400


@pytest.fixture
def leaky_setup():
    """A frequency encoding: counting over the whole frame lets a fold's validation rows see their own future."""
    rng = np.random.default_rng(0)
    cat = rng.integers(0, 8, N)
    df = pd.DataFrame({"t": np.arange(N), "cat": cat})
    y = (cat == 3).astype(float) * 2 + rng.normal(0, 0.3, N)

    def freq(fit_df, transform_df):
        """Occurrence count of each row's category WITHIN fit_df."""
        return transform_df["cat"].map(fit_df.groupby("cat")["cat"].size()).fillna(0.0).to_numpy(dtype=float)

    return df, y, freq


def _run(df, y, freq, **kw):
    """The detector under the fixture's estimator/scoring."""
    return detect_expanding_window_feature_leakage(df, "t", y, freq, lambda: Ridge(), n_splits=4, scoring="r2", auto_remediate=True, **kw)


class TestTheFlagIsMeasuredNotAssumed:
    """A verification that cannot fail is not a verification."""

    def test_a_broken_remediation_is_reported_as_unverified(self, leaky_setup, monkeypatch):
        """Stitch the LEAKY full-dataset values into the remediated series instead of the honest ones. The
        remediation has then done nothing at all, and the flag must say so."""
        df, y, freq = leaky_setup
        real = ewl._score_expanding_folds

        def sabotaged(df_sorted, y_sorted, chunks, n_splits, leaky_values, fn, factory, scoring, remediated_sorted=None):
            """Fill the remediation buffer with the suspect series rather than the per-fold honest refit."""
            out = real(df_sorted, y_sorted, chunks, n_splits, leaky_values, fn, factory, scoring, remediated_sorted)
            if remediated_sorted is not None:
                remediated_sorted[:] = leaky_values
            return out

        monkeypatch.setattr(ewl, "_score_expanding_folds", sabotaged)
        res = _run(df, y, freq)
        assert res["remediation_verified"] is False, "a remediation that changed nothing was certified as verified"

    def test_the_residual_gap_is_reported(self, leaky_setup):
        """The number behind the flag, so a caller can see how close to the tolerance it landed."""
        res = _run(*leaky_setup)
        assert np.isfinite(res["remediation_inflation"])

    def test_the_flag_agrees_with_the_number(self, leaky_setup):
        """A flag derived from a different quantity than the one reported would be worse than none."""
        res = _run(*leaky_setup)
        assert res["remediation_verified"] == (res["remediation_inflation"] <= ewl._LEAK_TOLERANCE)

    def test_the_verification_is_not_a_tautology(self, leaky_setup, monkeypatch):
        """The old callback made the two branches identical; assert they now see different arrays."""
        df, y, freq = leaky_setup
        seen = []
        real = ewl._score_expanding_folds

        def spy(df_sorted, y_sorted, chunks, n_splits, leaky_values, *a, **k):
            """Record each pass's candidate series."""
            seen.append(np.asarray(leaky_values, dtype=float).copy())
            return real(df_sorted, y_sorted, chunks, n_splits, leaky_values, *a, **k)

        monkeypatch.setattr(ewl, "_score_expanding_folds", spy)
        _run(df, y, freq)
        assert len(seen) == 2 and not np.array_equal(seen[0], seen[1]), "the verification pass re-scored the same series"


class TestTheRemediationItselfStillWorks:
    """The fix must not weaken what the function already did correctly."""

    def test_a_genuine_leak_is_still_detected(self, leaky_setup):
        """Whole-frame frequency counts on time-ordered folds are the failure this module exists to catch."""
        assert _run(*leaky_setup)["leak_detected"]

    def test_a_correct_remediation_verifies(self, leaky_setup):
        """The honest stitching removes the inflation, and the check confirms it rather than asserting it."""
        assert _run(*leaky_setup)["remediation_verified"]

    def test_the_remediated_feature_is_in_original_row_order(self):
        """`remediated_feature[order]` is what the verification consumes, so the round-trip has to hold."""
        rng = np.random.default_rng(1)
        t = rng.permutation(N)  # deliberately NOT pre-sorted
        cat = rng.integers(0, 6, N)
        df = pd.DataFrame({"t": t, "cat": cat})
        y = (cat == 2).astype(float) + rng.normal(0, 0.4, N)

        def freq(fit_df, transform_df):
            """Occurrence count within fit_df."""
            return transform_df["cat"].map(fit_df.groupby("cat")["cat"].size()).fillna(0.0).to_numpy(dtype=float)

        res = _run(df, y, freq)
        assert res["remediated_feature"].shape == (N,) and np.isfinite(res["remediated_feature"]).all()

    def test_detection_only_mode_reports_no_remediation_keys(self, leaky_setup):
        """Unchanged contract for the default path."""
        df, y, freq = leaky_setup
        res = detect_expanding_window_feature_leakage(df, "t", y, freq, lambda: Ridge(), n_splits=4, scoring="r2")
        assert "remediation_verified" not in res and "remediated_feature" not in res
