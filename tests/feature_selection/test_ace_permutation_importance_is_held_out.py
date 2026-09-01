"""`ace_select(importance="permutation")` scored permutation importance on the rows it had just fitted on.

The docstrings describe it twice as held-out PFI, and that is the whole reason to choose the mode: sklearn's
permutation importance is advertised as unbiased where impurity importance skews toward high-cardinality
columns. Computed in-sample against the default fully-grown 120-tree forest it is not: the forest memorises its
training set, so permuting a high-cardinality CONTRAST column also produces a large importance drop.

That matters because the acceptance bar is the 100th percentile over the contrast importances. Memorised
contrasts inflate the bar, and genuinely relevant but low-cardinality real features then fail the one-sided
t-test -- the caller gets a smaller-than-correct accepted set, with no warning, in the mode advertised as
removing exactly that bias.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.ace import _one_replicate_importances, _pfi_split, ace_select


class TestTheModelDoesNotScoreItsOwnTrainingRows:
    """The defect, stated as a property of the fit."""

    def test_the_replicate_fits_and_scores_on_disjoint_rows(self):
        """Recorded directly off the estimator the replicate uses."""
        from sklearn.ensemble import RandomForestClassifier

        seen: dict = {}

        class _Spy(RandomForestClassifier):
            """Records the row count it was fitted on and the rows it was asked to score."""

            def fit(self, X, y, **kw):
                """Record the fit size, then fit normally."""
                seen["fit_n"] = len(X)
                return super().fit(X, y, **kw)

            def predict(self, X):
                """Record the largest scoring size seen."""
                seen["score_n"] = max(seen.get("score_n", 0), len(X))
                return super().predict(X)

        rng = np.random.default_rng(0)
        n = 400
        X = rng.normal(0, 1, (n, 4))
        y = (X[:, 0] > 0).astype(int)
        _one_replicate_importances(_Spy(n_estimators=10, random_state=0), X, y, "permutation", 2, rng, 0)
        assert seen["fit_n"] < n, "the replicate fitted on every row, leaving nothing held out"
        assert seen["score_n"] <= n - seen["fit_n"], (seen["fit_n"], seen["score_n"])

    def test_native_importance_still_fits_on_everything(self):
        """`feature_importances_` needs no holdout; that path must be unchanged."""
        from sklearn.ensemble import RandomForestClassifier

        seen: dict = {}

        class _Spy(RandomForestClassifier):
            """Records the row count it was fitted on."""

            def fit(self, X, y, **kw):
                """Record the fit size, then fit normally."""
                seen["fit_n"] = len(X)
                return super().fit(X, y, **kw)

        rng = np.random.default_rng(1)
        n = 400
        X = rng.normal(0, 1, (n, 4))
        y = (X[:, 0] > 0).astype(int)
        _one_replicate_importances(_Spy(n_estimators=10, random_state=0), X, y, "native", 2, rng, 0)
        assert seen["fit_n"] == n

    def test_a_tiny_input_degrades_instead_of_crashing(self):
        """Too few rows to hold any out must fall back to the in-sample score, not raise."""
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.default_rng(2)
        X = rng.normal(0, 1, (6, 2))
        y = np.array([0, 1, 0, 1, 0, 1])
        real, contrast = _one_replicate_importances(RandomForestClassifier(n_estimators=5, random_state=0), X, y, "permutation", 2, rng, 0)
        assert real.shape == (2,) and contrast.shape == (2,)

    def test_the_split_is_disjoint_and_covers_every_row(self):
        """A split that dropped or duplicated rows would bias the estimate in a different way."""
        rng = np.random.default_rng(3)
        y = np.random.default_rng(4).integers(0, 2, 200)
        fit_idx, score_idx = _pfi_split(200, y, rng)
        assert set(fit_idx).isdisjoint(score_idx) and len(set(fit_idx) | set(score_idx)) == 200

    def test_the_split_varies_across_replicates(self):
        """Averaging over one arbitrary split is not the same as averaging over draws."""
        rng = np.random.default_rng(5)
        y = np.random.default_rng(6).integers(0, 2, 200)
        first = set(_pfi_split(200, y, rng)[1])
        second = set(_pfi_split(200, y, rng)[1])
        assert first != second

    def test_too_few_rows_returns_no_split(self):
        """The documented degradation path."""
        assert _pfi_split(4, np.array([0, 1, 0, 1]), np.random.default_rng(7)) == (None, None)


class TestTheSelectionStillWorks:
    """Removing the bias must not remove the signal."""

    def test_a_planted_signal_is_accepted(self):
        """A feature that genuinely predicts y must survive the contrast bar."""
        rng = np.random.default_rng(8)
        n = 600
        signal = rng.normal(0, 1, n)
        X = np.column_stack([signal, rng.normal(0, 1, n), rng.integers(0, 200, n).astype(float)])
        y = (signal + rng.normal(0, 0.3, n) > 0).astype(int)
        res = ace_select(X, y, importance="permutation", n_replicates=5, n_perm_repeats=2, feature_names=["signal", "noise", "high_card"])
        assert "signal" in set(res.selected_features)

    def test_pure_noise_is_not_accepted(self):
        """The contrast bar must still reject."""
        rng = np.random.default_rng(9)
        n = 400
        X = rng.normal(0, 1, (n, 3))
        y = rng.integers(0, 2, n)
        res = ace_select(X, y, importance="permutation", n_replicates=5, n_perm_repeats=2, feature_names=["a", "b", "c"])
        assert len(res.selected_features) <= 1, res.selected_features
