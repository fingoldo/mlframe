"""Three contract breaks where the code and its own docstring disagreed.

  * `justify_estimator(refit=False)` documents "``est`` is returned unfitted" but returned `None`, which is the
    same value the below-threshold rejection returns. A caller branching on `if fitted_model is None: fall back
    to random sampling` therefore abandoned ML-guided sampling on a PASSING gate, and `get_model` cached that
    `None` as the model.
  * `EarlyStoppingWrapper.max_runtime_mins` is documented and computed into a deadline that the `staged` backend
    was never handed, so its stage sweep ran to the full budget regardless of the clock.
  * `get_models_raw_predictions(trained_models, X, Y)` never read `Y`, yet required it positionally -- inviting
    the reading that predictions are scored against it, and accepting a misaligned array silently.
"""

from __future__ import annotations

import warnings

import time

import numpy as np
import pytest


class TestAPassingGateIsDistinguishableFromARejection:
    """`None` is the rejection sentinel, so a passing gate must not return it."""

    def _run(self, refit, min_score):
        """A trivially learnable regression, so the CV gate's verdict is decided by `min_score` alone."""
        from sklearn.linear_model import LinearRegression

        from mlframe.models.tuning_rules import justify_estimator

        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3))
        y = X @ np.array([2.0, -1.0, 0.5]) + rng.normal(0, 0.01, 200)
        return justify_estimator(LinearRegression(), X, y, cv=3, refit=refit, min_score=min_score, random_state=0)

    def test_a_passing_gate_with_refit_false_returns_the_estimator(self):
        """The docstring's promise; it returned None, which reads as "the gate rejected it"."""
        est, score = self._run(refit=False, min_score=0.5)
        assert score >= 0.5
        assert est is not None, "a passing gate is indistinguishable from a rejection"

    def test_the_returned_estimator_is_unfitted(self):
        """ "Returned unfitted" is the other half of the promise -- refit=False must not have fit it."""
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted

        est, _ = self._run(refit=False, min_score=0.5)
        with pytest.raises(NotFittedError):
            check_is_fitted(est)

    def test_a_failing_gate_still_returns_none(self):
        """The rejection sentinel has to keep meaning what it means."""
        est, score = self._run(refit=False, min_score=1.5)
        assert est is None and score < 1.5

    def test_a_passing_gate_with_refit_true_returns_a_fitted_estimator(self):
        """The path that already worked must not regress."""
        from sklearn.utils.validation import check_is_fitted

        est, _ = self._run(refit=True, min_score=0.5)
        check_is_fitted(est)


class TestTheStagedBackendHonoursTheRuntimeCap:
    """The other two backends check the deadline; this one was never given it."""

    def test_the_deadline_reaches_the_staged_fit(self, monkeypatch):
        """A wrapper call with `max_runtime_mins` set must FORWARD it to the staged fit, not drop it.

        Observed through a spy on `_fit_staged` rather than by searching `fit`'s source for the call text,
        which broke on any harmless rewrite of that line while passing for an implementation that computed a
        deadline and then discarded it.
        """
        from sklearn.ensemble import GradientBoostingClassifier

        from mlframe.estimators.early_stopping import EarlyStoppingWrapper

        seen: list = []
        real = EarlyStoppingWrapper._fit_staged

        def _spy(self, X_train, y_train, X_val, y_val, scoring, deadline, *args, **kwargs):
            """Record the deadline the wrapper handed down, then run the real staged fit."""
            seen.append(deadline)
            return real(self, X_train, y_train, X_val, y_val, scoring, deadline, *args, **kwargs)

        monkeypatch.setattr(EarlyStoppingWrapper, "_fit_staged", _spy)

        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 4))
        y = (X[:, 0] + rng.normal(0, 0.5, 200) > 0).astype(int)
        kw = dict(base_model=GradientBoostingClassifier(random_state=0), max_iter=5, patience=100)

        EarlyStoppingWrapper(**kw).fit(X, y)
        assert len(seen) == 1, f"_fit_staged should be called exactly once per fit, got {len(seen)} call(s)"
        assert seen[0] == float("inf"), f"with no max_runtime_mins the stage sweep must be unbounded, got {seen[0]!r}"

        seen.clear()
        before = time.monotonic()
        EarlyStoppingWrapper(**kw, max_runtime_mins=10.0).fit(X, y)
        assert len(seen) == 1, f"_fit_staged should be called exactly once per fit, got {len(seen)} call(s)"
        # A real, finite deadline roughly 10 minutes out -- not inf, and not a value the wrapper computed and
        # then discarded on the way down.
        assert np.isfinite(seen[0]), f"max_runtime_mins was set but the staged fit received {seen[0]!r}"
        assert before + 60.0 < seen[0] < before + 11 * 60.0, f"the forwarded deadline is not ~10 minutes out: {seen[0] - before:.1f}s from now"

    def test_an_already_expired_deadline_stops_the_stage_sweep_early(self):
        """The observable consequence: the sweep must not walk all `max_iter` stages."""
        from sklearn.ensemble import GradientBoostingClassifier

        from mlframe.estimators.early_stopping import EarlyStoppingWrapper

        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 4))
        y = (X[:, 0] + rng.normal(0, 0.5, 300) > 0).astype(int)
        kw = dict(base_model=GradientBoostingClassifier(random_state=0), max_iter=40, patience=100)
        unbounded = EarlyStoppingWrapper(**kw)
        bounded = EarlyStoppingWrapper(**kw, max_runtime_mins=1e-9)
        unbounded.fit(X, y)
        bounded.fit(X, y)
        # Measured: 40 stages unbounded (patience never trips), 1 with an already-expired deadline.
        assert unbounded.n_iterations_ == 40, unbounded.n_iterations_
        assert bounded.n_iterations_ < unbounded.n_iterations_, (bounded.n_iterations_, unbounded.n_iterations_)

    def test_a_generous_deadline_does_not_truncate(self):
        """It must not stop early just because a deadline exists."""
        from sklearn.ensemble import GradientBoostingClassifier

        from mlframe.estimators.early_stopping import EarlyStoppingWrapper

        rng = np.random.default_rng(1)
        X = rng.normal(size=(300, 4))
        y = (X[:, 0] + rng.normal(0, 0.5, 300) > 0).astype(int)
        a = EarlyStoppingWrapper(base_model=GradientBoostingClassifier(random_state=0), max_iter=20, patience=100)
        b = EarlyStoppingWrapper(base_model=GradientBoostingClassifier(random_state=0), max_iter=20, patience=100, max_runtime_mins=60)
        a.fit(X, y)
        b.fit(X, y)
        assert a.n_iterations_ == b.n_iterations_


class TestTheUnusedGroundTruthParameter:
    """It was required, discarded, and accepted any length."""

    def _models_and_X(self):
        """One fitted regressor keyed by name, plus a frame whose columns match its fitted feature names."""
        import pandas as pd
        from sklearn.linear_model import LinearRegression

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(50, 2)), columns=["a", "b"])
        m = LinearRegression().fit(X, rng.normal(size=50))
        return {"m": m}, X

    def test_it_is_optional_now(self):
        """A public entry point must not require an argument it discards."""
        models, X = self._models_and_X()
        from mlframe.inference.predict import get_models_raw_predictions

        assert set(get_models_raw_predictions(models, X)) == {"m"}

    def test_passing_it_warns(self):
        """Silence would leave the "predictions are scored against Y" reading intact."""
        models, X = self._models_and_X()
        from mlframe.inference.predict import get_models_raw_predictions

        with pytest.warns(DeprecationWarning, match="deprecated and unused"):
            get_models_raw_predictions(models, X, np.zeros(len(X)))

    def test_a_misaligned_ground_truth_now_raises(self):
        """Previously it had no error and no effect, which is the worst of both."""
        models, X = self._models_and_X()
        from mlframe.inference.predict import get_models_raw_predictions

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match="rows but X has"):
                get_models_raw_predictions(models, X, np.zeros(len(X) + 1))

    def test_the_predictions_are_unchanged_by_it(self):
        """It never affected the result; that must stay true."""
        models, X = self._models_and_X()
        from mlframe.inference.predict import get_models_raw_predictions

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with_y = get_models_raw_predictions(models, X, np.zeros(len(X)))
        np.testing.assert_array_equal(get_models_raw_predictions(models, X)["m"], with_y["m"])
