"""Regression tests for freshly-landed fixes in
``mlframe.feature_selection.wrappers._helpers``.

Covers:
  - [9]  importance_getter='permutation' now forwards get_feature_importances'
         own ``n_repeats`` and ``random_state`` to sklearn.permutation_importance
         instead of hardcoding n_repeats=5 / random_state=0 (dead params pre-fix).
  - [10] importance_getter='boruta_shap' requires ``data`` (X): raises a clear
         ValueError when data is None instead of silently feeding ``target`` in
         as the feature matrix; and the incompatible arfs/GrootCV ImportError
         fallback was removed (BorutaShap-only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.wrappers._helpers import get_feature_importances


def _fit_tiny_rf(n_features: int = 4, seed: int = 0):
    """Fit a tiny RandomForestClassifier; returns (model, X_df, y, feature_names)."""
    sklearn_ensemble = pytest.importorskip("sklearn.ensemble")
    rng = np.random.default_rng(seed)
    n = 60
    X = rng.normal(size=(n, n_features))
    # Make label depend on first column so model.score is meaningful.
    y = (X[:, 0] + 0.1 * rng.normal(size=n) > 0).astype(int)
    cols = [f"f{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=cols)
    model = sklearn_ensemble.RandomForestClassifier(n_estimators=10, random_state=seed).fit(X_df, y)
    return model, X_df, y, cols


# ----------------------------------------------------------------------- [9]


class TestPermutationForwardsParams:
    def test_permutation_forwards_n_repeats_and_random_state(self, monkeypatch):
        """The 'permutation' path must forward the function's n_repeats and the
        caller's random_state into sklearn.inspection.permutation_importance.
        Pre-fix both were hardcoded (5 / 0), so n_repeats was dead and the
        caller's seed never reached the shuffles."""
        model, X_df, y, cols = _fit_tiny_rf()

        captured = {}

        def _spy(model_, data_, target_, *, n_repeats, random_state, n_jobs, **kw):
            captured["n_repeats"] = n_repeats
            captured["random_state"] = random_state

            # Return an object shaped like sklearn's Bunch result.
            class _Res:
                importances_mean = np.zeros(len(cols), dtype=float)

            return _Res()

        # get_feature_importances does a local ``from sklearn.inspection import
        # permutation_importance`` -> patch the source symbol.
        import sklearn.inspection as _si

        monkeypatch.setattr(_si, "permutation_importance", _spy)

        get_feature_importances(
            model=model,
            current_features=cols,
            importance_getter="permutation",
            data=X_df,
            target=y,
            n_repeats=11,
            random_state=42,
        )
        assert captured["n_repeats"] == 11, "n_repeats not forwarded (pre-fix it was hardcoded to 5)"
        assert captured["random_state"] == 42, "random_state not forwarded (pre-fix it was hardcoded to 0)"

    def test_permutation_default_random_state_is_zero(self, monkeypatch):
        """Default random_state stays 0 so legacy behaviour is preserved when
        the caller does not pass one."""
        model, X_df, y, cols = _fit_tiny_rf()
        captured = {}

        def _spy(model_, data_, target_, *, n_repeats, random_state, n_jobs, **kw):
            captured["n_repeats"] = n_repeats
            captured["random_state"] = random_state

            class _Res:
                importances_mean = np.zeros(len(cols), dtype=float)

            return _Res()

        import sklearn.inspection as _si

        monkeypatch.setattr(_si, "permutation_importance", _spy)

        get_feature_importances(
            model=model,
            current_features=cols,
            importance_getter="permutation",
            data=X_df,
            target=y,
        )
        # Defaults: n_repeats=5 (function default), random_state=0 (function default).
        assert captured["n_repeats"] == 5
        assert captured["random_state"] == 0

    def test_permutation_real_seed_changes_result(self):
        """End-to-end: two distinct random_state values should be able to yield
        different permutation importances (seed now reaches the shuffles)."""
        pytest.importorskip("sklearn.inspection")
        model, X_df, y, cols = _fit_tiny_rf(n_features=4, seed=1)
        d0 = get_feature_importances(
            model=model,
            current_features=cols,
            importance_getter="permutation",
            data=X_df,
            target=y,
            n_repeats=2,
            random_state=0,
        )
        d0b = get_feature_importances(
            model=model,
            current_features=cols,
            importance_getter="permutation",
            data=X_df,
            target=y,
            n_repeats=2,
            random_state=0,
        )
        d1 = get_feature_importances(
            model=model,
            current_features=cols,
            importance_getter="permutation",
            data=X_df,
            target=y,
            n_repeats=2,
            random_state=12345,
        )
        v0 = np.array([d0[c] for c in cols])
        v0b = np.array([d0b[c] for c in cols])
        v1 = np.array([d1[c] for c in cols])
        # Same seed -> reproducible.
        assert np.allclose(v0, v0b), "same random_state must reproduce importances"
        # Different seed -> shuffles differ, so importances should not be identical.
        assert not np.allclose(v0, v1), "different random_state must reach the permutation shuffles"


# ---------------------------------------------------------------------- [10]


class TestBorutaShapRequiresData:
    def test_boruta_shap_data_none_raises_valueerror(self):
        """boruta_shap must require data (X). Pre-fix, data=None silently fed
        ``target`` in as the feature matrix (X=data if data is not None else
        target). Now it raises a clear ValueError mentioning data, regardless
        of whether the optional BorutaShap package is installed (the data guard
        fires before the import)."""
        model, _X_df, y, cols = _fit_tiny_rf()
        with pytest.raises(ValueError) as excinfo:
            get_feature_importances(
                model=model,
                current_features=cols,
                importance_getter="boruta_shap",
                data=None,
                target=y,
            )
        assert "data" in str(excinfo.value).lower()

    def test_boruta_shap_target_none_still_raises(self):
        """target is still required (pre-existing guard preserved)."""
        model, X_df, _y, cols = _fit_tiny_rf()
        with pytest.raises(ValueError) as excinfo:
            get_feature_importances(
                model=model,
                current_features=cols,
                importance_getter="boruta_shap",
                data=X_df,
                target=None,
            )
        assert "target" in str(excinfo.value).lower()

    def test_boruta_shap_missing_lib_message_is_borutashap_only(self, monkeypatch):
        """When BorutaShap is not installed, the ImportError message must point
        only at BorutaShap (the incompatible arfs/GrootCV fallback was removed).
        We simulate the missing package by making the import fail."""
        # Skip if BorutaShap is actually importable (then the import succeeds and
        # the path proceeds to .fit, which is out of scope for this unit test).
        try:
            import BorutaShap  # noqa: F401

            pytest.skip("BorutaShap is installed; missing-lib branch not exercised")
        except ImportError:
            pass

        model, X_df, y, cols = _fit_tiny_rf()
        with pytest.raises(ImportError) as excinfo:
            get_feature_importances(
                model=model,
                current_features=cols,
                importance_getter="boruta_shap",
                data=X_df,
                target=y,
            )
        msg = str(excinfo.value)
        assert "BorutaShap" in msg
        # arfs is no longer offered as a fallback in the message.
        assert "arfs" not in msg.lower()
