"""Shared sklearn-API contract battery for the stability-family selectors.

Covers the API contracts that the per-selector test files DO NOT already pin
(see the per-file notes below); duplicated contracts are intentionally omitted.

Selectors under test
--------------------
``StabilityMRMR`` (filters/stability.py) and ``MRMRTreeRescued``
(filters/_mrmr_tree_rescue.py) are sklearn transformers (BaseEstimator +
TransformerMixin); ``StabilityFESelector`` (filters/_stability_fe.py) is the
same but its fit refits MRMR ``n_bootstraps`` times + one full fit, so its
battery is marked ``@slow``. ``heterogeneous_relevance_vote``
(hetero_vote.py) is a plain function, so only the two function-level concepts
(same-seed determinism + NaN-in-X policy) apply to it.

Gap scope (already covered elsewhere -> NOT repeated here)
---------------------------------------------------------
* StabilityMRMR: transform-time name/width validation
  (test_stability_transform_validation.py), fitted-attribute population +
  n_jobs parity (test_stability_coverage.py), ctor param validation
  (test_stability_input_validation.py).
* MRMRTreeRescued: pickle round-trip + get_support/support_ extension +
  transform-column consistency (test_mrmr_tree_rescue.py:52-60).
* hetero_vote: keeps-signal/drops-noise + skill-weighting equivalence
  (test_hetero_vote.py).

This file adds ONLY the missing battery members: NotFitted-before-transform,
fit-returns-self, n_features_in_, transform-shape-vs-support, clone +
get_params/set_params round-trip, sklearn Pipeline integration, and pickle
round-trip with transform-equality (pickle for StabilityMRMR /
StabilityFESelector; MRMRTreeRescued already has pickle, so its row is omitted
here).

PROD BUG surfaced: ``MRMRTreeRescued.__init__(self, *args, ..., **kwargs)``
uses varargs, which makes sklearn ``clone`` / ``get_params`` / ``set_params``
raise ``RuntimeError`` ("estimators should always specify their parameters in
the signature of their __init__"). Base ``MRMR`` clones fine (explicit
signature). The clone/get_params/set_params test for MRMRTreeRescued is
written to the CORRECT behaviour and xfailed (strict=False) so it flips green
the moment the ctor is given an explicit signature.
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from tests.feature_selection.conftest import fast_subset, is_fast_mode  # noqa: F401


# --------------------------------------------------------------------------
# Shared small synthetic data (n <= 400, fixed seed). Two informative
# columns drive a linear binary target so each selector recovers a non-empty
# support without a heavy fit.
# --------------------------------------------------------------------------


def _binary_frame(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.standard_normal(n) for c in ["a", "b", "c", "d", "e"]})
    y = pd.Series((X["a"] + X["b"] + 0.3 * rng.standard_normal(n) > 0).astype(np.int64))
    return X, y


def _make_stability_mrmr():
    """Cheap StabilityMRMR: 3 bootstraps, 70% subsample, threshold 0 so the
    support is non-empty (every touched feature kept) -> transform width > 0."""
    from mlframe.feature_selection.filters.stability import StabilityMRMR
    from mlframe.feature_selection.filters.mrmr import MRMR

    return StabilityMRMR(
        estimator=MRMR(verbose=0),
        n_bootstraps=3,
        sample_fraction=0.7,
        support_threshold=0.0,
        random_state=0,
    )


def _make_mrmr_tree_rescued():
    from mlframe.feature_selection.filters import MRMRTreeRescued

    return MRMRTreeRescued(verbose=0, fe_max_steps=0, random_seed=0)


def _make_stability_fe():
    from mlframe.feature_selection.filters._stability_fe import StabilityFESelector

    return StabilityFESelector(
        base_mrmr_params={"verbose": 0, "fe_max_steps": 0, "random_seed": 0},
        n_bootstraps=2,
        sample_fraction=0.7,
        support_threshold=0.0,
        random_state=0,
    )


# ==========================================================================
# StabilityMRMR battery
# ==========================================================================


class TestStabilityMRMRContract:
    def test_is_base_estimator_transformer(self):
        sel = _make_stability_mrmr()
        assert isinstance(sel, BaseEstimator)
        assert isinstance(sel, TransformerMixin)

    def test_transform_before_fit_raises(self):
        """NotFitted contract: transform before fit must raise (no silent
        pass-through). StabilityMRMR has no fitted ``support_`` yet, so the
        positional slice raises an AttributeError (an error, not a no-op)."""
        sel = _make_stability_mrmr()
        X, _ = _binary_frame()
        with pytest.raises((NotFittedError, AttributeError)):
            sel.transform(X)

    def test_fit_returns_self_and_sets_n_features_in(self):
        sel = _make_stability_mrmr()
        X, y = _binary_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = sel.fit(X, y)
        assert out is sel
        assert sel.n_features_in_ == X.shape[1]

    def test_transform_shape_matches_support(self):
        sel = _make_stability_mrmr()
        X, y = _binary_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel.fit(X, y)
        Xt = sel.transform(X)
        assert Xt.shape[0] == X.shape[0]
        assert Xt.shape[1] == sel.support_.size

    def test_clone_get_params_set_params_round_trip(self):
        sel = _make_stability_mrmr()
        params = sel.get_params(deep=False)
        assert params["n_bootstraps"] == 3
        assert params["sample_fraction"] == 0.7
        assert "estimator" in params
        cloned = clone(sel)
        assert cloned.get_params(deep=False)["n_bootstraps"] == 3
        assert cloned.get_params(deep=False)["support_threshold"] == 0.0
        # An unfitted clone carries no fitted state.
        assert not hasattr(cloned, "support_")
        sel.set_params(n_bootstraps=5, support_threshold=0.4)
        assert sel.n_bootstraps == 5
        assert sel.support_threshold == 0.4

    @pytest.mark.slow
    def test_sklearn_pipeline_integration(self):
        """[selector, LogisticRegression] Pipeline fits + predicts -- the
        production wiring path through sklearn."""
        X, y = _binary_frame()
        pipe = Pipeline(
            [
                ("sel", _make_stability_mrmr()),
                ("clf", LogisticRegression(max_iter=500, random_state=0)),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X, y)
            preds = pipe.predict(X)
        assert len(preds) == len(y)
        assert set(np.unique(preds)).issubset(set(np.unique(y)))

    @pytest.mark.slow
    def test_pickle_round_trip_transform_equality(self):
        sel = _make_stability_mrmr()
        X, y = _binary_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel.fit(X, y)
        before = np.asarray(sel.transform(X))
        restored = pickle.loads(pickle.dumps(sel))  # nosec B301 -- round-trip of a locally-created, trusted object
        after = np.asarray(restored.transform(X))
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(restored.support_, sel.support_)


# ==========================================================================
# MRMRTreeRescued battery (clone/get_params is a PROD BUG -> xfail)
# ==========================================================================


class TestMRMRTreeRescuedContract:
    def test_is_base_estimator_transformer(self):
        sel = _make_mrmr_tree_rescued()
        assert isinstance(sel, BaseEstimator)
        assert isinstance(sel, TransformerMixin)

    def test_transform_before_fit_raises(self):
        """MRMRTreeRescued inherits MRMR.transform which raises NotFittedError
        when ``support_`` / ``feature_names_in_`` are absent."""
        sel = _make_mrmr_tree_rescued()
        X, _ = _binary_frame()
        with pytest.raises(NotFittedError):
            sel.transform(X)

    def test_fit_returns_self_and_sets_n_features_in(self):
        sel = _make_mrmr_tree_rescued()
        X, y = _binary_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = sel.fit(X, y)
        assert out is sel
        assert sel.n_features_in_ == X.shape[1]

    def test_transform_shape_matches_support(self):
        """The rescue extends ``support_`` only, so transform width equals the
        (extended) support size when no engineered recipes are produced
        (fe_max_steps=0)."""
        sel = _make_mrmr_tree_rescued()
        X, y = _binary_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel.fit(X, y)
        Xt = sel.transform(X)
        assert Xt.shape[0] == X.shape[0]
        assert Xt.shape[1] == sel.support_.size

    def test_clone_get_params_set_params_round_trip(self):
        """``_get_param_names`` reports MRMR's params + the tree-rescue params, so sklearn clone /
        get_params / set_params round-trip despite the varargs-forwarding ctor."""
        sel = _make_mrmr_tree_rescued()
        sel.set_params(tree_rescue_top_k=15)
        params = sel.get_params(deep=False)
        assert params["tree_rescue_top_k"] == 15
        assert "verbose" in params  # a forwarded MRMR param is visible too
        cloned = clone(sel)
        assert cloned.get_params(deep=False)["tree_rescue_top_k"] == 15

    @pytest.mark.slow
    def test_sklearn_pipeline_integration(self):
        """Pipeline fit does NOT clone its steps, so the varargs ctor bug does
        not block this path -- [selector, LogisticRegression] fits + predicts."""
        X, y = _binary_frame()
        pipe = Pipeline(
            [
                ("sel", _make_mrmr_tree_rescued()),
                ("clf", LogisticRegression(max_iter=500, random_state=0)),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X, y)
            preds = pipe.predict(X)
        assert len(preds) == len(y)
        assert set(np.unique(preds)).issubset(set(np.unique(y)))


# ==========================================================================
# StabilityFESelector battery (heavy: refits MRMR n_bootstraps + full fit)
# ==========================================================================


@pytest.mark.slow
class TestStabilityFESelectorContract:
    def test_is_base_estimator_transformer(self):
        sel = _make_stability_fe()
        assert isinstance(sel, BaseEstimator)
        assert isinstance(sel, TransformerMixin)

    def test_transform_before_fit_raises(self):
        """StabilityFESelector.transform guards on ``full_mrmr_`` and raises a
        RuntimeError when called before fit."""
        sel = _make_stability_fe()
        X, _ = _binary_frame()
        with pytest.raises((RuntimeError, NotFittedError, AttributeError)):
            sel.transform(X)

    def test_fit_returns_self(self):
        sel = _make_stability_fe()
        X, y = _binary_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = sel.fit(X, y)
        assert out is sel
        assert hasattr(sel, "full_mrmr_")
        assert hasattr(sel, "stable_set_")

    def test_clone_get_params_set_params_round_trip(self):
        sel = _make_stability_fe()
        params = sel.get_params(deep=False)
        assert params["n_bootstraps"] == 2
        assert params["sample_fraction"] == 0.7
        cloned = clone(sel)
        assert cloned.get_params(deep=False)["n_bootstraps"] == 2
        assert not hasattr(cloned, "full_mrmr_")
        sel.set_params(n_bootstraps=4, support_threshold=0.5)
        assert sel.n_bootstraps == 4
        assert sel.support_threshold == 0.5

    def test_pickle_round_trip_transform_equality(self):
        sel = _make_stability_fe()
        X, y = _binary_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel.fit(X, y)
        before = np.asarray(sel.transform(X))
        restored = pickle.loads(pickle.dumps(sel))  # nosec B301 -- round-trip of a locally-created, trusted object
        after = np.asarray(restored.transform(X))
        np.testing.assert_array_equal(before, after)
        assert restored.stable_set_ == sel.stable_set_

    def test_sklearn_pipeline_integration(self):
        X, y = _binary_frame()
        pipe = Pipeline(
            [
                ("sel", _make_stability_fe()),
                ("clf", LogisticRegression(max_iter=500, random_state=0)),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X, y)
            preds = pipe.predict(X)
        assert len(preds) == len(y)


# ==========================================================================
# heterogeneous_relevance_vote: two function-level contracts
# ==========================================================================


def _hetero_small_panel():
    """A 2-member cheap panel (small RF + scaled logistic) so the determinism
    + NaN tests run in ~0.5s instead of ~7s with the default 120-tree panel."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    return {
        "tree": RandomForestClassifier(n_estimators=30, random_state=0),
        "linear": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
    }


def _hetero_frame(n: int = 250, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"c{i}": rng.standard_normal(n) for i in range(8)})
    y = pd.Series((X["c0"] + X["c1"] + 0.3 * rng.standard_normal(n) > 0).astype(np.int64))
    return X, y


def test_hetero_vote_same_seed_identical_votes():
    """Same ``random_state`` -> byte-identical accepted set + vote_fraction.
    Pins the determinism contract (shadow permutations + importances are
    seeded off ``random_state + trial``)."""
    pytest.importorskip("sklearn")
    from mlframe.feature_selection.hetero_vote import heterogeneous_relevance_vote

    X, y = _hetero_frame()
    panel = _hetero_small_panel()
    a1, i1 = heterogeneous_relevance_vote(X, y, classification=True, n_shadow_trials=2, models=panel, random_state=0)
    a2, i2 = heterogeneous_relevance_vote(X, y, classification=True, n_shadow_trials=2, models=_hetero_small_panel(), random_state=0)
    assert a1 == a2
    assert i1["vote_fraction"] == i2["vote_fraction"]


def test_hetero_vote_nan_in_x_raises():
    """NaN-in-X policy pinned: the default sklearn panel members reject NaN, so
    the vote propagates a ValueError rather than silently dropping rows /
    imputing. This documents the contract (callers must clean NaN upstream)."""
    pytest.importorskip("sklearn")
    from mlframe.feature_selection.hetero_vote import heterogeneous_relevance_vote

    X, y = _hetero_frame()
    X_nan = X.copy()
    X_nan.iloc[0, 0] = np.nan
    panel = _hetero_small_panel()
    with pytest.raises(ValueError, match="NaN"):
        heterogeneous_relevance_vote(X_nan, y, classification=True, n_shadow_trials=2, models=panel, random_state=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--no-cov", "-p", "no:randomly", "-p", "no:cacheprovider"]))
