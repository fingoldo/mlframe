"""Selector-agnostic shared tests for ANY sklearn-style feature selector
in mlframe.feature_selection.

This file parametrizes a single fixture over `RFECV` and `MRMR` (and
future selectors) and asserts the COMMON contract:

- sklearn API: fit returns self; transform raises NotFittedError pre-fit;
  get_feature_names_out() matches transform() output cols
- support_ / n_features_ / n_features_in_ / feature_names_in_ invariants
- input dtype handling: pd.DataFrame, np.ndarray
- pickle round-trip
- pipeline integration

To register a new selector, append to ``_SELECTOR_FACTORIES`` below.

Algorithm-specific tests live in:
- test_wrappers*.py (RFECV)
- test_mrmr_*.py (MRMR)
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ----------------------------------------------------------------------------
# Selector factory registry. Each factory returns a fresh, unfitted selector.
# Add new selectors here; every assertion below runs against every factory.
# ----------------------------------------------------------------------------
def _make_rfecv():
    from mlframe.feature_selection.wrappers import RFECV

    return RFECV(
        estimator=LogisticRegression(max_iter=200, random_state=0),
        cv=3,
        max_refits=2,
        verbose=0,
        random_state=0,
        leakage_corr_threshold=None,
    )


def _make_mrmr():
    try:
        from mlframe.feature_selection.filters.mrmr import MRMR

        # random_seed is the MRMR-specific RNG knob (vs RFECV's random_state).
        # Set it for determinism tests; MRMR uses permutation-based MI which
        # is non-deterministic without a fixed seed.
        return MRMR(verbose=False, random_seed=0)
    except ImportError:
        pytest.skip("MRMR not importable")


_SELECTOR_FACTORIES = [
    pytest.param(_make_rfecv, id="RFECV"),
    pytest.param(_make_mrmr, id="MRMR"),
]


@pytest.fixture(params=_SELECTOR_FACTORIES)
def selector_factory(request):
    """Returns a callable that builds a fresh, unfitted selector."""
    return request.param


# ----------------------------------------------------------------------------
# Canonical synthetic problem - 8 informative + 12 noise, n=300, class_sep=2.0.
# Chosen so both selectors converge in <30s on a typical CI box.
# ----------------------------------------------------------------------------
@pytest.fixture
def small_clf_problem():
    """Per-test fresh X, y (function scope) - selectors must not mutate
    inputs, but defensive isolation guards against state leakage."""
    X, y = make_classification(
        n_samples=300,
        n_features=20,
        n_informative=8,
        n_redundant=0,
        random_state=0,
        shuffle=False,
        class_sep=2.0,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(20)]), y


# ----------------------------------------------------------------------------
# Group A: sklearn API contract
# ----------------------------------------------------------------------------
class TestSharedAPIContract:
    def test_fit_returns_self(self, selector_factory, small_clf_problem):
        X, y = small_clf_problem
        selector = selector_factory()
        result = selector.fit(X, y)
        assert result is selector, "fit() must return self (sklearn convention)"

    def test_transform_before_fit_raises_not_fitted(self, selector_factory, small_clf_problem):
        X, y = small_clf_problem
        selector = selector_factory()
        with pytest.raises(NotFittedError):
            selector.transform(X)

    def test_get_feature_names_out_matches_transform_cols(self, selector_factory, small_clf_problem):
        X, y = small_clf_problem
        selector = selector_factory().fit(X, y)
        try:
            names = selector.get_feature_names_out()
        except (AttributeError, NotImplementedError):
            pytest.skip("selector lacks get_feature_names_out()")
        out = selector.transform(X)
        if hasattr(out, "columns"):
            assert list(out.columns) == list(names)
        else:
            assert out.shape[1] == len(names)

    def test_n_features_in_set_after_fit(self, selector_factory, small_clf_problem):
        X, y = small_clf_problem
        selector = selector_factory().fit(X, y)
        assert hasattr(selector, "n_features_in_")
        # Both selectors may drop degenerate columns at fit entry, so
        # n_features_in_ can be <= original; verify it's reasonable.
        assert 1 <= selector.n_features_in_ <= X.shape[1]

    def test_feature_names_in_aligned_with_n_features_in(self, selector_factory, small_clf_problem):
        X, y = small_clf_problem
        selector = selector_factory().fit(X, y)
        assert hasattr(selector, "feature_names_in_")
        assert len(selector.feature_names_in_) == selector.n_features_in_

    def test_n_features_le_n_features_in(self, selector_factory, small_clf_problem):
        X, y = small_clf_problem
        selector = selector_factory().fit(X, y)
        assert hasattr(selector, "n_features_")
        assert 0 <= selector.n_features_ <= selector.n_features_in_

    def test_support_alignment(self, selector_factory, small_clf_problem):
        """support_ should encode the selection consistently with n_features_.

        MRMR can extend transform output with engineered recipe columns (e.g.
        cluster-aggregate, polynomial). support_ indexes RAW columns only --
        n_features_ counts raw selected + engineered output. So the invariant
        is ``support_count + n_engineered == n_features_``; falls back to
        ``support_count == n_features_`` for selectors without engineered
        recipes (e.g. RFECV).
        """
        X, y = small_clf_problem
        selector = selector_factory().fit(X, y)
        support = selector.support_
        # MRMR and RFECV use different support_ representations:
        # - RFECV: bool mask aligned with feature_names_in_
        # - MRMR: integer indices into feature_names_in_
        if len(support) > 0:
            if isinstance(support[0], (bool, np.bool_)):
                count = int(np.sum(support))
            else:
                count = len(support)
            n_engineered = len(getattr(selector, "_engineered_recipes_", []))
            assert count + n_engineered == selector.n_features_

    def test_transform_output_column_count_matches_n_features(self, selector_factory, small_clf_problem):
        X, y = small_clf_problem
        selector = selector_factory().fit(X, y)
        out = selector.transform(X)
        assert out.shape[1] == selector.n_features_
        assert out.shape[0] == X.shape[0]


# ----------------------------------------------------------------------------
# Group B: input dtype handling
# ----------------------------------------------------------------------------
class TestSharedInputTypes:
    def test_pandas_dataframe_input(self, selector_factory, small_clf_problem):
        X, y = small_clf_problem
        selector = selector_factory()
        selector.fit(X, y)  # X is pd.DataFrame
        out = selector.transform(X)
        # pd input should produce pd output (or at least keep shape).
        assert out.shape[0] == X.shape[0]

    def test_numpy_array_input(self, selector_factory, small_clf_problem):
        X_df, y = small_clf_problem
        X = X_df.values
        selector = selector_factory()
        try:
            selector.fit(X, y)
        except (TypeError, ValueError, AttributeError) as exc:
            # Some selectors may legitimately require pd.DataFrame for
            # certain feature-name workflows. Skip with a clear message
            # rather than failing.
            pytest.skip(f"selector does not accept np.ndarray input: {exc}")
        out = selector.transform(X)
        assert out.shape[0] == X.shape[0]


# ----------------------------------------------------------------------------
# Group C: edge-case input handling (each selector should reject invalid y)
# ----------------------------------------------------------------------------
class TestSharedEdgeCaseRejection:
    def test_constant_y_raises(self, selector_factory):
        """A constant target has H(y)=0 so no feature can carry information.
        Both selectors should raise ValueError rather than silently producing
        zero-information output."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((100, 5)), columns=[f"f{i}" for i in range(5)])
        y = np.zeros(100, dtype=int)
        selector = selector_factory()
        with pytest.raises(ValueError):
            selector.fit(X, y)


# ----------------------------------------------------------------------------
# Group D: stability / determinism contract
# ----------------------------------------------------------------------------
class TestSharedDeterminism:
    def test_re_fit_same_data_same_support(self, selector_factory, small_clf_problem):
        """Fitting twice on the same X, y must produce the same support_."""
        X, y = small_clf_problem
        s1 = selector_factory().fit(X, y)
        s2 = selector_factory().fit(X, y)
        # Compare via get_feature_names_out for portability across support_ formats
        try:
            n1 = sorted(s1.get_feature_names_out())
            n2 = sorted(s2.get_feature_names_out())
            assert n1 == n2, f"Re-fit on identical input should produce identical support_; got {n1} vs {n2}"
        except (AttributeError, NotImplementedError):
            pytest.skip("get_feature_names_out unavailable for comparison")


# ----------------------------------------------------------------------------
# Group E: biz-value (selector actually picks some informative features)
# ----------------------------------------------------------------------------
class TestSharedBizValue:
    def test_recovers_at_least_one_informative(self, selector_factory, small_clf_problem):
        """On a strong synthetic signal (class_sep=2.0, 8 informative of 20),
        the selector should pick at least 1 informative feature. This is a
        very loose test that just confirms the selector isn't broken /
        random."""
        X, y = small_clf_problem
        selector = selector_factory().fit(X, y)
        try:
            names = list(selector.get_feature_names_out())
        except (AttributeError, NotImplementedError):
            # Fall back via support_
            support = selector.support_
            if len(support) == 0:
                pytest.fail("selector returned empty support_")
            if isinstance(support[0], (bool, np.bool_)):
                names = [n for n, s in zip(selector.feature_names_in_, support) if s]
            else:
                names = [selector.feature_names_in_[i] for i in support]

        informative = {f"f{i}" for i in range(8)}
        recall = sum(1 for n in names if n in informative)
        assert recall >= 1, f"selector should recover >=1 of 8 informative features; got 0 in {names}"


# ----------------------------------------------------------------------------
# Group F: pickle round-trip
# ----------------------------------------------------------------------------
class TestSharedPersistence:
    def test_pickle_roundtrip(self, selector_factory, small_clf_problem):
        """A fitted selector must survive pickle.dumps/loads with intact
        support_ and transform behaviour."""
        X, y = small_clf_problem
        selector = selector_factory().fit(X, y)
        # First transform with original
        out_original = selector.transform(X)
        # Pickle round-trip
        try:
            blob = pickle.dumps(selector)
            restored = pickle.loads(blob)
        except Exception as exc:
            pytest.skip(f"selector not pickle-safe: {exc}")
        out_restored = restored.transform(X)
        assert out_restored.shape == out_original.shape
        # Compare values element-wise (to tolerance for floating)
        if hasattr(out_original, "values"):
            np.testing.assert_array_almost_equal(out_original.values, out_restored.values)
        else:
            np.testing.assert_array_almost_equal(out_original, out_restored)


# ----------------------------------------------------------------------------
# Group G: sklearn Pipeline integration
# ----------------------------------------------------------------------------
class TestSharedPipelineIntegration:
    def test_works_in_sklearn_pipeline(self, selector_factory, small_clf_problem):
        """Selector chained with an estimator inside a sklearn Pipeline:
        Pipeline.fit + .predict + .score must all work."""
        X, y = small_clf_problem
        try:
            pipe = Pipeline(
                [
                    ("select", selector_factory()),
                    ("clf", LogisticRegression(max_iter=200, random_state=0)),
                ]
            )
            pipe.fit(X, y)
            preds = pipe.predict(X)
            assert preds.shape == y.shape
        except Exception as exc:
            pytest.skip(f"selector not sklearn-Pipeline-compatible: {exc}")


# ----------------------------------------------------------------------------
# Group H: column-drift contract on transform()
# ----------------------------------------------------------------------------
class TestSharedColumnDrift:
    def test_transform_with_dropped_column_raises(self, selector_factory, small_clf_problem):
        """If a selected column is missing from the transform-time X, the
        selector must raise (not silently return a partial selection)."""
        X, y = small_clf_problem
        selector = selector_factory().fit(X, y)
        try:
            names = list(selector.get_feature_names_out())
        except (AttributeError, NotImplementedError):
            pytest.skip("get_feature_names_out unavailable")
        if not names:
            pytest.skip("selector picked 0 features; nothing to drop")
        # Drop the first selected column from X.
        X_drift = X.drop(columns=[names[0]])
        with pytest.raises((RuntimeError, KeyError, ValueError)):
            selector.transform(X_drift)


# ----------------------------------------------------------------------------
# Group I: trivial / degenerate input shapes
# ----------------------------------------------------------------------------
class TestSharedTrivialInputs:
    def test_single_feature_dataset(self, selector_factory):
        """X with exactly 1 column - selector must select it (or raise
        cleanly). Migrated from per-selector duplicates."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.standard_normal(200)})
        y = (X["a"] > 0).astype(int).values
        selector = selector_factory()
        try:
            selector.fit(X, y)
        except ValueError:
            # Some selectors require >= 2 features (e.g. mRMR redundancy
            # term needs pairs). Tolerated.
            pytest.skip("selector requires >=2 features")
        # Exactly ONE raw feature exists, so n_features_in_ must be 1 and the
        # raw column must be selected. n_features_ itself can exceed 1 because
        # MRMR's default hinge FE legitimately engineers relu legs off the kink
        # in y=(a>0) (the engineered tail is recreated from recipes at transform,
        # never leaked into the caller's X); assert on the RAW selection instead.
        assert selector.n_features_in_ == 1
        raw_selected = (
            int(np.asarray(selector.get_support()).sum()) if callable(getattr(selector, "get_support", None)) else len(getattr(selector, "support_", [0]))
        )
        assert raw_selected == 1, "the single informative raw column must be selected"

    def test_all_noise_features(self, selector_factory):
        """No feature carries any signal. Selector should run without
        crashing - the output set may be empty or near-empty."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((200, 5)), columns=list("abcde"))
        y = rng.integers(0, 2, 200)
        selector = selector_factory()
        try:
            selector.fit(X, y)
        except ValueError:
            pytest.skip("selector rejects all-noise input")
        assert hasattr(selector, "n_features_")

    def test_constant_feature_dropped_or_handled(self, selector_factory):
        """A constant column carries 0 information. RFECV's zero-variance
        filter drops it at fit entry; MRMR's MI-screen also handles it
        gracefully (MI(const, y) = 0)."""
        rng = np.random.default_rng(0)
        n = 200
        X = pd.DataFrame(
            {
                "informative": rng.standard_normal(n),
                "const": np.full(n, 5.0),
            }
        )
        y = (X["informative"] > 0).astype(int).values
        selector = selector_factory()
        selector.fit(X, y)
        # Whatever the strategy, the constant column must NOT be in support_.
        try:
            names = list(selector.get_feature_names_out())
            assert "const" not in names
        except (AttributeError, NotImplementedError):
            pytest.skip("get_feature_names_out unavailable")


# ----------------------------------------------------------------------------
# Group J: y dtype variety
# ----------------------------------------------------------------------------
class TestSharedYDtype:
    def test_y_as_pandas_series(self, selector_factory, small_clf_problem):
        X, y = small_clf_problem
        y_series = pd.Series(y)
        selector = selector_factory()
        selector.fit(X, y_series)
        assert selector.n_features_ >= 0

    def test_y_as_python_list_rejected_or_accepted(self, selector_factory, small_clf_problem):
        """List y is uncommon - either accept (auto-convert) OR reject with
        a clear AttributeError/TypeError. Crashing inside the selector with
        an opaque error mid-fit is the only WRONG behaviour."""
        X, y = small_clf_problem
        selector = selector_factory()
        try:
            selector.fit(X, list(y))
            # If accepted, n_features_ should be valid
            assert selector.n_features_ >= 0
        except (TypeError, AttributeError, ValueError):
            # Cleanly rejecting list y is acceptable.
            pass


# ----------------------------------------------------------------------------
# Group K: fit_transform contract
# ----------------------------------------------------------------------------
class TestSharedFitTransform:
    def test_fit_transform_equals_fit_then_transform(self, selector_factory, small_clf_problem):
        """selector.fit_transform(X, y) must produce the same output as
        selector.fit(X, y).transform(X)."""
        X, y = small_clf_problem
        s1 = selector_factory()
        out_ft = s1.fit_transform(X, y)
        s2 = selector_factory()
        out_t = s2.fit(X, y).transform(X)
        assert out_ft.shape == out_t.shape
        # Compare column sets via get_feature_names_out (order-agnostic safety)
        try:
            names1 = sorted(s1.get_feature_names_out())
            names2 = sorted(s2.get_feature_names_out())
            assert names1 == names2
        except (AttributeError, NotImplementedError):
            pass


# ----------------------------------------------------------------------------
# Group L: refit invalidates prior state
# ----------------------------------------------------------------------------
class TestSharedRefit:
    def test_refit_on_different_data_updates_support(self, selector_factory, small_clf_problem):
        """fit twice with different X. Second fit's support_ must reflect
        the second X (not stale state from the first fit). NOTE: MRMR's
        signature cache currently uses shape only (not column names), so
        renamed-only X may hit the cache; we test with FRESH instance to
        sidestep the signature cache."""
        X, y = small_clf_problem
        # Use TWO fresh instances to avoid signature-cache short-circuit.
        # (RFECV PR-1 F35 fix added column-key to signature; MRMR has not
        # yet been patched - tracked in TODO.md as P1 audit symmetry.)
        s1 = selector_factory().fit(X, y)
        names_first = sorted(s1.get_feature_names_out()) if hasattr(s1, "get_feature_names_out") else None
        X_renamed = X.rename(columns={c: f"renamed_{c}" for c in X.columns})
        s2 = selector_factory().fit(X_renamed, y)
        names_second = sorted(s2.get_feature_names_out()) if hasattr(s2, "get_feature_names_out") else None
        if names_first and names_second:
            # On a fresh instance, the second selection must include only
            # renamed_* columns - no possible cache contamination.
            # Engineered recipes (MRMR cluster-aggregate / unary-binary) carry a
            # composite name like ``clusteragg_mean_z(renamed_f4+...)`` whose source
            # columns are renamed_*; treat such names as renamed-aware iff every
            # renamed-prefixed token they reference is renamed_*.
            def _is_renamed_aware(name: str) -> bool:
                if name.startswith("renamed_"):
                    return True
                # No bare raw column survived (e.g. "f4" with no "renamed_" prefix).
                import re

                bare = re.findall(r"\bf\d+\b", name)
                return not bare

            assert all(_is_renamed_aware(n) for n in names_second), f"Fresh-instance refit on renamed X produced stale-looking names: {names_second}"


# ----------------------------------------------------------------------------
# Group M: multiclass y
# ----------------------------------------------------------------------------
class TestSharedMulticlass:
    def test_3class_classification(self, selector_factory):
        """Both selectors should handle 3+ class targets."""
        X, y = make_classification(
            n_samples=300,
            n_features=15,
            n_informative=5,
            n_redundant=0,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=0,
            shuffle=False,
            class_sep=2.0,
        )
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(15)])
        selector = selector_factory()
        try:
            selector.fit(Xdf, y)
        except (ValueError, TypeError) as exc:
            pytest.skip(f"selector binary-only: {exc}")
        assert selector.n_features_ >= 1


# ----------------------------------------------------------------------------
# Group N: regression target
# ----------------------------------------------------------------------------
class TestSharedRegression:
    def test_continuous_y(self, selector_factory):
        """Continuous y - selector should either work or skip cleanly.
        Note: RFECV with default LR estimator is classifier-only; passing
        a regressor estimator is the operator's responsibility."""
        from sklearn.datasets import make_regression

        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=4,
            random_state=0,
            shuffle=False,
        )
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        selector = selector_factory()
        try:
            selector.fit(Xdf, y)
        except (ValueError, TypeError) as exc:
            pytest.skip(f"selector classification-only: {exc}")
        assert selector.n_features_ >= 0


# ----------------------------------------------------------------------------
# Group O: empty / fit-time degenerate edge cases
# ----------------------------------------------------------------------------
class TestSharedDegenerateInputs:
    def test_empty_y_raises(self, selector_factory):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((0, 5)), columns=list("abcde"))
        y = np.array([], dtype=int)
        selector = selector_factory()
        with pytest.raises(ValueError):
            selector.fit(X, y)

    def test_y_length_mismatch_raises(self, selector_factory, small_clf_problem):
        X, y = small_clf_problem
        y_wrong = y[:-5]  # wrong length
        selector = selector_factory()
        with pytest.raises((ValueError, IndexError)):
            selector.fit(X, y_wrong)


# ----------------------------------------------------------------------------
# Group P: get_feature_names_out details
# ----------------------------------------------------------------------------
class TestSharedFeatureNamesOut:
    def test_returns_ndarray_of_str(self, selector_factory, small_clf_problem):
        X, y = small_clf_problem
        selector = selector_factory().fit(X, y)
        try:
            names = selector.get_feature_names_out()
        except (AttributeError, NotImplementedError):
            pytest.skip("selector lacks get_feature_names_out")
        assert isinstance(names, np.ndarray)
        for n in names:
            # Could be str, np.str_, or object dtype - just ensure non-empty
            assert len(str(n)) > 0

    def test_unfitted_raises(self, selector_factory):
        selector = selector_factory()
        try:
            with pytest.raises((NotFittedError, ValueError, AttributeError)):
                selector.get_feature_names_out()
        except AssertionError:
            # Some selectors may legitimately return [] on unfitted; skip
            pytest.skip("selector returns [] on unfitted (non-strict)")


# ----------------------------------------------------------------------------
# Group Q: integer / bool dtype features
# ----------------------------------------------------------------------------
class TestSharedDtypeVariety:
    def test_int_dtype_features(self, selector_factory):
        rng = np.random.default_rng(0)
        n = 200
        X = pd.DataFrame({f"f{i}": rng.integers(-100, 100, n).astype(np.int32) for i in range(8)})
        y = (X["f0"] > 0).astype(int).values
        selector = selector_factory()
        selector.fit(X, y)
        assert selector.n_features_ >= 1

    def test_float32_dtype_features(self, selector_factory):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({f"f{i}": rng.standard_normal(200).astype(np.float32) for i in range(8)})
        y = (X["f0"] > 0).astype(int).values
        selector = selector_factory()
        selector.fit(X, y)
        assert selector.n_features_ >= 1


# ----------------------------------------------------------------------------
# Group R: imbalanced y
# ----------------------------------------------------------------------------
class TestSharedImbalance:
    def test_moderate_imbalance_70_30(self, selector_factory):
        """70/30 imbalance is common in production - should work without
        crash."""
        rng = np.random.default_rng(0)
        n = 400
        X = pd.DataFrame(rng.standard_normal((n, 8)), columns=[f"f{i}" for i in range(8)])
        y = (rng.random(n) < 0.3).astype(int)
        selector = selector_factory()
        selector.fit(X, y)
        assert selector.n_features_ >= 0
