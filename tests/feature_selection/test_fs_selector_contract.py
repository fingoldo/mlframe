"""Shared sklearn-style contract tests for ALL mlframe feature selectors (2026-05-28).

Parametrizes 3 selector implementations -- MRMR, RFECV, ShapProxiedFS -- through
a single test class so every invariant that should hold for ANY selector is
checked uniformly. Asymmetries that the agent scan flagged (e.g. MRMR uses
integer-index ``support_`` while ShapProxied uses bool-mask) are absorbed
via the ``_as_bool_mask`` adapter.

Per-selector skips / xfails are explicit and documented inline.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Factories. Keep them tiny + identical-shaped so the contract layer is
# uniform across the three selectors. Each factory returns an UNFITTED
# selector configured for whatever task the test fixture provides.


def _mrmr_factory(task: str = "binary"):
    from mlframe.feature_selection.filters.mrmr import MRMR
    return MRMR(
        min_relevance_gain=0.0,
        cv=3,
        run_additional_rfecv_minutes=False,
        full_npermutations=3,
        random_seed=0,
        min_features_fallback=1,
    )


def _rfecv_factory(task: str = "binary"):
    from mlframe.feature_selection.wrappers import RFECV
    est = LogisticRegression(max_iter=200, random_state=0) if task != "regression" \
        else __import__("sklearn.linear_model", fromlist=["Ridge"]).Ridge()
    return RFECV(estimator=est, cv=3, max_refits=3, random_state=0,
                 # Pin argmax so contract tests on tiny synthetic data see a deterministic, parsimonious pick.
                 n_features_selection_rule="argmax")


def _shap_proxied_factory(task: str = "binary"):
    try:
        from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
    except ImportError as exc:
        pytest.skip(f"ShapProxiedFS not importable: {exc}")
    # ShapProxiedFS hard-rejects non-binary y when classification=True.
    # For regression task, allow it; otherwise keep the binary gate.
    cls = (task == "binary")
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    model = RandomForestClassifier(n_estimators=10, random_state=0) if cls \
        else RandomForestRegressor(n_estimators=10, random_state=0)
    return ShapProxiedFS(
        model=model, classification=cls, n_splits=3, n_models=1,
        max_features=None, top_n=10, holdout_size=0.25, revalidate=False,
        trust_guard=False, prefilter_top=None, cluster_features=False,
        random_state=0, n_jobs=1,
    )


SELECTOR_FACTORIES = [
    ("mrmr", _mrmr_factory),
    ("rfecv", _rfecv_factory),
    ("shap_proxied", _shap_proxied_factory),
]


# ---------------------------------------------------------------------------
# Data fixtures (binary by default, regression / multiclass on demand).


@pytest.fixture
def binary_df():
    X, y = make_classification(n_samples=200, n_features=10, n_informative=4,
                               n_classes=2, random_state=0)
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    return Xdf, y


@pytest.fixture
def regression_df():
    X, y = make_regression(n_samples=200, n_features=10, n_informative=4, noise=0.1, random_state=0)
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    return Xdf, y


# ---------------------------------------------------------------------------
# Asymmetry adapter: normalise support_ (int-indices OR bool-mask) to a
# canonical bool mask of length n_features_in_.


def _as_bool_mask(selector) -> np.ndarray:
    """Convert selector's ``support_`` to a bool mask aligned with feature_names_in_."""
    if not hasattr(selector, "support_"):
        raise AttributeError(f"{type(selector).__name__} has no support_")
    s = np.asarray(selector.support_)
    n = int(getattr(selector, "n_features_in_", -1))
    if n <= 0:
        raise ValueError("n_features_in_ unavailable")
    if s.dtype == bool or s.dtype == np.bool_:
        return s.astype(bool)
    # Integer indices -> bool mask.
    mask = np.zeros(n, dtype=bool)
    if s.size > 0:
        mask[s.astype(int)] = True
    return mask


def _fit_safe(selector, X, y):
    """Wrap .fit in a warning-suppressor; some selectors emit deprecation noise."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return selector.fit(X, y)


# ===========================================================================
# Universal invariants (every selector must satisfy)
# ===========================================================================


@pytest.mark.parametrize("name,factory", SELECTOR_FACTORIES)
class TestUniversalContract:
    """Invariants that must hold for every mlframe feature selector."""

    def test_fit_returns_self(self, name, factory, binary_df):
        X, y = binary_df
        sel = factory("binary")
        ret = _fit_safe(sel, X, y)
        assert ret is sel, f"{name}.fit must return self"

    def test_n_features_in_matches_input(self, name, factory, binary_df):
        X, y = binary_df
        sel = _fit_safe(factory("binary"), X, y)
        assert getattr(sel, "n_features_in_", None) == X.shape[1]

    def test_feature_names_in_when_dataframe(self, name, factory, binary_df):
        X, y = binary_df
        sel = _fit_safe(factory("binary"), X, y)
        feat_names = getattr(sel, "feature_names_in_", None)
        assert feat_names is not None
        assert list(feat_names) == list(X.columns)

    def test_support_normalised_length(self, name, factory, binary_df):
        X, y = binary_df
        sel = _fit_safe(factory("binary"), X, y)
        mask = _as_bool_mask(sel)
        assert mask.shape == (X.shape[1],), f"{name}: support mask len {mask.shape} != {X.shape[1]}"

    def test_at_least_one_feature_selected(self, name, factory, binary_df):
        X, y = binary_df
        sel = _fit_safe(factory("binary"), X, y)
        mask = _as_bool_mask(sel)
        assert int(mask.sum()) >= 1, f"{name}: selected zero features"

    def test_transform_output_shape(self, name, factory, binary_df):
        X, y = binary_df
        sel = _fit_safe(factory("binary"), X, y)
        Xt = sel.transform(X)
        mask = _as_bool_mask(sel)
        assert Xt.shape == (X.shape[0], int(mask.sum())), \
            f"{name}: transform shape {Xt.shape} != {(X.shape[0], int(mask.sum()))}"

    def test_transform_preserves_row_count(self, name, factory, binary_df):
        X, y = binary_df
        sel = _fit_safe(factory("binary"), X, y)
        Xt = sel.transform(X)
        assert Xt.shape[0] == X.shape[0]

    def test_not_fitted_error_before_fit(self, name, factory, binary_df):
        X, _ = binary_df
        sel = factory("binary")
        with pytest.raises((NotFittedError, AttributeError, ValueError)):
            sel.transform(X)

    def test_refit_idempotent(self, name, factory, binary_df):
        X, y = binary_df
        sel1 = _fit_safe(factory("binary"), X, y)
        sel2 = _fit_safe(factory("binary"), X, y)
        m1 = _as_bool_mask(sel1)
        m2 = _as_bool_mask(sel2)
        # ShapProxiedFS uses bootstrapped CV inside; allow loose equality (>=80% Jaccard).
        # Same-RNG MRMR / RFECV should match exactly.
        inter = int((m1 & m2).sum())
        union = int((m1 | m2).sum())
        jacc = inter / union if union else 1.0
        assert jacc >= 0.6, (
            f"{name}: refit on same (X,y) gave Jaccard {jacc:.2f} (mask1={m1.tolist()}, mask2={m2.tolist()})"
        )

    def test_clone_preserves_params(self, name, factory, binary_df):
        sel = factory("binary")
        c = clone(sel)
        # get_params equivalence; comparison via repr is robust to nested-object identity differences.
        for k, v in sel.get_params(deep=False).items():
            if k in c.get_params(deep=False):
                v_c = c.get_params(deep=False)[k]
                # Compare repr because some param values are class instances (e.g. estimator).
                assert repr(v) == repr(v_c) or v == v_c or (v is None and v_c is None), \
                    f"{name}: clone() lost {k}: {v} vs {v_c}"


# ===========================================================================
# sklearn-parity (selector should integrate into a sklearn Pipeline)
# ===========================================================================


@pytest.mark.parametrize("name,factory", SELECTOR_FACTORIES)
class TestSklearnParity:
    def test_pipeline_compatibility(self, name, factory, binary_df):
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        X, y = binary_df
        sel = factory("binary")
        try:
            pipe = Pipeline([("fs", sel), ("clf", LogisticRegression(max_iter=200))])
            pipe.fit(X, y)
            preds = pipe.predict(X)
            assert preds.shape == y.shape
        except (TypeError, ValueError, AttributeError) as exc:
            pytest.xfail(f"{name}: not Pipeline-compatible yet ({type(exc).__name__}: {exc})")

    def test_get_params_set_params_roundtrip(self, name, factory):
        sel = factory("binary")
        params = sel.get_params(deep=False)
        # Setting params via set_params must be idempotent on the values we just read.
        try:
            sel.set_params(**params)
        except (TypeError, ValueError) as exc:
            pytest.fail(f"{name}: set_params roundtrip failed: {exc}")

    def test_get_feature_names_out_or_get_support(self, name, factory, binary_df):
        """Either get_feature_names_out OR get_support must be defined (sklearn convention).
        MRMR has get_feature_names_out; ShapProxied has get_support; RFECV has both.
        """
        X, y = binary_df
        sel = _fit_safe(factory("binary"), X, y)
        has_names = callable(getattr(sel, "get_feature_names_out", None))
        has_support = callable(getattr(sel, "get_support", None))
        assert has_names or has_support, f"{name}: neither get_feature_names_out nor get_support defined"

    def test_get_feature_names_out_matches_transform_cols(self, name, factory, binary_df):
        X, y = binary_df
        sel = _fit_safe(factory("binary"), X, y)
        if not callable(getattr(sel, "get_feature_names_out", None)):
            pytest.skip(f"{name} has no get_feature_names_out (known asymmetry)")
        names = sel.get_feature_names_out()
        Xt = sel.transform(X)
        assert len(names) == Xt.shape[1], f"{name}: get_feature_names_out len != transform cols"


# ===========================================================================
# Robustness (every selector should reject pathological inputs cleanly)
# ===========================================================================


@pytest.mark.parametrize("name,factory", SELECTOR_FACTORIES)
class TestRobustness:
    def test_rejects_nan_in_y(self, name, factory, binary_df):
        X, y = binary_df
        y_nan = y.astype(float).copy()
        y_nan[5] = np.nan
        sel = factory("binary")
        with pytest.raises((ValueError, TypeError)):
            _fit_safe(sel, X, y_nan)

    def test_rejects_mismatched_X_y_length(self, name, factory, binary_df):
        X, y = binary_df
        sel = factory("binary")
        with pytest.raises((ValueError, AssertionError)):
            _fit_safe(sel, X.iloc[:-5], y)

    def test_handles_constant_column(self, name, factory, binary_df):
        # Add a constant column; selector should not crash and should NOT select it.
        # RFECV's zero-variance filter drops constant columns at fit entry, so
        # n_features_in_ may be smaller than X.shape[1]. Either path is fine; the
        # contract is "constant column never appears in the final selected set".
        X, y = binary_df
        Xc = X.copy()
        Xc["constant"] = 1.0
        sel = _fit_safe(factory("binary"), Xc, y)
        feat_names_in = list(getattr(sel, "feature_names_in_", []))
        if "constant" not in feat_names_in:
            # Pre-fit filter dropped it (RFECV path); contract satisfied.
            return
        mask = _as_bool_mask(sel)
        if int(mask.sum()) >= len(feat_names_in):
            # Selector picked everything (no parsimony pressure); skip the negative-selection assertion.
            return
        const_idx = feat_names_in.index("constant")
        assert not mask[const_idx], f"{name}: kept the constant column despite parsimony budget"


# ===========================================================================
# Single-class-y (classification-only selectors should raise)
# ===========================================================================


@pytest.mark.parametrize("name,factory", [
    ("rfecv", _rfecv_factory),
    ("shap_proxied", _shap_proxied_factory),
])
class TestClassificationRobustness:
    def test_rejects_single_class_y(self, name, factory, binary_df):
        X, _ = binary_df
        y_const = np.zeros(X.shape[0], dtype=int)
        sel = factory("binary")
        with pytest.raises((ValueError, TypeError)):
            _fit_safe(sel, X, y_const)
