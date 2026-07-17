"""Phase 4 feature tests for RFECV: stability metrics (N1), 1-SE confidence
interval on best N (N2), get_feature_names_out (N4), must_include hybrid (N5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold

from mlframe.feature_selection.wrappers import RFECV
from tests.training.synthetic import make_sklearn_classification_df


@pytest.fixture(scope="module")
def small_clf_data():
    Xdf, y, _ = make_sklearn_classification_df(
        n_samples=300,
        n_features=15,
        n_informative=5,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=2.0,
        shuffle=False,
        seed=0,
    )
    return Xdf, y


@pytest.fixture(scope="module")
def fitted_rfecv(small_clf_data):
    X, y = small_clf_data
    rfecv = RFECV(
        estimator=LogisticRegression(max_iter=400, random_state=0),
        cv=3,
        max_refits=8,
        verbose=0,
        random_state=0,
    )
    rfecv.fit(X, y)
    return rfecv, X, y


# ----------------------------------------------------------------------------
# N4: get_feature_names_out()
# ----------------------------------------------------------------------------
class TestN4_GetFeatureNamesOut:
    def test_returns_selected_columns(self, fitted_rfecv):
        rfecv, X, y = fitted_rfecv
        names = rfecv.get_feature_names_out()
        assert isinstance(names, np.ndarray)
        assert len(names) == rfecv.n_features_
        # Each name must be a column in the input frame
        assert all(n in X.columns for n in names)

    def test_unfitted_raises(self):
        rfecv = RFECV(estimator=LogisticRegression())
        with pytest.raises(ValueError, match="not fitted"):
            rfecv.get_feature_names_out()

    def test_aligned_with_transform_output(self, fitted_rfecv):
        rfecv, X, y = fitted_rfecv
        names = rfecv.get_feature_names_out()
        out = rfecv.transform(X)
        assert list(out.columns) == list(names)


# ----------------------------------------------------------------------------
# N1: selection_stability_
# ----------------------------------------------------------------------------
class TestN1_SelectionStability:
    def test_jaccard_in_unit_range(self, fitted_rfecv):
        rfecv, X, y = fitted_rfecv
        s = rfecv.selection_stability_(metric="jaccard")
        if not np.isnan(s):
            assert 0.0 <= s <= 1.0

    def test_dice_in_unit_range(self, fitted_rfecv):
        rfecv, X, y = fitted_rfecv
        s = rfecv.selection_stability_(metric="dice")
        if not np.isnan(s):
            assert 0.0 <= s <= 1.0

    def test_kuncheva_in_unit_range(self, fitted_rfecv):
        rfecv, X, y = fitted_rfecv
        s = rfecv.selection_stability_(metric="kuncheva")
        if not np.isnan(s):
            assert 0.0 <= s <= 1.0

    def test_unknown_metric_raises(self, fitted_rfecv):
        rfecv, X, y = fitted_rfecv
        with pytest.raises(ValueError, match="Unknown stability metric"):
            rfecv.selection_stability_(metric="bogus")

    def test_unfitted_raises(self):
        rfecv = RFECV(estimator=LogisticRegression())
        with pytest.raises(ValueError, match="not fitted"):
            rfecv.selection_stability_()


# ----------------------------------------------------------------------------
# N2: n_features_one_se_
# ----------------------------------------------------------------------------
class TestN2_OneSeRule:
    def test_one_se_at_most_optimal(self, fitted_rfecv):
        """1-SE rule must return a count <= the variance-blind optimum, since
        it picks the SMALLEST N within the SE band."""
        rfecv, X, y = fitted_rfecv
        n_one_se = rfecv.n_features_one_se_()
        # Find the variance-blind argmax of cv_mean_perf
        nfs = np.asarray(rfecv.cv_results_["nfeatures"])
        means = np.asarray(rfecv.cv_results_["cv_mean_perf"])
        nonzero = nfs > 0
        if nonzero.any():
            argmax_n = int(nfs[nonzero][np.argmax(means[nonzero])])
            assert n_one_se <= argmax_n, f"1-SE rule N={n_one_se} must be <= variance-blind argmax N={argmax_n}"

    def test_one_se_returns_positive(self, fitted_rfecv):
        rfecv, X, y = fitted_rfecv
        assert rfecv.n_features_one_se_() >= 1


# ----------------------------------------------------------------------------
# N5: must_include hybrid
# ----------------------------------------------------------------------------
class TestN5_MustInclude:
    def test_must_include_features_always_in_support(self):
        rng = np.random.default_rng(0)
        n, p = 200, 12
        X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])
        y = ((X["f0"] + X["f1"]) > 0).astype(int).values
        # Force-keep 'f7' (a noise feature) - it MUST end up in support_
        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=400, random_state=0),
            cv=3,
            max_refits=6,
            verbose=0,
            must_include=["f7"],
        )
        rfecv.fit(X, y)
        names = list(rfecv.get_feature_names_out())
        assert "f7" in names, f"must_include='f7' missing from final selection: {names}"

    def test_must_include_validation_on_invalid_name(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((100, 5)), columns=list("abcde"))
        y = (X["a"] > 0).astype(int).values
        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=400, random_state=0),
            cv=3,
            verbose=0,
            must_include=["a", "nonexistent"],
        )
        with pytest.raises(ValueError, match="must_include contains entries not in X"):
            rfecv.fit(X, y)

    def test_must_include_doesnt_break_voting(self):
        """The voting / FI aggregation must still work with must_include
        active. If FI runs are all empty (because must_include exhausted
        the search universe) we expect graceful degradation."""
        rng = np.random.default_rng(0)
        n, p = 150, 6
        X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"g{i}" for i in range(p)])
        y = (X["g0"] > 0).astype(int).values
        rfecv = RFECV(
            estimator=LogisticRegression(max_iter=400, random_state=0),
            cv=3,
            max_refits=4,
            verbose=0,
            must_include=["g0", "g1", "g2"],  # half the universe pinned
        )
        rfecv.fit(X, y)
        sel = list(rfecv.get_feature_names_out())
        # The 3 pinned features must all be in the final selection
        assert {"g0", "g1", "g2"}.issubset(set(sel))
