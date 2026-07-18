"""Regression: ``CompositeTargetEstimator.from_fitted_inner`` must inherit
``feature_names_in_`` from the already-fitted inner estimator.

Pre-fix path (iter-340 50k seed=4987 cb-regression, axes
dim_reducer=PCA composite_disc=True):
- ``run_composite_post_processing`` builds wrappers via
  :meth:`CompositeTargetEstimator.from_fitted_inner`. Only the ``.fit``
  path used to populate ``feature_names_in_`` from ``X.columns``;
  ``from_fitted_inner`` skipped that line.
- At predict, ``predict_from_models`` looks up
  ``getattr(model, "feature_names_in_")`` to drive the column-subset /
  ``df_pre_pipeline`` fallback path (predict.py:1134-1194). Wrapper
  exposed nothing -> branch skipped -> the wrapper was fed the
  post-extensions pca-only / svd-only frame while its inner was
  trained on the raw-plus-extension frame.
- CatBoost then raised ``At position 0 should be feature with name x0
  (found pca0)`` and aborted every composite + dim_reducer combo.

Post-fix: ``from_fitted_inner`` mirrors ``.fit``'s capture: read
``feature_names_in_`` / ``feature_names_`` off the inner and stash a list
on the wrapper. The predict-side resolver now sees the raw column list
and routes the call to the raw-plus-extension frame.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.composite import CompositeTargetEstimator


class _StubInnerWithSklearnNames:
    """Smallest fake fitted estimator that mirrors the sklearn surface."""

    def __init__(self, feature_names):
        self.feature_names_in_ = np.array(list(feature_names), dtype=object)
        self.is_fitted_ = True

    def predict(self, X):
        """Predict."""
        return np.zeros(getattr(X, "shape", (0,))[0])


class _StubInnerWithCatBoostNames:
    """CatBoost exposes ``feature_names_`` (no ``_in_`` suffix)."""

    def __init__(self, feature_names):
        self.feature_names_ = list(feature_names)

    def predict(self, X):
        """Predict."""
        return np.zeros(getattr(X, "shape", (0,))[0])


class _StubInnerNoNames:
    """Groups tests covering stub inner no names."""
    def predict(self, X):
        """Predict."""
        return np.zeros(getattr(X, "shape", (0,))[0])


def _build_wrapper(stub_inner, base_column="pca4"):
    """Build wrapper."""
    return CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=stub_inner,
        transform_name="linear_residual",
        base_column=base_column,
        transform_fitted_params={"alpha": 1.0, "beta": 0.0, "mad_eff": 1.0},
        y_train=np.linspace(-1.0, 1.0, 64),
    )


def test_from_fitted_inner_inherits_sklearn_style_feature_names():
    """From fitted inner inherits sklearn style feature names."""
    cols = ["x0", "x1", "pca0", "pca1", "pca2", "pca3", "pca4"]
    inner = _StubInnerWithSklearnNames(cols)
    wrapper = _build_wrapper(inner)
    assert hasattr(wrapper, "feature_names_in_"), (
        "from_fitted_inner must mirror .fit's feature_names_in_ capture; "
        "without it predict_from_models cannot resolve the column subset "
        "and routes the post-extensions frame to the inner (iter-340)."
    )
    assert list(wrapper.feature_names_in_) == cols


def test_from_fitted_inner_inherits_catboost_style_feature_names():
    """From fitted inner inherits catboost style feature names."""
    cols = ["x0", "x1", "x2", "pca0", "pca1"]
    inner = _StubInnerWithCatBoostNames(cols)
    wrapper = _build_wrapper(inner)
    assert hasattr(wrapper, "feature_names_in_")
    assert list(wrapper.feature_names_in_) == cols


def test_from_fitted_inner_handles_inner_without_feature_names():
    """No feature-names attr on the inner -> wrapper does not raise and
    does not invent a list. predict_from_models gracefully falls back to
    its other lookups."""
    inner = _StubInnerNoNames()
    wrapper = _build_wrapper(inner)
    assert getattr(wrapper, "feature_names_in_", None) is None


def test_from_fitted_inner_feature_names_is_list_not_ndarray():
    """The downstream caller in predict.py builds a string set from the
    resolved expected list; arrays of dtype=object compare elementwise
    rather than as the iterable predict.py expects. Pinning to ``list``
    here matches the fit-path contract (``list(X.columns)``)."""
    inner = _StubInnerWithSklearnNames(["x0", "x1", "pca0"])
    wrapper = _build_wrapper(inner)
    assert isinstance(wrapper.feature_names_in_, list)
