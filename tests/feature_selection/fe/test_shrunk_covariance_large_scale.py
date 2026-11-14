"""Ledoit-Wolf covariance fits survive a column whose scale squares past the float32 range.

The estimator squares its inputs, so fitted on float32 data a column beyond about 1e10 overflowed the covariance to NaN and sklearn raised
"Input contains NaN". Engineered operands such as reciprocals and products reach that scale, and the Mahalanobis-density stage then dropped its columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_engineering.transformer.class_mahalanobis import _shrunk_covariance
from mlframe.feature_engineering.transformer.lda_projection import _fisher_lda
from mlframe.feature_selection.filters._mahalanobis_density_fe import generate_mahalanobis_density_features, mahalanobis_density_feature


def _two_columns(scale: float, n: int = 500) -> np.ndarray:
    """A unit-scale column beside one multiplied by ``scale``, in float32 as the engineered frames are."""
    rng = np.random.default_rng(0)
    return np.column_stack([rng.standard_normal(n), rng.standard_normal(n) * scale]).astype(np.float32)


@pytest.mark.parametrize("scale", [1e10, 1e20, 1e30])
def test_density_features_are_produced_for_a_huge_scale_column(scale):
    """The joint density score is fitted and finite instead of the stage raising."""
    X = pd.DataFrame(_two_columns(scale), columns=["a", "b"])
    enc, payload = generate_mahalanobis_density_features(X, ["a", "b"])
    assert enc.shape == (500, 1)
    assert np.isfinite(enc.to_numpy()).all()
    assert np.isfinite(payload["Sigma_inv"]).all()


@pytest.mark.parametrize("scale", [1e10, 1e30])
def test_mahalanobis_distance_is_finite_for_a_huge_scale_column(scale):
    """The standalone distance is finite for the same input."""
    X = _two_columns(scale)
    assert np.isfinite(mahalanobis_density_feature(X)).all()


@pytest.mark.parametrize("scale", [1e10, 1e30])
def test_class_mahalanobis_covariance_is_finite_for_a_huge_scale_column(scale):
    """The class-conditional Mahalanobis covariance and its inverse are finite."""
    mean, inv_cov = _shrunk_covariance(_two_columns(scale))
    assert np.isfinite(mean).all()
    assert np.isfinite(inv_cov).all()


@pytest.mark.parametrize("scale", [1e10, 1e30])
def test_fisher_lda_direction_is_finite_for_a_huge_scale_column(scale):
    """The pooled Ledoit-Wolf covariance of the Fisher direction does not overflow."""
    X = _two_columns(scale)
    w, c = _fisher_lda(X[:250], X[250:] + 1.0)
    assert np.isfinite(w).all()
    assert np.isfinite(c)
