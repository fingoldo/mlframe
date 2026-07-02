"""Edge-case coverage for ``mlframe.calibration.post.BinaryPostCalibrator``.

Uses lightweight fake calibrators (no heavy optional deps) to exercise the adapter's
uniform interface: fit-time metadata, 1D<->2D reshaping, output clipping, the
transform->predict method fallback, empty / NaN inputs, the ``Top``-prefix 2D routing,
and the venn-abers lazy-import-absent dispatch.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.calibration.post import BinaryPostCalibrator


class _IdentityCal:
    """Returns the (positive-class) probability unchanged as a 1D array."""

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float).ravel()


class _OverflowCal:
    """Emits out-of-range values so the adapter's [0,1] clip is exercised."""

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float).ravel() * 5.0 - 2.0


class _PredictOnlyCal:
    """Exposes ``predict`` but not ``transform`` -> adapter must fall back to predict."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float).ravel()


class TopFoo:
    # Name must start with "Top" so _calibrator_needs_2d_probs's prefix check fires.
    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)


@pytest.fixture
def calib_data():
    p = np.array([0.1, 0.4, 0.6, 0.9])
    y = np.array([0, 0, 1, 1])
    return p, y


def test_fit_sets_sklearn_metadata_1d(calib_data):
    p, y = calib_data
    c = BinaryPostCalibrator(_IdentityCal()).fit(p, y)
    assert c.n_features_in_ == 1
    assert c.classes_.tolist() == [0, 1]


def test_fit_sets_n_features_in_from_2d_width(calib_data):
    p, y = calib_data
    p2d = np.column_stack([1 - p, p])
    c = BinaryPostCalibrator(_IdentityCal()).fit(p2d, y)
    assert c.n_features_in_ == 2


def test_postcalibrate_returns_2d_rows_sum_to_one(calib_data):
    p, y = calib_data
    c = BinaryPostCalibrator(_IdentityCal()).fit(p, y)
    out = c.postcalibrate_probs(p)
    assert out.shape == (4, 2)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    np.testing.assert_allclose(out[:, 1], p)


def test_2d_input_reduced_to_positive_column(calib_data):
    p, y = calib_data
    p2d = np.column_stack([1 - p, p])
    c = BinaryPostCalibrator(_IdentityCal()).fit(p2d, y)
    out = c.postcalibrate_probs(p2d)
    np.testing.assert_allclose(out[:, 1], p)


def test_out_of_range_calibrator_output_is_clipped(calib_data):
    p, y = calib_data
    c = BinaryPostCalibrator(_OverflowCal()).fit(p, y)
    out = c.postcalibrate_probs(p)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_transform_falls_back_to_predict(calib_data):
    p, y = calib_data
    c = BinaryPostCalibrator(_PredictOnlyCal(), transform_method_name="transform").fit(p, y)
    assert c._resolved_transform_method_name == "predict"
    out = c.postcalibrate_probs(p)
    assert out.shape == (4, 2)


def test_empty_input_returns_empty_2d(calib_data):
    p, y = calib_data
    c = BinaryPostCalibrator(_IdentityCal()).fit(p, y)
    out = c.postcalibrate_probs(np.array([]))
    assert out.shape == (0, 2)


def test_nan_input_propagates_without_crash(calib_data):
    p, y = calib_data
    c = BinaryPostCalibrator(_IdentityCal()).fit(p, y)
    out = c.postcalibrate_probs(np.array([0.2, np.nan, 0.8, 0.5]))
    assert out.shape == (4, 2)
    assert np.isnan(out).any()


def test_predict_proba_alias_matches_postcalibrate(calib_data):
    p, y = calib_data
    c = BinaryPostCalibrator(_IdentityCal()).fit(p, y)
    np.testing.assert_allclose(c.predict_proba(p), c.postcalibrate_probs(p))


def test_top_prefix_routes_to_2d_and_identity_does_not():
    # A "Top*"-named calibrator preserves the 2D shape; a plain 1D calibrator does not.
    assert BinaryPostCalibrator._calibrator_needs_2d_probs(TopFoo()) is True
    assert BinaryPostCalibrator._calibrator_needs_2d_probs(_IdentityCal()) is False


def test_is_venn_abers_false_when_dep_absent_or_unrelated():
    # Lazy-import guard: an unrelated calibrator is never routed to the venn-abers branch
    # (returns False whether or not venn_abers is installed).
    assert BinaryPostCalibrator._is_venn_abers(_IdentityCal()) is False
