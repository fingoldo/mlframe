"""compute_per_column_rff(X_query=...): the held-out frame is scaled with the TRAIN-fitted RobustScaler.

Two separate calls, one per split, each fit their own scaler, so a test frame whose spread differs from the train frame lands on a different scale than
the one the booster was trained on. A benchmark built that way collapsed boosters from R2 0.997 to 0.4 on a 768-row dataset without any code being wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import RobustScaler

from mlframe.feature_engineering.transformer.per_column_rff import compute_per_column_rff

_KW = dict(seed=7, d_embed_per_column=4, sigma_scale=1.0)


def _frames():
    """A train frame and a test frame with a different location and spread."""
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((400, 6)).astype(np.float32)
    X_te = (3.0 * rng.standard_normal((150, 6)) + 5.0).astype(np.float32)  # different location and spread than train
    return X_tr, X_te


def test_query_rows_use_the_train_fitted_scaler():
    """The query projection equals projecting the manually train-scaled frame with standardize off."""
    X_tr, X_te = _frames()
    got = compute_per_column_rff(X_tr, standardize=True, X_query=X_te, **_KW).to_numpy()
    scaled = RobustScaler().fit(X_tr).transform(X_te).astype(np.float32)
    expected = compute_per_column_rff(scaled, standardize=False, **_KW).to_numpy()
    assert got.shape == (150, 6 * 2 * 4)
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_independent_calls_per_split_disagree_with_the_query_mode():
    """The pitfall the option removes: a separate call on the shifted test frame scales it by its own statistics."""
    X_tr, X_te = _frames()
    separate = compute_per_column_rff(X_te, standardize=True, **_KW).to_numpy()
    replayed = compute_per_column_rff(X_tr, standardize=True, X_query=X_te, **_KW).to_numpy()
    assert np.abs(separate - replayed).max() > 0.1


def test_query_equal_to_train_matches_the_default_call():
    """With X_query=X the result is the historical in-sample projection of X."""
    X_tr, _ = _frames()
    default = compute_per_column_rff(X_tr, standardize=True, **_KW).to_numpy()
    same = compute_per_column_rff(X_tr, standardize=True, X_query=X_tr, **_KW).to_numpy()
    np.testing.assert_allclose(default, same, atol=1e-6)


def test_query_with_a_different_width_is_rejected():
    """A query frame with fewer columns than the train frame cannot be projected and says so."""
    X_tr, X_te = _frames()
    with pytest.raises(ValueError, match="6 columns"):
        compute_per_column_rff(X_tr, X_query=X_te[:, :5], **_KW)
