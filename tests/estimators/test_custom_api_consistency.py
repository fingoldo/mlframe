"""Public-API consistency / validation regression tests for mlframe.estimators.

* API9  -- ArithmAvgClassifier clips X into [0,1] like GeomAvgClassifier, so
  out-of-[0,1] inputs yield valid probabilities.
* API29 -- Arithm/Geom fit raises when nprobs > n_features (short/empty slice).
* API11 -- IdentityClassifier.predict_proba multiclass returns clipped,
  row-normalised, classes_-aligned probabilities.
* API30 -- MyDecorrelator.transform returns the same type as its input.
* API28 -- EarlyStoppingWrapper._split shuffles/stratifies so a sorted-by-class
  dataset does not produce a single-class validation fold.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest

from mlframe.estimators.custom import (
    ArithmAvgClassifier,
    GeomAvgClassifier,
    IdentityClassifier,
    MyDecorrelator,
)

# --------------------------------------------------------------------------- #
# API9: ArithmAvgClassifier clips out-of-[0,1] inputs
# --------------------------------------------------------------------------- #


def test_arithm_clips_out_of_range_to_valid_probs():
    """Arithm clips out of range to valid probs."""
    clf = ArithmAvgClassifier(nprobs=2)
    X_fit = np.array([[0.2, 0.3], [0.6, 0.7], [0.1, 0.9]])
    clf.fit(X_fit, np.array([0, 1, 0]))
    # Out-of-[0,1] feature values that previously produced negative / >1 probs.
    X = np.array([[-0.5, 0.3], [1.8, 1.4], [0.2, -2.0]])
    proba = clf.predict_proba(X)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    assert np.allclose(proba.sum(axis=1), 1.0)


# --------------------------------------------------------------------------- #
# API29: nprobs > n_features guard (Arithm + Geom)
# --------------------------------------------------------------------------- #


def test_arithm_fit_raises_on_nprobs_exceeding_features():
    """Arithm fit raises on nprobs exceeding features."""
    clf = ArithmAvgClassifier(nprobs=5)
    with pytest.raises(ValueError):
        clf.fit(np.array([[0.2, 0.3], [0.6, 0.7]]), np.array([0, 1]))


def test_geom_fit_raises_on_nprobs_exceeding_features():
    """Geom fit raises on nprobs exceeding features."""
    clf = GeomAvgClassifier(nprobs=5)
    with pytest.raises(ValueError):
        clf.fit(np.array([[0.2, 0.3], [0.6, 0.7]]), np.array([0, 1]))


# --------------------------------------------------------------------------- #
# API11: IdentityClassifier multiclass predict_proba
# --------------------------------------------------------------------------- #


def test_identity_classifier_multiclass_probs_normalised_and_aligned():
    """Identity classifier multiclass probs normalised and aligned."""
    clf = IdentityClassifier(feature_indices=[0, 1, 2])
    # 3 classes; feature block carries raw (unnormalised, partly out-of-range) per-class scores.
    y = np.array([0, 1, 2, 0])
    X_fit = pd.DataFrame({"a": [0.1, 0.2, 0.3, 0.4], "b": [0.5, 0.6, 0.7, 0.8], "c": [0.9, 1.0, 0.2, 0.3]})
    clf.fit(X_fit, y)
    assert list(clf.classes_) == [0, 1, 2]
    X = pd.DataFrame({"a": [2.0, -1.0], "b": [0.5, 0.5], "c": [0.5, 0.5]})
    proba = clf.predict_proba(X)
    assert proba.shape == (2, 3)  # one column per class
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    assert np.allclose(proba.sum(axis=1), 1.0)  # rows sum to 1


# --------------------------------------------------------------------------- #
# API30: MyDecorrelator preserves input type
# --------------------------------------------------------------------------- #


def test_decorrelator_ndarray_in_ndarray_out():
    """Decorrelator ndarray in ndarray out."""
    rng = np.random.RandomState(0)
    base = rng.normal(size=(50, 1))
    # col1 ~ col0 (correlated), col2 independent.
    X = np.hstack([base, base * 1.0 + rng.normal(scale=1e-3, size=(50, 1)), rng.normal(size=(50, 1))])
    dec = MyDecorrelator(threshold=0.9)
    dec.fit(X)
    out = dec.transform(X)
    assert isinstance(out, np.ndarray)
    assert out.shape[1] < X.shape[1]  # at least one correlated column dropped


def test_decorrelator_dataframe_in_dataframe_out():
    """Decorrelator dataframe in dataframe out."""
    rng = np.random.RandomState(0)
    base = rng.normal(size=50)
    df = pd.DataFrame({"a": base, "b": base + 1e-3 * rng.normal(size=50), "c": rng.normal(size=50)})
    dec = MyDecorrelator(threshold=0.9)
    dec.fit(df)
    out = dec.transform(df)
    assert isinstance(out, pd.DataFrame)


# --------------------------------------------------------------------------- #
# API28: EarlyStoppingWrapper._split shuffles/stratifies
# --------------------------------------------------------------------------- #


def test_early_stopping_split_not_single_class_on_sorted_data():
    """Early stopping split not single class on sorted data."""
    from sklearn.linear_model import SGDClassifier

    from mlframe.estimators.early_stopping import EarlyStoppingWrapper

    # Data sorted by class: last validation_fraction rows are ALL class 1 under the old last-rows split.
    y = np.array([0] * 50 + [1] * 50)
    X = np.arange(100, dtype=float).reshape(-1, 1)
    wrapper = EarlyStoppingWrapper(
        base_model=SGDClassifier(max_iter=1),
        max_iter=2,
        validation_fraction=0.2,
        random_state=42,
    )
    wrapper._is_regressor = False
    _, _, _, y_val = wrapper._split(X, y)
    assert len(np.unique(y_val)) == 2, "validation fold must not be single-class on sorted-by-class data"
