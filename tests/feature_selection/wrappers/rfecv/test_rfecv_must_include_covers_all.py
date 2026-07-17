"""Regression test (audit4-P1): when ``must_include`` covers EVERY column, the RFECV search universe (the
must_include complement) is empty, so the optimiser evaluates only ``nfeatures==0`` and sets ``support_=[]``.
The must_include glue in ``_finalize`` was gated on ``len(support_)>0``, so it was SKIPPED on that empty
support_ -- silently DROPPING the pinned features, leaving ``n_features_=0`` and a shape-inconsistent
(length-0) ``support_`` despite pins being requested. The pins must always survive.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.wrappers import RFECV


def _data(n_samples=120, n_features=4, seed=3):
    """Helper that data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = (X[:, :2].sum(axis=1) + 0.3 * rng.standard_normal(n_samples) > 0).astype(int)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)]), y


def _fit(**overrides):
    """Helper that fit."""
    base = dict(
        estimator=LogisticRegression(max_iter=300, random_state=0),
        max_refits=4,
        max_noimproving_iters=2,
        verbose=0,
        optimizer_plotting="No",
        random_state=42,
    )
    base.update(overrides)
    return RFECV(**base)


def test_must_include_covering_all_columns_retains_pins():
    """Must include covering all columns retains pins."""
    X, y = _data(n_features=4)
    all_cols = list(X.columns)  # must_include == the whole universe -> empty search complement
    rfecv = _fit(must_include=all_cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rfecv.fit(X, y)

    selected = rfecv.get_feature_names_out().tolist()
    assert set(selected) >= set(all_cols), f"pinned features silently dropped: got {selected}"
    assert rfecv.n_features_ == len(all_cols), f"n_features_={rfecv.n_features_}, expected {len(all_cols)}"
    assert len(rfecv.support_) > 0, "support_ is empty despite must_include pins"
    # transform must be shape-consistent with the retained pins.
    assert rfecv.transform(X).shape[1] == len(all_cols)


def test_must_include_subset_still_retained_when_optimiser_picks_nothing_else():
    """A partial must_include over a weak/near-noise universe: even if the optimiser adds nothing, the pins
    survive (and n_features_ >= number of pins)."""
    X, y = _data(n_features=6, seed=7)
    pins = ["f4", "f5"]
    rfecv = _fit(must_include=pins)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rfecv.fit(X, y)
    selected = set(rfecv.get_feature_names_out().tolist())
    assert set(pins) <= selected, f"pins missing: {selected}"
    assert rfecv.n_features_ >= len(pins)
