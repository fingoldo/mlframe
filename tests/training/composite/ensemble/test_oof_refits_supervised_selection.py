"""An OOF fold never scores a component through a feature selector that saw the fold's holdout targets (EST-17).

The entry's pre_pipeline is fitted on the full train, and when it holds a supervised selector (MRMR, RFECV, BorutaShap)
that fit chose features with y on every train row - each OOF fold's holdout rows included. Reusing it made the OOF score
of a feature-selected component optimistic next to a plain one, and the NNLS stack weighted it up on that basis. On a CV
fold the pipeline is now refit on the fold's own train rows; unsupervised steps keep the cheap reuse, and the external
holdout path, whose rows the full-train fit never saw, is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.training.composite import compute_oof_holdout_predictions
from mlframe.training.composite.post_shim import PrePipelinePredictShim

_FITS: list[np.ndarray] = []


class _RecordingSelector(TransformerMixin, BaseEstimator):
    """A supervised selector stand-in: keeps every column, and records the targets each fit was given."""

    def fit(self, X, y=None):
        """Record ``y`` (a supervised selector reads it) and keep all columns."""
        _FITS.append(np.asarray(y, dtype=np.float64).copy())
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        """Every column is kept."""
        return np.asarray(X, dtype=np.float64)

    def get_support(self, indices: bool = False):
        """The mask a fitted selector exposes; its presence is what marks the step as supervised selection."""
        mask = np.ones(self.n_features_in_, dtype=bool)
        return np.flatnonzero(mask) if indices else mask


def _data(n: int = 240, seed: int = 0):
    """A plain linear target on three features."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = X.to_numpy() @ np.array([1.0, -2.0, 0.5]) + rng.normal(0.0, 0.1, n)
    return X, y


def _component(X, y, steps):
    """A shim-wrapped component whose pre_pipeline was fitted on the full train, as the suite leaves it."""
    pp = Pipeline(steps).fit(X, y)
    inner = LinearRegression().fit(pp.transform(X), y)
    return PrePipelinePredictShim(model=inner, pre_pipeline=pp)


def test_a_kfold_oof_never_fits_the_selector_on_a_folds_holdout_targets():
    """Every selector fit during the K-fold OOF saw only targets from that fold's train rows."""
    X, y = _data()
    comp = _component(X, y, [("scale", StandardScaler()), ("select", _RecordingSelector())])
    _FITS.clear()
    _P, y_h, _names, rows = compute_oof_holdout_predictions(
        component_models=[comp], component_names=["fs"], component_specs=[None], train_X=X, y_train_full=y,
        base_train_full_per_spec={}, holdout_frac=0.3, random_state=0, kfold=3, return_rows=True,
    )
    assert len(_FITS) == 3, "the OOF reused the full-train selector instead of refitting it on each of the 3 folds"
    assert rows is not None
    # The targets are continuous, so a value identifies its row. A fold's holdout is what its fit did NOT see, and the three
    # holdouts must partition the OOF rows: then no fit ever saw the targets of the rows it went on to predict.
    oof_values = set(np.round(y[rows], 12))
    unseen = [oof_values - set(np.round(fit_y, 12)) for fit_y in _FITS]
    assert all(unseen), "a fold's selector was fitted on every OOF row, its own holdout included"
    assert sum(len(u) for u in unseen) == len(oof_values) and set().union(*unseen) == oof_values


def test_an_unsupervised_pipeline_keeps_the_reuse():
    """A scaler-only pipeline is still reused as fitted: refitting it would cost time and change nothing about honesty."""
    X, y = _data()
    comp = _component(X, y, [("scale", StandardScaler())])
    pp = comp.pre_pipeline
    mean_before = pp.named_steps["scale"].mean_.copy()
    compute_oof_holdout_predictions(
        component_models=[comp], component_names=["plain"], component_specs=[None], train_X=X, y_train_full=y,
        base_train_full_per_spec={}, holdout_frac=0.3, random_state=0, kfold=3,
    )
    np.testing.assert_array_equal(pp.named_steps["scale"].mean_, mean_before)


def test_the_shared_pipeline_is_refit_once_per_fold_not_once_per_component():
    """Two components built on one fitted pipeline share its fold refit through the per-fold memo."""
    X, y = _data()
    pp = Pipeline([("select", _RecordingSelector())]).fit(X, y)
    comps = [PrePipelinePredictShim(model=LinearRegression().fit(pp.transform(X), y), pre_pipeline=pp) for _ in range(2)]
    _FITS.clear()
    compute_oof_holdout_predictions(
        component_models=comps, component_names=["a", "b"], component_specs=[None, None], train_X=X, y_train_full=y,
        base_train_full_per_spec={}, holdout_frac=0.3, random_state=0, kfold=3,
    )
    assert len(_FITS) == 3, f"expected one refit per fold, got {len(_FITS)}"


def test_the_external_holdout_path_keeps_the_full_train_fit():
    """With an external holdout the full-train fit never saw the holdout rows, so nothing is refit."""
    X, y = _data()
    X_val, y_val = _data(n=80, seed=1)
    comp = _component(X, y, [("select", _RecordingSelector())])
    _FITS.clear()
    compute_oof_holdout_predictions(
        component_models=[comp], component_names=["fs"], component_specs=[None], train_X=X, y_train_full=y,
        base_train_full_per_spec={}, holdout_frac=0.3, random_state=0, external_holdout_X=X_val, external_holdout_y=y_val,
    )
    assert _FITS == [], "the external path refit a selector that had never seen the holdout"
