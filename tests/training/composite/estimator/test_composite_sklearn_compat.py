"""Tests for native sklearn Pipeline / TransformedTargetRegressor integration
of composite targets (mlframe.training.composite.sklearn_compat).

Covers the two integration shapes:
1. make_composite_regressor as a Pipeline final step (fit + predict + clone).
2. CompositeTargetTransformer plugged into TransformedTargetRegressor via the
   func / inverse_func pair (base transform) and via transformer= (unary),
   plus the get_feature_names_out passthrough and clone / pickle round-trips.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.training.composite.estimator import CompositeTargetEstimator
from mlframe.training.composite.sklearn_compat import (
    CompositeTargetTransformer,
    make_composite_regressor,
)


# ---- smoke import ----------------------------------------------------

def test_module_smoke_import():
    import importlib

    m = importlib.import_module("mlframe.training.composite.sklearn_compat")
    assert hasattr(m, "make_composite_regressor")
    assert hasattr(m, "CompositeTargetTransformer")
    assert set(m.__all__) == {"make_composite_regressor", "CompositeTargetTransformer"}


# ---- fixtures --------------------------------------------------------

def _synth(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = rng.normal(5.0, 1.0, n)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    X = pd.DataFrame({"base": base, "f1": f1, "f2": f2})
    y = base + 0.5 * f1 - 0.3 * f2 + rng.normal(0.0, 0.1, n)
    return X, y


# ---- path 1: make_composite_regressor --------------------------------

def test_make_composite_regressor_returns_estimator():
    reg = make_composite_regressor(LinearRegression(), "diff", "base")
    assert isinstance(reg, CompositeTargetEstimator)
    assert reg.transform_name == "diff"
    assert reg.base_column == "base"


def test_make_composite_regressor_eager_validates_unknown_transform():
    from mlframe.training.composite.transforms import UnknownTransformError

    with pytest.raises(UnknownTransformError):
        make_composite_regressor(LinearRegression(), "not_a_transform", "base")


def test_make_composite_regressor_requires_base_for_base_transform():
    with pytest.raises(ValueError, match="requires a base"):
        make_composite_regressor(LinearRegression(), "diff", "")


def test_make_composite_regressor_unary_no_base_ok():
    reg = make_composite_regressor(LinearRegression(), "cbrt_y", "")
    assert isinstance(reg, CompositeTargetEstimator)


def test_make_composite_regressor_passes_through_kwargs():
    reg = make_composite_regressor(
        LinearRegression(), "diff", "base", fallback_predict="nan", drop_invalid_rows=False
    )
    assert reg.fallback_predict == "nan"
    assert reg.drop_invalid_rows is False


def test_pipeline_with_composite_regressor_fits_and_predicts():
    # Scaler over feature columns, composite regressor as final step. The base
    # column survives the scaler (StandardScaler keeps column names off; we use
    # a frame-preserving setup so the wrapper can pull 'base' at predict).
    X, y = _synth()
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler().set_output(transform="pandas")),
            ("reg", make_composite_regressor(LinearRegression(), "diff", "base")),
        ]
    )
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == (len(y),)
    # diff over the dominant base + a linear model should recover y closely.
    assert np.mean(np.abs(pred - y)) < 0.5


def test_make_composite_regressor_clone_roundtrips():
    reg = make_composite_regressor(LinearRegression(), "diff", "base")
    cloned = clone(reg)
    assert isinstance(cloned, CompositeTargetEstimator)
    assert cloned.transform_name == "diff"
    assert cloned.base_column == "base"
    # cloned is unfitted and independently fittable
    X, y = _synth()
    cloned.fit(X, y)
    assert cloned.predict(X).shape == (len(y),)


# ---- path 2: CompositeTargetTransformer ------------------------------

def test_transformer_forward_inverse_roundtrip_base():
    X, y = _synth()
    tr = CompositeTargetTransformer("diff", "base")
    tr.fit(y, X)
    T = tr.transform(y)
    y_back = tr.inverse_transform(T)
    assert np.max(np.abs(y_back - y)) < 1e-9


def test_transformer_forward_inverse_roundtrip_unary():
    _, y = _synth()
    y_pos = np.abs(y) + 1.0
    tr = CompositeTargetTransformer("cbrt_y")
    tr.fit(y_pos)
    T = tr.transform(y_pos)
    y_back = tr.inverse_transform(T)
    assert np.max(np.abs(y_back - y_pos)) < 1e-8


def test_transformer_func_inverse_func_are_bound_methods():
    tr = CompositeTargetTransformer("diff", "base")
    # property returns the bound methods (picklable, no closure)
    assert tr.func == tr.transform
    assert tr.inverse_func == tr.inverse_transform


def test_ttr_base_transform_via_func_inverse_func():
    X, y = _synth()
    Xf = X.drop(columns=["base"])
    tr = CompositeTargetTransformer("diff", "base")
    tr.fit(y, X)
    # check_inverse=False: TTR's invertibility probe runs on a SUBSAMPLE, which
    # would break the positional base replay; we disable it (the transform is
    # exactly invertible by construction).
    ttr = TransformedTargetRegressor(
        regressor=LinearRegression(),
        func=tr.func,
        inverse_func=tr.inverse_func,
        check_inverse=False,
    )
    ttr.fit(Xf, y)
    pred = ttr.predict(Xf)
    assert pred.shape == (len(y),)
    assert np.mean(np.abs(pred - y)) < 0.5


def test_ttr_unary_transform_via_transformer_instance():
    X, y = _synth()
    Xf = X.drop(columns=["base"])
    y_pos = np.abs(y) + 1.0
    # Unary transform needs no base, so the TTR-managed transformer.fit(y) path
    # (which never passes X) works end-to-end.
    ttr = TransformedTargetRegressor(
        regressor=LinearRegression(),
        transformer=CompositeTargetTransformer("cbrt_y"),
        check_inverse=False,
    )
    ttr.fit(Xf, y_pos)
    pred = ttr.predict(Xf)
    assert pred.shape == (len(y_pos),)
    assert np.all(np.isfinite(pred))


def test_transformer_clone_roundtrips():
    tr = CompositeTargetTransformer("diff", "base", feature_names_out=["z"])
    cloned = clone(tr)
    assert isinstance(cloned, CompositeTargetTransformer)
    assert cloned.transform_name == "diff"
    assert cloned.base_column == "base"
    assert list(cloned.feature_names_out) == ["z"]


def test_transformer_pickle_roundtrips():
    import pickle

    X, y = _synth()
    tr = CompositeTargetTransformer("diff", "base")
    tr.fit(y, X)
    tr2 = pickle.loads(pickle.dumps(tr))
    assert np.allclose(tr2.transform(y), tr.transform(y))


def test_get_feature_names_out_configured():
    tr = CompositeTargetTransformer("diff", "base", feature_names_out=["a", "b"])
    out = tr.get_feature_names_out()
    assert list(out) == ["a", "b"]


def test_get_feature_names_out_echoes_input():
    tr = CompositeTargetTransformer("diff", "base")
    out = tr.get_feature_names_out(["c1", "c2", "c3"])
    assert list(out) == ["c1", "c2", "c3"]


def test_get_feature_names_out_default_synthetic():
    tr = CompositeTargetTransformer("diff", "base")
    out = tr.get_feature_names_out()
    assert list(out) == ["diff_target"]


def test_transformer_base_array_path():
    X, y = _synth()
    base = X["base"].to_numpy()
    tr = CompositeTargetTransformer("diff", base=base)
    tr.fit(y)  # no X needed when base array supplied
    y_back = tr.inverse_transform(tr.transform(y))
    assert np.max(np.abs(y_back - y)) < 1e-9


def test_transformer_base_length_mismatch_raises():
    _, y = _synth(n=100)
    tr = CompositeTargetTransformer("diff", base=np.arange(50, dtype=float))
    with pytest.raises(ValueError, match="base length"):
        tr.fit(y)


def test_transformer_missing_base_raises():
    _, y = _synth()
    tr = CompositeTargetTransformer("diff")  # base transform, no base anywhere
    with pytest.raises(ValueError, match="requires a base"):
        tr.fit(y)


def test_transformer_replay_row_count_mismatch_raises():
    X, y = _synth(n=200)
    tr = CompositeTargetTransformer("diff", "base")
    tr.fit(y, X)
    # Asking the forward for a different row count than fit breaks positional
    # base replay -- must raise, never silently mis-align.
    with pytest.raises(ValueError, match="replayed positionally"):
        tr.transform(y[:100])


# ---- biz_value: composite target measurably beats raw on a base-dominated DGP ----

def test_biz_val_composite_transformer_beats_identity_on_base_dominated_target():
    """On a target that is ~base + small residual, modelling the diff residual
    (composite target) yields a much lower holdout RMSE than learning y directly
    from the NON-base features. Floor is a 2x RMSE improvement; the structural
    win is large because the base carries ~90% of the target variance and the
    diff transform hands that part to the inverse for free."""
    rng = np.random.default_rng(7)
    n = 2000
    base = rng.normal(10.0, 3.0, n)
    f1 = rng.normal(size=n)
    X = pd.DataFrame({"base": base, "f1": f1})
    y = base + 0.4 * f1 + rng.normal(0.0, 0.2, n)
    Xf = X.drop(columns=["base"])

    ntr = 1500
    Xtr, Xte = X.iloc[:ntr], X.iloc[ntr:]
    Xftr, Xfte = Xf.iloc[:ntr], Xf.iloc[ntr:]
    ytr, yte = y[:ntr], y[ntr:]

    # Composite: model diff over base, invert.
    comp = make_composite_regressor(LinearRegression(), "diff", "base")
    comp.fit(Xtr, ytr)
    rmse_comp = float(np.sqrt(np.mean((comp.predict(Xte) - yte) ** 2)))

    # Identity baseline: learn y directly from non-base features only.
    base_model = LinearRegression().fit(Xftr, ytr)
    rmse_id = float(np.sqrt(np.mean((base_model.predict(Xfte) - yte) ** 2)))

    assert rmse_comp < rmse_id / 2.0, (
        f"composite diff RMSE {rmse_comp:.4f} should be < half the identity "
        f"RMSE {rmse_id:.4f} on a base-dominated target"
    )
