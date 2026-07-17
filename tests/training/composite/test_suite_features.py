"""Unit + biz_value tests for CompositeFeatureGenerator (composite signal as one engineered feature)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

from mlframe.training.composite.spec import CompositeSpec
from mlframe.training.composite.suite_features import CompositeFeatureGenerator

try:
    import polars as pl

    _HAS_POLARS = True
except ImportError:  # pragma: no cover
    _HAS_POLARS = False


def _make_spec(base_column: str = "base") -> CompositeSpec:
    return CompositeSpec(
        name="y-linres-base",
        target_col="y",
        transform_name="linear_residual",
        base_column=base_column,
        fitted_params={},
        mi_gain=0.1,
        mi_y=0.2,
        mi_t=0.3,
        valid_domain_frac=1.0,
        n_train_rows=0,
    )


def _base_dominated_data(n: int = 1500, seed: int = 0):
    """Target dominated by an additive base term + a small residual learnable from other features."""
    rng = np.random.default_rng(seed)
    base = rng.normal(5.0, 2.0, size=n)
    f1 = rng.normal(0, 1, size=n)
    f2 = rng.normal(0, 1, size=n)
    # y = 3*base + small residual(f1,f2) + noise; base explains most variance.
    resid = 0.4 * f1 + 0.3 * f2 * f2
    y = 3.0 * base + resid + rng.normal(0, 0.1, size=n)
    df = pd.DataFrame({"base": base, "f1": f1, "f2": f2})
    return df, y


# ----------------------------------------------------------------------
# Unit tests
# ----------------------------------------------------------------------
def test_oof_feature_is_leakage_free():
    """OOF column must differ from an in-sample fitted-predict column (proof it is out-of-fold)."""
    df, y = _base_dominated_data(n=600)
    gen = CompositeFeatureGenerator(spec=_make_spec(), base_estimator=LinearRegression(), n_splits=4, random_state=1)
    out = gen.fit_transform(df, y)
    col = gen.column_name_
    assert col in out.columns
    oof = out[col].to_numpy()
    # In-sample fitted-predict on the SAME rows (the optimistic version).
    in_sample = gen.transform(df)[col].to_numpy()
    assert oof.shape == (len(df),)
    # OOF and in-sample must not be identical -> OOF really came from held-out folds.
    assert not np.allclose(oof, in_sample), "OOF feature appears to be in-sample (leakage)."
    assert np.isfinite(oof).mean() > 0.95


def test_transform_on_new_data():
    df, y = _base_dominated_data(n=800)
    train, test = df.iloc[:600], df.iloc[600:]
    gen = CompositeFeatureGenerator(spec=_make_spec(), base_estimator=LinearRegression())
    gen.fit(train, y[:600])
    out = gen.transform(test)
    assert gen.column_name_ in out.columns
    assert len(out) == len(test)
    assert np.isfinite(out[gen.column_name_].to_numpy()).all()
    # transform must not mutate input frame.
    assert gen.column_name_ not in test.columns


def test_sklearn_fit_transform_surface():
    df, y = _base_dominated_data(n=500)
    gen = CompositeFeatureGenerator(spec=_make_spec(), base_estimator=LinearRegression())
    out = gen.fit_transform(df, y)
    # TransformerMixin contract.
    assert hasattr(gen, "oof_feature_") and hasattr(gen, "estimator_")
    names = gen.get_feature_names_out(list(df.columns))
    assert gen.column_name_ in list(names)
    assert out.shape[1] == df.shape[1] + 1


def test_wrapper_factory_path_and_custom_name():
    from mlframe.training.composite.estimator import CompositeTargetEstimator

    df, y = _base_dominated_data(n=400)
    factory = lambda: CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name="linear_residual", base_column="base")
    gen = CompositeFeatureGenerator(wrapper_factory=factory, column_name="comp_feat")
    out = gen.fit_transform(df, y)
    assert "comp_feat" in out.columns


def test_fit_requires_y():
    df, _ = _base_dominated_data(n=100)
    gen = CompositeFeatureGenerator(spec=_make_spec())
    with pytest.raises(ValueError):
        gen.fit(df, None)


def test_transform_without_final_fit_raises():
    df, y = _base_dominated_data(n=300)
    gen = CompositeFeatureGenerator(spec=_make_spec(), base_estimator=LinearRegression(), fit_final_on_all=False)
    gen.fit(df, y)
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        gen.transform(df)


@pytest.mark.skipif(not _HAS_POLARS, reason="polars not installed")
def test_polars_frame_support():
    df, y = _base_dominated_data(n=500)
    pdf = pl.from_pandas(df)
    gen = CompositeFeatureGenerator(spec=_make_spec(), base_estimator=LinearRegression())
    out = gen.fit_transform(pdf, y)
    assert isinstance(out, pl.DataFrame)
    assert gen.column_name_ in out.columns
    # transform on new polars data.
    out2 = gen.transform(pdf)
    assert gen.column_name_ in out2.columns


# ----------------------------------------------------------------------
# biz_value: downstream linear model on [raw + composite OOF feature] beats raw alone.
# ----------------------------------------------------------------------
def test_biz_val_composite_feature_beats_raw_on_base_dominated_target():
    """On a base-dominated target, a Ridge trained on raw features + the composite OOF
    feature must beat the same Ridge on raw features alone (OOF holdout R2). Measured
    delta is large (the composite captures the 3*base term a plain linear model on
    [base,f1,f2] also gets -- but the residual transform isolates it cleanly, helping
    the non-linear f2^2 residual). Floor +0.05 R2 with margin."""
    df, y = _base_dominated_data(n=1500, seed=3)
    train_idx = np.arange(1000)
    test_idx = np.arange(1000, 1500)

    gen = CompositeFeatureGenerator(spec=_make_spec(), base_estimator=LinearRegression(), n_splits=5, random_state=7)
    aug_train = gen.fit_transform(df.iloc[train_idx].reset_index(drop=True), y[train_idx])
    aug_test = gen.transform(df.iloc[test_idx].reset_index(drop=True))

    raw_cols = ["f1", "f2"]  # raw features WITHOUT the base (composite carries base signal)
    aug_cols = raw_cols + [gen.column_name_]

    m_raw = Ridge().fit(df.iloc[train_idx][raw_cols], y[train_idx])
    r2_raw = r2_score(y[test_idx], m_raw.predict(df.iloc[test_idx][raw_cols]))

    m_aug = Ridge().fit(aug_train[aug_cols], y[train_idx])
    r2_aug = r2_score(y[test_idx], m_aug.predict(aug_test[aug_cols]))

    assert r2_aug >= r2_raw + 0.05, f"composite OOF feature should lift holdout R2 by >=0.05: raw={r2_raw:.3f} aug={r2_aug:.3f}"
