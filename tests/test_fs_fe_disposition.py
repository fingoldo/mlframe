"""Regression tests for the FS / FE disposition pass.

Each test below maps to a per-finding fix in the audit. Naming follows
``test_<area>_<short_description>`` so pytest -k filtering stays usable.
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# FS-1: MRMR no longer wrapped in SimpleImputer; NaN handled natively.
# ----------------------------------------------------------------------------


def test_fs1_build_pre_pipelines_mrmr_unwrapped_handles_nan():
    """``_build_pre_pipelines`` must produce a bare MRMR (no Pipeline+SimpleImputer wrap).

    The pre-fix wrapping silently imputed NaN at fit-time which routed NaN-aware downstream
    backends (CB/LGB/XGB) onto the imputed feature instead of the original NaN signal.
    """
    from mlframe.training.core._setup_helpers import _build_pre_pipelines
    from mlframe.feature_selection.filters import MRMR
    from sklearn.pipeline import Pipeline

    pre_pipelines, pre_pipeline_names = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=None,
        rfecv_models_params=None,
        use_mrmr_fs=True,
        mrmr_kwargs={"fe_max_steps": 0, "verbose": 0},
        custom_pre_pipelines=None,
    )
    mrmr_objs = [p for p, n in zip(pre_pipelines, pre_pipeline_names) if "MRMR" in n]
    assert len(mrmr_objs) == 1
    assert isinstance(mrmr_objs[0], MRMR), f"expected bare MRMR, got {type(mrmr_objs[0]).__name__}"
    assert not isinstance(mrmr_objs[0], Pipeline), "MRMR must not be wrapped in a Pipeline"


def test_fs1_mrmr_fits_on_nan_inputs_natively():
    """MRMR.fit must accept NaN values in X without raising (uses nan_strategy)."""
    from mlframe.feature_selection.filters import MRMR

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    # Sprinkle ~5% NaN into two columns.
    X[rng.choice(200, 10, replace=False), 0] = np.nan
    X[rng.choice(200, 10, replace=False), 2] = np.nan
    y = (rng.normal(size=200) > 0).astype(int)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])

    mrmr = MRMR(fe_max_steps=0, verbose=0)
    mrmr.fit(X_df, y)  # must not raise
    assert mrmr.n_features_ >= 0


# ----------------------------------------------------------------------------
# FS-3: FeatureSelectionConfig validates mrmr_kwargs and rfecv_kwargs.
# ----------------------------------------------------------------------------


def test_fs3_feature_selection_config_rejects_unknown_mrmr_kwarg():
    """Typo'd MRMR kwarg (``fe_max_step`` instead of ``fe_max_steps``) must fail at config build."""
    from mlframe.training.configs import FeatureSelectionConfig
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=r"mrmr_kwargs.*unknown key"):
        FeatureSelectionConfig(use_mrmr_fs=True, mrmr_kwargs={"fe_max_step": 3})


def test_fs3_feature_selection_config_rejects_unknown_rfecv_kwarg():
    """Typo'd RFECV kwarg must fail at config build."""
    from mlframe.training.configs import FeatureSelectionConfig
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=r"rfecv_kwargs.*unknown key"):
        FeatureSelectionConfig(rfecv_kwargs={"max_runtime_min": 5})  # missing trailing 's'


def test_fs3_feature_selection_config_accepts_valid_kwargs():
    """Valid kwargs (cv_n_splits + a real RFECV.__init__ arg) must pass validation."""
    from mlframe.training.configs import FeatureSelectionConfig

    cfg = FeatureSelectionConfig(
        use_mrmr_fs=True,
        mrmr_kwargs={"fe_max_steps": 1, "verbose": 0},
        rfecv_kwargs={"max_runtime_mins": 5.0, "cv_n_splits": 3},
    )
    assert cfg.mrmr_kwargs["fe_max_steps"] == 1


# ----------------------------------------------------------------------------
# FS-5: custom_pre_pipelines are cloned before insertion.
# ----------------------------------------------------------------------------


def test_fs5_custom_pre_pipelines_cloned_before_insertion():
    """A user-supplied transformer should be CLONED into the returned list,
    not appended by identity (otherwise state leaks across the model loop)."""
    from mlframe.training.core._setup_helpers import _build_pre_pipelines
    from sklearn.preprocessing import StandardScaler

    user_pipeline = StandardScaler()
    pre_pipelines, pre_pipeline_names = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=None,
        rfecv_models_params=None,
        use_mrmr_fs=False,
        mrmr_kwargs=None,
        custom_pre_pipelines={"scaler_a": user_pipeline},
    )
    assert len(pre_pipelines) == 1
    assert pre_pipelines[0] is not user_pipeline, "custom_pre_pipelines must be cloned, not stored by identity"
    assert type(pre_pipelines[0]).__name__ == "StandardScaler"


# ----------------------------------------------------------------------------
# FS-Low: TimeSeriesSplit auto-detect works on polars frames.
# ----------------------------------------------------------------------------


def test_fs_low_rfecv_timeseries_autodetect_polars():
    """RFECV's monotonic-datetime auto-detect now triggers for polars frames carrying
    exactly one sorted datetime column (mirror of the pandas DatetimeIndex path)."""
    pl = pytest.importorskip("polars")
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit
    from mlframe.feature_selection.wrappers._rfecv import RFECV

    n = 60
    # One sorted datetime column + a few numerics + a target.
    X = pl.DataFrame({
        "ts": pl.datetime_range(
            start=pl.datetime(2024, 1, 1),
            end=pl.datetime(2024, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        ),
        "f0": np.arange(n, dtype=np.float64),
        "f1": np.linspace(0, 1, n),
    })
    y = (np.arange(n) * 0.5 + np.random.default_rng(0).normal(0, 0.1, n)).astype(np.float64)

    # Pass cv=3 (numeric) so the auto-detect block fires.
    rfecv = RFECV(estimator=Ridge(), cv=3, min_features_to_select=1, max_runtime_mins=1)
    # Probe through the constructor path that triggers auto-detect: call fit and inspect cv on the
    # resulting estimator. RFECV.fit assigns the resolved cv to self.cv_ in mlframe; if not, check the
    # ``cv`` argument was upgraded by reading the logger output. Here we use the public reflected
    # ``cv`` after fit if available.
    try:
        rfecv.fit(X, y)
    except Exception:
        # Some sklearn / polars-version combos require an explicit pandas conversion;
        # the auto-detect log path is still exercised before fit attempts iteration.
        pass
    # The fitted attribute ``cv_`` (if present) or the resolved cv on the wrapper carries the
    # TimeSeriesSplit instance when auto-detect triggered.
    _cv = getattr(rfecv, "cv_", None) or getattr(rfecv, "cv", None)
    assert isinstance(_cv, TimeSeriesSplit), (
        f"expected TimeSeriesSplit auto-detect on polars datetime input, got {type(_cv).__name__}"
    )


# ----------------------------------------------------------------------------
# FS-Low: run_additional_rfecv_minutes now has a regression branch.
# ----------------------------------------------------------------------------


def test_fs_low_run_additional_rfecv_has_regression_branch():
    """Source-level guarantee: the regression branch (CatBoostRegressor + CB_REGR) is now wired
    into the additional-RFECV pass in MRMR.fit. Pre-fix the else of ``if len(y)/nunique>100`` was
    silently empty so regression callers got no benefit from run_additional_rfecv_minutes."""
    import inspect
    from mlframe.feature_selection.filters import mrmr as _mrmr_mod

    src = inspect.getsource(_mrmr_mod)
    assert "CatBoostRegressor" in src, "regression branch must import CatBoostRegressor"
    assert "CB_REGR" in src, "regression branch must reference configs.CB_REGR"


# ----------------------------------------------------------------------------
# FE-1: Enum union in _align_xgb_cat_categories no longer leaks test categories.
# ----------------------------------------------------------------------------


def test_fe1_pandas_cat_union_no_test_leak():
    """Categories that appear ONLY in test_df must NOT propagate into train_df after alignment."""
    from mlframe.training._eval_helpers import _align_xgb_cat_categories

    train = pd.DataFrame({"c": pd.Categorical(["a", "b", "a", "b"])})
    val = pd.DataFrame({"c": pd.Categorical(["a", "b"])})
    # ``z`` is a category ONLY in test; it must not leak into train.cat.categories.
    test = pd.DataFrame({"c": pd.Categorical(["a", "z"])})

    train_out, val_out, test_out = _align_xgb_cat_categories(
        model_type_name="xgb", train_df=train, val_df=val, test_df=test
    )
    train_cats = set(train_out["c"].cat.categories)
    assert "z" not in train_cats, (
        f"test-only category 'z' leaked into train categories: {train_cats}"
    )


def test_fe1_pandas_cat_union_picks_up_val_only_categories():
    """Categories that appear in val_df (not train) SHOULD be added to the union (val is
    authorised because the user uses it for early stopping/model selection)."""
    from mlframe.training._eval_helpers import _align_xgb_cat_categories

    train = pd.DataFrame({"c": pd.Categorical(["a", "b"])})
    val = pd.DataFrame({"c": pd.Categorical(["a", "x"])})  # 'x' only in val
    test = None

    train_out, _, _ = _align_xgb_cat_categories(
        model_type_name="xgb", train_df=train, val_df=val, test_df=test
    )
    assert "x" in set(train_out["c"].cat.categories)


# ----------------------------------------------------------------------------
# FE-4: WARN log when CB + ordinal + cat_features.
# ----------------------------------------------------------------------------


def test_fe4_cb_ordinal_cat_features_warns(caplog):
    """When mlframe_models contains 'cb' AND categorical_encoding='ordinal' AND
    cat_features is non-empty, _phase_fit_pipeline emits a logger.warning."""
    pl = pytest.importorskip("polars")
    import logging
    from mlframe.training.configs import (
        PreprocessingBackendConfig,
        PreprocessingConfig,
        FeatureTypesConfig,
    )
    from mlframe.training.core._phase_helpers import _phase_fit_pipeline

    # ``c`` is a polars Utf8/String column = an auto-detectable cat candidate; the WARN at the top
    # of _phase_fit_pipeline reads cat-like columns directly from the train_df schema.
    train = pl.DataFrame({"c": ["a", "b", "a", "b"], "x": [1.0, 2.0, 3.0, 4.0]})

    pipe_cfg = PreprocessingBackendConfig(
        categorical_encoding="ordinal", skip_categorical_encoding=False, scaler_name=None
    )
    prep_cfg = PreprocessingConfig()
    ft_cfg = FeatureTypesConfig()

    caplog.set_level(logging.WARNING, logger="mlframe.training.core._phase_helpers")
    try:
        _phase_fit_pipeline(
            train_df=train, val_df=None, test_df=None,
            mlframe_models=["cb"],
            pipeline_config=pipe_cfg,
            preprocessing_config=prep_cfg,
            feature_types_config=ft_cfg,
            preprocessing_extensions=None,
            metadata={},
            verbose=True,
        )
    except Exception:
        # Even if a downstream step fails on this synthetic frame, the WARN
        # must have already fired before the failure (the check is at the
        # very top of _phase_fit_pipeline, right after categorical-encoding
        # auto-detection).
        pass

    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    matched = [m for m in msgs if "CatBoost is in mlframe_models" in m and "ordinal" in m]
    assert matched, f"expected CB+ordinal+cat_features WARN, got: {msgs}"
