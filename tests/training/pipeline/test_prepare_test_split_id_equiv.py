"""Regression: a stale ``_mlframe_identity_equivalent`` flag must not let
``_prepare_test_split`` skip the test transform for a column-reducing pipeline.

The flag is computed on the TRAIN transform and stored on the pipeline object;
when a pipeline / bare feature-selector object is reused across rounds (e.g. a
non-FS round sets it True, then the MRMR round reuses the same tree pipeline),
it goes stale. With the stale ``True`` the whole transform block in
``_prepare_test_split`` was gated off, so a RAW test frame was handed to a model
trained on the FS-reduced features -> ``XGBoost feature_names mismatch`` at
reporting (surfaced by fuzz ``c0143``: use_mrmr_fs + mrmr_cat_fe, the no-value-
transform models lgb/xgb).

The fix: when a FITTED feature selector is present AND the test frame still
carries the raw input width, the transform is not a no-op -> ignore the stale
flag and transform. Already-transformed frames (narrower than the fitted input
width) and unfitted-identity pipelines are unaffected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif


def _fitted_reducing_selector(n_in=8, k=4, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(200, n_in)), columns=[f"f{i}" for i in range(n_in)])
    y = (X["f0"] + X["f1"] > 0).astype(int)
    sel = SelectKBest(f_classif, k=k).fit(X, y)
    assert getattr(sel, "n_features_in_", None) == n_in
    return sel


def test_stale_identity_flag_does_not_skip_reducing_test_transform():
    from mlframe.training.pipeline._pipeline_helpers import _prepare_test_split

    sel = _fitted_reducing_selector(n_in=8, k=4)
    # Simulate the stale flag: a column-reducing selector wrongly marked identity.
    sel._mlframe_identity_equivalent = True

    rng = np.random.default_rng(1)
    test_df = pd.DataFrame(rng.normal(size=(50, 8)), columns=[f"f{i}" for i in range(8)])
    test_target = rng.integers(0, 2, size=50)

    out_test_df, _out_target, _cols = _prepare_test_split(
        df=None,
        test_df=test_df,
        test_idx=None,
        test_target=test_target,
        target=None,
        real_drop_columns=[],
        model=object(),  # any non-None: the transform block is gated on model is not None
        pre_pipeline=sel,
        skip_pre_pipeline_transform=False,
        skip_preprocessing=False,
        selector_passthrough_cols=None,
    )

    # Post-fix: the selector transformed the raw 8-col test frame down to 4.
    # Pre-fix the stale identity flag skipped the transform -> 8 columns remained.
    assert out_test_df.shape[1] == 4, (
        f"test frame must be reduced 8->4 by the fitted selector; got "
        f"{out_test_df.shape[1]} columns (stale identity flag skipped the transform)"
    )


def test_already_transformed_test_frame_is_not_double_transformed():
    """Guard: when the test frame is ALREADY the selector's narrow output width
    (a skip-path cache hit), the is_raw check keeps it untouched even with the
    identity flag forced off -- no double-transform / shape error."""
    from mlframe.training.pipeline._pipeline_helpers import _prepare_test_split

    sel = _fitted_reducing_selector(n_in=8, k=4)
    sel._mlframe_identity_equivalent = True

    rng = np.random.default_rng(2)
    # Already 4-wide (transformed) -> not raw input width (8).
    already = pd.DataFrame(rng.normal(size=(50, 4)), columns=[f"s{i}" for i in range(4)])
    test_target = rng.integers(0, 2, size=50)

    out_test_df, _t, _c = _prepare_test_split(
        df=None, test_df=already, test_idx=None, test_target=test_target,
        target=None, real_drop_columns=[], model=object(), pre_pipeline=sel,
        skip_pre_pipeline_transform=True, skip_preprocessing=False,
        selector_passthrough_cols=None,
    )
    # Left as-is (4 columns); no attempt to re-run the 8->4 selector on a 4-col frame.
    assert out_test_df.shape[1] == 4


class _ZeroVarPrefilterSelector(TransformerMixin, BaseEstimator):
    """Minimal stand-in for the RFECV / MRMR name-keyed selector contract that
    reproduces the c0026 width gap: a zero-variance PRE-FILTER drops constant
    columns at fit entry, so ``feature_names_in_`` (and thus ``n_features_in_``)
    records FEWER features than the raw frame the suite later hands to
    ``_prepare_test_split`` as the test frame. ``support_`` then selects a subset
    of those. ``transform`` selects the kept columns BY NAME (mirrors
    ``RFECV.transform``), so it works on the raw frame -- the bug was purely in
    the gating that decided whether to CALL transform at all.
    """

    def fit(self, X, y=None):
        # Zero-variance pre-filter (drops constant columns) -> recorded input
        # width is narrower than the caller's raw frame.
        nunique = X.nunique(axis=0)
        kept_in = [c for c in X.columns if nunique[c] > 1]
        self.feature_names_in_ = np.asarray(kept_in, dtype=object)
        self.n_features_in_ = len(kept_in)
        # Select roughly half of the surviving columns.
        n_keep = max(1, len(kept_in) // 2)
        self.support_ = np.array([i < n_keep for i in range(len(kept_in))], dtype=bool)
        self.n_features_ = int(self.support_.sum())
        return self

    def get_support(self):
        return self.support_

    def transform(self, X, y=None):
        selected = [c for c, keep in zip(self.feature_names_in_, self.support_) if keep]
        return X[selected]


def test_stale_identity_skip_when_input_prefilter_narrows_recorded_width():
    """c0026: a selector whose recorded input width (post-zero-variance-filter)
    is SMALLER than the raw test frame must STILL transform the raw frame.

    The prior heuristic compared ``test_df.shape[1]`` against the selector's
    ``n_features_in_``. With a zero-variance pre-filter, ``n_features_in_`` < raw
    width, so a raw frame compared UNEQUAL and was misclassified as already-
    transformed -> the transform was skipped -> the model (trained on the
    selected subset) received the full raw frame -> LightGBMError feature-count
    mismatch. The fix discriminates on the selector's OUTPUT width instead.
    """
    from mlframe.training.pipeline._pipeline_helpers import _prepare_test_split

    rng = np.random.default_rng(7)
    # 7 raw cols: one is constant (zero-variance) -> the selector records 6 input
    # features and selects 3 of them. Raw test frame keeps all 7 -- exactly the
    # c0026 7 -> {6 recorded} -> {3 selected} shape.
    cols = [f"num_{i}" for i in range(6)] + ["num_zero"]
    X = pd.DataFrame(rng.normal(size=(200, 7)), columns=cols)
    X["num_zero"] = 0.0  # constant column the zero-variance pre-filter drops
    y = (X["num_0"] + X["num_1"] > 0).astype(int)

    sel = _ZeroVarPrefilterSelector().fit(X, y)
    assert sel.n_features_in_ == 6  # zero-var col dropped from recorded input
    n_selected = int(sel.support_.sum())
    assert n_selected < 7  # selector reduces below the raw width

    # Stale flag set True (the reuse-across-rounds scenario) -- with the bug this
    # short-circuited the whole transform block.
    sel._mlframe_identity_equivalent = True

    test_df = pd.DataFrame(rng.normal(size=(50, 7)), columns=cols)
    test_df["num_zero"] = 0.0
    test_target = rng.integers(0, 2, size=50)

    out_test_df, _t, _c = _prepare_test_split(
        df=None,
        test_df=test_df,
        test_idx=None,
        test_target=test_target,
        target=None,
        real_drop_columns=[],
        model=object(),
        pre_pipeline=sel,
        skip_pre_pipeline_transform=False,
        skip_preprocessing=False,
        selector_passthrough_cols=None,
    )

    # Post-fix: the raw 7-col test frame is reduced to the selected width.
    # Pre-fix: stayed at 7 (n_features_in_=6 != 7 -> "already transformed" -> skip).
    assert out_test_df.shape[1] == n_selected, (
        f"raw 7-col test frame must be reduced to the selector's {n_selected} "
        f"selected columns; got {out_test_df.shape[1]} (the zero-variance "
        f"pre-filter narrowed n_features_in_ below the raw width and the gating "
        f"wrongly skipped the transform)"
    )


def test_lgb_reuse_shim_reports_on_full_frame_after_fs_no_feature_mismatch(tmp_path):
    """End-to-end c0026 sensor: an LGB-dataset-reuse model trained on the
    FS-reduced frame must report/predict on the fuller raw frame without raising
    ``LightGBMError: number of features in data (N) != training (M)``.

    Drives the real ``LGBMClassifierWithDatasetReuse`` shim through the same
    ``_prepare_test_split`` gating that c0026 hit: fit the model on the selector's
    reduced columns, then route a raw (wider) test frame through
    ``_prepare_test_split`` and call ``predict_proba`` on the result -- which must
    succeed because the gating now transforms the raw frame down to the trained
    feature set.
    """
    pytest.importorskip("lightgbm")
    from mlframe.training.lgb_shim import LGBMClassifierWithDatasetReuse
    from mlframe.training.pipeline._pipeline_helpers import _prepare_test_split

    rng = np.random.default_rng(11)
    cols = [f"num_{i}" for i in range(6)] + ["num_zero"]
    X = pd.DataFrame(rng.normal(size=(400, 7)), columns=cols)
    X["num_zero"] = 0.0
    y = (X["num_0"] + X["num_1"] > 0).astype(int)

    sel = _ZeroVarPrefilterSelector().fit(X, y)
    selected = [c for c, keep in zip(sel.feature_names_in_, sel.support_) if keep]

    # Train the LGB-reuse model on the FS-reduced frame (the trained feature set).
    model = LGBMClassifierWithDatasetReuse(n_estimators=10, verbose=-1)
    model.fit(X[selected], y.to_numpy())
    assert model._Booster.num_feature() == len(selected)

    # Stale identity flag -- the c0026 reuse-across-rounds condition.
    sel._mlframe_identity_equivalent = True

    test_df = pd.DataFrame(rng.normal(size=(60, 7)), columns=cols)
    test_df["num_zero"] = 0.0
    test_target = rng.integers(0, 2, size=60)

    out_test_df, _t, _c = _prepare_test_split(
        df=None,
        test_df=test_df,
        test_idx=None,
        test_target=test_target,
        target=None,
        real_drop_columns=[],
        model=model,
        pre_pipeline=sel,
        skip_pre_pipeline_transform=False,
        skip_preprocessing=False,
        selector_passthrough_cols=None,
    )

    # The reporting frame must be aligned to the model's trained feature set.
    assert out_test_df.shape[1] == len(selected)
    assert list(out_test_df.columns) == selected

    # The actual c0026 failure point: predict on the prepared test frame. Pre-fix
    # the raw 7-col frame reached here and LightGBM raised the feature mismatch.
    probs = model.predict_proba(out_test_df)
    assert probs.shape[0] == 60
