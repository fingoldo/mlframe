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
from sklearn.feature_selection import SelectKBest, f_classif


def _fitted_reducing_selector(n_in=8, k=4, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(200, n_in)), columns=[f"f{i}" for i in range(n_in)])
    y = (X["f0"] + X["f1"] > 0).astype(int)
    sel = SelectKBest(f_classif, k=k).fit(X, y)
    assert getattr(sel, "n_features_in_", None) == n_in
    return sel


def test_stale_identity_flag_does_not_skip_reducing_test_transform():
    from mlframe.training._pipeline_helpers import _prepare_test_split

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
    from mlframe.training._pipeline_helpers import _prepare_test_split

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
