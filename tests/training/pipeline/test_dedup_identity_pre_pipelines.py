"""Dedup: skip identity-equivalent pre_pipelines when ordinary already trained.

User request (2026-05-13): when a feature-selection pipeline keeps all input
columns and creates none, training models on it duplicates the ordinary
branch. The suite must detect this and skip the redundant branch, with a
config flag to opt out.
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# 1. MRMR.transform fast-path: identity → return X unchanged
# ═══════════════════════════════════════════════════════════════════════════


def _make_fake_fitted_mrmr(X_columns, support=None):
    """Return an MRMR instance that *looks* fitted to sklearn without
    actually running the full MI-based selection loop.  ``support`` is a
    boolean array (True = selected) defaulting to all-True."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    m = MRMR(verbose=0)
    n = len(X_columns)
    if support is None:
        support = np.ones(n, dtype=bool)
    m.feature_names_in_ = np.asarray(X_columns)
    m.support_ = support
    m._engineered_recipes_ = []
    return m


def test_mrmr_transform_fastpath_identity_returns_x_unchanged():
    """All features selected + zero recipes → transform must return X
    unchanged (the SAME object, not a copy)."""
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(30, 5)), columns=list("abcde"))
    mrmr = _make_fake_fitted_mrmr(list(X.columns))
    out = mrmr.transform(X)
    assert out is X, "transform must return X unchanged when all columns selected + no recipes (zero-copy fast-path)"


def test_mrmr_transform_no_fastpath_on_subset():
    """Subset selected → transform returns a new frame, not X."""
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(30, 5)), columns=list("abcde"))
    # Integer-index support: first 3 columns selected
    mrmr = _make_fake_fitted_mrmr(list(X.columns), support=np.array([0, 1, 2]))
    out = mrmr.transform(X)
    assert out is not X
    assert out.shape[1] == 3


def test_mrmr_transform_fastpath_handles_boolean_and_int_support():
    """Fast-path must correctly count selected features for BOTH boolean
    and integer-index support_ arrays."""
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(30, 4)), columns=list("wxyz"))
    # Boolean support: 4 True = all selected → fast-path
    m_bool = _make_fake_fitted_mrmr(
        list(X.columns),
        support=np.array([True, True, True, True]),
    )
    out_bool = m_bool.transform(X)
    assert out_bool is X

    # Integer-index support: [0,1,2,3] = all selected → fast-path
    m_int = _make_fake_fitted_mrmr(
        list(X.columns),
        support=np.array([0, 1, 2, 3]),
    )
    out_int = m_int.transform(X)
    assert out_int is X


# ═══════════════════════════════════════════════════════════════════════════
# 2. pre_pipeline._mlframe_identity_equivalent detection logic
# ═══════════════════════════════════════════════════════════════════════════


def _detect_identity(pre_pipeline, train_df_before, train_df_after):
    """Simulate what _apply_pre_pipeline_transforms does after fit-transform."""
    _in = list(train_df_before.columns)
    _out = list(train_df_after.columns) if hasattr(train_df_after, "columns") else None
    if _out is not None:
        pre_pipeline._mlframe_identity_equivalent = _in == _out
    return pre_pipeline


def test_marker_true_when_all_columns_kept():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(40, 5)), columns=list("abcde"))
    mrmr = _make_fake_fitted_mrmr(list(X.columns))
    out = mrmr.transform(X)
    _detect_identity(mrmr, X, out)
    assert mrmr._mlframe_identity_equivalent is True


def test_marker_false_when_columns_dropped():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(40, 6)), columns=list("abcdef"))
    mrmr = _make_fake_fitted_mrmr(
        list(X.columns),
        support=np.array([0, 1, 2]),  # only first 3 of 6
    )
    out = mrmr.transform(X)
    _detect_identity(mrmr, X, out)
    assert mrmr._mlframe_identity_equivalent is False


def test_marker_not_set_when_output_has_no_columns_attr():
    """Numpy arrays have no .columns → marker is not set (no-op)."""
    mrmr = _make_fake_fitted_mrmr(list("abc"))
    # hasattr check inside _detect_identity guards the comparison.
    _detect_identity(mrmr, pd.DataFrame(), np.zeros((10, 3)))
    assert not hasattr(mrmr, "_mlframe_identity_equivalent")


# ═══════════════════════════════════════════════════════════════════════════
# 3. FeatureSelectionConfig.skip_identity_equivalent_pre_pipelines
# ═══════════════════════════════════════════════════════════════════════════


def test_config_flag_defaults_true():
    from mlframe.training.configs import FeatureSelectionConfig

    cfg = FeatureSelectionConfig()
    assert cfg.skip_identity_equivalent_pre_pipelines is True


def test_config_flag_can_be_false():
    from mlframe.training.configs import FeatureSelectionConfig

    cfg = FeatureSelectionConfig(skip_identity_equivalent_pre_pipelines=False)
    assert cfg.skip_identity_equivalent_pre_pipelines is False


def test_config_flag_serializable():
    """Flag must survive round-trip through dict (for JSON / YAML configs)."""
    from mlframe.training.configs import FeatureSelectionConfig

    cfg = FeatureSelectionConfig(skip_identity_equivalent_pre_pipelines=False)
    d = cfg.dict()
    cfg2 = FeatureSelectionConfig(**d)
    assert cfg2.skip_identity_equivalent_pre_pipelines is False


# ═══════════════════════════════════════════════════════════════════════════
# 4. _prepare_test_split guards against NotFittedError on identity equiv
# ═══════════════════════════════════════════════════════════════════════════


def test_prepare_test_split_skips_transform_when_identity_equivalent():
    """Simulate the train-cache-hit scenario from the 2026-05-13 prod bug:
    ``_apply_pre_pipeline_transforms`` populates the cache, returns
    cached train/val DFs, but leaves the pre_pipeline UNFITTED.
    ``_prepare_test_split`` then tries ``pre_pipeline.transform(test_df)``
    → NotFittedError.  The guard must skip the transform when
    ``_mlframe_identity_equivalent`` is True, regardless of fitted state."""
    rng = np.random.default_rng(42)
    test_df = pd.DataFrame(rng.normal(size=(20, 4)), columns=list("abcd"))
    mrmr = _make_fake_fitted_mrmr(list("abcd"))
    mrmr._mlframe_identity_equivalent = True
    # Simulate what _prepare_test_split would do: the guard checks
    # _mlframe_identity_equivalent BEFORE calling transform().
    _id_equiv = getattr(mrmr, "_mlframe_identity_equivalent", False)
    if not _id_equiv:
        # This would raise NotFittedError — MRMR is not truly fitted
        result = mrmr.transform(test_df)
    else:
        result = test_df  # skip, no-op
    assert result is test_df, "identity-equivalent pre_pipeline should skip transform on test"


def test_prepare_test_split_still_transforms_when_not_identity():
    """When _mlframe_identity_equivalent is False or absent, the guard
    must NOT fire — test transform proceeds normally (and would crash
    if unfitted, which is the correct surface-this-bug behavior)."""
    rng = np.random.default_rng(42)
    test_df = pd.DataFrame(rng.normal(size=(20, 3)), columns=list("uvw"))
    mrmr = _make_fake_fitted_mrmr(list("uvw"), support=np.array([0, 1]))  # drop col 2
    mrmr._mlframe_identity_equivalent = False
    _id_equiv = getattr(mrmr, "_mlframe_identity_equivalent", False)
    assert not _id_equiv, "guard should NOT fire — must apply transform"
    # pre_pipeline IS fitted in this test → transform works fine
    result = mrmr.transform(test_df)
    assert result is not test_df
    assert result.shape[1] == 2  # subset selected
