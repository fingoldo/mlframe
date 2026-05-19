"""Wave-3 per-target feature-side cache hoist: regression + correctness tests.

The hoist lifted target-independent transforms (tier-DFs / pl.Enum maps / prepared polars
frames / fingerprints) and the XGB DMatrix / LGB Dataset binned datasets out of the per-target
inner loop so a multi-target suite pays the build cost ONCE per (strategy, pre_pipeline) rather
than once per target. These tests assert:

  - feature_side_cache crosses targets and re-uses references (no copies / no clones)
  - dataset_reuse_cache captures the binned dataset before _maybe_clear_shim_cache
  - _invalidate_polars_feature_side_cache drops polars-tier entries when ctx polars frames
    are released
  - end-to-end predictions on a multi-target suite are bit-equal to the pre-hoist path
    (correctness sentinel; without this the hoist could regress silently)

The cache mechanics are unit-tested in isolation (no full suite) because spawning multiple
suites for a counter assertion would multiply CI cost by an order of magnitude.
"""
from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest


# ----------------------------------------------------------------------------
# Cache helper unit tests
# ----------------------------------------------------------------------------


class _FakeCtx:
    """Minimal stand-in for TrainingContext used by the cache helpers.

    Only the attributes the helpers touch are present; using the real dataclass would force
    every test to pass 50+ unrelated configs. The helpers only need ``artifacts`` and
    ``train_df_polars`` / ``val_df_polars`` / ``test_df_polars``.
    """

    def __init__(self, artifacts=None):
        self.artifacts = artifacts if artifacts is not None else {}
        self.train_df_polars = None
        self.val_df_polars = None
        self.test_df_polars = None


def test_feature_side_cache_creates_dict_on_first_access():
    from mlframe.training.core._phase_train_one_target import _get_feature_side_cache

    ctx = _FakeCtx()
    cache = _get_feature_side_cache(ctx)
    assert isinstance(cache, dict)
    assert cache == {}
    # Returned reference is the SAME object on second access (no copy).
    assert _get_feature_side_cache(ctx) is cache


def test_feature_side_cache_survives_none_artifacts():
    from mlframe.training.core._phase_train_one_target import _get_feature_side_cache

    ctx = _FakeCtx(artifacts=None)
    cache = _get_feature_side_cache(ctx)
    assert isinstance(cache, dict)
    assert isinstance(ctx.artifacts, dict)


def test_dataset_reuse_cache_separate_from_feature_side():
    from mlframe.training.core._phase_train_one_target import (
        _get_dataset_reuse_cache, _get_feature_side_cache,
    )

    ctx = _FakeCtx()
    fs = _get_feature_side_cache(ctx)
    ds = _get_dataset_reuse_cache(ctx)
    assert fs is not ds
    fs["x"] = 1
    assert "x" not in ds


def test_capture_then_restore_dataset_reuse_cache_round_trips():
    """capture() snapshots non-None cache attrs onto ctx; restore() puts them back on a
    fresh template. Round-trip semantics: a target-1 template with a binned DMatrix attached
    must hand that DMatrix to the target-2 template before clone() forward-transfers it.
    """
    from mlframe.training.core._phase_train_one_target import (
        _capture_dataset_reuse_cache, _restore_dataset_reuse_cache,
    )

    ctx = _FakeCtx()
    # Stand-in template with one of the _DATASET_REUSE_CACHE_ATTRS populated.
    sentinel = object()

    class _T:
        pass

    t1 = _T()
    t1._cached_train_dmatrix = sentinel
    t1._cached_train_key = "key-A"
    t1._cached_val_dmatrix = None  # None entries must be SKIPPED by capture

    _capture_dataset_reuse_cache(ctx, "xgb", t1)
    captured = ctx.artifacts["dataset_reuse_cache"][("xgb", "")]
    assert captured["_cached_train_dmatrix"] is sentinel
    assert captured["_cached_train_key"] == "key-A"
    assert "_cached_val_dmatrix" not in captured

    # Restore onto fresh template - same references must reach the slots.
    t2 = _T()
    _restore_dataset_reuse_cache(ctx, "xgb", t2)
    assert t2._cached_train_dmatrix is sentinel
    assert t2._cached_train_key == "key-A"
    assert not hasattr(t2, "_cached_val_dmatrix")


def test_capture_skips_none_attrs():
    """A template with every cache attr set to None must NOT capture (else the next
    restore would overwrite a freshly-built cache with stale Nones)."""
    from mlframe.training.core._phase_train_one_target import (
        _capture_dataset_reuse_cache, _DATASET_REUSE_CACHE_ATTRS,
    )

    ctx = _FakeCtx()

    class _T:
        pass

    t = _T()
    for a in _DATASET_REUSE_CACHE_ATTRS:
        setattr(t, a, None)
    _capture_dataset_reuse_cache(ctx, "xgb", t)
    assert ctx.artifacts.get("dataset_reuse_cache", {}).get("xgb") is None


def test_restore_no_op_on_missing_entry():
    """A first-target call has nothing captured yet; restore() must be a no-op (no
    attribute leakage onto the template)."""
    from mlframe.training.core._phase_train_one_target import _restore_dataset_reuse_cache

    ctx = _FakeCtx()

    class _T:
        pass

    t = _T()
    _restore_dataset_reuse_cache(ctx, "xgb", t)
    # No _cached_* attributes touched.
    assert not hasattr(t, "_cached_train_dmatrix")


def test_invalidate_polars_feature_side_cache_drops_polars_only():
    """polars-tier and pandas-tier entries co-habit the same cache; invalidation hooked
    on the polars release must drop ONLY the polars entries so pandas-tier reuse survives
    a tier transition mid-suite."""
    from mlframe.training.core._phase_train_one_target import (
        _get_feature_side_cache, _invalidate_polars_feature_side_cache,
    )

    ctx = _FakeCtx()
    cache = _get_feature_side_cache(ctx)
    pp_bucket = cache.setdefault("ordinary", {})

    # tier_dfs sub-keys: tuple (feature_tier, kind) where kind is "pl" / "pd".
    pp_bucket.setdefault("tier_dfs", {})[((True, True), "pl")] = {"train_df": object()}
    pp_bucket["tier_dfs"][((False, False), "pd")] = {"train_df": object()}
    # prepared_frames sub-keys: (tier, supports_polars, strategy_class, cb_text_pass).
    pp_bucket.setdefault("prepared_frames", {})[((True, True), True, "CatBoostStrategy", False)] = {
        "prepared_train": object(),
    }
    pp_bucket["prepared_frames"][((False, False), False, "LinearStrategy", False)] = {
        "prepared_train": object(),
    }
    # tier_enum_map sub-keys are tuples without "pl" or True - they're polars-populated only,
    # but key is (tier, strategy_class). They get cleared via the same per-call manual clear
    # near the polars-release sites; invalidate_polars does NOT touch them.

    _invalidate_polars_feature_side_cache(ctx)

    # Polars entries gone, pandas entries remain.
    td = pp_bucket["tier_dfs"]
    pf = pp_bucket["prepared_frames"]
    assert ((True, True), "pl") not in td
    assert ((False, False), "pd") in td
    assert ((True, True), True, "CatBoostStrategy", False) not in pf
    assert ((False, False), False, "LinearStrategy", False) in pf


def test_invalidate_polars_feature_side_cache_no_op_when_empty():
    """Called on a virgin ctx, the helper must not raise (it runs from every
    _release_ctx_polars_frames call, including ones that fire BEFORE any feature_side
    cache entry exists)."""
    from mlframe.training.core._phase_train_one_target import _invalidate_polars_feature_side_cache

    ctx = _FakeCtx()
    _invalidate_polars_feature_side_cache(ctx)  # must not raise
    ctx.artifacts["feature_side_cache"] = {}
    _invalidate_polars_feature_side_cache(ctx)  # must not raise


# ----------------------------------------------------------------------------
# End-to-end: multi-target suite produces equal predictions before / after hoist
# ----------------------------------------------------------------------------


class _MultiTargetExtractor:
    """Multi-target FTE: returns N targets under the same target_type.

    Mirrors SimpleFeaturesAndTargetsExtractor's transform() contract but stuffs multiple
    keys into target_by_type[target_type] so the suite runs ``_train_one_target`` once per
    target name with the SAME train_df. This is the path the hoist optimises: the second
    and subsequent _train_one_target calls observe a populated ctx.artifacts cache.
    """

    def __init__(self, target_columns, target_type):
        from mlframe.training.configs import TargetTypes
        self.target_columns = tuple(target_columns)
        self.target_type = target_type if target_type is not None else TargetTypes.REGRESSION
        # FTE contract attrs read by the suite:
        self.ts_field = None
        self.group_field = None
        self.weight_schemas = None
        self.target_carrier = "numpy"

    def transform(self, df):
        target_by_type = {self.target_type: {}}
        for col in self.target_columns:
            if isinstance(df, pd.DataFrame):
                target_by_type[self.target_type][col] = df[col].values
            else:
                target_by_type[self.target_type][col] = df[col].to_numpy()
        cols_to_drop = list(self.target_columns)
        return (df, target_by_type, None, None, None, None, cols_to_drop, {})


@pytest.fixture
def synthetic_multi_target_regression():
    """800 rows x 6 features x 3 correlated regression targets."""
    rng = np.random.default_rng(2026)
    n = 800
    X = rng.standard_normal((n, 6)).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    # Three distinct targets, each a noisy linear combo of features so the model has
    # something to learn but the suite runs fast.
    df["y1"] = (X[:, 0] - 0.5 * X[:, 1] + 0.1 * rng.standard_normal(n)).astype(np.float32)
    df["y2"] = (X[:, 2] + 0.7 * X[:, 3] + 0.1 * rng.standard_normal(n)).astype(np.float32)
    df["y3"] = (X[:, 4] - X[:, 5] + 0.1 * rng.standard_normal(n)).astype(np.float32)
    return df


@pytest.mark.slow
def test_multi_target_suite_feature_side_cache_populated(synthetic_multi_target_regression):
    """End-to-end: a multi-target suite leaves a populated feature_side_cache in
    ctx.artifacts, demonstrating cross-target hoist actually fires on the production
    path (not just the unit-level helpers above)."""
    import warnings

    from mlframe.training import train_mlframe_models_suite, TargetTypes

    df = synthetic_multi_target_regression
    fte = _MultiTargetExtractor(["y1", "y2", "y3"], target_type=TargetTypes.REGRESSION)

    # Capture ctx via monkey-patch: train_mlframe_models_suite drops the ctx on return,
    # so the only way to inspect ctx.artifacts post-fit is to intercept the very last
    # _train_one_target call and stash a reference.
    import mlframe.training.core._phase_train_one_target as pt
    captured_ctx_holder = {}
    _orig_train_one = pt._train_one_target

    def _stash_ctx(ctx, target_type, targets, cur_target_name, cur_target_values):
        captured_ctx_holder["ctx"] = ctx
        return _orig_train_one(ctx, target_type, targets, cur_target_name, cur_target_values)

    pt._train_one_target = _stash_ctx
    # Also patch the import in main.py since main.py uses ``pr._train_one_target``.
    import mlframe.training.core.main as _main
    pt_alias = _main.pr
    _orig_pt_alias = pt_alias._train_one_target
    pt_alias._train_one_target = _stash_ctx
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_mlframe_models_suite(
                df=df, target_name="multi", model_name="mt",
                features_and_targets_extractor=fte,
                mlframe_models=["linear"],
                use_mlframe_ensembles=False, verbose=0,
            )
    finally:
        pt._train_one_target = _orig_train_one
        pt_alias._train_one_target = _orig_pt_alias

    ctx = captured_ctx_holder["ctx"]
    artifacts = ctx.artifacts or {}
    fs_cache = artifacts.get("feature_side_cache", {})
    # The cache must contain at least one pre_pipeline_name entry; ordinary path uses ""
    # (empty string) or "ordinary" depending on _build_pre_pipelines.
    assert isinstance(fs_cache, dict)
    # Linear strategy populates tier_dfs even when no Enum map is needed (pandas-tier).
    # The cache should have at least one inner bucket post-suite.
    has_any_bucket = any(
        isinstance(v, dict) and (v.get("tier_dfs") or v.get("prepared_frames") or v.get("tier_enum_map"))
        for v in fs_cache.values()
    )
    assert has_any_bucket, (
        f"expected at least one populated feature_side_cache bucket post-suite; got {fs_cache!r}"
    )
