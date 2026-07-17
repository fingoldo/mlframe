"""Wave-4 pre-screen + seed-defaulting tests (A1-05, A1-11, A1-03).

A1-05: apply_drops across train/val/test mirrors is atomic (all or none) -- a per-mirror failure must NOT leave
       some mirrors dropped and others not (schema drift).
A1-11: group-aware split with an empty protected set SKIPS the pre-screen rather than risking the group key.
A1-03: MRMR/RFECV selector seeds default from the split seed when the operator didn't pin one.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd

from mlframe.training.core._phase_train_one_target_pre_screen import _maybe_run_unsupervised_pre_screen


class _FSCfg:
    """Minimal feature-selection config stub for the pre-screen helper under test."""

    pre_screen_unsupervised = True
    pre_screen_variance_threshold = 0.0
    pre_screen_null_fraction_threshold = 0.99


class _SplitCfg:
    """Minimal split-config stub exposing the group-awareness fields the helper reads."""

    def __init__(self, use_groups=False, group_field=None, random_seed=42):
        self.use_groups = use_groups
        self.group_field = group_field
        self.random_seed = random_seed


def _make_ctx(train, use_groups=False, group_field=None, protected_group=True):
    """Minimal training-context stub carrying the mirrors/config the pre-screen helper reads and mutates."""
    ctx = types.SimpleNamespace()
    ctx.feature_selection_config = _FSCfg()
    ctx._pre_screen_done = False
    ctx._pre_screen_dropped_cols = None
    ctx.filtered_train_df = train
    ctx.filtered_val_df = train.copy()
    ctx.train_df_pd = train.copy()
    ctx.val_df_pd = train.copy()
    ctx.test_df_pd = train.copy()
    ctx.train_df_polars = None
    ctx.val_df_polars = None
    ctx.test_df_polars = None
    ctx.cat_features = None
    ctx.text_features = None
    ctx.embedding_features = None
    ctx.group_id_col = group_field if (use_groups and protected_group) else None
    ctx.ts_field = None
    ctx.extractor = None
    ctx.split_config = _SplitCfg(use_groups=use_groups, group_field=group_field)
    ctx.target_by_type = {}
    ctx.metadata = {}
    ctx.verbose = 0
    return ctx


def test_a1_11_group_split_empty_protected_skips_prescreen():
    """A group-aware split with an empty protected set must SKIP the pre-screen (drop nothing)."""
    rng = np.random.RandomState(0)
    train = pd.DataFrame({"gid": np.arange(200).astype(str), "a": rng.randn(200), "const": np.zeros(200)})
    # use_groups=True but protected set empty (no group_id_col / extractor / split_config.group_field).
    ctx = _make_ctx(train, use_groups=True, group_field=None, protected_group=False)
    _maybe_run_unsupervised_pre_screen(ctx, {})
    assert ctx._pre_screen_done is True
    assert ctx._pre_screen_dropped_cols == []  # SKIPPED -> nothing dropped
    # No mirror had its constant column dropped.
    assert "const" in ctx.filtered_train_df.columns
    assert "const" in ctx.test_df_pd.columns


def test_a1_11_group_split_with_protected_runs_prescreen():
    """A group-aware split WITH a protected group column runs the pre-screen and protects that column."""
    rng = np.random.RandomState(0)
    train = pd.DataFrame({"gid": np.arange(200).astype(str), "a": rng.randn(200), "const": np.zeros(200)})
    ctx = _make_ctx(train, use_groups=True, group_field="gid", protected_group=True)
    _maybe_run_unsupervised_pre_screen(ctx, {})
    assert ctx._pre_screen_done is True
    # const (variance 0) should be dropped; gid protected.
    assert "const" not in ctx.filtered_train_df.columns
    assert "gid" in ctx.filtered_train_df.columns


def test_a1_05_apply_drops_atomic_on_mirror_failure(monkeypatch):
    """A failure applying drops to one train/val/test mirror must leave ALL mirrors untouched (atomic)."""
    rng = np.random.RandomState(0)
    train = pd.DataFrame({"a": rng.randn(200), "const": np.zeros(200)})
    ctx = _make_ctx(train, use_groups=False)

    import mlframe.feature_selection.pre_screen as ps_mod

    real_apply = ps_mod.apply_drops
    calls = {"n": 0}

    def _flaky_apply(frame, drops):
        """Delegate to the real apply_drops, except raise on the 3rd mirror to simulate a partial failure."""
        calls["n"] += 1
        if calls["n"] == 3:  # fail on the third mirror
            raise RuntimeError("simulated mirror drop failure")
        return real_apply(frame, drops)

    monkeypatch.setattr(ps_mod, "apply_drops", _flaky_apply)
    _maybe_run_unsupervised_pre_screen(ctx, {})
    # Atomic: because one mirror's apply_drops raised, NO mirror gets the drop applied (outer except skips).
    assert ctx._pre_screen_done is True
    assert ctx._pre_screen_dropped_cols == []
    for attr in ("filtered_train_df", "filtered_val_df", "train_df_pd", "val_df_pd", "test_df_pd"):
        f = getattr(ctx, attr)
        assert "const" in f.columns, f"{attr} should NOT be partially dropped (schema-drift guard)"


def test_a1_03_mrmr_seed_defaults_from_split_seed():
    """MRMR's random_seed defaults from the split's fs_random_seed when the operator didn't pin one."""
    from mlframe.training.core._setup_helpers_pre_pipelines import _build_pre_pipelines

    pps, _names = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=[],
        rfecv_models_params={},
        use_mrmr_fs=True,
        mrmr_kwargs={},
        fs_random_seed=123,
    )
    mrmr = next(p for p in pps if p is not None)
    assert mrmr.random_seed == 123


def test_a1_03_mrmr_explicit_seed_wins_over_split_seed():
    """An explicit MRMR random_seed in mrmr_kwargs wins over the split's fs_random_seed default."""
    from mlframe.training.core._setup_helpers_pre_pipelines import _build_pre_pipelines

    pps, _names = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=[],
        rfecv_models_params={},
        use_mrmr_fs=True,
        mrmr_kwargs={"random_seed": 7},
        fs_random_seed=123,
    )
    mrmr = next(p for p in pps if p is not None)
    assert mrmr.random_seed == 7


def test_a1_01_group_split_leaves_mrmr_strict_groups_off_by_default():
    """A group-aware split does NOT force ``strict_groups=True`` on MRMR (documented, deliberate:
    MRMR's MI estimator is group-naive, so forcing the hard NotImplementedError-on-mismatch would
    abort the whole suite over a group-naive RANKING honesty caveat -- disproportionate; MRMR already
    surfaces this via a UserWarning + ``groups_ignored_`` on fit). An operator wanting the hard stop
    opts in explicitly via ``mrmr_kwargs={"strict_groups": True}``, covered by the sibling test below."""
    from mlframe.training.core._setup_helpers_pre_pipelines import _build_pre_pipelines

    pps, _names = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=[],
        rfecv_models_params={},
        use_mrmr_fs=True,
        mrmr_kwargs={},
        fs_use_groups=True,
    )
    mrmr = next(p for p in pps if p is not None)
    assert mrmr.strict_groups is False


def test_a1_01_group_split_honors_explicit_mrmr_strict_groups_opt_in():
    """An operator can still opt into the hard stop explicitly via ``mrmr_kwargs``."""
    from mlframe.training.core._setup_helpers_pre_pipelines import _build_pre_pipelines

    pps, _names = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=[],
        rfecv_models_params={},
        use_mrmr_fs=True,
        mrmr_kwargs={"strict_groups": True},
        fs_use_groups=True,
    )
    mrmr = next(p for p in pps if p is not None)
    assert mrmr.strict_groups is True
