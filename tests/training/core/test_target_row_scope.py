"""Narrowing the training context to one target's labelled rows, and putting it back."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.configs import TargetTypes, TrainingBehaviorConfig
from mlframe.training.core._target_row_decisions import TARGET_ROWS_METADATA_KEY, rows_for_target, working_target
from mlframe.training.core._target_row_scope import ROW_REGISTRY, TargetRowScopeError, active_rows, check_rows_in_scope, target_row_scope
from mlframe.training.core._target_rows import build_target_rows, splits_of
from mlframe.training.core._training_context import TrainingContext

N = 20


def _ctx(od: bool = False, calib: bool = True) -> TrainingContext:
    """A context over N rows: train 0..9, val 10..13, test 14..17, calib 18..19, frames labelled by global position."""
    ctx = TrainingContext()
    ctx.behavior_config = TrainingBehaviorConfig(min_labelled_train_rows=3, min_labelled_val_rows=2, min_labelled_test_rows=2, min_labelled_calib_rows=2)
    full = pd.DataFrame({"x": np.arange(N, dtype=float)})
    ctx.train_idx, ctx.val_idx, ctx.test_idx = np.arange(10), np.arange(10, 14), np.arange(14, 18)
    ctx.train_df_pd, ctx.val_df_pd, ctx.test_df_pd = full.iloc[:10], full.iloc[10:14], full.iloc[14:18]
    ctx.train_df_polars, ctx.val_df_polars = pl.from_pandas(full.iloc[:10]), pl.from_pandas(full.iloc[10:14])
    ctx.test_df_polars = pl.from_pandas(full.iloc[14:18])
    if calib:
        ctx.calib_idx, ctx.calib_df = np.arange(18, 20), pl.from_pandas(full.iloc[18:20])
    if od:  # outlier detection dropped train row 3 and val row 11
        ctx.train_od_idx = np.arange(10) != 3
        ctx.val_od_idx = np.arange(4) != 1
        ctx.filtered_train_idx, ctx.filtered_val_idx = ctx.train_idx[ctx.train_od_idx], ctx.val_idx[ctx.val_od_idx]
        ctx.filtered_train_df, ctx.filtered_val_df = ctx.train_df_pd[ctx.train_od_idx], ctx.val_df_pd[ctx.val_od_idx]
        ctx.train_df_polars = ctx.train_df_polars.filter(pl.Series(ctx.train_od_idx))
        ctx.val_df_polars = ctx.val_df_polars.filter(pl.Series(ctx.val_od_idx))
    else:
        ctx.filtered_train_idx, ctx.filtered_val_idx = ctx.train_idx, ctx.val_idx
        ctx.filtered_train_df, ctx.filtered_val_df = ctx.train_df_pd, ctx.val_df_pd
    ctx.group_ids = np.arange(N)
    return ctx


MASK = np.ones(N, dtype=bool)
MASK[[1, 3, 5, 11, 15, 19]] = False


def test_every_context_field_is_classified():
    """A new TrainingContext field must say how a narrowed scope treats it, or a consumer reads it un-narrowed."""
    fields = {f.name for f in dataclasses.fields(TrainingContext)}
    assert fields - set(ROW_REGISTRY) == set(), "classify these fields in _target_row_scope.ROW_REGISTRY"
    assert set(ROW_REGISTRY) - fields == set(), "ROW_REGISTRY names fields TrainingContext no longer has"


@pytest.mark.parametrize("od", [False, True])
def test_frames_and_indices_are_narrowed_together(od):
    ctx = _ctx(od=od)
    rows = build_target_rows(MASK, splits_of(ctx))
    with target_row_scope(ctx, rows):
        assert active_rows() is rows
        np.testing.assert_array_equal(ctx.train_idx, [0, 2, 4, 6, 7, 8, 9])
        np.testing.assert_array_equal(ctx.filtered_train_idx, [0, 2, 4, 6, 7, 8, 9])  # OD dropped 3, unlabelled anyway
        np.testing.assert_array_equal(ctx.val_idx, [10, 12, 13])
        np.testing.assert_array_equal(ctx.filtered_val_idx, [10, 12, 13])  # OD dropped 11, unlabelled anyway
        for frame, idx in ((ctx.train_df_pd, ctx.train_idx), (ctx.filtered_train_df, ctx.filtered_train_idx), (ctx.val_df_pd, ctx.val_idx),
                           (ctx.test_df_pd, ctx.test_idx)):
            np.testing.assert_array_equal(frame["x"].to_numpy(), idx)
        np.testing.assert_array_equal(ctx.train_df_polars["x"].to_numpy(), ctx.filtered_train_idx)
        np.testing.assert_array_equal(ctx.calib_df["x"].to_numpy(), ctx.calib_idx)
        if od:
            assert ctx.train_od_idx.size == ctx.train_idx.size and ctx.train_od_idx.all()
        else:
            assert ctx.filtered_train_df is ctx.train_df_pd, "a frame that is one object for two fields stays one object"
        assert ctx.group_ids.size == N, "full-length arrays are indexed by the narrowed indices, not sliced"


def test_everything_is_put_back_on_exit_even_after_an_exception():
    ctx = _ctx(od=True)
    before = {f.name: getattr(ctx, f.name) for f in dataclasses.fields(ctx)}
    rows = build_target_rows(MASK, splits_of(ctx))
    with pytest.raises(RuntimeError, match="boom"):
        with target_row_scope(ctx, rows):
            ctx.models["regression"] = {"y": "model"}  # merge-back: kept
            ctx.pipeline = "fitted on narrowed rows"  # shared: restored
            ctx._pandas_view_cache[1] = "view"  # scoped cache: restored
            raise RuntimeError("boom")
    assert active_rows() is None
    assert before
    for name, value in before.items():
        if name == "models":
            continue
        assert getattr(ctx, name) is value, name
    assert ctx.models == {"regression": {"y": "model"}}


def test_a_frame_the_body_released_stays_released():
    ctx = _ctx()
    with target_row_scope(ctx, build_target_rows(MASK, splits_of(ctx))):
        ctx.train_df_polars = None  # the body frees polars frames once pandas takes over
    assert ctx.train_df_polars is None
    assert ctx.val_df_polars is not None


def test_a_fully_labelled_target_changes_nothing():
    ctx = _ctx()
    before = {f.name: getattr(ctx, f.name) for f in dataclasses.fields(ctx)}
    with target_row_scope(ctx, None):
        assert before
        for name, value in before.items():
            assert getattr(ctx, name) is value, name
        assert active_rows() is None


def test_a_frame_misaligned_with_its_split_is_refused():
    ctx = _ctx()
    ctx.val_df_pd = ctx.val_df_pd.iloc[:3]
    with pytest.raises(TargetRowScopeError, match="val_df_pd"):
        with target_row_scope(ctx, build_target_rows(MASK, splits_of(ctx))):
            pass


def test_a_fit_that_would_see_an_unlabelled_row_is_refused():
    ctx = _ctx()
    rows = build_target_rows(MASK, splits_of(ctx))
    with target_row_scope(ctx, rows):
        check_rows_in_scope({"train_idx": ctx.train_idx, "val_idx": ctx.val_idx, "train_target": np.ones(3)})
        with pytest.raises(TargetRowScopeError, match="train_idx"):
            check_rows_in_scope({"train_idx": np.arange(10)})
        with pytest.raises(TargetRowScopeError, match="val_target"):
            check_rows_in_scope({"val_target": np.array([1.0, np.nan])})
    check_rows_in_scope({"train_idx": np.arange(10)})  # outside a scope nothing is checked


def test_the_pipeline_cache_key_carries_the_rows_only_inside_a_scope():
    from mlframe.training.core._phase_train_one_target_cache_helpers import compute_model_pipeline_cache_key

    class _Strategy:
        supports_polars = False
        requires_imputation = requires_scaling = requires_encoding = False

        def feature_tier(self):
            return (False, False)

    kwargs = dict(strategy=_Strategy(), pre_pipeline_name="", cat_features=[], text_features=[], embedding_features=[], train_df_polars=None,
                  cur_target_name="y", current_train_target=None, _compute_pipeline_cache_key=lambda *a, **k: "base")
    assert compute_model_pipeline_cache_key(**kwargs) == "base"
    ctx = _ctx()
    rows = build_target_rows(MASK, splits_of(ctx))
    with target_row_scope(ctx, rows):
        assert compute_model_pipeline_cache_key(**kwargs) == f"base_rows{rows.signature[:8]}"


# ---- per-target decisions ------------------------------------------------------------------------------------------


def _decide(ctx, values, target_type=TargetTypes.REGRESSION):
    metadata: dict = {}
    rows, working, train_it = rows_for_target(ctx, target_type, "y", values, metadata, {})
    return rows, working, train_it, metadata[TARGET_ROWS_METADATA_KEY]["regression/y" if target_type == TargetTypes.REGRESSION else f"{target_type}/y"]


def test_too_few_labelled_train_rows_skips_the_target():
    y = np.full(N, np.nan)
    y[[0, 1, 10, 11, 14, 15]] = 1.0
    rows, _, train_it, record = _decide(_ctx(), y)
    assert not train_it and rows is None and "min_labelled_train_rows" in record["skipped"]


def test_too_few_val_rows_trains_without_val_and_too_few_calib_rows_without_calib():
    y = np.arange(N, dtype=float)
    y[[11, 12, 13, 19]] = np.nan
    rows, _, train_it, _record = _decide(_ctx(), y)
    assert train_it and rows.idx["val_idx"].size == 0 and rows.idx["filtered_val_idx"].size == 0
    assert rows.dropped == {"val_idx", "filtered_val_idx", "calib_idx"}
    ctx = _ctx()
    with target_row_scope(ctx, rows):
        assert ctx.val_idx.size == 0 and ctx.val_df_pd is None, "no val is the suite's val_size=0 shape: an empty index, no frame"
        assert ctx.calib_idx is None and ctx.calib_df is None, "no calib is None, as when calib_size is 0"


def test_too_few_test_rows_marks_the_metrics_low_n():
    y = np.arange(N, dtype=float)
    y[[14, 15, 16]] = np.nan
    rows, _, train_it, record = _decide(_ctx(), y)
    assert train_it and record["low_n"] is True and rows.idx["test_idx"].size == 1


def test_a_classification_target_left_with_one_class_is_skipped():
    y = np.zeros(N)
    y[[1, 2, 3]] = np.nan
    y[15] = 1.0  # the only positive is in test
    _rows, _, train_it, record = _decide(_ctx(), y, TargetTypes.BINARY_CLASSIFICATION)
    assert not train_it and "class" in record["skipped"]


def test_rows_of_a_class_train_never_saw_are_left_out_of_scoring():
    y = np.tile([0.0, 1.0], N // 2)
    y[5] = np.nan
    y[[15, 16]] = 2.0  # class 2 only in test
    rows, working, train_it, record = _decide(_ctx(), y, TargetTypes.MULTICLASS_CLASSIFICATION)
    assert train_it and record["rows_of_classes_absent_from_train"] == 2
    assert not np.isin(rows.idx["test_idx"], [15, 16]).any()
    assert working.dtype.kind == "i" and not np.isnan(working.astype(float)).any()


def test_the_working_classification_target_is_integer_and_regression_keeps_nan():
    y = np.array([0.0, 1.0, np.nan, 1.0])
    mask = ~np.isnan(y)
    cls = working_target(y, mask, TargetTypes.BINARY_CLASSIFICATION, "y")
    assert cls.dtype == np.int8 and cls[mask].tolist() == [0, 1, 1]
    assert working_target(y, mask, TargetTypes.REGRESSION, "y") is y


def test_a_split_left_with_no_labelled_row_has_no_frame_and_comes_back_whole():
    """A right-censored target with no labelled test row sees test as the suite's test_size=0 does; its frames return."""
    ctx = _ctx()
    mask = np.ones(N, dtype=bool)
    mask[14:18] = False
    full_test = ctx.test_df_pd
    with target_row_scope(ctx, build_target_rows(mask, splits_of(ctx))):
        assert ctx.test_idx.size == 0 and ctx.test_df_pd is None and ctx.test_df_polars is None
    assert ctx.test_df_pd is full_test and ctx.test_df_polars is not None


def test_split_details_and_the_record_describe_the_labelled_rows_and_flag_censored_holdouts():
    """A target whose newest outcomes are not known yet: its test window and counts are its own, and it is flagged."""
    ctx = _ctx()
    ctx.timestamps = pd.date_range("2024-01-01", periods=N, freq="D").to_numpy()
    y = np.arange(N, dtype=float)
    y[15:18] = np.nan  # 1 of 4 test rows labelled, all of train
    rows, _, train_it, record = _decide(ctx, y)
    assert train_it and record["time_window"]["test_idx"] == "2024-01-15/2024-01-15"
    assert any("censored" in note for note in record["notes"])
    with target_row_scope(ctx, rows):
        assert ctx.test_details == "2024-01-15/2024-01-15 labelled 1/4"
