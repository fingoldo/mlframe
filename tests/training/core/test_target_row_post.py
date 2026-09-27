"""Post-loop helpers: split arguments narrowed per target, recurrent training per row scope."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.configs import TargetTypes, TrainingBehaviorConfig
from mlframe.training.core._target_row_decisions import TARGET_ROWS_METADATA_KEY
from mlframe.training.core._target_row_post import narrow_split_args, train_recurrent_by_rows
from mlframe.training.core._target_row_scope import active_rows
from mlframe.training.core._target_rows import build_target_rows
from mlframe.training.core._training_context import TrainingContext

N = 20


def _ctx():
    ctx = TrainingContext()
    ctx.behavior_config = TrainingBehaviorConfig(min_labelled_train_rows=3, min_labelled_val_rows=1, min_labelled_test_rows=1, min_labelled_calib_rows=1)
    ctx.train_idx, ctx.val_idx, ctx.test_idx = np.arange(10), np.arange(10, 15), np.arange(15, 20)
    ctx.filtered_train_idx, ctx.filtered_val_idx = ctx.train_idx, ctx.val_idx
    return ctx


def test_split_arguments_are_narrowed_frames_indices_and_sequences_together():
    mask = np.ones(N, dtype=bool)
    mask[[2, 11, 16]] = False
    ctx = _ctx()
    rows = build_target_rows(mask, {"train_idx": ctx.train_idx, "val_idx": ctx.val_idx, "test_idx": ctx.test_idx})
    frame = pd.DataFrame({"x": np.arange(N, dtype=float)})
    args = dict(
        train_idx=ctx.train_idx, val_idx=ctx.val_idx, test_idx=ctx.test_idx, filtered_train_idx=ctx.train_idx,
        train_df_pd=frame.iloc[:10], filtered_train_df=frame.iloc[:10], test_df_pd=frame.iloc[15:],
        train_sequences=[f"s{i}" for i in range(10)], other="kept",
    )
    out = narrow_split_args(rows, args)
    assert 2 not in out["train_idx"] and 2 not in out["filtered_train_idx"] and 16 not in out["test_idx"]
    assert out["train_df_pd"]["x"].tolist() == out["train_idx"].tolist() == out["filtered_train_df"]["x"].tolist()
    assert "s2" not in out["train_sequences"] and len(out["train_sequences"]) == 9
    assert out["other"] == "kept" and narrow_split_args(None, args) == args


def test_recurrent_training_runs_each_target_with_gaps_in_its_own_scope_and_skips_skipped_targets():
    ctx = _ctx()
    y_part = np.arange(N, dtype=float)
    y_part[[1, 12]] = np.nan
    targets = {TargetTypes.REGRESSION: {"full": np.arange(N, dtype=float), "part": y_part, "gone": np.full(N, np.nan)}}
    metadata = {TARGET_ROWS_METADATA_KEY: {"regression/part": {"signature": "x"}, "regression/gone": {"skipped": "too few"}}}
    calls = []

    def fake_train(*, target_by_type, models, train_idx, **kwargs):
        names = [n for named in target_by_type.values() for n in named]
        calls.append((names, np.asarray(train_idx), active_rows() is not None))
        return models

    train_recurrent_by_rows(ctx, fake_train, targets, metadata, models={}, train_idx=ctx.train_idx, val_idx=ctx.val_idx, test_idx=ctx.test_idx, ctx=ctx)
    assert [c[0] for c in calls] == [["full"], ["part"]]
    assert calls[0][1].tolist() == list(range(10)) and not calls[0][2]
    assert 1 not in calls[1][1] and calls[1][2]
