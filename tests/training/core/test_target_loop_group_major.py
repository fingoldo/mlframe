"""Targets with missing labels train group by group: one narrowing per distinct set of rows, outputs in the given order."""

from __future__ import annotations

import types

import numpy as np

from mlframe.training.configs import TargetTypes, TrainingBehaviorConfig
from mlframe.training.core import _main_train_suite_target_loop as loop
from mlframe.training.core._target_row_scope import active_rows
from mlframe.training.core._training_context import TrainingContext

N = 40


def test_groups_train_largest_first_narrow_once_and_models_keep_the_given_order(monkeypatch):
    """Groups train largest first narrow once and models keep the given order."""
    ctx = TrainingContext()
    ctx.behavior_config = TrainingBehaviorConfig(min_labelled_train_rows=3, min_labelled_val_rows=1, min_labelled_test_rows=1, min_labelled_calib_rows=1)
    ctx.train_idx, ctx.val_idx, ctx.test_idx = np.arange(30), np.arange(30, 35), np.arange(35, 40)
    ctx.filtered_train_idx, ctx.filtered_val_idx = ctx.train_idx, ctx.val_idx
    ctx._pre_screen_done = True
    few = np.arange(N, dtype=float)
    few[::2] = np.nan  # half the rows
    many = np.arange(N, dtype=float)
    many[::10] = np.nan  # a tenth of the rows
    targets = {TargetTypes.REGRESSION: {"a": np.arange(N, dtype=float), "b": few, "c": many, "d": few * 2}}

    entries = []
    real_scope = loop.target_row_scope

    def counting_scope(c, rows):
        """Record each scope entry's rows, then delegate."""
        entries.append(rows)
        return real_scope(c, rows)

    monkeypatch.setattr(loop, "target_row_scope", counting_scope)
    trained = []

    def train_one(c, tt, targets_, name, values):
        """Record the model name and whether a row scope is active, and store the model."""
        trained.append((name, active_rows() is not None))
        c.models.setdefault(tt, {})[name] = name

    loop.train_every_target(ctx, targets, {}, types.SimpleNamespace(_train_one_target=train_one))
    assert [name for name, _ in trained] == ["a", "c", "b", "d"]
    assert [scoped for _, scoped in trained] == [False, True, True, True]
    assert len(entries) == 3 and entries[0] is None, "one scope for the fully labelled targets, one per group of rows"
    assert list(ctx.models[TargetTypes.REGRESSION]) == ["a", "b", "c", "d"]
