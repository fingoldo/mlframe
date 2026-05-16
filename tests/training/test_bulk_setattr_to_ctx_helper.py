"""Tests for ``_bulk_setattr_to_ctx`` helper used by ``core/main.py`` to mirror local variables onto the suite TrainingContext.

The helper exists to catch partial-migration slot bugs (e.g. the prior ``train_df_pandas_pre`` miss) at call time
with a clear KeyError rather than as an obscure ``AttributeError: 'NoneType' has no attribute 'foo'`` deeper in
the suite. These tests pin the assign-all and fail-loud-on-missing contracts plus the natural last-write-wins
re-assignment behaviour.
"""

from __future__ import annotations

import pytest

from mlframe.training.core._misc_helpers import _bulk_setattr_to_ctx


class _DummyCtx:
    """Bare attribute bag standing in for TrainingContext; setattr-friendly, no slot restrictions."""


def test_bulk_setattr_assigns_all_names():
    ctx = _DummyCtx()
    names = ("train_idx", "val_idx", "train_df")
    values = {"train_idx": [0, 1, 2], "val_idx": [3, 4], "train_df": "frame-A", "unused": 99}

    _bulk_setattr_to_ctx(ctx, names, values)

    assert ctx.train_idx == [0, 1, 2]
    assert ctx.val_idx == [3, 4]
    assert ctx.train_df == "frame-A"
    # Keys not listed in `names` must not leak onto ctx; otherwise the helper would silently smuggle
    # unrelated locals (e.g. loop counters, scratch vars) into the suite-wide context.
    assert not hasattr(ctx, "unused")


def test_bulk_setattr_raises_keyerror_on_missing_name():
    ctx = _DummyCtx()
    names = ("train_idx", "val_idx", "train_df_pandas_pre")
    # Deliberately omit ``train_df_pandas_pre`` to simulate the partial-migration slot bug the helper guards.
    values = {"train_idx": [0, 1, 2], "val_idx": [3, 4]}

    with pytest.raises(KeyError) as excinfo:
        _bulk_setattr_to_ctx(ctx, names, values)

    # Missing name must appear in the error message so the maintainer can grep straight to the slot
    # that the calling phase forgot to bind, instead of debugging a downstream AttributeError.
    assert "train_df_pandas_pre" in str(excinfo.value)
    # And ctx must NOT be partially populated when the call fails -- otherwise a half-mirrored ctx
    # would mask the bug at the next read site.
    assert not hasattr(ctx, "train_idx")
    assert not hasattr(ctx, "val_idx")


def test_bulk_setattr_idempotent_on_reassignment():
    ctx = _DummyCtx()
    names = ("train_df", "val_df")
    first = {"train_df": "first-train", "val_df": "first-val"}
    second = {"train_df": "second-train", "val_df": "second-val"}

    _bulk_setattr_to_ctx(ctx, names, first)
    assert ctx.train_df == "first-train"
    assert ctx.val_df == "first-val"

    _bulk_setattr_to_ctx(ctx, names, second)
    # Last-write-wins: second call overwrites first, no surprise residue or merge attempt.
    assert ctx.train_df == "second-train"
    assert ctx.val_df == "second-val"
