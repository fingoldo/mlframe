"""Regression: ``_run_target_distribution_analyzer`` must return a 4-tuple on
EVERY path.

The caller in ``_main_train_suite`` unpacks the result as
``hyperparams_config, train_df, val_df, test_df = _run_target_distribution_analyzer(...)``.
The function's main path returns that 4-tuple, but the disabled / empty-target
early-return guard returned the bare ``hyperparams_config`` (stale from before
the function grew to also thread the frames), so any combo with
``enable_target_distribution_analyzer=False`` or an empty ``target_by_type``
raised ``ValueError: too many values to unpack (expected 4)`` in
``train_mlframe_models_suite`` -- a broad regression surfaced across many
non-MRMR fuzz combos (c0004 / c0031 / c0035 / c0047 / c0088 / c0099 / ...).
"""

from __future__ import annotations


def test_early_return_is_four_tuple_with_frames_passed_through():
    from mlframe.training.core._main_train_suite_target_distribution import (
        _run_target_distribution_analyzer,
    )

    sentinel = object()
    out = _run_target_distribution_analyzer(
        enable_target_distribution_analyzer=False,  # forces the early-return guard
        target_by_type={},
        train_idx=None,
        group_ids=None,
        timestamps=None,
        train_df="TRAIN",
        verbose=False,
        metadata={},
        hyperparams_config=sentinel,
        ctx=None,
        val_df="VAL",
        test_df="TEST",
    )
    assert isinstance(out, tuple) and len(out) == 4, (
        f"_run_target_distribution_analyzer must return a 4-tuple (hyperparams_config, train_df, val_df, test_df); got {out!r}"
    )
    hp, tr, va, te = out
    assert hp is sentinel, "hyperparams_config must pass through unchanged"
    assert (tr, va, te) == ("TRAIN", "VAL", "TEST"), "train/val/test frames must pass through unchanged when the analyzer is off"


def test_empty_target_by_type_also_four_tuple():
    """The guard is ``not (enabled and target_by_type)`` -- an empty
    ``target_by_type`` must take the same 4-tuple early return even when the
    flag is on."""
    from mlframe.training.core._main_train_suite_target_distribution import (
        _run_target_distribution_analyzer,
    )

    out = _run_target_distribution_analyzer(
        enable_target_distribution_analyzer=True,
        target_by_type={},  # empty -> early return
        train_idx=None,
        group_ids=None,
        timestamps=None,
        train_df="TRAIN",
        verbose=False,
        metadata={},
        hyperparams_config="HP",
        ctx=None,
        val_df="VAL",
        test_df="TEST",
    )
    assert isinstance(out, tuple) and len(out) == 4
    assert out == ("HP", "TRAIN", "VAL", "TEST")
