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
    """Early return is four tuple with frames passed through."""
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
    assert (
        isinstance(out, tuple) and len(out) == 4
    ), f"_run_target_distribution_analyzer must return a 4-tuple (hyperparams_config, train_df, val_df, test_df); got {out!r}"
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


def test_auto_drop_after_feature_analyzer_drops_columns_on_pandas_and_polars():
    """``_maybe_auto_drop_after_feature_analyzer`` must actually drop the analyzer's flagged
    columns on both pandas and polars frames.

    BUG FOUND AND FIXED (2026-08-02, incidental to a profiling cycle): ``getattr(train_df,
    "columns", []) or []`` forced ``bool()`` on the ``or`` operator's left side, and pandas
    raises ``ValueError: The truth value of a Index is ambiguous`` for any multi-column
    ``DataFrame.columns`` -- so auto-drop crashed (caught by the caller's best-effort
    ``except``, silently falling back to the full column set) on EVERY pandas ``train_df``.
    A second, independent bug in the same function: the drop helper called ``df.drop(present)``
    -- pandas' ``.drop()`` defaults to axis=0 (drops ROWS by index label, not columns), so even
    past the first bug this raised ``KeyError`` (never caught by the ``except TypeError`` that
    was meant to catch it) instead of actually dropping columns. Pins both fixes: pandas
    now drops via ``columns=``, polars falls back to the positional form its ``.drop()``
    actually accepts."""
    import pandas as pd
    import polars as pl

    from mlframe.training.core._main_train_suite_target_distribution import (
        _maybe_auto_drop_after_feature_analyzer,
    )

    class _FakeReport:
        """Minimal stand-in for the feature-distribution report: flags column 'b' as a drop candidate."""

        drop_candidates = ["b"]
        diagnostics: dict = {}

    class _BehaviorConfig:
        """Minimal stand-in for behavior_config: opts into candidate-list auto-drop, near-dup drop disabled."""

        auto_drop_distribution_analyzer_candidates = True
        auto_drop_near_duplicate_threshold = 2.0  # > 1.0 -> the near-duplicate branch is a no-op

    for make_df in (
        lambda: pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}),
        lambda: pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}),
    ):
        df = make_df()
        train_df, val_df, test_df, dropped = _maybe_auto_drop_after_feature_analyzer(
            fd_report=_FakeReport(),
            train_df=df,
            val_df=df,
            test_df=df,
            behavior_config=_BehaviorConfig(),
            metadata={},
            verbose=False,
        )
        assert dropped == ["b"], f"expected 'b' to be dropped for {type(df).__name__}; got {dropped!r}"
        for out in (train_df, val_df, test_df):
            assert list(out.columns) == ["a", "c"], f"expected columns ['a', 'c'] for {type(df).__name__}; got {list(out.columns)!r}"
