"""Sensor: SimpleFeaturesAndTargetsExtractor.transform() must reset per-call
state so a reused instance doesn't leak state from suite call N into N+1.

Pre-fix shape (cross-suite metadata corruption wave P0 #1 + #2):
- columns_to_drop is a set initialised once at __init__. build_targets does
  ``self.columns_to_drop.add(col)`` for every target column. A caller pattern
  ``fte = SimpleFeaturesAndTargetsExtractor(...); for fold in folds: ...``
  carried N-1's target columns into call N -- if a target from call 1 was a
  FEATURE column in call 2 (different target_name), it silently dropped from
  features in call 2.

- ftextractor_emitted_columns is a dict initialised once. add_features does
  ``self.ftextractor_emitted_columns[self.ts_field] = derived`` on every call.
  If the caller swaps ts_field between suite calls, stale entries from the
  previous ts_field persist into the next call's metadata, leaking through to
  predict-side ``skip these source columns`` decisions.

Post-fix: transform() resets both at entry. The user-supplied initial
columns_to_drop is captured in a snapshot on first transform() entry and re-applied
each call so caller-supplied drops persist while build_targets's per-call adds don't.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor


def _make_df(extra_cols=None):
    rng = np.random.default_rng(42)
    cols = {
        "x0": rng.normal(size=100),
        "x1": rng.normal(size=100),
        "y_a": rng.integers(0, 2, size=100),
        "y_b": rng.integers(0, 2, size=100),
    }
    if extra_cols:
        for k, v in extra_cols.items():
            cols[k] = v
    return pd.DataFrame(cols)


def test_columns_to_drop_reset_between_transform_calls():
    """REGRESSION: second transform() call must NOT see y_a in columns_to_drop
    from the first call when the second is configured to target only y_b."""
    fte = SimpleFeaturesAndTargetsExtractor(
        classification_targets=["y_a"],
    )
    df = _make_df()
    _ = fte.transform(df)
    assert "y_a" in fte.columns_to_drop

    # Reconfigure for a different target. (Plain attr-set is the public-API pattern
    # the agent finding cites; the FTE doesn't expose a setter for these.)
    fte.classification_targets = ["y_b"]
    _ = fte.transform(df)
    # columns_to_drop now must reflect ONLY this call's target, not the prior y_a.
    assert "y_b" in fte.columns_to_drop
    assert "y_a" not in fte.columns_to_drop, "columns_to_drop leaked 'y_a' from prior transform() call. Pre-fix bug regression."


def test_columns_to_drop_initial_user_supplied_preserved():
    """A user-supplied columns_to_drop must persist across transform() calls
    (it's the caller's intent, not per-call state)."""
    fte = SimpleFeaturesAndTargetsExtractor(
        classification_targets=["y_a"],
        columns_to_drop={"x0"},  # user wants x0 always dropped
    )
    df = _make_df()
    _ = fte.transform(df)
    assert "x0" in fte.columns_to_drop
    assert "y_a" in fte.columns_to_drop  # added by build_targets

    # Second call must still have x0 (user-supplied) AND y_a (this call's target).
    _ = fte.transform(df)
    assert "x0" in fte.columns_to_drop, "user-supplied initial drop column lost on second transform"
    assert "y_a" in fte.columns_to_drop


def test_ftextractor_emitted_columns_reset_between_calls():
    """ftextractor_emitted_columns must NOT carry stale entries from a prior call."""
    import numpy as _np

    df = _make_df({"ts1": pd.date_range("2024-01-01", periods=100, freq="D")})
    fte = SimpleFeaturesAndTargetsExtractor(
        classification_targets=["y_a"],
        ts_field="ts1",
        datetime_features={"year": _np.int16},
    )
    _ = fte.transform(df)
    assert "ts1" in fte.ftextractor_emitted_columns

    # Now swap ts_field to a different column AND change datetime_features off.
    df2 = _make_df()
    fte.ts_field = None
    fte.datetime_features = None
    _ = fte.transform(df2)
    # The 'ts1' entry from the prior call must be gone.
    assert "ts1" not in fte.ftextractor_emitted_columns, (
        f"stale ts1 entry leaked from prior call: {dict(fte.ftextractor_emitted_columns)}. Pre-fix bug regression."
    )


def test_columns_to_drop_initial_none_handled():
    """Edge: caller passes columns_to_drop=None (the default). Must NOT crash;
    snapshot becomes empty set and only this-call adds appear after transform()."""
    fte = SimpleFeaturesAndTargetsExtractor(
        classification_targets=["y_a"],
        columns_to_drop=None,  # explicit None
    )
    df = _make_df()
    _ = fte.transform(df)
    # Snapshot must exist (initialized to empty set when input was None).
    assert hasattr(fte, "_initial_columns_to_drop_snapshot")
    assert fte._initial_columns_to_drop_snapshot == set()
    # After transform, only this-call's target should be in the set.
    assert fte.columns_to_drop == {"y_a"}


def test_transform_idempotent_across_three_calls():
    """Sanity: three identical transform() calls must produce identical
    columns_to_drop + ftextractor_emitted_columns shape (no monotonic growth)."""
    df = _make_df()
    fte = SimpleFeaturesAndTargetsExtractor(classification_targets=["y_a"])
    _ = fte.transform(df)
    snap1 = (set(fte.columns_to_drop), dict(fte.ftextractor_emitted_columns))
    _ = fte.transform(df)
    snap2 = (set(fte.columns_to_drop), dict(fte.ftextractor_emitted_columns))
    _ = fte.transform(df)
    snap3 = (set(fte.columns_to_drop), dict(fte.ftextractor_emitted_columns))
    assert snap1 == snap2 == snap3, f"transform() state not idempotent across three calls:\n  call1: {snap1}\n  call2: {snap2}\n  call3: {snap3}"
